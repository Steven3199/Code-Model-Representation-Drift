import itertools
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5Config,
    T5EncoderModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


START_TIME = time.time()


# ================= 1. Config =================
MODEL_TYPE = "RoBERTa"  # 可选: "codebert", "GraphCodeBERT", "codeT5"
SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

# Set RUN_ALL_MODELS=True to run continual-learning evaluation for all four models in order.
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]

DATA_DIR = "/opt/data/private/codes/Java250/Java250/token/pre-trained"
SAVE_DIR = "/opt/data/private/codes/Java250/Java250/token/RQ4_cl_results"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASET_NAME = "Java250"
RQ1_MODEL_DIR = os.path.dirname(DATA_DIR)

NUM_LABELS = 250

ID_TRAIN_FILE = "train.jsonl"
ID_TEST_FILE = "id_test.jsonl"
OOD_TRAIN_FILE = "ood_train.jsonl"
OOD_TEST_FILE = "ood_test_new.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 8
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

TOP_K = 5
SEED = 42
CLASSIFIER_HEAD_TYPE = "auto"  # auto / single / mlp_seq / mlp_named
ENCODER_POOLING = "auto"  # auto / cls / mean
# 与刚刚 RQ1 自适应分类代码保持一致：auto 时 CodeT5 使用 mean pooling，CodeBERT/GraphCodeBERT 使用 CLS。
# 如果你的旧 RQ1 CodeT5 是 first-token 训练的，可以手动改为 "cls"。
ENCODER_POOLING = "auto"  # auto / cls / mean

# Naive fine-tuning baseline is not included in this script because it has already been run separately.
RUN_NAIVE_FT = False

# Continual-learning hyperparameters.
# EWC 在之前的参数搜索中对结果几乎没有影响，因此这里只保留一个代表性设置。
# 选择 10.0 是因为它是原搜索空间中最小的 EWC 正则强度，避免引入过强约束。
EWC_LAMBDAS = [10.0]

# Replay 使用前一轮实验中 OOD Macro-F1 最优的配置：
# replay_lambda=0.5, replay_batch_size=4, replay_buffer_size=1000.
REPLAY_CONFIGS = [
    {"lambda": 0.5, "batch_size": 4},
]

# Replay buffer sampled from ID-train / previous-task training data.
REPLAY_BUFFER_SIZES = [1000]
REPLAY_BUFFER_SEED = SEED

ALL_MODELS_METRICS_FILE = os.path.join(SAVE_DIR, "rq4_cl_best_replay_ewc10_metrics_all_models.json")


def canonical_model_type(model_type):
    aliases = {
        "codebert": "codebert",
        "code-bert": "codebert",
        "graphcodebert": "GraphCodeBERT",
        "graph-codebert": "GraphCodeBERT",
        "codet5": "codeT5",
        "code-t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
        "roberta": "RoBERTa",
        "roberta-base": "RoBERTa",
    }
    key = str(model_type).strip()
    if key in SUPPORTED_MODEL_TYPES:
        return key
    lower = key.lower()
    if lower in aliases:
        return aliases[lower]
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {SUPPORTED_MODEL_TYPES}")


def resolve_model_path(model_type):
    return MODEL_PATHS[canonical_model_type(model_type)]


def build_rq1_best_model_path(model_type):
    model_type = canonical_model_type(model_type)
    return os.path.join(RQ1_MODEL_DIR, f"best_model_{model_type}_{DATASET_NAME}.bin")


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, RQ1_BEST_MODEL, METRICS_FILE

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = resolve_model_path(MODEL_TYPE)
    RQ1_BEST_MODEL = build_rq1_best_model_path(MODEL_TYPE)
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq4_cl_best_replay_ewc10_metrics_{MODEL_TYPE}.json")


configure_model_runtime(MODEL_TYPE)


# ================= 2. Utilities =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def format_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.1f}s"


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    return checkpoint


def strip_prefix(key):
    if key.startswith("module."):
        key = key.replace("module.", "", 1)
    if key.startswith("model."):
        key = key.replace("model.", "", 1)
    return key


def infer_head_type_from_state_dict(state_dict):
    """
    自动判断 RQ1 checkpoint 的分类头结构。

    注意：
    - 新版 RQ1 自适应代码保存的是 single head: classifier.weight/bias。
    - 旧版 ClassifierHead 可能保存 classifier.out_proj.weight/bias，但没有 dense，
      这种本质也是 single linear head，需要映射到 classifier.weight/bias。
    - 真正的 named MLP 需要 dense + out_proj。
    """
    keys = [strip_prefix(key) for key in state_dict.keys()]

    if any(key.startswith("classification_head.dense.") or key.startswith("classification_head.out_proj.") for key in keys):
        return "mlp_named"

    if any(key.startswith("classifier.0.") or key.startswith("classifier.3.") for key in keys):
        return "mlp_seq"

    has_classifier_dense = any(key.startswith("classifier.dense.") for key in keys)
    has_classifier_out_proj = any(key.startswith("classifier.out_proj.") for key in keys)
    if has_classifier_dense:
        return "mlp_named"

    if any(key in ("classifier.weight", "classifier.bias", "score.weight", "score.bias") for key in keys):
        return "single"

    # 兼容旧版 RQ1 ClassifierHead: dropout + out_proj，无 dense，本质是单层分类头。
    if has_classifier_out_proj:
        return "single"

    # 没有任何可识别分类头时先按 single 构建，后续最终分类层检查会报错。
    return "single"

def adapt_state_dict_keys(state_dict, head_type):
    new_dict = {}
    for key, value in state_dict.items():
        new_key = strip_prefix(key)

        if new_key.startswith("roberta."):
            new_key = new_key.replace("roberta.", "encoder.", 1)
        elif new_key.startswith("bert."):
            new_key = new_key.replace("bert.", "encoder.", 1)
        elif new_key.startswith("transformer."):
            new_key = new_key.replace("transformer.", "encoder.", 1)

        if new_key == "score.weight":
            new_key = "classifier.weight"
        elif new_key == "score.bias":
            new_key = "classifier.bias"

        if head_type == "single":
            if new_key.startswith("classifier.out_proj."):
                new_key = new_key.replace("classifier.out_proj.", "classifier.", 1)
            elif new_key.startswith("classifier.3."):
                new_key = new_key.replace("classifier.3.", "classifier.", 1)
        elif head_type == "mlp_seq":
            if new_key.startswith("classifier.dense."):
                new_key = new_key.replace("classifier.dense.", "classifier.0.", 1)
            elif new_key.startswith("classifier.out_proj."):
                new_key = new_key.replace("classifier.out_proj.", "classifier.3.", 1)
        elif head_type == "mlp_named":
            if new_key.startswith("classifier.0."):
                new_key = new_key.replace("classifier.0.", "classifier.dense.", 1)
            elif new_key.startswith("classifier.3."):
                new_key = new_key.replace("classifier.3.", "classifier.out_proj.", 1)

        new_dict[new_key] = value
    return new_dict



def filter_compatible_state_dict(model, state_dict):
    """只加载 key 存在且 shape 一致的参数，避免分类头/标签数不匹配时静默错误。"""
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model", tuple(value.shape) if hasattr(value, "shape") else None, None))
            continue
        if hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append((key, "shape_mismatch", tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible[key] = value
    return compatible, skipped


def get_required_final_classifier_keys(head_type):
    """强制确保最终分类输出层从 RQ1 checkpoint 完整加载。"""
    if head_type == "single":
        return ["classifier.weight", "classifier.bias"]
    if head_type == "mlp_named":
        return ["classifier.out_proj.weight", "classifier.out_proj.bias"]
    if head_type == "mlp_seq":
        return ["classifier.3.weight", "classifier.3.bias"]
    raise ValueError(f"Unsupported head_type: {head_type}")


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    model_type_lower = model_type.lower()
    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")
    # 与新版 RQ1 自适应代码一致：CodeT5 mean pooling；CodeBERT/GraphCodeBERT CLS pooling。
    return "mean" if "t5" in model_type_lower else "cls"

def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    model_type_lower = model_type.lower()

    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")

    # Match RQ1: CodeT5/PLBART use mean pooling; CodeBERT/GraphCodeBERT use CLS pooling.
    return "mean" if ("t5" in model_type_lower or "plbart" in model_type_lower) else "cls"


def get_code(js):
    return js.get("func") or js.get("code") or js.get("function") or ""


def get_label(js):
    label = js.get("label")
    if label is None:
        label = js.get("target")
    return label


# ================= 3. Model =================
class NamedMLPHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        nn.init.eye_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, features):
        # features 是已经池化后的 [batch, hidden] 代码级向量。
        x = self.dropout(features)
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class CodeEncoder(nn.Module):
    def __init__(self, model_path, num_labels, model_type, head_type="single", pooling="auto", load_pretrained=True):
        super().__init__()
        self.model_type = model_type.lower()
        self.head_type = head_type
        self.pooling_strategy = resolve_pooling_strategy(model_type, pooling)

        if "t5" in self.model_type:
            if load_pretrained:
                self.encoder = T5EncoderModel.from_pretrained(model_path)
            else:
                config = T5Config.from_pretrained(model_path)
                self.encoder = T5EncoderModel(config)
            hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            if load_pretrained:
                base_model = AutoModel.from_pretrained(model_path)
            else:
                config = AutoConfig.from_pretrained(model_path)
                base_model = AutoModel.from_config(config)
            if not hasattr(base_model, "get_encoder"):
                raise ValueError(
                    "PLBART model should expose get_encoder(); "
                    f"please check MODEL_PATH={model_path}"
                )
            self.encoder = base_model.get_encoder()
            hidden_size = getattr(base_model.config, "d_model", None)
            if hidden_size is None:
                hidden_size = getattr(base_model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot infer PLBART hidden size from model config.")
        else:
            if load_pretrained:
                self.encoder = AutoModel.from_pretrained(model_path)
            else:
                config = AutoConfig.from_pretrained(model_path)
                self.encoder = AutoModel.from_config(config)
            hidden_size = self.encoder.config.hidden_size

        if head_type == "single":
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden_size, num_labels)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        elif head_type == "mlp_named":
            self.classifier = NamedMLPHead(hidden_size, num_labels)
        else:
            self.head_type = "mlp_seq"
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_labels),
            )
            nn.init.eye_(self.classifier[0].weight)
            nn.init.zeros_(self.classifier[0].bias)
            nn.init.xavier_uniform_(self.classifier[3].weight)
            nn.init.zeros_(self.classifier[3].bias)

    def pool_encoder_output(self, outputs, attention_mask):
        if self.pooling_strategy == "mean":
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / denom
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_features = self.pool_encoder_output(outputs, attention_mask)
        if self.head_type == "single":
            logits = self.classifier(self.dropout(pooled_features))
        elif self.head_type == "mlp_named":
            logits = self.classifier(pooled_features)
        else:
            logits = self.classifier(pooled_features)
        return type("ModelOutput", (object,), {"logits": logits})


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        raise FileNotFoundError(f"RQ1 best model not found: {RQ1_BEST_MODEL}")

    raw_state = unwrap_checkpoint(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    if not isinstance(raw_state, dict):
        raise RuntimeError("Loaded RQ1 checkpoint is not a state_dict-like object.")

    head_type = infer_head_type_from_state_dict(raw_state) if CLASSIFIER_HEAD_TYPE == "auto" else CLASSIFIER_HEAD_TYPE
    if head_type not in ("single", "mlp_seq", "mlp_named"):
        raise ValueError(f"Unsupported classifier head_type: {head_type}")

    model = CodeEncoder(
        MODEL_PATH,
        NUM_LABELS,
        MODEL_TYPE,
        head_type=head_type,
        pooling=ENCODER_POOLING,
        load_pretrained=False,
    ).to(DEVICE)

    adapted_state = adapt_state_dict_keys(raw_state, head_type=head_type)
    compatible_state, skipped = filter_compatible_state_dict(model, adapted_state)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)

    matched = [key for key in compatible_state.keys() if key in model.state_dict()]
    encoder_matched = [key for key in matched if key.startswith("encoder.")]
    classifier_matched = [key for key in matched if key.startswith("classifier.")]

    print(
        f">>> Loaded RQ1 model: model_type={MODEL_TYPE}, head_type={model.head_type}, "
        f"pooling={model.pooling_strategy}, keys={len(adapted_state)}, matched={len(matched)}, "
        f"encoder_matched={len(encoder_matched)}, classifier_matched={len(classifier_matched)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
    )

    skipped_classifier = [item for item in skipped if item[0].startswith("classifier.")]
    if skipped_classifier:
        print(">>> Skipped classifier keys due to mismatch/missing:")
        for item in skipped_classifier[:30]:
            print("   ", item)

    missing_show = [key for key in missing if "pooler" not in key][:20]
    if missing_show:
        print(">>> Missing keys preview:", missing_show)
    if unexpected:
        print(">>> Unexpected keys preview:", unexpected[:20])

    print(f">>> Classifier matched keys: {len(classifier_matched)}")
    print(">>> Classifier matched key preview:", classifier_matched[:20])
    if len(encoder_matched) == 0:
        raise RuntimeError(
            "No encoder parameters were loaded from the RQ1 checkpoint. "
            "Check MODEL_TYPE, MODEL_PATH, and RQ1_BEST_MODEL."
        )
    if len(classifier_matched) == 0:
        raise RuntimeError("No classifier parameters were loaded from the RQ1 checkpoint. Check CLASSIFIER_HEAD_TYPE.")

    required_final_keys = get_required_final_classifier_keys(model.head_type)
    missing_final_keys = [key for key in required_final_keys if key not in matched]
    print(">>> Required final classifier keys:", required_final_keys)
    if missing_final_keys:
        raise RuntimeError(
            "Final classifier output layer was not fully loaded from the RQ1 checkpoint. "
            f"Missing final classifier keys: {missing_final_keys}. "
            "Please check RQ1_BEST_MODEL, NUM_LABELS, MODEL_TYPE, and CLASSIFIER_HEAD_TYPE."
        )
    print(">>> Final classifier output layer fully loaded.")

    model.eval()
    return model


# ================= 4. Data =================
class CodeDataset(Dataset):
    def __init__(self, file_name, tokenizer):
        self.examples = []
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Loading {file_name}", leave=False)):
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code_text = get_code(js)
                label = get_label(js)
                if not code_text or label is None:
                    continue
                label = int(label)
                if label < 0 or label >= NUM_LABELS:
                    continue
                inputs = tokenizer(
                    code_text,
                    max_length=MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                self.examples.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "label": torch.tensor(label, dtype=torch.long),
                    "idx": js.get("idx", js.get("id", str(line_idx))),
                })
        print(f">>> Loaded {len(self.examples)} samples from {file_name}")
        label_values = [int(x["label"]) for x in self.examples]
        if label_values:
            print(f">>> 0-based label check: min={min(label_values)}, max={max(label_values)}, expected=[0, {NUM_LABELS - 1}]")
        print(">>> Label distribution preview:", dict(Counter(label_values).most_common(10)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_fixed_ood_datasets(tokenizer):
    print(f">>> Using fixed OOD train file: {OOD_TRAIN_FILE}")
    print(f">>> Using fixed OOD test file : {OOD_TEST_FILE}")
    return CodeDataset(OOD_TRAIN_FILE, tokenizer), CodeDataset(OOD_TEST_FILE, tokenizer)


def build_replay_buffer_dataset(id_train_ds, buffer_size=None, seed=42):
    """Build an ID replay buffer from previous-task/ID training samples."""
    if buffer_size is None or buffer_size <= 0 or buffer_size >= len(id_train_ds):
        indices = list(range(len(id_train_ds)))
        print(f">>> Replay buffer: using full ID train as rehearsal set, size={len(indices)}")
        return id_train_ds, indices

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(id_train_ds), size=int(buffer_size), replace=False).tolist()
    indices = sorted(indices)
    print(f">>> Replay buffer: sampled {len(indices)} / {len(id_train_ds)} ID-train samples")
    return Subset(id_train_ds, indices), indices


# ================= 5. Continual learning helpers =================
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        self.old_params = {name: param.clone().detach() for name, param in self.params.items()}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {name: torch.zeros_like(param) for name, param in self.params.items()}
        self.model.eval()
        for batch in tqdm(dataloader, desc="Computing Fisher on ID train", leave=False):
            self.model.zero_grad()
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            ).logits
            loss = F.cross_entropy(logits, batch["label"].to(self.device))
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += (param.grad.detach() ** 2) / max(len(dataloader), 1)
        return fisher

    def penalty(self, model):
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.old_params[name]) ** 2).sum()
        return loss


# ================= 6. Evaluation / Training =================
@torch.no_grad()
def evaluate(model, dataloader, desc="Evaluating"):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        ).logits
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())

    if not all_logits:
        return {"accuracy": 0.0, "macro_f1": 0.0, "top5_acc": 0.0, "num_samples": 0}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    preds = torch.argmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    k = min(TOP_K, all_logits.size(1))
    _, topk = all_logits.topk(k, dim=1)
    return {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "macro_f1": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        "top5_acc": float(topk.eq(all_labels.view(-1, 1)).any(dim=1).float().mean().item()),
        "num_samples": int(len(labels_np)),
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    epoch,
    strategy,
    trial_name,
    ewc_tool=None,
    replay_iter=None,
    ewc_lambda=0.0,
    replay_lambda=0.0,
):
    model.train()
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    ewc_penalty_sum = 0.0
    weighted_ewc_penalty_sum = 0.0
    replay_loss_sum = 0.0
    weighted_replay_loss_sum = 0.0
    steps = 0

    pbar = tqdm(dataloader, desc=f"[{trial_name}] Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        ).logits
        ce_loss = F.cross_entropy(logits, batch["label"].to(DEVICE))
        loss = ce_loss

        ewc_penalty = torch.tensor(0.0, device=DEVICE)
        weighted_ewc_penalty = torch.tensor(0.0, device=DEVICE)
        if strategy == "ewc" and ewc_tool is not None:
            ewc_penalty = ewc_tool.penalty(model)
            weighted_ewc_penalty = ewc_lambda * ewc_penalty
            loss = loss + weighted_ewc_penalty

        replay_loss = torch.tensor(0.0, device=DEVICE)
        weighted_replay_loss = torch.tensor(0.0, device=DEVICE)
        if strategy == "replay" and replay_iter is not None:
            id_batch = next(replay_iter)
            id_logits = model(
                id_batch["input_ids"].to(DEVICE),
                id_batch["attention_mask"].to(DEVICE),
            ).logits
            replay_loss = F.cross_entropy(id_logits, id_batch["label"].to(DEVICE))
            weighted_replay_loss = replay_lambda * replay_loss
            loss = loss + weighted_replay_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss_sum += float(loss.item())
        ce_loss_sum += float(ce_loss.detach().item())
        ewc_penalty_sum += float(ewc_penalty.detach().item())
        weighted_ewc_penalty_sum += float(weighted_ewc_penalty.detach().item())
        replay_loss_sum += float(replay_loss.detach().item())
        weighted_replay_loss_sum += float(weighted_replay_loss.detach().item())
        steps += 1

        postfix = {
            "loss": f"{loss.item():.4f}",
            "ce": f"{ce_loss.detach().item():.4f}",
        }
        if strategy == "ewc":
            postfix["ewc*w"] = f"{weighted_ewc_penalty.detach().item():.2e}"
        if strategy == "replay":
            postfix["rep*w"] = f"{weighted_replay_loss.detach().item():.4f}"
        pbar.set_postfix(postfix)

    denom = max(steps, 1)
    stats = {
        "total_loss": total_loss_sum / denom,
        "ce_loss": ce_loss_sum / denom,
        "ewc_penalty_raw": ewc_penalty_sum / denom,
        "ewc_penalty_weighted": weighted_ewc_penalty_sum / denom,
        "replay_loss_raw": replay_loss_sum / denom,
        "replay_loss_weighted": weighted_replay_loss_sum / denom,
        "steps": int(steps),
    }

    if strategy == "ewc":
        print(
            f">>> [{trial_name}] Epoch {epoch} loss breakdown | "
            f"CE={stats['ce_loss']:.6f} | "
            f"EWC_raw={stats['ewc_penalty_raw']:.6e} | "
            f"lambda*EWC={stats['ewc_penalty_weighted']:.6e} | "
            f"Total={stats['total_loss']:.6f}"
        )
    elif strategy == "replay":
        print(
            f">>> [{trial_name}] Epoch {epoch} loss breakdown | "
            f"OOD_CE={stats['ce_loss']:.6f} | "
            f"Replay_CE={stats['replay_loss_raw']:.6f} | "
            f"lambda*Replay={stats['replay_loss_weighted']:.6f} | "
            f"Total={stats['total_loss']:.6f}"
        )
    else:
        print(
            f">>> [{trial_name}] Epoch {epoch} loss breakdown | "
            f"OOD_CE={stats['ce_loss']:.6f} | Total={stats['total_loss']:.6f}"
        )

    return stats


def format_float_tag(value):
    return str(value).replace(".", "p")


def build_ewc_trial_name(ewc_lambda):
    return f"ewc_lambda_{format_float_tag(ewc_lambda)}"


def build_replay_trial_name(replay_lambda, replay_batch_size, replay_buffer_size):
    return (
        f"replay_lambda_{format_float_tag(replay_lambda)}"
        f"_bs_{replay_batch_size}"
        f"_buf_{replay_buffer_size}"
    )


def print_epoch_metrics_table(trial_name, epoch_metrics):
    print("\n" + "=" * 150)
    print(f"RQ4 Continual Learning Epoch Metrics - {trial_name}")
    print("=" * 150)
    print(
        f"{'Epoch':<7} | {'Loss':<10} | {'CE':<10} | {'EWC*w':<10} | {'Replay*w':<10} | "
        f"{'ID Acc':<10} | {'ID F1':<10} | {'ID Top5':<10} | "
        f"{'OOD Acc':<10} | {'OOD F1':<10} | {'OOD Top5':<10} | {'Best':<5}"
    )
    print("-" * 150)
    for row in epoch_metrics:
        id_eval = row["id_test"]
        ood_eval = row["ood_test"]
        loss_stats = row.get("loss_breakdown", {})
        best_mark = "*" if row.get("is_best") else ""
        print(
            f"{row['epoch']:<7} | {row['train_loss']:<10.4f} | "
            f"{loss_stats.get('ce_loss', 0.0):<10.4f} | "
            f"{loss_stats.get('ewc_penalty_weighted', 0.0):<10.2e} | "
            f"{loss_stats.get('replay_loss_weighted', 0.0):<10.4f} | "
            f"{id_eval['accuracy']:<10.4f} | {id_eval['macro_f1']:<10.4f} | {id_eval['top5_acc']:<10.4f} | "
            f"{ood_eval['accuracy']:<10.4f} | {ood_eval['macro_f1']:<10.4f} | {ood_eval['top5_acc']:<10.4f} | "
            f"{best_mark:<5}"
        )
    print("=" * 150)


def print_final_table(results):
    print("\n" + "=" * 110)
    print("RQ4 Continual Learning Final Results")
    print("=" * 110)
    print(
        f"{'Strategy/Config':<28} | {'ID Acc':<10} | {'ID F1':<10} | {'ID Top5':<10} | "
        f"{'OOD Acc':<10} | {'OOD F1':<10} | {'OOD Top5':<10}"
    )
    print("-" * 110)
    for strategy in ["ewc", "replay"]:
        for trial_name, trial_result in results.get(strategy, {}).items():
            final = trial_result["final"]
            id_eval = final["id_test"]
            ood_eval = final["ood_test"]
            print(
                f"{trial_name:<28} | "
                f"{id_eval['accuracy']:<10.4f} | {id_eval['macro_f1']:<10.4f} | {id_eval['top5_acc']:<10.4f} | "
                f"{ood_eval['accuracy']:<10.4f} | {ood_eval['macro_f1']:<10.4f} | {ood_eval['top5_acc']:<10.4f}"
            )
    print("=" * 110)


# ================= 7. Main =================
def main(model_type=None):
    if model_type is not None:
        configure_model_runtime(model_type)
    else:
        configure_model_runtime(MODEL_TYPE)
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("=" * 100)
    print("RQ4-CL: fixed OOD train/test files, representative EWC and best Replay configuration")
    print("=" * 100)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("SUPPORTED_MODEL_TYPES:", SUPPORTED_MODEL_TYPES)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("OOD_TRAIN_FILE:", OOD_TRAIN_FILE)
    print("OOD_TEST_FILE:", OOD_TEST_FILE)
    print("CLASSIFIER_HEAD_TYPE:", CLASSIFIER_HEAD_TYPE)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("EWC_LAMBDAS:", EWC_LAMBDAS)
    print("REPLAY_CONFIGS:", REPLAY_CONFIGS)
    print("REPLAY_BUFFER_SIZES:", REPLAY_BUFFER_SIZES)
    print("=" * 100)

    id_train_ds = CodeDataset(ID_TRAIN_FILE, tokenizer)
    id_test_ds = CodeDataset(ID_TEST_FILE, tokenizer)
    ood_train_ds, ood_test_ds = load_fixed_ood_datasets(tokenizer)

    ood_train_loader = DataLoader(ood_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    ood_test_loader = DataLoader(ood_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    id_train_loader = DataLoader(id_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    replay_buffers = {}
    for buffer_size in REPLAY_BUFFER_SIZES:
        replay_buffer_ds, replay_buffer_indices = build_replay_buffer_dataset(
            id_train_ds,
            buffer_size=buffer_size,
            seed=REPLAY_BUFFER_SEED,
        )
        replay_buffers[int(buffer_size)] = {
            "dataset": replay_buffer_ds,
            "indices": replay_buffer_indices,
            "actual_size": len(replay_buffer_ds),
        }

    trial_configs = []

    for ewc_lambda in EWC_LAMBDAS:
        trial_configs.append({
            "strategy": "ewc",
            "trial_name": build_ewc_trial_name(ewc_lambda),
            "ewc_lambda": float(ewc_lambda),
            "replay_lambda": 0.0,
            "replay_batch_size": None,
            "replay_buffer_size": 0,
        })

    for cfg in REPLAY_CONFIGS:
        replay_lambda = float(cfg["lambda"])
        replay_batch_size = int(cfg["batch_size"])
        for replay_buffer_size in REPLAY_BUFFER_SIZES:
            replay_buffer_size = int(replay_buffer_size)
            trial_configs.append({
                "strategy": "replay",
                "trial_name": build_replay_trial_name(replay_lambda, replay_batch_size, replay_buffer_size),
                "ewc_lambda": 0.0,
                "replay_lambda": replay_lambda,
                "replay_batch_size": replay_batch_size,
                "replay_buffer_size": replay_buffer_size,
                "replay_buffer_actual_size": replay_buffers[replay_buffer_size]["actual_size"],
            })

    results = {"ewc": {}, "replay": {}}
    best_by_strategy = {}
    best_overall = None

    for trial_cfg in trial_configs:
        strategy = trial_cfg["strategy"]
        trial_name = trial_cfg["trial_name"]
        strategy_start = time.time()
        set_seed(SEED)

        print("\n" + "=" * 80)
        print(f"RQ4 Continual Learning - {trial_name}")
        print("=" * 80)
        print("trial_config:", trial_cfg)

        model = load_rq1_model()

        ewc_tool = None
        replay_iter = None
        if strategy == "ewc":
            print(f">>> [EWC] Computing Fisher matrix on ID train; lambda={trial_cfg['ewc_lambda']}...")
            ewc_tool = EWC(model, id_train_loader, DEVICE)
        elif strategy == "replay":
            buffer_info = replay_buffers[int(trial_cfg["replay_buffer_size"])]
            id_replay_loader = DataLoader(
                buffer_info["dataset"],
                batch_size=trial_cfg["replay_batch_size"],
                shuffle=True,
            )
            replay_iter = itertools.cycle(id_replay_loader)
            print(
                f">>> [Replay] Using ID replay buffer: configured_size={trial_cfg['replay_buffer_size']}, "
                f"actual_size={buffer_info['actual_size']}, "
                f"batch_size={trial_cfg['replay_batch_size']}, lambda={trial_cfg['replay_lambda']}"
            )

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = len(ood_train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * WARMUP_RATIO),
            num_training_steps=total_steps,
        )

        train_losses = []
        loss_breakdowns = []
        epoch_metrics = []
        best_epoch = 0
        best_ood_macro_f1 = -1.0
        final_model_path = os.path.join(SAVE_DIR, f"final_cl_{trial_name}_{MODEL_TYPE}.bin")
        best_epoch_model_path = None

        for epoch in range(1, EPOCHS + 1):
            loss_stats = train_one_epoch(
                model,
                ood_train_loader,
                optimizer,
                scheduler,
                epoch,
                strategy,
                trial_name,
                ewc_tool=ewc_tool,
                replay_iter=replay_iter,
                ewc_lambda=trial_cfg["ewc_lambda"],
                replay_lambda=trial_cfg["replay_lambda"],
            )
            loss = float(loss_stats["total_loss"])
            train_losses.append(loss)
            loss_breakdowns.append(loss_stats)

            epoch_path = os.path.join(SAVE_DIR, f"cl_{trial_name}_{MODEL_TYPE}_epoch_{epoch}.bin")
            torch.save(model.state_dict(), epoch_path)
            print(f"{trial_name} Epoch {epoch} | Train loss={loss:.4f}")

            epoch_id_eval = evaluate(model, id_test_loader, desc=f"[{trial_name}] Epoch {epoch} ID Test")
            epoch_ood_eval = evaluate(model, ood_test_loader, desc=f"[{trial_name}] Epoch {epoch} OOD Test")
            epoch_record = {
                "epoch": int(epoch),
                "train_loss": loss,
                "loss_breakdown": loss_stats,
                "id_test": epoch_id_eval,
                "ood_test": epoch_ood_eval,
                "is_best": False,
            }
            epoch_metrics.append(epoch_record)

            # NOTE: Still follows the original protocol: select best epoch by OOD test Macro-F1.
            # For publication-grade strict evaluation, replace this with an OOD validation split.
            if epoch_ood_eval["macro_f1"] > best_ood_macro_f1:
                best_ood_macro_f1 = float(epoch_ood_eval["macro_f1"])
                best_epoch = epoch
                best_epoch_model_path = epoch_path
                torch.save(model.state_dict(), final_model_path)
                for item in epoch_metrics:
                    item["is_best"] = item["epoch"] == best_epoch
                print(
                    f">>> New best {trial_name} by OOD test Macro-F1: "
                    f"epoch={best_epoch}, ood_macro_f1={best_ood_macro_f1:.4f}"
                )

        if best_epoch == 0:
            torch.save(model.state_dict(), final_model_path)
            best_epoch = EPOCHS
            best_epoch_model_path = final_model_path
            best_ood_macro_f1 = epoch_metrics[-1]["ood_test"]["macro_f1"] if epoch_metrics else float("nan")
            if epoch_metrics:
                epoch_metrics[-1]["is_best"] = True

        model.load_state_dict(safe_torch_load(final_model_path, map_location=DEVICE))
        final_id_eval = evaluate(model, id_test_loader, desc=f"[{trial_name}] Best Epoch ID Test")
        final_ood_eval = evaluate(model, ood_test_loader, desc=f"[{trial_name}] Best Epoch OOD Test")
        print_epoch_metrics_table(trial_name, epoch_metrics)

        results[strategy][trial_name] = {
            "config": trial_cfg,
            "final": {
                "selection_metric": "highest_ood_test_new_macro_f1",
                "best_epoch": int(best_epoch),
                "best_ood_macro_f1": float(best_ood_macro_f1),
                "id_test": final_id_eval,
                "ood_test": final_ood_eval,
                "train_losses": train_losses,
                "loss_breakdowns": loss_breakdowns,
                "epoch_metrics": epoch_metrics,
            },
            "files": {
                "final_model": final_model_path,
                "best_epoch_model": best_epoch_model_path,
            },
            "time_seconds": time.time() - strategy_start,
        }

        trial_summary = {
            "strategy": strategy,
            "trial_name": trial_name,
            "config": trial_cfg,
            "best_epoch": int(best_epoch),
            "best_ood_macro_f1": float(best_ood_macro_f1),
            "id_test": final_id_eval,
            "ood_test": final_ood_eval,
            "final_model": final_model_path,
        }

        if strategy not in best_by_strategy or final_ood_eval["macro_f1"] > best_by_strategy[strategy]["ood_test"]["macro_f1"]:
            best_by_strategy[strategy] = trial_summary
            best_strategy_model_path = os.path.join(SAVE_DIR, f"best_cl_{strategy}_{MODEL_TYPE}.bin")
            torch.save(model.state_dict(), best_strategy_model_path)
            best_by_strategy[strategy]["best_strategy_model"] = best_strategy_model_path

        if best_overall is None or final_ood_eval["macro_f1"] > best_overall["ood_test"]["macro_f1"]:
            best_overall = dict(trial_summary)
            best_overall_model_path = os.path.join(SAVE_DIR, f"best_cl_overall_{MODEL_TYPE}.bin")
            torch.save(model.state_dict(), best_overall_model_path)
            best_overall["best_overall_model"] = best_overall_model_path

        print(
            f">>> [{trial_name}] Done | "
            f"BestEpoch={best_epoch} | "
            f"ID F1={final_id_eval['macro_f1']:.4f} | "
            f"OOD F1={final_ood_eval['macro_f1']:.4f} | "
            f"Time={format_time(results[strategy][trial_name]['time_seconds'])}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_final_table(results)

    metrics = {
        "task": "Code Classification",
        "method": "RQ4-CL fixed OOD train/test files with representative EWC and best Replay configuration; 0-based labels; adaptive encoder pooling aligned with RQ1",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "rq1_best_model": RQ1_BEST_MODEL,
        "data_dir": DATA_DIR,
        "dataset_name": DATASET_NAME,
        "supported_model_types": SUPPORTED_MODEL_TYPES,
        "id_train_file": ID_TRAIN_FILE,
        "id_test_file": ID_TEST_FILE,
        "ood_train_file": OOD_TRAIN_FILE,
        "ood_test_file": OOD_TEST_FILE,
        "classifier_head_type": CLASSIFIER_HEAD_TYPE,
        "encoder_pooling": ENCODER_POOLING,
        "encoder_pooling_actual": resolve_pooling_strategy(MODEL_TYPE, ENCODER_POOLING),
        "label_format": "0-based labels in [0, NUM_LABELS-1]",
        "ewc_lambdas": EWC_LAMBDAS,
        "replay_configs": REPLAY_CONFIGS,
        "replay_buffer": {
            "source": "ID train / previous-task training data",
            "configured_sizes": REPLAY_BUFFER_SIZES,
            "buffers": {
                str(size): {
                    "actual_size": info["actual_size"],
                    "indices": info["indices"],
                }
                for size, info in replay_buffers.items()
            },
        },
        "split": {
            "ood_train_size": len(ood_train_ds),
            "ood_test_size": len(ood_test_ds),
        },
        "selection_metric": "highest_ood_test_new_macro_f1",
        "best_by_strategy": best_by_strategy,
        "best_overall": best_overall,
        "results": results,
        "files": {
            "metrics_file": METRICS_FILE,
        },
        "time": {
            "total_seconds": time.time() - START_TIME,
            "total_readable": format_time(time.time() - START_TIME),
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Metrics file:", METRICS_FILE)
    print("Total time:", format_time(time.time() - START_TIME))
    return metrics


if __name__ == "__main__":
    if RUN_ALL_MODELS:
        all_results = []
        for run_model_type in MODEL_TYPES_TO_RUN:
            print("\n\n" + "#" * 120)
            print(f"Running RQ4 continual-learning evaluation for model: {run_model_type}")
            print("#" * 120)
            all_results.append(main(run_model_type))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        with open(ALL_MODELS_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": "Code Classification",
                    "method": "RQ4-CL fixed OOD train/test files with representative EWC and best Replay configuration",
                    "dataset_name": DATASET_NAME,
                    "models": MODEL_TYPES_TO_RUN,
                    "results": all_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("All-model metrics file:", ALL_MODELS_METRICS_FILE)
    else:
        main(MODEL_TYPE)
