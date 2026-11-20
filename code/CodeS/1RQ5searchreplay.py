import json
import os
import random
import time
import itertools
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5Config,
    T5EncoderModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm


"""
================================================================================
RQ4-CL-Replay + RQ3-guided Geometry-aware OOD Split

目标：
  在最佳 Replay 持续学习配置基础上，使用 RQ5/RQ3 几何感知方法重新构造
  OOD train / OOD test_new。

核心流程：
  1) 从同一个 RQ1 best model 出发；
  2) 读取 OOD_POOL_FILE，例如 ood_test.jsonl；
  3) 用 RQ1 模型计算每个 OOD 类的几何困难度：
       entropy 高        -> hard
       alignment 低      -> hard
       anisotropy 高     -> hard
       effective rank 低 -> hard
  4) 根据几何困难度，在固定全局比例 80%/20% 下生成：
       geometry_replay_ood_train.jsonl
       geometry_replay_ood_test_new.jsonl
     其中困难类倾向保留更多训练样本，容易类倾向分配更多测试样本；
  5) 在 geometry OOD train 上做 Replay continual learning；
  6) 在 ID test 和 geometry OOD test_new 上评估。

注意：
  - 标签默认 0-based，不做 label-1。
  - CodeBERT/GraphCodeBERT 默认使用 CLS pooling。
  - CodeT5 默认使用 mean pooling。
  - 分类头自动兼容 single / mlp_seq / mlp_named / legacy out_proj-only。
  - 当前脚本仍沿用原 RQ4 协议：按 OOD test_new Macro-F1 选最佳 epoch。
    如果用于严格论文实验，建议把 geometry OOD train 再划出 validation，
    只把 geometry OOD test_new 留作最终测试。
================================================================================
"""

START_TIME = time.time()


# ============================================================
# 1. Config
# ============================================================
MODEL_TYPE = "RoBERTa"  # options: "codebert", "GraphCodeBERT", "codeT5"
MODEL_PATH = "/opt/data/private/RoBERTa"

DATA_DIR = "/opt/data/private/codes/Java250/Java250/cst/pre-trained"
SAVE_DIR = "/opt/data/private/codes/Java250/Java250/cst/RQ4_replay_geometry_split_results"
os.makedirs(SAVE_DIR, exist_ok=True)

RQ1_BEST_MODEL = "/opt/data/private/codes/Java250/Java250/cst/best_model_RoBERTa_Java250.bin"

ID_TRAIN_FILE = "train.jsonl"
ID_TEST_FILE = "id_test.jsonl"

# 用这个 OOD pool 重新生成 OOD train / OOD test_new。
# 对于 token/cst shift，通常设置为原始大 OOD 池，例如 ood_test.jsonl。
OOD_POOL_FILE = "ood_test.jsonl"

GEOMETRY_OOD_TRAIN_FILE = os.path.join(SAVE_DIR, "geometry_replay_ood_train.jsonl")
GEOMETRY_OOD_TEST_FILE = os.path.join(SAVE_DIR, "geometry_replay_ood_test_new.jsonl")
SPLIT_REPORT_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geometry_split_report_{MODEL_TYPE}.json")
METRICS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geometry_split_metrics_{MODEL_TYPE}.json")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, f"best_rq4_replay_geometry_{MODEL_TYPE}.bin")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 250
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 8
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
TOP_K = 5
SEED = 42

# 与 RQ1 自适应分类代码保持一致：
#   auto: CodeT5 -> mean pooling; CodeBERT/GraphCodeBERT -> CLS pooling
#   也可手动设为 "cls" 或 "mean"，用于强制与旧 RQ1 checkpoint 对齐。
ENCODER_POOLING = "auto"
CLASSIFIER_HEAD_TYPE = "auto"  # auto / single / mlp_seq / mlp_named

# Best Replay configuration from previous RQ4 result.
REPLAY_LAMBDA = 0.5
REPLAY_BATCH_SIZE = 4
REPLAY_BUFFER_SIZE = 1000
REPLAY_BUFFER_SEED = SEED

# Geometry split settings.
SHIFT_TYPE = "token"  # choices: "token", "cst", "time", "user"
ALIGNMENT_MODE = "auto"  # choices: "auto", "pair", "label_centroid"
MIN_CLASS_SAMPLES_FOR_GEOMETRY = 3
MIN_TRAIN_PER_CLASS = 2

# Fixed global OOD train/test ratio.
FIXED_TRAIN_RATIO = 0.80
FIXED_TEST_RATIO = 0.20

# Allocation preferences under the fixed 20% test budget.
# hard <= middle <= easy: hard classes keep more training samples.
TEST_RATIO_HARD = 0.08
TEST_RATIO_MIDDLE = 0.10
TEST_RATIO_EASY = 0.15

DEFAULT_SCORE_WEIGHTS = {
    "entropy": 0.25,
    "alignment": 0.25,
    "anisotropy": 0.25,
    "rank": 0.25,
}

MAX_ANISO_SAMPLES_PER_CLASS = 3000
MAX_RANK_SAMPLES_PER_CLASS = 8000


# SmartSearch settings: NO TPE / NO Optuna.
# Stage-1: cheap probing with fixed logical geometry configs.
# Stage-2: medium-budget Replay training, using top Stage-1 + surrogate-selected configs.
# Stage-3: full-budget Replay training for top configs.
SMART_STAGE1_TRIALS = 24
SMART_STAGE1_EPOCHS = 3
SMART_STAGE2_TRIALS = 16
SMART_STAGE2_EPOCHS = 5
SMART_STAGE2_FROM_STAGE1 = 6
SMART_STAGE2_RANDOM = 3
SMART_FINAL_TRIALS = 5
SMART_FINAL_EPOCHS = EPOCHS
SMART_SURROGATE_POOL_SIZE = 500
SURROGATE_MIN_SAMPLES = 8
SURROGATE_N_ESTIMATORS = 128
SURROGATE_EXPLORATION_WEIGHT = 0.15
SEARCH_SELECT_METRIC = "ood_gain_id_preserve"
SEARCH_OOD_F1_WEIGHT = 0.65
SEARCH_ID_F1_WEIGHT = 0.35

# RoBERTa often has a weaker correlation between very-short-budget and
# full-budget training than code-pretrained encoders.  Rank candidates by
# both their post-training OOD quality and the improvement over the same
# split before Replay training.  Penalize ID forgetting explicitly.
SEARCH_POST_OOD_WEIGHT = 0.70
SEARCH_OOD_GAIN_WEIGHT = 0.30
SEARCH_ID_DROP_PENALTY = 0.50

SEARCH_WEIGHT_BOUNDS = {
    "entropy": (0.0, 1.0),
    "alignment": (0.0, 1.0),
    "anisotropy": (0.0, 1.0),
    "rank": (0.0, 1.0),
}
SEARCH_RATIO_BOUNDS = {
    "hard": (0.02, 0.20),
    "middle": (0.04, 0.30),
    "easy": (0.06, 0.40),
}

SMART_SEARCH_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geo_smartsearch_results_{MODEL_TYPE}.json")
SMART_TOP_CONFIGS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geo_smartsearch_top_configs_{MODEL_TYPE}.json")


# ============================================================
# 2. Utilities
# ============================================================
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
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key.replace(prefix, "", 1)
    return key


def normalized_keys(state_dict):
    return [strip_prefix(k) for k in state_dict.keys()]


def infer_head_type_from_state_dict(state_dict):
    """
    自动判断分类头结构。

    兼容几类 RQ1 checkpoint:
      single:
        classifier.weight / classifier.bias
        score.weight / score.bias
        classifier.out_proj.weight / classifier.out_proj.bias 但无 dense
        classification_head.out_proj.* 但无 dense
      mlp_seq:
        classifier.0.* and classifier.3.*
      mlp_named:
        classifier.dense.* and classifier.out_proj.*
        classification_head.dense.* and classification_head.out_proj.*
    """
    keys = normalized_keys(state_dict)

    has_seq = any(k.startswith("classifier.0.") or k.startswith("classifier.3.") for k in keys)
    if has_seq:
        return "mlp_seq"

    has_named_dense = any(k.startswith("classifier.dense.") or k.startswith("classification_head.dense.") for k in keys)
    has_named_out = any(k.startswith("classifier.out_proj.") or k.startswith("classification_head.out_proj.") for k in keys)
    if has_named_dense and has_named_out:
        return "mlp_named"

    if any(k in ("classifier.weight", "classifier.bias", "score.weight", "score.bias") for k in keys):
        return "single"

    # Legacy RQ1: ClassifierHead only had dropout + out_proj.
    if has_named_out:
        return "single"

    return "single"


def adapt_state_dict_keys(state_dict, head_type):
    new_dict = {}

    for key, value in state_dict.items():
        new_key = strip_prefix(key)

        # encoder prefix adaptation
        if new_key.startswith("roberta."):
            new_key = new_key.replace("roberta.", "encoder.", 1)
        elif new_key.startswith("bert."):
            new_key = new_key.replace("bert.", "encoder.", 1)
        elif new_key.startswith("transformer."):
            new_key = new_key.replace("transformer.", "encoder.", 1)
        elif new_key.startswith("encoder.encoder."):
            new_key = new_key.replace("encoder.encoder.", "encoder.", 1)

        # classification_head -> classifier
        if new_key.startswith("classification_head.dense."):
            new_key = new_key.replace("classification_head.dense.", "classifier.dense.", 1)
        elif new_key.startswith("classification_head.out_proj."):
            new_key = new_key.replace("classification_head.out_proj.", "classifier.out_proj.", 1)

        # score -> classifier
        if new_key == "score.weight":
            new_key = "classifier.weight"
        elif new_key == "score.bias":
            new_key = "classifier.bias"

        # map checkpoint head keys to current model head keys
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
    if head_type == "single":
        return ["classifier.weight", "classifier.bias"]
    if head_type == "mlp_named":
        return ["classifier.out_proj.weight", "classifier.out_proj.bias"]
    if head_type == "mlp_seq":
        return ["classifier.3.weight", "classifier.3.bias"]
    raise ValueError(f"Unsupported head_type: {head_type}")


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")
    return "mean" if "t5" in model_type.lower() else "cls"


def get_code(js):
    return js.get("func") or js.get("code") or js.get("function") or js.get("source") or ""


def get_label(js):
    label = js.get("label")
    if label is None:
        label = js.get("target")
    return label


def get_idx(js, fallback):
    idx = js.get("idx")
    if idx is None:
        idx = js.get("id")
    if idx is None:
        idx = fallback
    return str(idx)


def resolve_data_path(file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(DATA_DIR, file_name)


def read_jsonl(path):
    data = []
    bad_lines = 0
    skipped_label = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            code = get_code(js)
            label = get_label(js)
            if not code or label is None:
                bad_lines += 1
                continue

            label = int(label)
            if label < 0 or label >= NUM_LABELS:
                skipped_label += 1
                continue

            js["idx"] = get_idx(js, line_idx)
            js["label"] = label  # 0-based label kept unchanged
            data.append(js)

    labels = [int(x["label"]) for x in data]
    print("=" * 100)
    print("JSONL loaded:", path)
    print("num_samples:", len(data))
    print("bad_lines:", bad_lines)
    print("skipped_label_out_of_range:", skipped_label)
    if labels:
        print(f"0-based label check: min={min(labels)}, max={max(labels)}, expected=[0, {NUM_LABELS - 1}]")
        print("label_distribution_preview:", dict(Counter(labels).most_common(10)))
    print("=" * 100)
    return data


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_distribution(data):
    return dict(Counter(int(x["label"]) for x in data))


def deterministic_indices(n, max_samples, seed=SEED):
    if max_samples is None or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def minmax_hard(values, higher_is_harder=True):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if finite.sum() == 0:
        return out
    v = arr[finite]
    vmin, vmax = float(v.min()), float(v.max())
    if abs(vmax - vmin) < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (v - vmin) / (vmax - vmin)
    if not higher_is_harder:
        out = 1.0 - out
    out[~finite] = 0.0
    return out


def normalize_positive_weights(weights):
    vals = {
        "entropy": max(0.0, float(weights.get("entropy", DEFAULT_SCORE_WEIGHTS["entropy"]))),
        "alignment": max(0.0, float(weights.get("alignment", DEFAULT_SCORE_WEIGHTS["alignment"]))),
        "anisotropy": max(0.0, float(weights.get("anisotropy", DEFAULT_SCORE_WEIGHTS["anisotropy"]))),
        "rank": max(0.0, float(weights.get("rank", DEFAULT_SCORE_WEIGHTS["rank"]))),
    }
    total = sum(vals.values())
    if total <= 1e-12:
        return dict(DEFAULT_SCORE_WEIGHTS)
    return {k: float(v / total) for k, v in vals.items()}


# ============================================================
# 3. Model
# ============================================================
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

    def prepare_final_features(self, pooled_features):
        x = self.dropout(pooled_features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x

    def forward(self, pooled_features):
        return self.out_proj(self.prepare_final_features(pooled_features))


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

    def prepare_classifier_features(self, pooled_features):
        if self.head_type == "single":
            return self.dropout(pooled_features)
        if self.head_type == "mlp_named":
            return self.classifier.prepare_final_features(pooled_features)
        x = self.classifier[0](pooled_features)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x

    def classify_final_features(self, final_features):
        if self.head_type == "single":
            return self.classifier(final_features)
        if self.head_type == "mlp_named":
            return self.classifier.out_proj(final_features)
        return self.classifier[3](final_features)

    def forward(self, input_ids, attention_mask, return_feat=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_features = self.pool_encoder_output(outputs, attention_mask)
        final_features = self.prepare_classifier_features(pooled_features)
        logits = self.classify_final_features(final_features)
        if return_feat:
            return logits, final_features
        return type("ModelOutput", (object,), {"logits": logits})


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        raise FileNotFoundError(f"RQ1 best model not found: {RQ1_BEST_MODEL}")

    raw_state = unwrap_checkpoint(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    if not isinstance(raw_state, dict):
        raise RuntimeError("Loaded checkpoint is not a state_dict-like object.")

    head_type = infer_head_type_from_state_dict(raw_state) if CLASSIFIER_HEAD_TYPE == "auto" else CLASSIFIER_HEAD_TYPE
    if head_type not in ("single", "mlp_seq", "mlp_named"):
        raise ValueError(f"Unsupported head_type: {head_type}")

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
        f"pooling={model.pooling_strategy}, matched={len(matched)}, "
        f"encoder_matched={len(encoder_matched)}, classifier_matched={len(classifier_matched)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
    )

    skipped_classifier = [item for item in skipped if item[0].startswith("classifier.")]
    if skipped_classifier:
        print(">>> Skipped classifier keys due to mismatch/missing:")
        for item in skipped_classifier[:30]:
            print("   ", item)

    missing_show = [key for key in missing if "pooler" not in key][:30]
    if missing_show:
        print(">>> Missing keys preview:", missing_show)
    if unexpected:
        print(">>> Unexpected keys preview:", unexpected[:30])

    if len(encoder_matched) == 0:
        raise RuntimeError("No encoder parameters were loaded from the RQ1 checkpoint.")
    if len(classifier_matched) == 0:
        raise RuntimeError("No classifier parameters were loaded from the RQ1 checkpoint.")

    required_final_keys = get_required_final_classifier_keys(model.head_type)
    missing_final_keys = [key for key in required_final_keys if key not in matched]
    print(">>> Required final classifier keys:", required_final_keys)
    if missing_final_keys:
        raise RuntimeError(
            "Final classifier output layer was not fully loaded from the RQ1 checkpoint. "
            f"Missing final classifier keys: {missing_final_keys}. "
            "Please check RQ1_BEST_MODEL, NUM_LABELS, and CLASSIFIER_HEAD_TYPE."
        )
    print(">>> Final classifier output layer fully loaded.")

    model.eval()
    return model


# ============================================================
# 4. Dataset
# ============================================================
class CodeDataset(Dataset):
    def __init__(self, source, tokenizer, name="dataset"):
        self.examples = []

        if isinstance(source, str):
            data = read_jsonl(resolve_data_path(source))
            source_name = source
        else:
            data = source
            source_name = name

        bad_lines = 0
        skipped_label = 0
        labels_seen = []

        for line_idx, js in enumerate(tqdm(data, desc=f"Tokenizing {source_name}", leave=False)):
            code = get_code(js)
            label = get_label(js)
            if not code or label is None:
                bad_lines += 1
                continue
            label = int(label)
            if label < 0 or label >= NUM_LABELS:
                skipped_label += 1
                continue
            labels_seen.append(label)

            inputs = tokenizer(
                code,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.examples.append({
                "idx": get_idx(js, line_idx),
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
                "label_int": label,
                "raw": js,
            })

        counter = Counter(x["label_int"] for x in self.examples)
        print("=" * 100)
        print(f"CodeDataset loaded: {source_name}")
        print("num_samples:", len(self.examples))
        print("bad_lines:", bad_lines)
        print("skipped_label_out_of_range:", skipped_label)
        if labels_seen:
            print(f"0-based label check: min={min(labels_seen)}, max={max(labels_seen)}, expected=[0, {NUM_LABELS - 1}]")
        print("label_distribution_preview:", dict(counter.most_common(10)))
        print("=" * 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# 5. Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, desc="Evaluating", save_path=None):
    model.eval()
    all_logits = []
    all_labels = []
    records = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        ).logits
        labels = batch["label"].to(DEVICE)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if save_path is not None:
            preds = torch.argmax(logits, dim=1)
            for i in range(labels.size(0)):
                records.append({
                    "idx": batch["idx"][i],
                    "label": int(labels[i].detach().cpu().item()),
                    "pred": int(preds[i].detach().cpu().item()),
                })

    if not all_logits:
        return {"accuracy": 0.0, "macro_f1": 0.0, "top5_acc": 0.0, "num_samples": 0}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    preds = torch.argmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    k = min(TOP_K, all_logits.size(1))
    _, topk = all_logits.topk(k, dim=1)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "macro_f1": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        "top5_acc": float(topk.eq(all_labels.view(-1, 1)).any(dim=1).float().mean().item()),
        "num_samples": int(len(labels_np)),
    }


# ============================================================
# 6. Geometry metrics and split
# ============================================================
@torch.no_grad()
def extract_embeddings_and_logits(model, dataloader, desc="Extracting embeddings"):
    model.eval()
    all_feats = []
    all_logits = []
    all_labels = []
    all_idxs = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        logits, feats = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            return_feat=True,
        )
        all_feats.append(feats.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())
        all_idxs.extend([str(x) for x in batch["idx"]])

    return (
        torch.cat(all_feats, dim=0),
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        np.asarray(all_idxs),
    )


def compute_anisotropy(embeddings):
    n = embeddings.size(0)
    if n <= 1:
        return 0.0
    idx = deterministic_indices(n, MAX_ANISO_SAMPLES_PER_CLASS, seed=SEED)
    x = embeddings[idx].float()
    x = F.normalize(x, p=2, dim=1)
    cos_mat = torch.mm(x, x.T)
    m = x.size(0)
    avg_sim = (cos_mat.sum() - m) / (m * (m - 1))
    return float(avg_sim.item())


def compute_effective_rank(embeddings):
    n = embeddings.size(0)
    if n <= 1:
        return 1.0
    idx = deterministic_indices(n, MAX_RANK_SAMPLES_PER_CLASS, seed=SEED)
    x = embeddings[idx].float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x)
    except Exception:
        s = torch.svd(x)[1]
    s = torch.clamp(s, min=1e-12)
    p = s / (s.sum() + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    return float(torch.exp(entropy).item())


def compute_pair_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs):
    id_norm = F.normalize(id_emb.float(), p=2, dim=1)
    ood_norm = F.normalize(ood_emb.float(), p=2, dim=1)
    id_map = {}
    for i in range(id_emb.size(0)):
        label = int(id_labels[i].item())
        id_map[(str(id_idxs[i]), label)] = i

    vals = defaultdict(list)
    matched = 0
    for j in range(ood_emb.size(0)):
        label = int(ood_labels[j].item())
        key = (str(ood_idxs[j]), label)
        if key not in id_map:
            continue
        i = id_map[key]
        sim = F.cosine_similarity(id_norm[i:i + 1], ood_norm[j:j + 1], dim=1).item()
        vals[label].append(sim)
        matched += 1

    return {int(c): float(np.mean(v)) for c, v in vals.items()}, matched


def compute_centroid_alignment_by_class(id_emb, id_labels, ood_emb, ood_labels):
    results = {}
    for c in range(NUM_LABELS):
        id_mask = id_labels == c
        ood_mask = ood_labels == c
        if id_mask.sum().item() == 0 or ood_mask.sum().item() == 0:
            continue
        id_centroid = id_emb[id_mask].float().mean(dim=0, keepdim=True)
        ood_centroid = ood_emb[ood_mask].float().mean(dim=0, keepdim=True)
        sim = F.cosine_similarity(
            F.normalize(id_centroid, p=2, dim=1),
            F.normalize(ood_centroid, p=2, dim=1),
            dim=1,
        ).item()
        results[int(c)] = float(sim)
    return results


def compute_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs):
    shift = SHIFT_TYPE.lower()
    mode = ALIGNMENT_MODE.lower()
    if mode == "pair" or (mode == "auto" and shift in ["token", "cst"]):
        pair_align, matched = compute_pair_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs)
        print(f">>> Pair-level alignment matched pairs: {matched}")
        if matched > 0:
            centroid_align = compute_centroid_alignment_by_class(id_emb, id_labels, ood_emb, ood_labels)
            centroid_align.update(pair_align)
            return centroid_align
        print("[Warning] No matched ID/OOD pairs. Falling back to label-centroid alignment.")
    return compute_centroid_alignment_by_class(id_emb, id_labels, ood_emb, ood_labels)


def compute_per_class_metrics(ood_emb, ood_logits, ood_labels, ood_idxs, id_emb, id_labels, id_idxs):
    results = {}
    alignment_by_class = compute_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs)

    for c in range(NUM_LABELS):
        ood_mask = ood_labels == c
        n_ood = int(ood_mask.sum().item())
        if n_ood < MIN_CLASS_SAMPLES_FOR_GEOMETRY:
            continue
        cls_emb = ood_emb[ood_mask]
        cls_logits = ood_logits[ood_mask]
        probs = F.softmax(cls_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
        alignment = alignment_by_class.get(int(c), float("nan"))
        anisotropy = compute_anisotropy(cls_emb)
        rank = compute_effective_rank(cls_emb)
        results[int(c)] = {
            "entropy": float(entropy),
            "alignment": float(alignment),
            "anisotropy": float(anisotropy),
            "rank": float(rank),
            "n_samples": int(n_ood),
        }
    return results


def compute_class_scores(class_metrics, weights=None):
    if not class_metrics:
        return {}
    weights = normalize_positive_weights(weights or DEFAULT_SCORE_WEIGHTS)
    classes = list(class_metrics.keys())

    ent = np.array([class_metrics[c]["entropy"] for c in classes], dtype=float)
    ali = np.array([class_metrics[c]["alignment"] for c in classes], dtype=float)
    ani = np.array([class_metrics[c]["anisotropy"] for c in classes], dtype=float)
    rnk = np.array([class_metrics[c]["rank"] for c in classes], dtype=float)

    ent_hard = minmax_hard(ent, higher_is_harder=True)
    ali_hard = minmax_hard(ali, higher_is_harder=False)
    ani_hard = minmax_hard(ani, higher_is_harder=True)
    rnk_hard = minmax_hard(rnk, higher_is_harder=False)

    scores = (
        weights["entropy"] * ent_hard
        + weights["alignment"] * ali_hard
        + weights["anisotropy"] * ani_hard
        + weights["rank"] * rnk_hard
    )

    for c, eh, ah, nh, rh, s in zip(classes, ent_hard, ali_hard, ani_hard, rnk_hard, scores):
        class_metrics[c]["entropy_hard"] = float(eh)
        class_metrics[c]["alignment_hard"] = float(ah)
        class_metrics[c]["anisotropy_hard"] = float(nh)
        class_metrics[c]["rank_hard"] = float(rh)
        class_metrics[c]["difficulty_score"] = float(s)

    return {int(c): float(s) for c, s in zip(classes, scores)}


def class_test_ratios_from_scores(class_scores):
    if not class_scores:
        return {}
    sorted_items = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    ratios = {}
    for rank, (c, _) in enumerate(sorted_items):
        percentile = rank / max(n - 1, 1)
        if percentile < 0.25:
            ratio = TEST_RATIO_HARD
        elif percentile < 0.75:
            ratio = TEST_RATIO_MIDDLE
        else:
            ratio = TEST_RATIO_EASY
        ratios[int(c)] = float(ratio)
    return ratios


def build_geometry_ood_train_test_split(ood_pool_raw, class_test_ratios):
    """
    Fixed global 80/20 split. class_test_ratios are allocation preferences only.
    """
    rng = np.random.RandomState(SEED)
    by_label = defaultdict(list)
    for item in ood_pool_raw:
        by_label[int(item["label"])].append(item)

    total_n = len(ood_pool_raw)
    target_test_total = int(round(total_n * FIXED_TEST_RATIO))
    labels = sorted(by_label.keys())

    per_class = {}
    total_capacity = 0
    for label in labels:
        items = list(by_label[label])
        rng.shuffle(items)
        n = len(items)
        max_test = max(0, n - MIN_TRAIN_PER_CLASS)
        total_capacity += max_test
        preference = float(class_test_ratios.get(label, TEST_RATIO_MIDDLE))
        per_class[label] = {
            "items": items,
            "n": n,
            "max_test": max_test,
            "preference": max(preference, 1e-12),
        }

    if target_test_total > total_capacity:
        print(
            f"[Warning] fixed test target {target_test_total} exceeds capacity {total_capacity}; "
            "using capacity instead."
        )
        target_test_total = total_capacity

    weights = {
        label: info["n"] * info["preference"] if info["max_test"] > 0 else 0.0
        for label, info in per_class.items()
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 1e-12:
        weights = {label: float(info["n"]) if info["max_test"] > 0 else 0.0 for label, info in per_class.items()}
        weight_sum = sum(weights.values())

    n_test_by_label = {}
    frac_records = []
    for label, info in per_class.items():
        if info["max_test"] <= 0 or weight_sum <= 1e-12:
            n_test_by_label[label] = 0
            continue
        raw = target_test_total * weights[label] / weight_sum
        clipped = min(raw, info["max_test"])
        base = int(np.floor(clipped))
        n_test_by_label[label] = base
        frac_records.append((clipped - base, label))

    current_total = sum(n_test_by_label.values())
    frac_records.sort(reverse=True)
    while current_total < target_test_total:
        changed = False
        for _, label in frac_records:
            if current_total >= target_test_total:
                break
            if n_test_by_label[label] < per_class[label]["max_test"]:
                n_test_by_label[label] += 1
                current_total += 1
                changed = True
        if not changed:
            break

    if current_total > target_test_total:
        frac_records_asc = sorted(frac_records)
        while current_total > target_test_total:
            changed = False
            for _, label in frac_records_asc:
                if current_total <= target_test_total:
                    break
                if n_test_by_label[label] > 0:
                    n_test_by_label[label] -= 1
                    current_total -= 1
                    changed = True
            if not changed:
                break

    train_data = []
    test_data = []
    report = {}
    for label in labels:
        info = per_class[label]
        items = info["items"]
        n_test = int(n_test_by_label[label])
        test_items = items[:n_test]
        train_items = items[n_test:]
        test_data.extend(test_items)
        train_data.extend(train_items)
        report[int(label)] = {
            "available": int(info["n"]),
            "train": int(len(train_items)),
            "test": int(len(test_items)),
            "test_ratio": float(len(test_items) / max(1, info["n"])),
            "allocation_preference": float(info["preference"]),
            "max_test_capacity": int(info["max_test"]),
        }

    rng.shuffle(train_data)
    rng.shuffle(test_data)

    actual_train_ratio = len(train_data) / max(1, total_n)
    actual_test_ratio = len(test_data) / max(1, total_n)
    print(
        f">>> Geometry OOD split: train={len(train_data)} ({actual_train_ratio:.4f}), "
        f"test={len(test_data)} ({actual_test_ratio:.4f}), target=0.80/0.20"
    )
    return train_data, test_data, report


def build_geometry_split(tokenizer):
    id_test_raw = read_jsonl(resolve_data_path(ID_TEST_FILE))
    ood_pool_raw = read_jsonl(resolve_data_path(OOD_POOL_FILE))
    if not ood_pool_raw:
        raise RuntimeError("No OOD pool samples loaded. Please check OOD_POOL_FILE.")

    id_test_ds = CodeDataset(id_test_raw, tokenizer, name="id_test_for_geometry")
    ood_pool_ds = CodeDataset(ood_pool_raw, tokenizer, name="ood_pool_for_geometry")
    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_pool_loader = DataLoader(ood_pool_ds, batch_size=BATCH_SIZE, shuffle=False)

    rq1_model = load_rq1_model()
    id_emb, _, id_labels, id_idxs = extract_embeddings_and_logits(
        rq1_model,
        id_test_loader,
        desc="Extract ID test embeddings for alignment",
    )
    ood_emb, ood_logits, ood_labels, ood_idxs = extract_embeddings_and_logits(
        rq1_model,
        ood_pool_loader,
        desc="Extract OOD pool embeddings",
    )

    class_metrics = compute_per_class_metrics(
        ood_emb,
        ood_logits,
        ood_labels,
        ood_idxs,
        id_emb,
        id_labels,
        id_idxs,
    )
    class_scores = compute_class_scores(class_metrics, DEFAULT_SCORE_WEIGHTS)
    class_test_ratios = class_test_ratios_from_scores(class_scores)
    train_raw, test_raw, split_report = build_geometry_ood_train_test_split(ood_pool_raw, class_test_ratios)

    write_jsonl(GEOMETRY_OOD_TRAIN_FILE, train_raw)
    write_jsonl(GEOMETRY_OOD_TEST_FILE, test_raw)

    sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    print(">>> Geometry class count:", len(class_metrics))
    print(">>> Top-10 hardest classes:")
    for c, score in sorted_scores[:10]:
        m = class_metrics[c]
        r = split_report.get(c, {})
        print(
            f"Class {c:<4} | score={score:.4f} | "
            f"test={r.get('test', 0)}/{r.get('available', 0)} | "
            f"test_ratio={r.get('test_ratio', 0.0):.3f} | "
            f"ent={m['entropy']:.3f} | align={m['alignment']:.3f} | "
            f"aniso={m['anisotropy']:.3f} | rank={m['rank']:.2f}"
        )

    with open(SPLIT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "RQ3-guided geometry-aware OOD train/test split for RQ4 Replay",
                "model_type": MODEL_TYPE,
                "model_path": MODEL_PATH,
                "rq1_best_model": RQ1_BEST_MODEL,
                "data_dir": DATA_DIR,
                "ood_pool_file": OOD_POOL_FILE,
                "geometry_ood_train_file": GEOMETRY_OOD_TRAIN_FILE,
                "geometry_ood_test_file": GEOMETRY_OOD_TEST_FILE,
                "fixed_global_split_ratio": {
                    "train_ratio": FIXED_TRAIN_RATIO,
                    "test_ratio": FIXED_TEST_RATIO,
                    "note": "class test ratios are allocation preferences under a fixed 20% test budget",
                },
                "score_weights": normalize_positive_weights(DEFAULT_SCORE_WEIGHTS),
                "test_ratio_hard": TEST_RATIO_HARD,
                "test_ratio_middle": TEST_RATIO_MIDDLE,
                "test_ratio_easy": TEST_RATIO_EASY,
                "class_metrics": class_metrics,
                "class_scores": class_scores,
                "class_test_ratios": class_test_ratios,
                "split_report": split_report,
                "geometry_ood_train_size": len(train_raw),
                "geometry_ood_test_size": len(test_raw),
                "geometry_ood_train_label_distribution": label_distribution(train_raw),
                "geometry_ood_test_label_distribution": label_distribution(test_raw),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    del rq1_model, id_emb, id_labels, id_idxs, ood_emb, ood_logits, ood_labels, ood_idxs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_raw, test_raw, split_report, class_metrics, class_scores


# ============================================================
# 7. Replay training
# ============================================================
def build_replay_buffer_dataset(id_train_ds, buffer_size=None, seed=42):
    if buffer_size is None or buffer_size <= 0 or buffer_size >= len(id_train_ds):
        indices = list(range(len(id_train_ds)))
        print(f">>> Replay buffer: using full ID train, size={len(indices)}")
        return id_train_ds, indices
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(id_train_ds), size=int(buffer_size), replace=False).tolist()
    indices = sorted(indices)
    print(f">>> Replay buffer: sampled {len(indices)} / {len(id_train_ds)} ID-train samples")
    return Subset(id_train_ds, indices), indices


def train_one_epoch_replay(model, ood_loader, replay_iter, optimizer, scheduler, epoch):
    model.train()
    total_loss_sum = 0.0
    ood_ce_sum = 0.0
    replay_ce_sum = 0.0
    weighted_replay_sum = 0.0
    steps = 0

    pbar = tqdm(ood_loader, desc=f"[Replay-Geo] Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()

        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        ).logits
        ood_ce = F.cross_entropy(logits, batch["label"].to(DEVICE))
        loss = ood_ce

        id_batch = next(replay_iter)
        id_logits = model(
            id_batch["input_ids"].to(DEVICE),
            id_batch["attention_mask"].to(DEVICE),
        ).logits
        replay_ce = F.cross_entropy(id_logits, id_batch["label"].to(DEVICE))
        weighted_replay = REPLAY_LAMBDA * replay_ce
        loss = loss + weighted_replay

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss_sum += float(loss.item())
        ood_ce_sum += float(ood_ce.detach().item())
        replay_ce_sum += float(replay_ce.detach().item())
        weighted_replay_sum += float(weighted_replay.detach().item())
        steps += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ood_ce": f"{ood_ce.detach().item():.4f}",
            "rep*w": f"{weighted_replay.detach().item():.4f}",
        })

    denom = max(steps, 1)
    stats = {
        "total_loss": total_loss_sum / denom,
        "ood_ce": ood_ce_sum / denom,
        "replay_ce": replay_ce_sum / denom,
        "replay_weighted": weighted_replay_sum / denom,
        "steps": int(steps),
    }
    print(
        f">>> [Replay-Geo] Epoch {epoch} loss breakdown | "
        f"OOD_CE={stats['ood_ce']:.6f} | "
        f"Replay_CE={stats['replay_ce']:.6f} | "
        f"lambda*Replay={stats['replay_weighted']:.6f} | "
        f"Total={stats['total_loss']:.6f}"
    )
    return stats


def print_epoch_table(epoch_metrics):
    print("\n" + "=" * 130)
    print("RQ4 Replay + Geometry OOD Split Epoch Metrics")
    print("=" * 130)
    print(
        f"{'Epoch':<7} | {'Loss':<10} | {'OOD_CE':<10} | {'Replay*w':<10} | "
        f"{'ID Acc':<10} | {'ID F1':<10} | {'ID Top5':<10} | "
        f"{'OOD Acc':<10} | {'OOD F1':<10} | {'OOD Top5':<10} | {'Best':<5}"
    )
    print("-" * 130)
    for row in epoch_metrics:
        id_eval = row["id_test"]
        ood_eval = row["ood_test"]
        loss_stats = row.get("loss_breakdown", {})
        best_mark = "*" if row.get("is_best") else ""
        print(
            f"{row['epoch']:<7} | {row['train_loss']:<10.4f} | "
            f"{loss_stats.get('ood_ce', 0.0):<10.4f} | "
            f"{loss_stats.get('replay_weighted', 0.0):<10.4f} | "
            f"{id_eval['accuracy']:<10.4f} | {id_eval['macro_f1']:<10.4f} | {id_eval['top5_acc']:<10.4f} | "
            f"{ood_eval['accuracy']:<10.4f} | {ood_eval['macro_f1']:<10.4f} | {ood_eval['top5_acc']:<10.4f} | "
            f"{best_mark:<5}"
        )
    print("=" * 130)



# ============================================================
# 8. SmartSearch utilities: geometry split search + Replay training
# ============================================================
def clip_float(value, low, high):
    return float(min(max(float(value), low), high))


def normalize_weight_config(weights):
    clipped = {
        name: clip_float(weights.get(name, DEFAULT_SCORE_WEIGHTS[name]), *bounds)
        for name, bounds in SEARCH_WEIGHT_BOUNDS.items()
    }
    return normalize_positive_weights(clipped)


def normalize_ratio_config(ratios):
    hard = clip_float(ratios.get("hard", TEST_RATIO_HARD), *SEARCH_RATIO_BOUNDS["hard"])
    middle = clip_float(ratios.get("middle", TEST_RATIO_MIDDLE), *SEARCH_RATIO_BOUNDS["middle"])
    easy = clip_float(ratios.get("easy", TEST_RATIO_EASY), *SEARCH_RATIO_BOUNDS["easy"])

    # Keep curriculum: hard classes get the smallest test allocation preference.
    middle = max(middle, hard)
    easy = max(easy, middle)
    return {"hard": float(hard), "middle": float(middle), "easy": float(easy)}


def make_search_config(name, weights, ratios):
    return {
        "name": str(name),
        "weights": normalize_weight_config(weights),
        "ratios": normalize_ratio_config(ratios),
    }


def config_signature(cfg, precision=5):
    values = []
    for key in ["entropy", "alignment", "anisotropy", "rank"]:
        values.append(round(float(cfg["weights"][key]), precision))
    for key in ["hard", "middle", "easy"]:
        values.append(round(float(cfg["ratios"][key]), precision))
    return tuple(values)


def class_test_ratios_from_scores(class_scores, ratio_config=None):
    """
    Map class difficulty to test allocation preferences.
    High-difficulty classes use the hard ratio, easy classes use the easy ratio.
    These ratios do NOT set the global test ratio; they only allocate the fixed 20% test budget.
    """
    if not class_scores:
        return {}
    ratio_config = normalize_ratio_config(ratio_config or {
        "hard": TEST_RATIO_HARD,
        "middle": TEST_RATIO_MIDDLE,
        "easy": TEST_RATIO_EASY,
    })
    sorted_items = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    ratios = {}
    for rank, (c, _) in enumerate(sorted_items):
        percentile = rank / max(n - 1, 1)
        if percentile < 0.25:
            ratio = ratio_config["hard"]
        elif percentile < 0.75:
            ratio = ratio_config["middle"]
        else:
            ratio = ratio_config["easy"]
        ratios[int(c)] = float(ratio)
    return ratios


def get_fixed_smart_stage1_configs():
    """
    24 interpretable SmartSearch configs: 4 weighting strategies × 6 allocation strategies.
    This is deterministic and contains no TPE/Optuna.
    """
    weight_templates = [
        ("balanced", {"entropy": 0.25, "alignment": 0.25, "anisotropy": 0.25, "rank": 0.25}),
        ("alignment_focus", {"entropy": 0.15, "alignment": 0.55, "anisotropy": 0.15, "rank": 0.15}),
        ("collapse_focus", {"entropy": 0.15, "alignment": 0.15, "anisotropy": 0.35, "rank": 0.35}),
        ("uncertainty_focus", {"entropy": 0.55, "alignment": 0.15, "anisotropy": 0.15, "rank": 0.15}),
    ]
    ratio_templates = [
        ("balanced_15", {"hard": 0.15, "middle": 0.15, "easy": 0.15}),
        ("default_curriculum", {"hard": 0.08, "middle": 0.10, "easy": 0.15}),
        ("strong_curriculum", {"hard": 0.05, "middle": 0.10, "easy": 0.20}),
        ("easy_more_test", {"hard": 0.06, "middle": 0.15, "easy": 0.30}),
        ("mid_easy_more_test", {"hard": 0.08, "middle": 0.20, "easy": 0.30}),
        ("high_test_all", {"hard": 0.15, "middle": 0.25, "easy": 0.35}),
    ]
    configs = []
    for w_name, weights in weight_templates:
        for r_name, ratios in ratio_templates:
            configs.append(make_search_config(f"{w_name}__{r_name}", weights, ratios))
    return configs


def sample_uniform_config(rng, name):
    weights = {k: rng.uniform(low, high) for k, (low, high) in SEARCH_WEIGHT_BOUNDS.items()}
    ratios = {k: rng.uniform(low, high) for k, (low, high) in SEARCH_RATIO_BOUNDS.items()}
    return make_search_config(name, weights, ratios)


def generate_candidate_configs(rng, count, prefix):
    if prefix == "stage1":
        return get_fixed_smart_stage1_configs()[:count]

    configs = []
    seen = set()
    attempts = 0
    while len(configs) < count and attempts < count * 30:
        attempts += 1
        cfg = sample_uniform_config(rng, f"{prefix}_{len(configs) + 1:03d}")
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(cfg)
    return configs


def split_feature_summary(split_report, train_size, test_size):
    if not split_report:
        return [float(train_size), float(test_size), 0.0, 0.0, 0.0, 0.0]
    available = []
    test_ratios = []
    for item in split_report.values():
        available.append(float(item.get("available", 0)))
        test_ratios.append(float(item.get("test_ratio", 0.0)))
    return [
        float(train_size),
        float(test_size),
        float(np.mean(available)),
        float(np.std(available)),
        float(np.mean(test_ratios)),
        float(np.std(test_ratios)),
    ]


def config_feature_vector(cfg, train_size=0, test_size=0, split_report=None):
    features = [
        float(cfg["weights"]["entropy"]),
        float(cfg["weights"]["alignment"]),
        float(cfg["weights"]["anisotropy"]),
        float(cfg["weights"]["rank"]),
        float(cfg["ratios"]["hard"]),
        float(cfg["ratios"]["middle"]),
        float(cfg["ratios"]["easy"]),
    ]
    features.extend(split_feature_summary(split_report, train_size, test_size))
    return features


def build_trial_split(cfg, class_metrics, ood_pool_raw):
    class_scores = compute_class_scores(class_metrics, cfg["weights"])
    class_test_ratios = class_test_ratios_from_scores(class_scores, cfg["ratios"])
    train_raw, test_raw, split_report = build_geometry_ood_train_test_split(ood_pool_raw, class_test_ratios)
    features = config_feature_vector(
        cfg,
        train_size=len(train_raw),
        test_size=len(test_raw),
        split_report=split_report,
    )
    return train_raw, test_raw, split_report, features, class_scores, class_test_ratios


def compute_search_select_score(id_eval, ood_eval, pre_id_eval=None, pre_ood_eval=None):
    id_macro_f1 = float(id_eval.get("macro_f1", 0.0))
    ood_macro_f1 = float(ood_eval.get("macro_f1", 0.0))
    id_accuracy = float(id_eval.get("accuracy", 0.0))
    ood_accuracy = float(ood_eval.get("accuracy", 0.0))
    id_top5 = float(id_eval.get("top5_acc", 0.0))
    ood_top5 = float(ood_eval.get("top5_acc", 0.0))

    pre_id_macro_f1 = float((pre_id_eval or {}).get("macro_f1", id_macro_f1))
    pre_ood_macro_f1 = float((pre_ood_eval or {}).get("macro_f1", ood_macro_f1))
    ood_macro_f1_gain = ood_macro_f1 - pre_ood_macro_f1
    id_macro_f1_change = id_macro_f1 - pre_id_macro_f1
    id_macro_f1_drop = max(0.0, -id_macro_f1_change)

    if SEARCH_SELECT_METRIC == "ood_macro_f1":
        score = ood_macro_f1
    elif SEARCH_SELECT_METRIC == "joint_macro_f1":
        total_weight = max(SEARCH_OOD_F1_WEIGHT + SEARCH_ID_F1_WEIGHT, 1e-8)
        score = (
            SEARCH_OOD_F1_WEIGHT * ood_macro_f1
            + SEARCH_ID_F1_WEIGHT * id_macro_f1
        ) / total_weight
    elif SEARCH_SELECT_METRIC == "joint_all":
        macro_score = 0.65 * ood_macro_f1 + 0.35 * id_macro_f1
        acc_score = 0.65 * ood_accuracy + 0.35 * id_accuracy
        top5_score = 0.65 * ood_top5 + 0.35 * id_top5
        score = 0.70 * macro_score + 0.20 * acc_score + 0.10 * top5_score
    elif SEARCH_SELECT_METRIC == "ood_gain_id_preserve":
        # The absolute term keeps the selected model useful, while the gain
        # term reduces the bias caused by different geometry-derived test
        # subsets.  Only degradation on ID is penalized; an ID improvement is
        # reported but is not allowed to dominate OOD adaptation.
        score = (
            SEARCH_POST_OOD_WEIGHT * ood_macro_f1
            + SEARCH_OOD_GAIN_WEIGHT * ood_macro_f1_gain
            - SEARCH_ID_DROP_PENALTY * id_macro_f1_drop
        )
    else:
        raise ValueError(
            "SEARCH_SELECT_METRIC must be one of: ood_macro_f1, joint_macro_f1, "
            "joint_all, ood_gain_id_preserve. "
            f"Got: {SEARCH_SELECT_METRIC}"
        )

    components = {
        "id_macro_f1": id_macro_f1,
        "ood_macro_f1": ood_macro_f1,
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "id_top5_acc": id_top5,
        "ood_top5_acc": ood_top5,
        "pre_id_macro_f1": pre_id_macro_f1,
        "pre_ood_macro_f1": pre_ood_macro_f1,
        "ood_macro_f1_gain": ood_macro_f1_gain,
        "id_macro_f1_change": id_macro_f1_change,
        "id_macro_f1_drop": id_macro_f1_drop,
        "search_select_metric": SEARCH_SELECT_METRIC,
        "search_post_ood_weight": float(SEARCH_POST_OOD_WEIGHT),
        "search_ood_gain_weight": float(SEARCH_OOD_GAIN_WEIGHT),
        "search_id_drop_penalty": float(SEARCH_ID_DROP_PENALTY),
        "search_ood_f1_weight": None if SEARCH_SELECT_METRIC == "ood_macro_f1" else float(SEARCH_OOD_F1_WEIGHT),
        "search_id_f1_weight": None if SEARCH_SELECT_METRIC == "ood_macro_f1" else float(SEARCH_ID_F1_WEIGHT),
    }
    return float(score), components


def fit_surrogate_model(search_results):
    usable = [x for x in search_results if x.get("features") is not None and x.get("select_score") is not None]
    if len(usable) < SURROGATE_MIN_SAMPLES:
        return None
    x = np.asarray([item["features"] for item in usable], dtype=float)
    y = np.asarray([item["select_score"] for item in usable], dtype=float)
    model = ExtraTreesRegressor(
        n_estimators=SURROGATE_N_ESTIMATORS,
        random_state=SEED,
        min_samples_leaf=2,
    )
    model.fit(x, y)
    return model


def surrogate_predict_with_uncertainty(model, features):
    x = np.asarray(features, dtype=float)
    tree_preds = np.asarray([tree.predict(x) for tree in model.estimators_])
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)
    acquisition = mean + SURROGATE_EXPLORATION_WEIGHT * std
    return mean, std, acquisition


def select_unique_configs(configs, limit, seen=None):
    seen = set() if seen is None else seen
    selected = []
    for cfg in configs:
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        selected.append(cfg)
        if len(selected) >= limit:
            break
    return selected


def train_replay_for_epochs(model, ood_train_loader, replay_buffer_ds, num_epochs, trial_name):
    replay_loader = DataLoader(replay_buffer_ds, batch_size=REPLAY_BATCH_SIZE, shuffle=True)
    replay_iter = itertools.cycle(replay_loader)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(ood_train_loader) * num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    train_losses = []
    loss_breakdowns = []
    for epoch in range(1, num_epochs + 1):
        loss_stats = train_one_epoch_replay(
            model,
            ood_train_loader,
            replay_iter,
            optimizer,
            scheduler,
            epoch,
        )
        train_losses.append(float(loss_stats["total_loss"]))
        loss_breakdowns.append(loss_stats)
        print(
            f">>> [{trial_name}] Epoch {epoch}/{num_epochs} | "
            f"loss={loss_stats['total_loss']:.4f} | "
            f"OOD_CE={loss_stats['ood_ce']:.4f} | "
            f"Replay*w={loss_stats['replay_weighted']:.4f}"
        )
    return model, train_losses, loss_breakdowns


def export_top_configs(search_results, path=SMART_TOP_CONFIGS_FILE, top_k=5):
    valid = [item for item in search_results if item.get("select_score") is not None]
    valid = sorted(valid, key=lambda x: x["select_score"], reverse=True)[:top_k]
    exported = []
    for item in valid:
        cfg = item["config"]
        exported.append({
            "trial": item.get("trial"),
            "stage": item.get("stage"),
            "select_metric": item.get("select_metric"),
            "select_score": float(item.get("select_score", 0.0)),
            "select_components": item.get("select_components"),
            "id_macro_f1": float(item.get("id_test", {}).get("macro_f1", 0.0)),
            "ood_macro_f1": float(item.get("ood_test", {}).get("macro_f1", 0.0)),
            "pretrain_ood_macro_f1": float(item.get("pretrain_ood_test", {}).get("macro_f1", 0.0)),
            "ood_macro_f1_gain": float(item.get("ood_macro_f1_gain", 0.0)),
            "id_macro_f1_change": float(item.get("id_macro_f1_change", 0.0)),
            "train_size": int(item.get("train_size", 0)),
            "ood_test_size": int(item.get("ood_test_size", 0)),
            "weights": cfg["weights"],
            "ratios": cfg["ratios"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Top SmartSearch configs. This file is for reporting/debugging only; no TPE/Optuna is used.",
            "top_k": int(top_k),
            "configs": exported,
        }, f, ensure_ascii=False, indent=2)
    print("Top SmartSearch configs saved:", path)
    return exported


# ============================================================
# 9. Main: SmartSearch without TPE
# ============================================================
def main():
    print("=" * 120)
    print("RQ4 Replay + Geometry SmartSearch (NO TPE / NO Optuna)")
    print("=" * 120)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("DATA_DIR:", DATA_DIR)
    print("SAVE_DIR:", SAVE_DIR)
    print("ID_TRAIN_FILE:", ID_TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_POOL_FILE:", OOD_POOL_FILE)
    print("NUM_LABELS:", NUM_LABELS)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("CLASSIFIER_HEAD_TYPE:", CLASSIFIER_HEAD_TYPE)
    print("Replay config:", {
        "lambda": REPLAY_LAMBDA,
        "batch_size": REPLAY_BATCH_SIZE,
        "buffer_size": REPLAY_BUFFER_SIZE,
    })
    print("SmartSearch config:", {
        "stage1_trials": SMART_STAGE1_TRIALS,
        "stage1_epochs": SMART_STAGE1_EPOCHS,
        "stage2_trials": SMART_STAGE2_TRIALS,
        "stage2_epochs": SMART_STAGE2_EPOCHS,
        "final_trials": SMART_FINAL_TRIALS,
        "final_epochs": SMART_FINAL_EPOCHS,
        "surrogate": "ExtraTreesRegressor",
        "tpe_optuna": False,
        "select_metric": SEARCH_SELECT_METRIC,
        "ood_f1_weight": SEARCH_OOD_F1_WEIGHT if SEARCH_SELECT_METRIC != "ood_macro_f1" else None,
        "id_f1_weight": SEARCH_ID_F1_WEIGHT if SEARCH_SELECT_METRIC != "ood_macro_f1" else None,
    })
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    rng = np.random.RandomState(SEED)

    print("\n" + "=" * 100)
    print("Phase 1: Load ID data and OOD pool")
    print("=" * 100)
    id_train_raw = read_jsonl(resolve_data_path(ID_TRAIN_FILE))
    id_test_raw = read_jsonl(resolve_data_path(ID_TEST_FILE))
    ood_pool_raw = read_jsonl(resolve_data_path(OOD_POOL_FILE))
    if not ood_pool_raw:
        raise RuntimeError("No OOD pool samples loaded. Please check OOD_POOL_FILE.")

    id_train_ds = CodeDataset(id_train_raw, tokenizer, name="id_train")
    id_test_ds = CodeDataset(id_test_raw, tokenizer, name="id_test")
    ood_pool_ds = CodeDataset(ood_pool_raw, tokenizer, name="ood_pool")

    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_pool_loader = DataLoader(ood_pool_ds, batch_size=BATCH_SIZE, shuffle=False)

    replay_buffer_ds, replay_buffer_indices = build_replay_buffer_dataset(
        id_train_ds,
        buffer_size=REPLAY_BUFFER_SIZE,
        seed=REPLAY_BUFFER_SEED,
    )

    print("\n" + "=" * 100)
    print("Phase 2: Compute RQ3 geometry metrics on OOD pool")
    print("=" * 100)
    rq1_model = load_rq1_model()
    rq1_id_baseline = evaluate(
        rq1_model,
        id_test_loader,
        desc="RQ1 pre-search ID baseline",
    )
    rq1_ood_pool_baseline = evaluate(
        rq1_model,
        ood_pool_loader,
        desc="RQ1 pre-search full OOD-pool baseline",
    )
    print(">>> RQ1 pre-search ID baseline:", rq1_id_baseline)
    print(">>> RQ1 pre-search full OOD-pool baseline:", rq1_ood_pool_baseline)
    id_emb, _, id_labels, id_idxs = extract_embeddings_and_logits(
        rq1_model,
        id_test_loader,
        desc="Extract ID test embeddings for alignment",
    )
    ood_emb, ood_logits, ood_labels, ood_idxs = extract_embeddings_and_logits(
        rq1_model,
        ood_pool_loader,
        desc="Extract OOD pool embeddings",
    )
    class_metrics = compute_per_class_metrics(
        ood_emb,
        ood_logits,
        ood_labels,
        ood_idxs,
        id_emb,
        id_labels,
        id_idxs,
    )
    print(">>> Geometry class count:", len(class_metrics))
    del rq1_model, id_emb, id_labels, id_idxs, ood_emb, ood_logits, ood_labels, ood_idxs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    search_results = []
    baseline_ood_cache = {}
    best_result = None
    best_train_raw = None
    best_test_raw = None
    best_split_report = None
    best_class_scores = None
    best_class_test_ratios = None

    def run_config(trial_idx, cfg, stage, num_epochs, eligible_for_best=False):
        nonlocal best_result, best_train_raw, best_test_raw, best_split_report, best_class_scores, best_class_test_ratios

        trial_name = f"{stage}_{trial_idx:03d}_{cfg['name']}"
        print("\n" + "-" * 120)
        print(f"Trial: {trial_name}")
        print("stage:", stage, "| replay_epochs:", num_epochs)
        print("weights:", cfg["weights"])
        print("ratios :", cfg["ratios"])
        print("-" * 120)

        set_seed(SEED)
        trial_train_raw, trial_test_raw, split_report, features, class_scores, class_test_ratios = build_trial_split(
            cfg,
            class_metrics,
            ood_pool_raw,
        )

        if not trial_train_raw or not trial_test_raw:
            print("Skip empty split.")
            return None

        train_ds = CodeDataset(trial_train_raw, tokenizer, name=f"{trial_name}_ood_train")
        test_ds = CodeDataset(trial_test_raw, tokenizer, name=f"{trial_name}_ood_test_new")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = load_rq1_model()

        # Measure how difficult this geometry-derived test subset already is
        # before training.  Cache by sample identity because several configs
        # can generate the same split.
        split_key = tuple(sorted((get_idx(x, i), int(x["label"])) for i, x in enumerate(trial_test_raw)))
        if split_key not in baseline_ood_cache:
            baseline_ood_cache[split_key] = evaluate(
                model,
                test_loader,
                desc=f"{trial_name} Pre-train OOD baseline",
            )
        pre_ood_eval = baseline_ood_cache[split_key]
        pre_id_eval = rq1_id_baseline

        trial_start = time.time()
        model, train_losses, loss_breakdowns = train_replay_for_epochs(
            model,
            train_loader,
            replay_buffer_ds,
            num_epochs,
            trial_name,
        )
        train_seconds = time.time() - trial_start

        id_eval = evaluate(model, id_test_loader, desc=f"{trial_name} ID Test")
        ood_eval = evaluate(model, test_loader, desc=f"{trial_name} OOD Test")
        select_score, select_components = compute_search_select_score(
            id_eval,
            ood_eval,
            pre_id_eval=pre_id_eval,
            pre_ood_eval=pre_ood_eval,
        )

        result = {
            "trial": trial_name,
            "stage": stage,
            "epochs": int(num_epochs),
            "config": cfg,
            "weights": cfg["weights"],
            "ratios": cfg["ratios"],
            "features": features,
            "train_size": len(trial_train_raw),
            "ood_test_size": len(trial_test_raw),
            "id_test": id_eval,
            "ood_test": ood_eval,
            "pretrain_id_test": pre_id_eval,
            "pretrain_ood_test": pre_ood_eval,
            "ood_macro_f1_gain": select_components["ood_macro_f1_gain"],
            "id_macro_f1_change": select_components["id_macro_f1_change"],
            "train_losses": train_losses,
            "loss_breakdowns": loss_breakdowns,
            "select_metric": SEARCH_SELECT_METRIC,
            "select_score": select_score,
            "select_components": select_components,
            "selection_set": "geometry_ood_test_new",
            "search_backend": "smart_surrogate_multifidelity_no_tpe",
            "train_seconds": float(train_seconds),
        }
        search_results.append(result)

        print(
            f"{trial_name} result | "
            f"ID F1={id_eval['macro_f1']:.4f} | "
            f"OOD F1={ood_eval['macro_f1']:.4f} | "
            f"OOD ΔF1={select_components['ood_macro_f1_gain']:+.4f} | "
            f"ID ΔF1={select_components['id_macro_f1_change']:+.4f} | "
            f"Select={select_score:.4f} | "
            f"OOD Acc={ood_eval['accuracy']:.4f} | "
            f"OOD Top5={ood_eval['top5_acc']:.4f} | "
            f"Time={format_time(train_seconds)}"
        )

        if eligible_for_best and (best_result is None or select_score > best_result["select_score"]):
            best_result = result
            best_train_raw = trial_train_raw
            best_test_raw = trial_test_raw
            best_split_report = split_report
            best_class_scores = class_scores
            best_class_test_ratios = class_test_ratios
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            write_jsonl(GEOMETRY_OOD_TRAIN_FILE, best_train_raw)
            write_jsonl(GEOMETRY_OOD_TEST_FILE, best_test_raw)
            print(f">>> New best full-budget Replay-Geo model saved: {BEST_MODEL_PATH}")
            print(f">>> Best geometry train/test files saved: {GEOMETRY_OOD_TRAIN_FILE}, {GEOMETRY_OOD_TEST_FILE}")

        with open(SMART_SEARCH_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)

        del model, train_ds, test_ds, train_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    print("\n" + "=" * 100)
    print("Phase 3: Stage-1 cheap probing with fixed logical configs")
    print("=" * 100)
    stage1_configs = generate_candidate_configs(rng, SMART_STAGE1_TRIALS, "stage1")
    stage1_results = []
    for idx, cfg in enumerate(stage1_configs, start=1):
        result = run_config(
            idx,
            cfg,
            stage="stage1_probe",
            num_epochs=SMART_STAGE1_EPOCHS,
            eligible_for_best=False,
        )
        if result is not None:
            stage1_results.append(result)

    if not stage1_results:
        raise RuntimeError("Stage-1 search finished without a valid trial.")

    print("\n" + "=" * 100)
    print("Phase 4: Train surrogate model and select Stage-2 candidates")
    print("=" * 100)
    surrogate = fit_surrogate_model(stage1_results)
    candidate_pool = generate_candidate_configs(rng, SMART_SURROGATE_POOL_SIZE, "surrogate_pool")
    pool_records = []
    for cfg in candidate_pool:
        train_raw, test_raw, split_report, features, _, _ = build_trial_split(cfg, class_metrics, ood_pool_raw)
        if not train_raw or not test_raw:
            continue
        pool_records.append({
            "config": cfg,
            "features": features,
            "train_size": len(train_raw),
            "ood_test_size": len(test_raw),
        })

    stage1_sorted = sorted(stage1_results, key=lambda item: item["select_score"], reverse=True)
    stage1_ood_sorted = sorted(
        stage1_results,
        key=lambda item: float(item.get("ood_test", {}).get("macro_f1", 0.0)),
        reverse=True,
    )
    stage1_gain_sorted = sorted(
        stage1_results,
        key=lambda item: float(item.get("ood_macro_f1_gain", -1e9)),
        reverse=True,
    )
    selected_seen = set()
    stage2_configs = select_unique_configs(
        [item["config"] for item in stage1_sorted],
        min(max(1, SMART_STAGE2_FROM_STAGE1 - 2), SMART_STAGE2_TRIALS),
        selected_seen,
    )
    if len(stage2_configs) < min(SMART_STAGE2_FROM_STAGE1, SMART_STAGE2_TRIALS):
        stage2_configs.extend(select_unique_configs(
            [item["config"] for item in stage1_ood_sorted],
            1,
            selected_seen,
        ))
    if len(stage2_configs) < min(SMART_STAGE2_FROM_STAGE1, SMART_STAGE2_TRIALS):
        stage2_configs.extend(select_unique_configs(
            [item["config"] for item in stage1_gain_sorted],
            1,
            selected_seen,
        ))

    if surrogate is not None and pool_records:
        pool_features = [item["features"] for item in pool_records]
        pred_mean, pred_std, acquisition = surrogate_predict_with_uncertainty(surrogate, pool_features)
        for item, mean_score, std_score, acq_score in zip(pool_records, pred_mean, pred_std, acquisition):
            item["surrogate_mean"] = float(mean_score)
            item["surrogate_std"] = float(std_score)
            item["surrogate_acquisition"] = float(acq_score)
        pool_records = sorted(pool_records, key=lambda item: item["surrogate_acquisition"], reverse=True)
        print("Surrogate trained with", len(stage1_results), "stage-1 probe trials.")
        print("Best surrogate candidate:", {
            "mean": pool_records[0]["surrogate_mean"],
            "std": pool_records[0]["surrogate_std"],
            "acquisition": pool_records[0]["surrogate_acquisition"],
            "config": pool_records[0]["config"],
        })
    else:
        print("Surrogate unavailable; using random/diverse Stage-2 candidates.")
        rng.shuffle(pool_records)

    surrogate_slots = max(0, SMART_STAGE2_TRIALS - SMART_STAGE2_RANDOM - len(stage2_configs))
    stage2_configs.extend(
        select_unique_configs(
            [item["config"] for item in pool_records],
            surrogate_slots,
            selected_seen,
        )
    )
    random_pool = list(pool_records)
    rng.shuffle(random_pool)
    stage2_configs.extend(
        select_unique_configs(
            [item["config"] for item in random_pool],
            SMART_STAGE2_TRIALS - len(stage2_configs),
            selected_seen,
        )
    )
    stage2_configs = stage2_configs[:SMART_STAGE2_TRIALS]
    print("Stage-2 selected configs:", len(stage2_configs))

    print("\n" + "=" * 100)
    print("Phase 5: Stage-2 medium-budget Replay training")
    print("=" * 100)
    stage2_results = []
    for idx, cfg in enumerate(stage2_configs, start=1):
        result = run_config(
            idx,
            cfg,
            stage="stage2_medium",
            num_epochs=SMART_STAGE2_EPOCHS,
            eligible_for_best=False,
        )
        if result is not None:
            stage2_results.append(result)

    if not stage2_results:
        print("WARNING: Stage-2 produced no valid result; falling back to best Stage-1 configs.")
        stage2_results = stage1_sorted[:SMART_FINAL_TRIALS]

    print("\n" + "=" * 100)
    print("Phase 6: Final full-budget Replay training for top candidates")
    print("=" * 100)
    final_seen = set()
    stage2_sorted = sorted(stage2_results, key=lambda item: item["select_score"], reverse=True)
    final_configs = select_unique_configs(
        [item["config"] for item in stage2_sorted],
        max(1, SMART_FINAL_TRIALS - 2),
        final_seen,
    )
    stage2_ood_sorted = sorted(
        stage2_results,
        key=lambda item: float(item.get("ood_test", {}).get("macro_f1", 0.0)),
        reverse=True,
    )
    stage2_gain_sorted = sorted(
        stage2_results,
        key=lambda item: float(item.get("ood_macro_f1_gain", -1e9)),
        reverse=True,
    )
    if len(final_configs) < SMART_FINAL_TRIALS:
        final_configs.extend(select_unique_configs(
            [item["config"] for item in stage2_ood_sorted],
            1,
            final_seen,
        ))
    if len(final_configs) < SMART_FINAL_TRIALS:
        final_configs.extend(select_unique_configs(
            [item["config"] for item in stage2_gain_sorted],
            1,
            final_seen,
        ))
    if len(final_configs) < SMART_FINAL_TRIALS:
        final_configs.extend(
            select_unique_configs(
                [item["config"] for item in stage1_sorted],
                SMART_FINAL_TRIALS - len(final_configs),
                final_seen,
            )
        )
    print("Final full-budget configs:", len(final_configs))

    final_results = []
    for idx, cfg in enumerate(final_configs, start=1):
        result = run_config(
            idx,
            cfg,
            stage="stage3_full",
            num_epochs=SMART_FINAL_EPOCHS,
            eligible_for_best=True,
        )
        if result is not None:
            final_results.append(result)

    if best_result is None:
        raise RuntimeError("SmartSearch finished without a full-budget best trial.")

    print("\n" + "=" * 100)
    print("Phase 7: Final evaluation of best SmartSearch Replay configuration")
    print("=" * 100)
    best_test_ds = CodeDataset(best_test_raw, tokenizer, name="best_geometry_ood_test_new")
    best_test_loader = DataLoader(best_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    best_model = load_rq1_model()
    best_model.load_state_dict(safe_torch_load(BEST_MODEL_PATH, map_location=DEVICE))

    id_pred_path = os.path.join(SAVE_DIR, f"predictions_id_test_replay_geo_smart_{MODEL_TYPE}.jsonl")
    ood_pred_path = os.path.join(SAVE_DIR, f"predictions_geometry_ood_test_replay_geo_smart_{MODEL_TYPE}.jsonl")
    final_id_eval = evaluate(best_model, id_test_loader, desc="Best Smart Replay-Geo ID Test", save_path=id_pred_path)
    final_ood_eval = evaluate(best_model, best_test_loader, desc="Best Smart Replay-Geo OOD Test", save_path=ood_pred_path)

    print("\n" + "=" * 120)
    print("RQ4 Replay + Geometry SmartSearch Final Results")
    print("=" * 120)
    print(f"{'Metric':<15} | {'ID Test':<12} | {'OOD Test':<12}")
    print("-" * 45)
    for key, name in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("top5_acc", "Top-5 Acc")]:
        print(f"{name:<15} | {final_id_eval[key]:<12.4f} | {final_ood_eval[key]:<12.4f}")
    print("=" * 120)
    print("Best full-budget trial:", best_result["trial"])
    print("Best select score:", best_result["select_score"])
    print("Best select components:", best_result.get("select_components"))
    print("Best weights:", best_result["weights"])
    print("Best ratios:", best_result["ratios"])
    print("Best model:", BEST_MODEL_PATH)

    top_configs = export_top_configs(search_results, SMART_TOP_CONFIGS_FILE, top_k=5)

    with open(SPLIT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "method": "RQ4 Replay + RQ3-guided geometry SmartSearch without TPE",
            "model_type": MODEL_TYPE,
            "model_path": MODEL_PATH,
            "rq1_best_model": RQ1_BEST_MODEL,
            "data_dir": DATA_DIR,
            "ood_pool_file": OOD_POOL_FILE,
            "best_trial": best_result,
            "best_split_report": best_split_report,
            "best_class_scores": best_class_scores,
            "best_class_test_ratios": best_class_test_ratios,
            "best_geometry_ood_train_file": GEOMETRY_OOD_TRAIN_FILE,
            "best_geometry_ood_test_file": GEOMETRY_OOD_TEST_FILE,
            "fixed_global_split_ratio": {
                "train_ratio": FIXED_TRAIN_RATIO,
                "test_ratio": FIXED_TEST_RATIO,
                "note": "hard/middle/easy ratios are allocation preferences only; global test ratio is fixed.",
            },
        }, f, ensure_ascii=False, indent=2)

    metrics = {
        "task": "Code Classification",
        "method": "RQ4 Replay with RQ3-guided Geometry SmartSearch, no TPE/Optuna",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "rq1_best_model": RQ1_BEST_MODEL,
        "data_dir": DATA_DIR,
        "id_train_file": ID_TRAIN_FILE,
        "id_test_file": ID_TEST_FILE,
        "ood_pool_file": OOD_POOL_FILE,
        "label_setting": "0-based labels; valid labels are [0, NUM_LABELS-1]",
        "num_labels": NUM_LABELS,
        "encoder_pooling": resolve_pooling_strategy(MODEL_TYPE, ENCODER_POOLING),
        "classifier_head_type": CLASSIFIER_HEAD_TYPE,
        "replay_config": {
            "lambda": REPLAY_LAMBDA,
            "batch_size": REPLAY_BATCH_SIZE,
            "buffer_size": REPLAY_BUFFER_SIZE,
            "buffer_source": "ID train / previous-task training data",
            "buffer_actual_size": len(replay_buffer_ds),
            "buffer_indices": replay_buffer_indices,
        },
        "smart_search": {
            "tpe_optuna": False,
            "stage1_trials": SMART_STAGE1_TRIALS,
            "stage1_epochs": SMART_STAGE1_EPOCHS,
            "stage2_trials": SMART_STAGE2_TRIALS,
            "stage2_epochs": SMART_STAGE2_EPOCHS,
            "stage2_from_stage1": SMART_STAGE2_FROM_STAGE1,
            "stage2_random": SMART_STAGE2_RANDOM,
            "final_trials": SMART_FINAL_TRIALS,
            "final_epochs": SMART_FINAL_EPOCHS,
            "surrogate_model": "ExtraTreesRegressor",
            "surrogate_pool_size": SMART_SURROGATE_POOL_SIZE,
            "select_metric": SEARCH_SELECT_METRIC,
            "select_score_definition": {
                "ood_macro_f1": "OOD Macro-F1 only",
                "joint_macro_f1": "0.65 * OOD Macro-F1 + 0.35 * ID Macro-F1",
                "ood_f1_weight": SEARCH_OOD_F1_WEIGHT if SEARCH_SELECT_METRIC != "ood_macro_f1" else None,
                "id_f1_weight": SEARCH_ID_F1_WEIGHT if SEARCH_SELECT_METRIC != "ood_macro_f1" else None,
                "note": "The surrogate and final best trial are selected by SEARCH_SELECT_METRIC.",
            },
            "selection_set": "geometry_ood_test_new",
        },
        "geometry_definition": {
            "shift_type": SHIFT_TYPE,
            "alignment_mode": ALIGNMENT_MODE,
            "entropy": "OOD sample-level prediction entropy averaged within class",
            "alignment": "token/cst use pair-level idx+label when available; otherwise label-centroid alignment",
            "anisotropy": "OOD class-wise average pairwise cosine similarity",
            "effective_rank": "OOD class-wise centered SVD effective rank",
            "standardization": "hard-oriented min-max normalization; positive weights normalized to sum 1",
        },
        "best_trial": best_result,
        "best_geometry_ood_train_size": len(best_train_raw),
        "best_geometry_ood_test_size": len(best_test_raw),
        "best_geometry_ood_train_label_distribution": label_distribution(best_train_raw),
        "best_geometry_ood_test_label_distribution": label_distribution(best_test_raw),
        "search_results": search_results,
        "final_evaluation": {
            "id_test": final_id_eval,
            "ood_test": final_ood_eval,
        },
        "files": {
            "best_model": BEST_MODEL_PATH,
            "best_geometry_ood_train": GEOMETRY_OOD_TRAIN_FILE,
            "best_geometry_ood_test": GEOMETRY_OOD_TEST_FILE,
            "search_results": SMART_SEARCH_RESULTS_FILE,
            "top_configs": SMART_TOP_CONFIGS_FILE,
            "split_report": SPLIT_REPORT_FILE,
            "id_predictions": id_pred_path,
            "ood_predictions": ood_pred_path,
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
    print("Search results:", SMART_SEARCH_RESULTS_FILE)
    print("Split report:", SPLIT_REPORT_FILE)
    print("Top configs:", SMART_TOP_CONFIGS_FILE)
    print("Total time:", format_time(time.time() - START_TIME))


if __name__ == "__main__":
    main()
