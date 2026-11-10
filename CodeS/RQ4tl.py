import os
import time
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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
MODEL_TYPE = "RoBERTa" # 可选: "codebert", "GraphCodeBERT", "codeT5"
SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

# Run all four models by default. Set False and edit MODEL_TYPE to run one model.
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_TYPE = "RoBERTa"
DATA_DIR = "/opt/data/private/codes/Java250/Java250/token/pre-trained"
SAVE_DIR = "/opt/data/private/codes/Java250/Java250/token/RQ4_tl_results"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASET_NAME = "Java250"
RQ1_MODEL_DIR = os.path.dirname(DATA_DIR)

ID_TEST_FILE = "id_test.jsonl"
OOD_TRAIN_FILE = "ood_train.jsonl"
OOD_TEST_FILE = "ood_test_new.jsonl"

NUM_LABELS = 250 #记得替换

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 8
LR_FT = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
TOP_K = 5
SEED = 42
CLASSIFIER_HEAD_TYPE = "auto"  # auto / single / mlp_seq / mlp_named
ENCODER_POOLING = "auto"  # auto / cls / mean

ALL_MODELS_METRICS_FILE = os.path.join(SAVE_DIR, "rq4_tl_fixed_ood_split_metrics_all_models.json")


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
    global MODEL_TYPE, MODEL_PATH, RQ1_BEST_MODEL, FINAL_MODEL_PATH, METRICS_FILE

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = resolve_model_path(MODEL_TYPE)
    RQ1_BEST_MODEL = build_rq1_best_model_path(MODEL_TYPE)
    FINAL_MODEL_PATH = os.path.join(SAVE_DIR, f"final_transfer_{MODEL_TYPE}.bin")
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq4_tl_fixed_ood_split_metrics_{MODEL_TYPE}.json")


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


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    model_type_lower = model_type.lower()

    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")

    return "mean" if ("t5" in model_type_lower or "plbart" in model_type_lower) else "cls"


def infer_head_type_from_state_dict(state_dict):
    keys = [strip_prefix(key) for key in state_dict.keys()]

    if any(key.startswith("classifier.0.") or key.startswith("classifier.3.") for key in keys):
        return "mlp_seq"
    if any(key.startswith("classifier.dense.") or key.startswith("classifier.out_proj.") for key in keys):
        return "mlp_named"
    if any(key in ("classifier.weight", "classifier.bias", "score.weight", "score.bias") for key in keys):
        return "single"
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

    def forward(self, features):
        x = features[:, 0, :] if features.dim() == 3 else features
        x = self.dropout(x)
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

    def pool_encoder_output(self, hidden_states, attention_mask):
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = torch.sum(hidden_states * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / denom
        return hidden_states[:, 0, :]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        features = self.pool_encoder_output(outputs.last_hidden_state, attention_mask)

        if self.head_type == "single":
            logits = self.classifier(self.dropout(features))
        elif self.head_type == "mlp_named":
            logits = self.classifier(features)
        else:
            logits = self.classifier(features)

        return type("ModelOutput", (object,), {"logits": logits})


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        raise FileNotFoundError(f"RQ1 best model not found: {RQ1_BEST_MODEL}")

    raw_state = unwrap_checkpoint(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    head_type = infer_head_type_from_state_dict(raw_state) if CLASSIFIER_HEAD_TYPE == "auto" else CLASSIFIER_HEAD_TYPE
    model = CodeEncoder(
        MODEL_PATH,
        NUM_LABELS,
        MODEL_TYPE,
        head_type=head_type,
        pooling=ENCODER_POOLING,
        load_pretrained=False,
    ).to(DEVICE)
    adapted_state = adapt_state_dict_keys(raw_state, head_type=head_type)
    missing, unexpected = model.load_state_dict(adapted_state, strict=False)
    matched = [key for key in adapted_state.keys() if key in model.state_dict()]
    encoder_matched = [key for key in matched if key.startswith("encoder.")]
    classifier_matched = [key for key in matched if key.startswith("classifier.")]

    print(
        f">>> Loaded RQ1 model: model_type={MODEL_TYPE}, head_type={head_type}, "
        f"pooling={model.pooling_strategy}, "
        f"keys={len(adapted_state)}, matched={len(matched)}, "
        f"encoder_matched={len(encoder_matched)}, classifier_matched={len(classifier_matched)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    missing_show = [key for key in missing if "pooler" not in key][:20]
    if missing_show:
        print(">>> Missing keys preview:", missing_show)
    if unexpected:
        print(">>> Unexpected keys preview:", unexpected[:20])

    if len(classifier_matched) == 0:
        raise RuntimeError(
            "No classifier parameters were loaded from the RQ1 checkpoint. "
            "Check CLASSIFIER_HEAD_TYPE and checkpoint classifier key names."
        )
    if len(encoder_matched) == 0:
        raise RuntimeError(
            "No encoder parameters were loaded from the RQ1 checkpoint. "
            "Check MODEL_TYPE, MODEL_PATH, and RQ1_BEST_MODEL."
        )

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
        print(">>> Label distribution preview:", dict(Counter(int(x["label"]) for x in self.examples).most_common(10)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_fixed_ood_datasets(tokenizer):
    print(f">>> Using fixed OOD train file: {OOD_TRAIN_FILE}")
    print(f">>> Using fixed OOD test file : {OOD_TEST_FILE}")
    return CodeDataset(OOD_TRAIN_FILE, tokenizer), CodeDataset(OOD_TEST_FILE, tokenizer)


# ================= 5. Evaluation / Training =================
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
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "top5_acc": 0.0,
            "num_samples": 0,
        }

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


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    steps = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        ).logits
        loss = F.cross_entropy(logits, batch["label"].to(DEVICE))
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(steps, 1)


def print_final_table(id_eval, ood_eval):
    print("\n" + "=" * 80)
    print("RQ4 Transfer Learning Final Results")
    print("=" * 80)
    print(f"{'Metric':<15} | {'RQ4 ID Test':<12} | {'RQ4 OOD Test':<12}")
    print("-" * 48)
    for key, name in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("top5_acc", "Top-5 Acc"),
    ]:
        print(f"{name:<15} | {id_eval[key]:<12.4f} | {ood_eval[key]:<12.4f}")
    print("=" * 80)


def print_epoch_metrics_table(epoch_metrics):
    print("\n" + "=" * 118)
    print("RQ4 Transfer Learning Epoch Metrics")
    print("=" * 118)
    print(
        f"{'Epoch':<7} | {'Loss':<10} | "
        f"{'ID Acc':<10} | {'ID F1':<10} | {'ID Top5':<10} | "
        f"{'OOD Acc':<10} | {'OOD F1':<10} | {'OOD Top5':<10} | {'Best':<5}"
    )
    print("-" * 118)
    for row in epoch_metrics:
        id_eval = row["id_test"]
        ood_eval = row["ood_test"]
        best_mark = "*" if row.get("is_best") else ""
        print(
            f"{row['epoch']:<7} | {row['train_loss']:<10.4f} | "
            f"{id_eval['accuracy']:<10.4f} | {id_eval['macro_f1']:<10.4f} | {id_eval['top5_acc']:<10.4f} | "
            f"{ood_eval['accuracy']:<10.4f} | {ood_eval['macro_f1']:<10.4f} | {ood_eval['top5_acc']:<10.4f} | "
            f"{best_mark:<5}"
        )
    print("=" * 118)


# ================= 6. Main =================
def main(model_type=None):
    if model_type is not None:
        configure_model_runtime(model_type)
    else:
        configure_model_runtime(MODEL_TYPE)
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("=" * 100)
    print("RQ4-TL: fixed OOD train/test files, no validation model selection")
    print("=" * 100)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("SUPPORTED_MODEL_TYPES:", SUPPORTED_MODEL_TYPES)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("OOD_TRAIN_FILE:", OOD_TRAIN_FILE)
    print("OOD_TEST_FILE:", OOD_TEST_FILE)
    print("CLASSIFIER_HEAD_TYPE:", CLASSIFIER_HEAD_TYPE)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("=" * 100)

    id_test_ds = CodeDataset(ID_TEST_FILE, tokenizer)
    ood_train_ds, ood_test_ds = load_fixed_ood_datasets(tokenizer)

    ood_train_loader = DataLoader(ood_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_test_loader = DataLoader(ood_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = load_rq1_model()

    optimizer = AdamW(model.parameters(), lr=LR_FT, weight_decay=WEIGHT_DECAY)
    total_steps = len(ood_train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    train_losses = []
    epoch_metrics = []
    best_epoch = 0
    best_ood_macro_f1 = -1.0
    best_epoch_model_path = None
    train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, ood_train_loader, optimizer, scheduler, epoch)
        train_losses.append(float(loss))
        epoch_path = os.path.join(SAVE_DIR, f"transfer_{MODEL_TYPE}_epoch_{epoch}.bin")
        torch.save(model.state_dict(), epoch_path)
        print(f"Epoch {epoch} | Train loss={loss:.4f}")

        epoch_id_eval = evaluate(model, id_test_loader, desc=f"Epoch {epoch} ID Test")
        epoch_ood_eval = evaluate(model, ood_test_loader, desc=f"Epoch {epoch} OOD Test")
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(loss),
            "id_test": epoch_id_eval,
            "ood_test": epoch_ood_eval,
            "is_best": False,
        }
        epoch_metrics.append(epoch_record)

        if epoch_ood_eval["macro_f1"] > best_ood_macro_f1:
            best_ood_macro_f1 = float(epoch_ood_eval["macro_f1"])
            best_epoch = epoch
            best_epoch_model_path = epoch_path
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            for item in epoch_metrics:
                item["is_best"] = item["epoch"] == best_epoch
            print(
                f">>> New best transfer epoch by OOD test Macro-F1: "
                f"epoch={best_epoch}, ood_macro_f1={best_ood_macro_f1:.4f}"
            )

    train_time = time.time() - train_start

    if best_epoch == 0:
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        best_epoch = EPOCHS
        best_epoch_model_path = FINAL_MODEL_PATH
        best_ood_macro_f1 = epoch_metrics[-1]["ood_test"]["macro_f1"] if epoch_metrics else float("nan")
        if epoch_metrics:
            epoch_metrics[-1]["is_best"] = True

    model.load_state_dict(safe_torch_load(FINAL_MODEL_PATH, map_location=DEVICE))
    print(f">>> Loaded best transfer model: epoch={best_epoch}, ood_macro_f1={best_ood_macro_f1:.4f}")

    final_id_eval = evaluate(model, id_test_loader, desc="Best Epoch ID Test")
    final_ood_eval = evaluate(model, ood_test_loader, desc="Best Epoch OOD Test")
    print_epoch_metrics_table(epoch_metrics)
    print_final_table(final_id_eval, final_ood_eval)

    metrics = {
        "task": "Code Classification",
        "method": "RQ4-TL fixed OOD train/test files without validation",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "rq1_best_model": RQ1_BEST_MODEL,
        "data_dir": DATA_DIR,
        "dataset_name": DATASET_NAME,
        "supported_model_types": SUPPORTED_MODEL_TYPES,
        "ood_train_file": OOD_TRAIN_FILE,
        "ood_test_file": OOD_TEST_FILE,
        "classifier_head_type": CLASSIFIER_HEAD_TYPE,
        "encoder_pooling_config": ENCODER_POOLING,
        "encoder_pooling_actual": model.pooling_strategy,
        "split": {
            "ood_train_size": len(ood_train_ds),
            "ood_test_size": len(ood_test_ds),
        },
        "final": {
            "selection_metric": "highest_ood_test_new_macro_f1",
            "best_epoch": int(best_epoch),
            "best_ood_macro_f1": float(best_ood_macro_f1),
            "id_test": final_id_eval,
            "ood_test": final_ood_eval,
            "train_losses": train_losses,
            "epoch_metrics": epoch_metrics,
        },
        "files": {
            "final_model": FINAL_MODEL_PATH,
            "best_epoch_model": best_epoch_model_path,
            "metrics_file": METRICS_FILE,
        },
        "time": {
            "train_seconds": train_time,
            "total_seconds": time.time() - START_TIME,
            "total_readable": format_time(time.time() - START_TIME),
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Final model:", FINAL_MODEL_PATH)
    print("Metrics file:", METRICS_FILE)
    print("Total time:", format_time(time.time() - START_TIME))
    return metrics


if __name__ == "__main__":
    if RUN_ALL_MODELS:
        all_results = []
        for run_model_type in MODEL_TYPES_TO_RUN:
            print("\n\n" + "#" * 120)
            print(f"Running RQ4 transfer learning for model: {run_model_type}")
            print("#" * 120)
            all_results.append(main(run_model_type))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        with open(ALL_MODELS_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": "Code Classification",
                    "method": "RQ4-TL fixed OOD train/test files without validation",
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
