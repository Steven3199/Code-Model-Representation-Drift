import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# ============================================================
# 1. Config: CodeS Code Classification RQ1
# ============================================================

# optional: "codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"
MODEL_TYPE = "GraphCodeBERT"
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}


def resolve_model_path(model_type):
    if model_type in MODEL_PATHS:
        return MODEL_PATHS[model_type]
    model_type_lower = model_type.lower()
    for name, path in MODEL_PATHS.items():
        if name.lower() == model_type_lower:
            return path
    return model_type


MODEL_PATH = resolve_model_path(MODEL_TYPE)

# Example paths. Change these according to dataset / shift type.
DATA_DIR = "/opt/data/private/codes/Python800/Python800/cst/pre-trained"
SAVE_DIR = "/opt/data/private/codes/Python800/Python800/cst"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASET_NAME = "Python800"
BEST_SAVE_NAME = f"best_model_{MODEL_TYPE}_{DATASET_NAME}.bin"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, BEST_SAVE_NAME)
METRICS_FILE = os.path.join(SAVE_DIR, f"rq1_metrics_{MODEL_TYPE}_{DATASET_NAME}.json")
CSV_FILE = os.path.join(SAVE_DIR, f"rq1_epoch_metrics_{MODEL_TYPE}_{DATASET_NAME}.csv")

TRAIN_FILE = "train.jsonl"
ID_TEST_FILE = "id_test.jsonl"
OOD_TEST_FILE = "ood_test.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

# All your CodeS labels are 0-based: Java250, Python75, Python800.
# Therefore labels are used directly and must be in [0, NUM_LABELS - 1].
NUM_LABELS = 800
TOP_K = 5

# Pooling strategy:
#   auto: CodeBERT / GraphCodeBERT / RoBERTa -> CLS/first-token; CodeT5 / PLBART -> mean pooling.
#   cls : force first-token pooling.
#   mean: force mean pooling.
ENCODER_POOLING = "auto"

NUM_WORKERS = 4
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"


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
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒"


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_data_path(file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(DATA_DIR, file_name)


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    model_type_lower = model_type.lower()

    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")

    # CodeT5 and PLBART are encoder-decoder style models and have no dedicated CLS token.
    # Mean pooling over non-padding tokens is a safer classification representation.
    # RoBERTa is encoder-only, so it follows the same CLS/first-token pooling as CodeBERT.
    return "mean" if ("t5" in model_type_lower or "plbart" in model_type_lower) else "cls"


def get_code_text(js):
    return (
        js.get("func")
        or js.get("code")
        or js.get("function")
        or js.get("source")
        or js.get("original_string")
        or ""
    )


def get_label(js):
    label = js.get("label")
    if label is None:
        label = js.get("target")
    return label


# ============================================================
# 3. Dataset
# ============================================================

class CodeDataset(Dataset):
    def __init__(self, file_name, tokenizer, name=None):
        self.examples = []
        self.path = resolve_data_path(file_name)
        self.name = name or os.path.basename(self.path)
        self.tokenizer = tokenizer

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Cannot find file: {self.path}")

        skipped_empty = 0
        skipped_json = 0
        skipped_label = 0
        labels_seen = []
        label_counter = Counter()

        with open(self.path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"加载 {self.name}", leave=False)):
                line = line.strip()
                if not line:
                    skipped_empty += 1
                    continue

                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    skipped_json += 1
                    continue

                code_text = get_code_text(js)
                label = get_label(js)

                if not code_text or label is None:
                    skipped_empty += 1
                    continue

                label = int(label)
                labels_seen.append(label)

                # Labels are 0-based in your datasets. We do not subtract 1.
                if label < 0 or label >= NUM_LABELS:
                    skipped_label += 1
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
                })
                label_counter[label] += 1

        print("=" * 100)
        print(f">>> CodeDataset loaded: {self.name}")
        print("path:", self.path)
        print("num_samples:", len(self.examples))
        print("skipped_empty_or_missing:", skipped_empty)
        print("skipped_json:", skipped_json)
        print("skipped_label_out_of_range:", skipped_label)
        if labels_seen:
            print("raw label min/max:", min(labels_seen), max(labels_seen))
        print("valid label min/max:", (min(label_counter.keys()), max(label_counter.keys())) if label_counter else None)
        print("num_unique_valid_labels:", len(label_counter))
        print("=" * 100)

        if len(self.examples) == 0:
            raise ValueError(f"No valid samples loaded from {self.path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# 4. Model: adaptive encoder + pooling + single linear head
# ============================================================

class BaselineModel(nn.Module):
    """
    RQ1 full fine-tuning classification model.

    Model adaptation:
      - CodeBERT / GraphCodeBERT / RoBERTa: AutoModel + CLS/first-token pooling.
      - CodeT5: T5EncoderModel + mean pooling by default.
      - PLBART: AutoModel.get_encoder() + mean pooling by default.

    Classifier head:
      - A single Linear layer named `classifier`.
      - State dict keys are classifier.weight and classifier.bias.
      - This is compatible with RQ2's `single` head auto-detection and final-head check.
    """
    def __init__(self, model_path, num_labels, model_type, pooling="auto"):
        super().__init__()
        self.model_type = model_type.lower()
        self.pooling_strategy = resolve_pooling_strategy(model_type, pooling)

        if "t5" in self.model_type:
            self.encoder = T5EncoderModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            base_model = AutoModel.from_pretrained(model_path)
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
            self.encoder = AutoModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        for param in self.parameters():
            param.requires_grad = True

    def pool_encoder_output(self, hidden_states, attention_mask):
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / denom
        return hidden_states[:, 0, :]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = self.pool_encoder_output(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return type("ModelOutput", (object,), {"logits": logits})


# ============================================================
# 5. Evaluation
# ============================================================

@torch.no_grad()
def evaluate_classification_metrics(model, dataloader, desc="分类评估"):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(input_ids, attention_mask).logits

        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = torch.argmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()

    acc = accuracy_score(labels_np, preds)
    macro_f1 = f1_score(labels_np, preds, average="macro", labels=list(range(NUM_LABELS)), zero_division=0)

    k = min(TOP_K, NUM_LABELS)
    _, topk = all_logits.topk(k, dim=1)
    topk_acc = topk.eq(all_labels.view(-1, 1)).any(dim=1).float().mean().item()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "top5_acc": float(topk_acc),
        "num_samples": int(len(all_labels)),
    }


def print_final_table(train_eval, id_eval, ood_eval):
    print("\n" + "=" * 90)
    print(f"{'指标类型':<15} | {'ID Train':<12} | {'ID Test':<12} | {'OOD Test':<12} | {'跌幅 ID-OOD':<12}")
    print("-" * 90)
    for metric_key, metric_name in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("top5_acc", "Top-5 Acc"),
    ]:
        print(
            f"{metric_name:<15} | "
            f"{train_eval[metric_key]:<12.4f} | "
            f"{id_eval[metric_key]:<12.4f} | "
            f"{ood_eval[metric_key]:<12.4f} | "
            f"{id_eval[metric_key] - ood_eval[metric_key]:<12.4f}"
        )
    print("=" * 90)


# ============================================================
# 6. Training
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    step_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(input_ids, attention_mask).logits
            loss = F.cross_entropy(logits, labels)

        if torch.isnan(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        step_count += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(step_count, 1)


# ============================================================
# 7. Main
# ============================================================

def main():
    program_start = time.time()

    print("=" * 100)
    print("CodeS RQ1: Adaptive Code Classification Fine-tuning")
    print("=" * 100)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("SUPPORTED_MODEL_TYPES:", list(MODEL_PATHS.keys()))
    print("DATA_DIR:", DATA_DIR)
    print("SAVE_DIR:", SAVE_DIR)
    print("BEST_MODEL_PATH:", BEST_MODEL_PATH)
    print("NUM_LABELS:", NUM_LABELS)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("DEVICE:", DEVICE)
    print("USE_AMP:", USE_AMP)
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = BaselineModel(MODEL_PATH, NUM_LABELS, MODEL_TYPE, pooling=ENCODER_POOLING).to(DEVICE)
    print(f">>> Actual encoder pooling strategy: {model.pooling_strategy}")
    print(">>> Classifier head type: single Linear, keys: classifier.weight / classifier.bias")

    print("\n>>> Loading datasets...")
    train_ds = CodeDataset(TRAIN_FILE, tokenizer, name="id_train")
    id_test_ds = CodeDataset(ID_TEST_FILE, tokenizer, name="id_test")
    ood_test_ds = CodeDataset(OOD_TEST_FILE, tokenizer, name="ood_test")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    eval_train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    eval_id_loader = DataLoader(
        id_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    eval_ood_loader = DataLoader(
        ood_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_f1 = -1.0
    best_epoch = 0
    epoch_records = []

    print("\n" + "=" * 40 + " 🚀 RQ1 开始基线全量微调 " + "=" * 40)
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)

        id_eval = evaluate_classification_metrics(model, eval_id_loader, desc=f"验证 Epoch {epoch} on ID Test")
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch} | "
            f"Loss: {train_loss:.4f} | "
            f"ID Acc: {id_eval['accuracy']:.4f} | "
            f"ID Macro-F1: {id_eval['macro_f1']:.4f} | "
            f"ID Top-{TOP_K}: {id_eval['top5_acc']:.4f} | "
            f"Epoch Time: {format_time(epoch_time)}"
        )

        epoch_save_path = os.path.join(SAVE_DIR, f"model_{MODEL_TYPE}_epoch_{epoch}.bin")
        torch.save(model.state_dict(), epoch_save_path)

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "id_accuracy": id_eval["accuracy"],
            "id_macro_f1": id_eval["macro_f1"],
            "id_top5_acc": id_eval["top5_acc"],
            "epoch_seconds": float(epoch_time),
            "epoch_readable": format_time(epoch_time),
            "checkpoint": epoch_save_path,
        }
        epoch_records.append(record)

        if id_eval["macro_f1"] > best_f1:
            best_f1 = id_eval["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✨ 发现新高！最佳模型已更新至: {BEST_MODEL_PATH}")

    train_time = time.time() - train_start

    print("\n" + "=" * 100)
    print(f"训练阶段总耗时: {format_time(train_time)}")
    print(f"正在加载第 {best_epoch} 轮 Best 模型执行最终 RQ1 评估...")
    print("=" * 100)

    if os.path.exists(BEST_MODEL_PATH):
        state_dict = safe_torch_load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("警告：未找到最佳模型，使用当前模型参数。")

    eval_start = time.time()
    train_eval = evaluate_classification_metrics(model, eval_train_loader, desc="评估 ID Train")
    id_eval = evaluate_classification_metrics(model, eval_id_loader, desc="评估 ID Test")
    ood_eval = evaluate_classification_metrics(model, eval_ood_loader, desc="评估 OOD Test")
    eval_time = time.time() - eval_start
    program_time = time.time() - program_start

    print_final_table(train_eval, id_eval, ood_eval)

    final_metrics = {
        "task": "Code Classification",
        "dataset": DATASET_NAME,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "data_dir": DATA_DIR,
        "num_labels": NUM_LABELS,
        "label_format": "0-based labels; labels are used directly without subtracting 1",
        "encoder_pooling_config": ENCODER_POOLING,
        "encoder_pooling_actual": model.pooling_strategy,
        "classifier_head_type": "single_linear",
        "classifier_state_dict_keys": ["classifier.weight", "classifier.bias"],
        "best_epoch": best_epoch,
        "best_id_macro_f1": best_f1,
        "epoch_records": epoch_records,
        "id_train": train_eval,
        "id_test": id_eval,
        "ood_test": ood_eval,
        "drop": {
            "accuracy_drop": id_eval["accuracy"] - ood_eval["accuracy"],
            "macro_f1_drop": id_eval["macro_f1"] - ood_eval["macro_f1"],
            "top5_acc_drop": id_eval["top5_acc"] - ood_eval["top5_acc"],
        },
        "time": {
            "train_seconds": float(train_time),
            "train_readable": format_time(train_time),
            "eval_seconds": float(eval_time),
            "eval_readable": format_time(eval_time),
            "program_seconds": float(program_time),
            "program_readable": format_time(program_time),
        },
        "files": {
            "best_model": BEST_MODEL_PATH,
            "metrics_file": METRICS_FILE,
            "epoch_csv": CSV_FILE,
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    with open(CSV_FILE, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,id_accuracy,id_macro_f1,id_top5_acc,epoch_seconds,checkpoint\n")
        for r in epoch_records:
            f.write(
                f"{r['epoch']},{r['train_loss']},{r['id_accuracy']},{r['id_macro_f1']},"
                f"{r['id_top5_acc']},{r['epoch_seconds']},{r['checkpoint']}\n"
            )

    print("\n" + "=" * 100)
    print("运行时间与文件保存信息")
    print("=" * 100)
    print("训练阶段总耗时:", format_time(train_time))
    print("最终评估总耗时:", format_time(eval_time))
    print("程序运行总耗时:", format_time(program_time))
    print("最佳模型保存于:", BEST_MODEL_PATH)
    print("汇总指标保存于:", METRICS_FILE)
    print("Epoch 指标保存于:", CSV_FILE)
    print("=" * 100)


if __name__ == "__main__":
    main()
