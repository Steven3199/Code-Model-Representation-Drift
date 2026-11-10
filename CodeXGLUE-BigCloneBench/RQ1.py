import os
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


# ============================================================
# 1. 配置区
# ============================================================

SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]

# 默认只运行 MODEL_TYPE 指定的一个模型；如需一次跑四个模型，改为 True。
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]

MODEL_TYPE = "RoBERTa"  # options: "codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}


def canonical_model_type(model_type):
    aliases = {
        "codebert": "codebert",
        "graphcodebert": "GraphCodeBERT",
        "codet5": "codeT5",
        "code_t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
        "roberta": "RoBERTa",
        "roberta-base": "RoBERTa",
    }
    key = str(model_type).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Supported: {SUPPORTED_MODEL_TYPES}")
    return aliases[key]


def resolve_model_path(model_type):
    return MODEL_PATHS[canonical_model_type(model_type)]


MODEL_TYPE = canonical_model_type(MODEL_TYPE)
MODEL_PATH = resolve_model_path(MODEL_TYPE)

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset"

TRAIN_FILE = os.path.join(DATA_DIR, "id_train.jsonl")
ID_TEST_FILE = os.path.join(DATA_DIR, "id_test.jsonl")

OOD_TEST_FILES = {
    "token": os.path.join(DATA_DIR, "ood_test_token.jsonl"),
    "cst": os.path.join(DATA_DIR, "ood_test_cst.jsonl")
}

SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/rq1_outputs"
BEST_SAVE_NAME = f"best_clone_{MODEL_TYPE}.bin"
METRICS_FILE = os.path.join(SAVE_DIR, f"rq1_clone_metrics_{MODEL_TYPE}.json")
ALL_MODELS_METRICS_FILE = os.path.join(SAVE_DIR, "rq1_clone_metrics_all_models.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 512
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

NUM_LABELS = 2

# 为了先跑通，建议限制样本数
# 如果你想全量训练，改成 None
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
#记得修改

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 工具函数
# ============================================================

def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, BEST_SAVE_NAME, METRICS_FILE
    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = resolve_model_path(MODEL_TYPE)
    BEST_SAVE_NAME = f"best_clone_{MODEL_TYPE}.bin"
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq1_clone_metrics_{MODEL_TYPE}.json")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒"


set_seed(SEED)


def get_label(js):
    label = js.get("label")
    if label is None:
        label = js.get("target")
    return label


# ============================================================
# 3. Lazy Dataset
# ============================================================

class CloneDataset(Dataset):
    """
    重要修改：
    不在 __init__ 中 tokenize 全部样本。
    只保存原始 code1/code2/label，训练时 __getitem__ 再 tokenize。
    这样可以避免内存爆炸。
    """
    def __init__(self, path, tokenizer, max_samples=None, shuffle_before_limit=False):
        self.path = path
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.examples = []

        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

        raw_labels = []

        print(f">>> 正在加载原始文本样本: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"读取 {os.path.basename(path)}", leave=False)):
                line = line.strip()
                if not line:
                    continue

                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    continue

                code1 = js.get("code1", "")
                code2 = js.get("code2", "")
                label = get_label(js)

                if not code1 or not code2 or label is None:
                    continue

                label = int(label)

                self.examples.append({
                    "code1": code1,
                    "code2": code2,
                    "label": label,
                    "idx1": js.get("idx1", ""),
                    "idx2": js.get("idx2", ""),
                    "shift_type": js.get("shift_type", "")
                })

                raw_labels.append(label)

        if shuffle_before_limit:
            random.shuffle(self.examples)

        if max_samples is not None and len(self.examples) > max_samples:
            self.examples = self.examples[:max_samples]

        raw_labels = [x["label"] for x in self.examples]
        counter = Counter(raw_labels)

        print(f">>> 成功加载 {len(self.examples)} 条样本: {path}")
        print(f">>> 标签分布: {dict(counter)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]

        inputs = self.tokenizer(
            item["code1"],
            item["code2"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        batch = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "idx1": item["idx1"],
            "idx2": item["idx2"],
            "shift_type": item["shift_type"]
        }
        if "token_type_ids" in inputs:
            batch["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        return batch


# ============================================================
# 4. 模型
# ============================================================

class CloneClassifier(nn.Module):
    def __init__(self, model_path, model_type, num_labels):
        super().__init__()
        self.model_type = model_type.lower()

        if "t5" in self.model_type:
            self.encoder = T5EncoderModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            base_model = AutoModel.from_pretrained(model_path)
            if not hasattr(base_model, "get_encoder"):
                raise ValueError("PLBART requires a seq2seq model with get_encoder().")
            self.encoder = base_model.get_encoder()
            hidden_size = getattr(base_model.config, "d_model", None)
            if hidden_size is None:
                hidden_size = getattr(base_model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot infer hidden size from PLBART config.")
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _pool(self, hidden_states, attention_mask):
        if "t5" in self.model_type or "plbart" in self.model_type:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        return hidden_states[:, 0, :]

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and "t5" not in self.model_type and "plbart" not in self.model_type:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits


# ============================================================
# 5. 训练与评估
# ============================================================

def evaluate(model, dataloader, desc="评估", save_path=None):
    model.eval()

    all_preds = []
    all_labels = []
    all_records = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            batch_size = labels.size(0)
            for i in range(batch_size):
                all_records.append({
                    "idx1": batch["idx1"][i],
                    "idx2": batch["idx2"][i],
                    "shift_type": batch["shift_type"][i],
                    "label": int(labels[i].cpu().item()),
                    "pred": int(preds[i].cpu().item()),
                    "prob_clone": float(probs[i, 1].cpu().item())
                })

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    label_counter = Counter(all_labels)
    pred_counter = Counter(all_preds)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_distribution": dict(label_counter),
        "pred_distribution": dict(pred_counter),
        "positive_label_ratio": sum(all_labels) / max(len(all_labels), 1),
        "positive_pred_ratio": sum(all_preds) / max(len(all_preds), 1),
        "num_samples": len(all_labels)
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch):
    model.train()

    total_loss = 0.0
    step_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
        loss = nn.functional.cross_entropy(logits, labels)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        step_count += 1

        pbar.set_postfix(loss=loss.item())

    return total_loss / max(step_count, 1)


def print_final_table(title, id_eval, ood_eval):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(
        f"{'指标':<15} | "
        f"{'ID Test':<12} | "
        f"{'OOD Test':<12} | "
        f"{'跌幅 ID-OOD':<12}"
    )
    print("-" * 100)

    print(
        f"{'Accuracy':<15} | "
        f"{id_eval['accuracy']:<12.4f} | "
        f"{ood_eval['accuracy']:<12.4f} | "
        f"{id_eval['accuracy'] - ood_eval['accuracy']:<12.4f}"
    )
    print(
        f"{'Precision':<15} | "
        f"{id_eval['precision']:<12.4f} | "
        f"{ood_eval['precision']:<12.4f} | "
        f"{id_eval['precision'] - ood_eval['precision']:<12.4f}"
    )
    print(
        f"{'Recall':<15} | "
        f"{id_eval['recall']:<12.4f} | "
        f"{ood_eval['recall']:<12.4f} | "
        f"{id_eval['recall'] - ood_eval['recall']:<12.4f}"
    )
    print(
        f"{'F1':<15} | "
        f"{id_eval['f1']:<12.4f} | "
        f"{ood_eval['f1']:<12.4f} | "
        f"{id_eval['f1'] - ood_eval['f1']:<12.4f}"
    )

    print("-" * 100)
    print("ID Test 标签分布:", id_eval["label_distribution"])
    print("ID Test 预测分布:", id_eval["pred_distribution"])
    print("OOD Test 标签分布:", ood_eval["label_distribution"])
    print("OOD Test 预测分布:", ood_eval["pred_distribution"])
    print(f"ID Test 正类真实比例: {id_eval['positive_label_ratio']:.4f}")
    print(f"ID Test 正类预测比例: {id_eval['positive_pred_ratio']:.4f}")
    print(f"OOD Test 正类真实比例: {ood_eval['positive_label_ratio']:.4f}")
    print(f"OOD Test 正类预测比例: {ood_eval['positive_pred_ratio']:.4f}")
    print("=" * 100)


# ============================================================
# 6. 主函数
# ============================================================

def main(model_type=None):
    configure_model_runtime(model_type or MODEL_TYPE)
    set_seed(SEED)
    program_start_time = time.time()

    print("=" * 100)
    print("CodeXGLUE BigCloneBench RQ1: Clone Detection under Token/CST Shift")
    print("=" * 100)
    print("SUPPORTED_MODEL_TYPES:", SUPPORTED_MODEL_TYPES)
    print("RUN_ALL_MODELS:", RUN_ALL_MODELS)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("TRAIN_FILE:", TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", OOD_TEST_FILES)
    print("MAX_TRAIN_SAMPLES:", MAX_TRAIN_SAMPLES)
    print("MAX_EVAL_SAMPLES:", MAX_EVAL_SAMPLES)
    print("SAVE_DIR:", SAVE_DIR)
    print("DEVICE:", DEVICE)
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    train_ds = CloneDataset(
        TRAIN_FILE,
        tokenizer,
        max_samples=MAX_TRAIN_SAMPLES,
        shuffle_before_limit=True
    )

    id_test_ds = CloneDataset(
        ID_TEST_FILE,
        tokenizer,
        max_samples=MAX_EVAL_SAMPLES,
        shuffle_before_limit=False
    )

    ood_datasets = {}
    for shift_name, path in OOD_TEST_FILES.items():
        ood_datasets[shift_name] = CloneDataset(
            path,
            tokenizer,
            max_samples=MAX_EVAL_SAMPLES,
            shuffle_before_limit=False
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    id_loader = DataLoader(
        id_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    ood_loaders = {
        shift_name: DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        for shift_name, ds in ood_datasets.items()
    }

    model = CloneClassifier(
        MODEL_PATH,
        MODEL_TYPE,
        NUM_LABELS
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = -1.0
    best_epoch = 0
    best_path = os.path.join(SAVE_DIR, BEST_SAVE_NAME)

    print("\n>>> 开始训练 BigCloneBench 克隆检测模型...")

    train_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch
        )

        id_eval = evaluate(
            model,
            id_loader,
            desc=f"验证 Epoch {epoch} on ID Test"
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch} | "
            f"Loss: {train_loss:.4f} | "
            f"ID Acc: {id_eval['accuracy']:.4f} | "
            f"ID P: {id_eval['precision']:.4f} | "
            f"ID R: {id_eval['recall']:.4f} | "
            f"ID F1: {id_eval['f1']:.4f} | "
            f"Epoch Time: {format_time(epoch_time)}"
        )

        epoch_save_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}_{MODEL_TYPE}.bin")
        torch.save(model.state_dict(), epoch_save_path)

        if id_eval["f1"] > best_f1:
            best_f1 = id_eval["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"✨ 新最佳模型已保存: {best_path}")

    train_time = time.time() - train_start_time

    print("\n" + "=" * 100)
    print(f"训练阶段总耗时: {format_time(train_time)}")
    print(f"加载第 {best_epoch} 轮最佳模型，执行最终 RQ1 评估...")
    print("=" * 100)

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    eval_start_time = time.time()

    id_pred_path = os.path.join(SAVE_DIR, f"predictions_id_test_{MODEL_TYPE}.jsonl")

    id_eval = evaluate(
        model,
        id_loader,
        desc="最终评估 ID Test",
        save_path=id_pred_path
    )

    ood_results = {}
    ood_pred_paths = {}

    for shift_name, loader in ood_loaders.items():
        pred_path = os.path.join(SAVE_DIR, f"predictions_ood_{shift_name}_{MODEL_TYPE}.jsonl")
        result = evaluate(
            model,
            loader,
            desc=f"最终评估 OOD Test ({shift_name})",
            save_path=pred_path
        )
        ood_results[shift_name] = result
        ood_pred_paths[shift_name] = pred_path

    eval_time = time.time() - eval_start_time
    program_time = time.time() - program_start_time

    final_metrics = {
        "task": "Clone Detection",
        "dataset": "CodeXGLUE BigCloneBench",
        "supported_model_types": SUPPORTED_MODEL_TYPES,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "best_epoch": best_epoch,
        "best_id_f1": best_f1,
        "id_test": id_eval,
        "ood_test": ood_results,
        "drop": {},
        "time": {
            "train_elapsed_seconds": train_time,
            "train_elapsed_readable": format_time(train_time),
            "eval_elapsed_seconds": eval_time,
            "eval_elapsed_readable": format_time(eval_time),
            "program_elapsed_seconds": program_time,
            "program_elapsed_readable": format_time(program_time)
        },
        "files": {
            "best_model": best_path,
            "id_predictions": id_pred_path,
            "ood_predictions": ood_pred_paths
        }
    }

    for shift_name, ood_eval in ood_results.items():
        final_metrics["drop"][shift_name] = {
            "accuracy_drop": id_eval["accuracy"] - ood_eval["accuracy"],
            "precision_drop": id_eval["precision"] - ood_eval["precision"],
            "recall_drop": id_eval["recall"] - ood_eval["recall"],
            "f1_drop": id_eval["f1"] - ood_eval["f1"]
        }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    for shift_name, ood_eval in ood_results.items():
        print_final_table(
            title=f"RQ1 最终结果：BigCloneBench Clone Detection - {shift_name.upper()} Shift",
            id_eval=id_eval,
            ood_eval=ood_eval
        )

    print("\n" + "=" * 100)
    print("运行时间与文件保存信息")
    print("=" * 100)
    print(f"训练阶段总耗时: {format_time(train_time)}")
    print(f"最终评估总耗时: {format_time(eval_time)}")
    print(f"程序运行总耗时: {format_time(program_time)}")
    print(f"最佳模型保存于: {best_path}")
    print(f"ID 预测结果保存于: {id_pred_path}")
    for shift_name, path in ood_pred_paths.items():
        print(f"OOD {shift_name} 预测结果保存于: {path}")
    print(f"汇总指标保存于: {METRICS_FILE}")
    print("=" * 100)


    return final_metrics


if __name__ == "__main__":
    if RUN_ALL_MODELS:
        all_results = {}
        for run_model_type in MODEL_TYPES_TO_RUN:
            run_model_type = canonical_model_type(run_model_type)
            print("\n" + "#" * 120)
            print(f"Running CodeXGLUE BigCloneBench RQ1 for model: {run_model_type}")
            print("#" * 120)
            all_results[run_model_type] = main(run_model_type)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        with open(ALL_MODELS_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"All-model metrics saved to: {ALL_MODELS_METRICS_FILE}")
    else:
        main(MODEL_TYPE)
