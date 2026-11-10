import os
import json
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, PLBartModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


# ============================================================
# 1. 配置区
# ============================================================

MODEL_TYPE = "RoBERTa"          # 可选: "codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa", "all"
MODEL_PATH = "/opt/data/private/RoBERTa"
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Defect-detection/dataset"

TRAIN_FILE = os.path.join(DATA_DIR, "id_train.jsonl")
ID_TEST_FILE = os.path.join(DATA_DIR, "id_test.jsonl")

# 同时评估两种 OOD
OOD_TEST_FILES = {
    "token": os.path.join(DATA_DIR, "ood_test_token.jsonl"),
    "cst": os.path.join(DATA_DIR, "ood_test_cst.jsonl"),
}

SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Defect-detection/rq1_outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 512
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

NUM_LABELS = 2

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 工具函数
# ============================================================

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒"


set_seed(SEED)


def get_code(js):
    return js.get("func") or js.get("code") or js.get("function") or ""


def get_label(js):
    label = js.get("target")
    if label is None:
        label = js.get("label")
    return label


def normalize_model_type(model_type):
    model_type_lower = model_type.lower()
    for model_name in MODEL_PATHS:
        if model_type_lower == model_name.lower():
            return model_name
    return model_type


def get_default_model_path(model_type):
    model_key = normalize_model_type(model_type)
    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return MODEL_PATHS[model_key]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE Defect Detection RQ1")
    parser.add_argument(
        "--model_type",
        default=MODEL_TYPE,
        choices=["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa", "all"],
        help="Model to train/evaluate. Use 'all' to run all five models sequentially.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local model path for single-model runs. Defaults to MODEL_PATHS[model_type].",
    )
    parser.add_argument(
        "--codebert_path",
        default=MODEL_PATHS["codebert"],
        help="Local CodeBERT path when --model_type all is used.",
    )
    parser.add_argument(
        "--GraphCodeBERT_path",
        default=MODEL_PATHS["GraphCodeBERT"],
        help="Local GraphCodeBERT path when --model_type all is used.",
    )
    parser.add_argument(
        "--codeT5_path",
        default=MODEL_PATHS["codeT5"],
        help="Local CodeT5 path when --model_type all is used.",
    )
    parser.add_argument(
        "--PLBART_path",
        default=MODEL_PATHS["PLBART"],
        help="Local PLBART path when --model_type all is used.",
    )
    parser.add_argument(
        "--RoBERTa_path",
        default=MODEL_PATHS["RoBERTa"],
        help="Local RoBERTa path when --model_type all is used.",
    )
    return parser


# ============================================================
# 3. 数据集
# ============================================================

class DefectDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer
        self.path = path

        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

        raw_labels = []

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"加载 {os.path.basename(path)}", leave=False)):
                line = line.strip()
                if not line:
                    continue

                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    continue

                code = get_code(js)
                label = get_label(js)

                if not code or label is None:
                    continue

                label = int(label)
                raw_labels.append(label)

                inputs = tokenizer(
                    code,
                    max_length=MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                self.examples.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "label": torch.tensor(label, dtype=torch.long),
                    "idx": js.get("idx", str(line_idx)),
                    "shift_type": js.get("shift_type", ""),
                    "code_preview": code[:300]
                })

        counter = Counter(raw_labels)
        print(f">>> 成功加载 {len(self.examples)} 条样本: {path}")
        print(f">>> 标签分布: {dict(counter)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# ============================================================
# 4. 模型
# ============================================================

class DefectClassifier(nn.Module):
    def __init__(self, model_path, model_type, num_labels):
        super().__init__()
        self.model_type = model_type.lower()
        self.model_key = normalize_model_type(model_type)
        self.pooling = "cls"

        if self.model_key == "codeT5":
            self.encoder = T5EncoderModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.d_model
        elif self.model_key == "PLBART":
            seq2seq_model = PLBartModel.from_pretrained(model_path)
            self.encoder = seq2seq_model.get_encoder()
            hidden_size = self.encoder.config.d_model
            self.pooling = "mean"
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def pool_hidden_state(self, last_hidden_state, attention_mask):
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            token_sum = (last_hidden_state * mask).sum(dim=1)
            token_count = mask.sum(dim=1).clamp(min=1e-6)
            return token_sum / token_count
        return last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_vec = self.pool_hidden_state(outputs.last_hidden_state, attention_mask)
        cls_vec = self.dropout(cls_vec)
        logits = self.classifier(cls_vec)

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
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            batch_size = labels.size(0)
            for i in range(batch_size):
                all_records.append({
                    "idx": batch["idx"][i],
                    "shift_type": batch["shift_type"][i],
                    "code_preview": batch["code_preview"][i],
                    "label": int(labels[i].cpu().item()),
                    "pred": int(preds[i].cpu().item()),
                    "prob_defective": float(probs[i, 1].cpu().item())
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
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask)
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


def print_result_row(metric_name, id_eval, ood_eval):
    print(
        f"{metric_name:<15} | "
        f"{id_eval[metric_name]:<12.4f} | "
        f"{ood_eval[metric_name]:<12.4f} | "
        f"{id_eval[metric_name] - ood_eval[metric_name]:<12.4f}"
    )


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

def run_experiment(model_type, model_path):
    set_seed(SEED)
    program_start_time = time.time()
    model_key = normalize_model_type(model_type)
    best_save_name = f"best_defect_{model_key}.bin"
    metrics_file = os.path.join(SAVE_DIR, f"rq1_defect_metrics_{model_key}_token_cst.json")

    print("=" * 100)
    print("CodeXGLUE Defect Detection RQ1")
    print("Evaluate Token Shift and CST Shift in one run")
    print("=" * 100)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("TRAIN_FILE:", TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", OOD_TEST_FILES)
    print("SAVE_DIR:", SAVE_DIR)
    print("DEVICE:", DEVICE)
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_ds = DefectDataset(TRAIN_FILE, tokenizer)
    id_test_ds = DefectDataset(ID_TEST_FILE, tokenizer)

    ood_datasets = {}
    for shift_name, path in OOD_TEST_FILES.items():
        ood_datasets[shift_name] = DefectDataset(path, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    id_loader = DataLoader(
        id_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    ood_loaders = {
        shift_name: DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        for shift_name, ds in ood_datasets.items()
    }

    model = DefectClassifier(
        model_path,
        model_type,
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
    best_path = os.path.join(SAVE_DIR, best_save_name)

    print("\n>>> 开始训练缺陷检测模型...")

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

        epoch_save_path = os.path.join(SAVE_DIR, f"checkpoint_{model_key}_epoch_{epoch}.bin")
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

    id_pred_path = os.path.join(SAVE_DIR, f"predictions_{model_key}_id_test.jsonl")

    id_eval = evaluate(
        model,
        id_loader,
        desc="最终评估 ID Test",
        save_path=id_pred_path
    )

    ood_results = {}
    ood_pred_paths = {}

    for shift_name, loader in ood_loaders.items():
        pred_path = os.path.join(SAVE_DIR, f"predictions_{model_key}_ood_{shift_name}.jsonl")
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
        "task": "Defect Detection",
        "dataset": "CodeXGLUE Defect Detection",
        "model_type": model_type,
        "model_path": model_path,
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

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    for shift_name, ood_eval in ood_results.items():
        print_final_table(
            title=f"RQ1 最终结果：CodeXGLUE Defect Detection - {shift_name.upper()} Shift",
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
    print(f"汇总指标保存于: {metrics_file}")
    print("=" * 100)


def main():
    args = build_arg_parser().parse_args()

    if args.model_type == "all":
        experiments = [
            ("codebert", args.codebert_path),
            ("GraphCodeBERT", args.GraphCodeBERT_path),
            ("codeT5", args.codeT5_path),
            ("PLBART", args.PLBART_path),
            ("RoBERTa", args.RoBERTa_path),
        ]
    else:
        experiments = [
            (args.model_type, args.model_path or get_default_model_path(args.model_type)),
        ]

    for model_type, model_path in experiments:
        run_experiment(model_type, model_path)


if __name__ == "__main__":
    main()
