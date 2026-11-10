# -*- coding: utf-8 -*-

import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import Counter
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


# ============================================================
# 1. 配置区
# ============================================================

# 选择语言："python" 或 "java"
#LANGUAGE = "python"
LANGUAGE = "java"

MODEL_TYPE = "RoBERTa"     # 选项: "codebert", "GraphCodeBERT", "codeT5"
# Supported model types: "codebert", "GraphCodeBERT", "codeT5", "PLBART".
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]


def resolve_model_path(model_type):
    if model_type in MODEL_PATHS:
        return MODEL_PATHS[model_type]
    model_type_lower = model_type.lower()
    for name, path in MODEL_PATHS.items():
        if name.lower() == model_type_lower:
            return path
    return model_type


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
    if key in MODEL_PATHS:
        return key
    lower = key.lower()
    if lower in aliases:
        return aliases[lower]
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {list(MODEL_PATHS.keys())}")


MODEL_TYPE = canonical_model_type(MODEL_TYPE)
MODEL_PATH = resolve_model_path(MODEL_TYPE)

DATA_ROOT = "/opt/data/private/CodeSearchNet"
DATA_DIR = os.path.join(DATA_ROOT, LANGUAGE)

TRAIN_FILE = os.path.join(DATA_DIR, "id_train_full.jsonl")
ID_TEST_FILE = os.path.join(DATA_DIR, "id_test.jsonl")

OOD_TEST_FILES = {
    #"token": os.path.join(DATA_DIR, "ood_test_token.jsonl"),
    "cst": os.path.join(DATA_DIR, "ood_test_cst.jsonl")
}

SAVE_DIR = os.path.join(DATA_DIR, "rq1_outputs")
BEST_SAVE_NAME = f"best_codesearch_{LANGUAGE}_{MODEL_TYPE}.bin"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, BEST_SAVE_NAME)
METRICS_FILE = os.path.join(SAVE_DIR, f"rq1_codesearch_metrics_{LANGUAGE}_{MODEL_TYPE}.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 32
LR = 5e-5
EPOCHS = 6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

# 评估样本数。代码搜索评估需要计算 N x N 相似度矩阵，太大会占内存。
# 如果想全量评估，改成 None，但不建议一开始就全量。
MAX_EVAL_SAMPLES = None

# 是否评估训练集前若干条，只作为过拟合参考，不作为 RQ1 核心指标
EVAL_TRAIN_SAMPLE = True
MAX_TRAIN_EVAL_SAMPLES = None

os.makedirs(SAVE_DIR, exist_ok=True)


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, BEST_SAVE_NAME, BEST_MODEL_PATH, METRICS_FILE
    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = resolve_model_path(MODEL_TYPE)
    BEST_SAVE_NAME = f"best_codesearch_{LANGUAGE}_{MODEL_TYPE}.bin"
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, BEST_SAVE_NAME)
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq1_codesearch_metrics_{LANGUAGE}_{MODEL_TYPE}.json")


# ============================================================
# 2. 工具函数
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒"


def safe_div(a, b):
    return a / max(b, 1)


set_seed(SEED)


# ============================================================
# 3. 数据集
# ============================================================

class CodeSearchDataset(Dataset):
    """
    CodeSearchNet 代码语义搜索数据集。

    输入:
        docstring / natural language
    候选:
        func / code

    注意：
    这里优先读取 func/code。
    如果 func/code 不存在，才退回 code_tokens。
    这样可以避免 OOD 文件中 func/code 已经 shift，
    但模型仍然读取旧 code_tokens 的问题。
    """

    def __init__(self, file_path, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        skipped = 0
        shift_counter = Counter()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"加载 {os.path.basename(file_path)}", leave=False)):
                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                nl_txt = self.get_nl_text(js)
                pl_txt = self.get_code_text(js)

                if not nl_txt or not pl_txt:
                    skipped += 1
                    continue

                # 防止部分 func/code 中残留 docstring
                # 只在非常明显时做轻量处理，不改变整体代码结构
                if nl_txt in pl_txt:
                    pl_txt = pl_txt.replace(nl_txt, "")
                    pl_txt = pl_txt.replace('""""""', "").replace("''''''", "")

                self.examples.append({
                    "nl": nl_txt,
                    "code": pl_txt,
                    "idx": js.get("idx", js.get("id", str(line_idx))),
                    "shift_type": js.get("shift_type", "id")
                })

                shift_counter[js.get("shift_type", "id")] += 1

        print(f">>> 成功加载 {len(self.examples)} 条样本: {file_path}")
        print(f">>> 跳过样本数: {skipped}")
        print(f">>> shift_type 分布: {dict(shift_counter)}")

    @staticmethod
    def get_nl_text(js):
        return (
            js.get("docstring")
            or js.get("doc")
            or js.get("nl")
            or js.get("query")
            or ""
        )

    @staticmethod
    def get_code_text(js):
        # 关键：优先使用 func/code
        code = js.get("func") or js.get("code") or js.get("function") or ""

        if not code and "code_tokens" in js:
            code = " ".join(js["code_tokens"])

        return code

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]

        nl_inputs = self.tokenizer(
            item["nl"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        pl_inputs = self.tokenizer(
            item["code"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "nl_input_ids": nl_inputs["input_ids"].squeeze(0),
            "nl_attention_mask": nl_inputs["attention_mask"].squeeze(0),
            "pl_input_ids": pl_inputs["input_ids"].squeeze(0),
            "pl_attention_mask": pl_inputs["attention_mask"].squeeze(0),
            "idx": item["idx"],
            "shift_type": item["shift_type"]
        }


# ============================================================
# 4. 模型
# ============================================================

class DualEncoderModel(nn.Module):
    def __init__(self, model_path, model_type):
        super().__init__()
        self.model_type = model_type.lower()

        if "t5" in self.model_type:
            self.encoder = T5EncoderModel.from_pretrained(model_path)
            self.hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            base_model = AutoModel.from_pretrained(model_path)
            if not hasattr(base_model, "get_encoder"):
                raise ValueError(
                    "PLBART model should expose get_encoder(); "
                    f"please check MODEL_PATH={model_path}"
                )
            self.encoder = base_model.get_encoder()
            self.hidden_size = getattr(base_model.config, "d_model", None)
            if self.hidden_size is None:
                self.hidden_size = getattr(base_model.config, "hidden_size", None)
            if self.hidden_size is None:
                raise ValueError("Cannot infer PLBART hidden size from model config.")
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
            self.hidden_size = self.encoder.config.hidden_size

    def get_vec(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if "t5" in self.model_type or "plbart" in self.model_type:
            # CodeT5 使用 mean pooling
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            vec = torch.sum(outputs.last_hidden_state * mask, dim=1)
            vec = vec / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            # CodeBERT / GraphCodeBERT / RoBERTa use CLS pooling
            vec = outputs.last_hidden_state[:, 0, :]

        return vec

    def forward(self, nl_ids, nl_mask, pl_ids, pl_mask):
        nl_vec = self.get_vec(nl_ids, nl_mask)
        pl_vec = self.get_vec(pl_ids, pl_mask)

        nl_vec = F.normalize(nl_vec, p=2, dim=1)
        pl_vec = F.normalize(pl_vec, p=2, dim=1)

        return nl_vec, pl_vec


# ============================================================
# 5. 评估函数
# ============================================================

def evaluate_search_metrics(model, dataloader, desc="搜索评估", save_path=None):
    model.eval()

    all_nl_vecs = []
    all_pl_vecs = []
    all_indices = []
    all_shift_types = []

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            nl_ids = batch["nl_input_ids"].to(DEVICE)
            nl_mask = batch["nl_attention_mask"].to(DEVICE)
            pl_ids = batch["pl_input_ids"].to(DEVICE)
            pl_mask = batch["pl_attention_mask"].to(DEVICE)

            with autocast(enabled=use_amp):
                nl_vec, pl_vec = model(nl_ids, nl_mask, pl_ids, pl_mask)

            all_nl_vecs.append(nl_vec.cpu())
            all_pl_vecs.append(pl_vec.cpu())
            all_indices.extend(batch["idx"])
            all_shift_types.extend(batch["shift_type"])

    if len(all_nl_vecs) == 0:
        return {
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "num_samples": 0
        }

    nl_vecs = torch.cat(all_nl_vecs, dim=0)
    pl_vecs = torch.cat(all_pl_vecs, dim=0)

    scores = torch.matmul(nl_vecs, pl_vecs.T).numpy()

    ranks = []
    records = []

    for i in range(len(scores)):
        score_row = scores[i]
        sorted_idx = np.argsort(score_row)[::-1]
        rank = int(np.where(sorted_idx == i)[0][0]) + 1
        ranks.append(rank)

        records.append({
            "idx": str(all_indices[i]),
            "shift_type": str(all_shift_types[i]),
            "rank": rank,
            "top1_index": int(sorted_idx[0]),
            "top1_score": float(score_row[sorted_idx[0]]),
            "gold_score": float(score_row[i])
        })

    ranks = np.array(ranks)

    result = {
        "mrr": float(np.mean(1.0 / ranks)),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "num_samples": int(len(ranks))
    }

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return result


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

    for key, name in [
        ("mrr", "MRR"),
        ("recall@1", "Recall@1"),
        ("recall@5", "Recall@5"),
        ("recall@10", "Recall@10")
    ]:
        print(
            f"{name:<15} | "
            f"{id_eval[key]:<12.4f} | "
            f"{ood_eval[key]:<12.4f} | "
            f"{id_eval[key] - ood_eval[key]:<12.4f}"
        )

    print("-" * 100)
    print("ID Test 样本数:", id_eval["num_samples"])
    print("OOD Test 样本数:", ood_eval["num_samples"])
    print("=" * 100)


# ============================================================
# 6. 训练函数
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch):
    model.train()

    total_loss = 0.0
    step_count = 0
    use_amp = DEVICE.type == "cuda"

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        nl_ids = batch["nl_input_ids"].to(DEVICE)
        nl_mask = batch["nl_attention_mask"].to(DEVICE)
        pl_ids = batch["pl_input_ids"].to(DEVICE)
        pl_mask = batch["pl_attention_mask"].to(DEVICE)

        with autocast(enabled=use_amp):
            nl_vec, pl_vec = model(nl_ids, nl_mask, pl_ids, pl_mask)

            logits = torch.matmul(nl_vec, pl_vec.T) * 20.0
            labels = torch.arange(nl_ids.size(0), device=DEVICE)

            loss_nl = F.cross_entropy(logits, labels)
            loss_pl = F.cross_entropy(logits.T, labels)
            loss = (loss_nl + loss_pl) / 2.0

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

        pbar.set_postfix(loss=loss.item())

    return total_loss / max(step_count, 1)


# ============================================================
# 7. 主函数
# ============================================================

def run_one_model(model_type=None):
    configure_model_runtime(model_type or MODEL_TYPE)
    program_start_time = time.time()

    print("=" * 100)
    print("CodeSearchNet RQ1: Code Search under Token/CST Shift")
    print("=" * 100)
    print("LANGUAGE:", LANGUAGE)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("RUN_ALL_MODELS:", RUN_ALL_MODELS)
    print("SUPPORTED_MODEL_TYPES:", list(MODEL_PATHS.keys()))
    print("DATA_DIR:", DATA_DIR)
    print("TRAIN_FILE:", TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", OOD_TEST_FILES)
    print("SAVE_DIR:", SAVE_DIR)
    print("DEVICE:", DEVICE)
    print("MAX_EVAL_SAMPLES:", MAX_EVAL_SAMPLES)
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("\n>>> 正在加载数据集...")
    train_ds = CodeSearchDataset(TRAIN_FILE, tokenizer)
    id_test_ds = CodeSearchDataset(ID_TEST_FILE, tokenizer)

    ood_datasets = {}
    for shift_name, path in OOD_TEST_FILES.items():
        ood_datasets[shift_name] = CodeSearchDataset(path, tokenizer)

    if len(train_ds) == 0:
        raise ValueError("训练集为空，请检查 id_train.jsonl")
    if len(id_test_ds) == 0:
        raise ValueError("ID 测试集为空，请检查 id_test.jsonl")

    # 评估子集
    if MAX_EVAL_SAMPLES is not None:
        id_eval_size = min(MAX_EVAL_SAMPLES, len(id_test_ds))
        id_eval_ds = Subset(id_test_ds, range(id_eval_size))
    else:
        id_eval_ds = id_test_ds

    ood_eval_datasets = {}
    for shift_name, ds in ood_datasets.items():
        if MAX_EVAL_SAMPLES is not None:
            eval_size = min(MAX_EVAL_SAMPLES, len(ds))
            ood_eval_datasets[shift_name] = Subset(ds, range(eval_size))
        else:
            ood_eval_datasets[shift_name] = ds

    if EVAL_TRAIN_SAMPLE:
        if MAX_TRAIN_EVAL_SAMPLES is not None:
            train_eval_size = min(MAX_TRAIN_EVAL_SAMPLES, len(train_ds))
            train_eval_ds = Subset(train_ds, range(train_eval_size))
        else:
            train_eval_size = len(train_ds)
            train_eval_ds = train_ds
    else:
        train_eval_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda")
    )

    id_eval_loader = DataLoader(
        id_eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda")
    )

    ood_eval_loaders = {
        shift_name: DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda")
        )
        for shift_name, ds in ood_eval_datasets.items()
    }

    if train_eval_ds is not None:
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda")
        )
    else:
        train_eval_loader = None

    model = DualEncoderModel(MODEL_PATH, MODEL_TYPE).to(DEVICE)

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

    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    best_mrr = -1.0
    best_epoch = 0

    print("\n>>> 开始微调训练 CodeSearchNet dual encoder...")

    train_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            epoch
        )

        id_eval = evaluate_search_metrics(
            model,
            id_eval_loader,
            desc=f"验证 Epoch {epoch} on ID Test"
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch} | "
            f"Loss: {train_loss:.4f} | "
            f"ID MRR: {id_eval['mrr']:.4f} | "
            f"R@1: {id_eval['recall@1']:.4f} | "
            f"R@5: {id_eval['recall@5']:.4f} | "
            f"R@10: {id_eval['recall@10']:.4f} | "
            f"Epoch Time: {format_time(epoch_time)}"
        )

        epoch_save_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}_{MODEL_TYPE}.bin")
        torch.save(model.state_dict(), epoch_save_path)

        if id_eval["mrr"] > best_mrr:
            best_mrr = id_eval["mrr"]
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✨ 新最佳模型已保存: {BEST_MODEL_PATH}")

    train_time = time.time() - train_start_time

    print("\n" + "=" * 100)
    print(f"训练阶段总耗时: {format_time(train_time)}")
    print(f"加载第 {best_epoch} 轮最佳模型，执行最终 RQ1 评估...")
    print("=" * 100)

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    else:
        print("警告：未找到最佳模型，使用当前模型参数。")

    eval_start_time = time.time()

    train_eval_result = None
    train_pred_path = None

    if train_eval_loader is not None:
        train_pred_path = os.path.join(SAVE_DIR, f"predictions_train_sample_{LANGUAGE}_{MODEL_TYPE}.jsonl")
        train_eval_result = evaluate_search_metrics(
            model,
            train_eval_loader,
            desc="最终评估 Train Sample",
            save_path=train_pred_path
        )

    id_pred_path = os.path.join(SAVE_DIR, f"predictions_id_test_{LANGUAGE}_{MODEL_TYPE}.jsonl")
    id_eval = evaluate_search_metrics(
        model,
        id_eval_loader,
        desc="最终评估 ID Test",
        save_path=id_pred_path
    )

    ood_results = {}
    ood_pred_paths = {}

    for shift_name, loader in ood_eval_loaders.items():
        pred_path = os.path.join(SAVE_DIR, f"predictions_ood_{shift_name}_{LANGUAGE}_{MODEL_TYPE}.jsonl")
        result = evaluate_search_metrics(
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
        "task": "Code Search",
        "dataset": f"CodeSearchNet-{LANGUAGE}",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "supported_model_types": list(MODEL_PATHS.keys()),
        "encoder_pooling": "mean" if ("t5" in MODEL_TYPE.lower() or "plbart" in MODEL_TYPE.lower()) else "cls",
        "language": LANGUAGE,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "best_epoch": best_epoch,
        "best_id_mrr": best_mrr,
        "train_sample": train_eval_result,
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
            "best_model": BEST_MODEL_PATH,
            "train_sample_predictions": train_pred_path,
            "id_predictions": id_pred_path,
            "ood_predictions": ood_pred_paths
        }
    }

    for shift_name, ood_eval in ood_results.items():
        final_metrics["drop"][shift_name] = {
            "mrr_drop": id_eval["mrr"] - ood_eval["mrr"],
            "recall@1_drop": id_eval["recall@1"] - ood_eval["recall@1"],
            "recall@5_drop": id_eval["recall@5"] - ood_eval["recall@5"],
            "recall@10_drop": id_eval["recall@10"] - ood_eval["recall@10"]
        }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    if train_eval_result is not None:
        print("\n" + "=" * 100)
        print("训练集样本参考结果，不作为 RQ1 核心指标")
        print("=" * 100)
        print(f"Train Sample MRR: {train_eval_result['mrr']:.4f}")
        print(f"Train Sample Recall@1: {train_eval_result['recall@1']:.4f}")
        print(f"Train Sample Recall@5: {train_eval_result['recall@5']:.4f}")
        print(f"Train Sample Recall@10: {train_eval_result['recall@10']:.4f}")
        print("=" * 100)

    for shift_name, ood_eval in ood_results.items():
        print_final_table(
            title=f"RQ1 最终结果：CodeSearchNet-{LANGUAGE} Code Search - {shift_name.upper()} Shift",
            id_eval=id_eval,
            ood_eval=ood_eval
        )

    print("\n" + "=" * 100)
    print("运行时间与文件保存信息")
    print("=" * 100)
    print(f"训练阶段总耗时: {format_time(train_time)}")
    print(f"最终评估总耗时: {format_time(eval_time)}")
    print(f"程序运行总耗时: {format_time(program_time)}")
    print(f"最佳模型保存于: {BEST_MODEL_PATH}")
    print(f"ID 预测结果保存于: {id_pred_path}")

    for shift_name, path in ood_pred_paths.items():
        print(f"OOD {shift_name} 预测结果保存于: {path}")

    print(f"汇总指标保存于: {METRICS_FILE}")
    print("=" * 100)
    return final_metrics


def main():
    models_to_run = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    all_results = []

    for run_model_type in models_to_run:
        try:
            all_results.append(run_one_model(run_model_type))
        except Exception as exc:
            if not RUN_ALL_MODELS:
                raise
            error_result = {
                "status": "failed",
                "model_type": run_model_type,
                "error": repr(exc),
            }
            all_results.append(error_result)
            print("\n" + "!" * 100)
            print(f"[Error] RQ1 CodeSearchNet failed for model={run_model_type}: {repr(exc)}")
            print("Continuing with next model because RUN_ALL_MODELS=True.")
            print("!" * 100)
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if RUN_ALL_MODELS:
        all_metrics_file = os.path.join(SAVE_DIR, f"rq1_codesearch_metrics_{LANGUAGE}_all_models.json")
        with open(all_metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": "Code Search",
                "dataset": f"CodeSearchNet-{LANGUAGE}",
                "models": models_to_run,
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        print("All-model metrics saved:", all_metrics_file)


if __name__ == "__main__":
    main()
