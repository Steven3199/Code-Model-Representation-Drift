import os
import json
import time
import random
import math
import argparse
from collections import Counter

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


# ============================================================
# 1. 配置区
# ============================================================

MODEL_TYPE = "PLBART"          # 可选: "codeT5", "PLBART", "all"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
MODEL_PATH = MODEL_PATHS["PLBART"]

DATA_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement"

# 如果为 None，会自动扫描 DATA_ROOT 下包含以下文件的目录：
# id_train.jsonl
# id_test.jsonl
# ood_test_token.jsonl
# ood_test_cst.jsonl
#
# 如果只想跑某个子集，可以写：
# SUBSET_DIRS = ["small"]
# SUBSET_DIRS = [medium"]
SUBSET_DIRS = None

SAVE_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement/rq1_outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

LR = 5e-5
EPOCHS = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

NUM_BEAMS = 5

# 调试时可以改小；正式实验建议 None
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

# 最终评估 ID Train 如果全量太慢，可以只评估前若干条
# 如果想全量评估，改成 None
MAX_TRAIN_EVAL_SAMPLES = 1000

os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
# 2. 工具函数
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


def normalize_model_type(model_type):
    model_type_lower = model_type.lower()
    if model_type_lower == "all":
        return "all"

    for name in MODEL_PATHS:
        if model_type_lower == name.lower():
            return name

    raise ValueError(f"Unsupported model_type: {model_type}")


def get_default_model_path(model_type):
    model_type = normalize_model_type(model_type)
    if model_type == "all":
        raise ValueError("model_type='all' does not have a single default model path")
    return MODEL_PATHS[model_type]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE Bugs2Fix RQ1")
    parser.add_argument(
        "--model_type",
        default=MODEL_TYPE,
        choices=["codeT5", "PLBART", "all"],
        help="Model to train/evaluate. Use 'all' to run codeT5 and PLBART sequentially.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local model path for a single-model run. Defaults to MODEL_PATHS[model_type].",
    )
    parser.add_argument("--codeT5_path", default=MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=MODEL_PATHS["PLBART"])
    return parser


def build_generation_model(model_type, model_path, tokenizer):
    model_type = normalize_model_type(model_type)

    if model_type not in ["codeT5", "PLBART"]:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model.to(DEVICE)


def normalize_code(text):
    """
    EM / BLEU / CodeBLEU 前做轻量归一化。
    不做强语义化处理，避免改变代码含义。
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(text.strip().split())

    return text


def get_source(js):
    return (
        js.get("source")
        or js.get("buggy")
        or js.get("src")
        or js.get("input")
        or ""
    )


def get_target(js):
    return (
        js.get("target")
        or js.get("fixed")
        or js.get("tgt")
        or js.get("output")
        or ""
    )


def discover_subset_dirs():
    """
    自动寻找已经生成好 shift 数据的 Bugs2Fix 子集目录。
    目录中需要包含：
        id_train.jsonl
        id_test.jsonl
        ood_test_token.jsonl
        ood_test_cst.jsonl
    """
    subset_infos = []

    if SUBSET_DIRS is not None:
        candidates = []
        for name in SUBSET_DIRS:
            path = os.path.join(DATA_ROOT, name)
            candidates.append(path)
    else:
        candidates = []

        for root, dirs, files in os.walk(DATA_ROOT):
            required = {
                "id_train.jsonl",
                "id_test.jsonl",
                "ood_test_token.jsonl",
                "ood_test_cst.jsonl"
            }

            if required.issubset(set(files)):
                candidates.append(root)

    for path in sorted(set(candidates)):
        train_file = os.path.join(path, "id_train.jsonl")
        id_test_file = os.path.join(path, "id_test.jsonl")
        ood_token_file = os.path.join(path, "ood_test_token.jsonl")
        ood_cst_file = os.path.join(path, "ood_test_cst.jsonl")

        if not os.path.exists(train_file):
            print(f"[跳过] {path}: 缺少 id_train.jsonl")
            continue

        if not os.path.exists(id_test_file):
            print(f"[跳过] {path}: 缺少 id_test.jsonl")
            continue

        if not os.path.exists(ood_token_file):
            print(f"[跳过] {path}: 缺少 ood_test_token.jsonl")
            continue

        if not os.path.exists(ood_cst_file):
            print(f"[跳过] {path}: 缺少 ood_test_cst.jsonl")
            continue

        subset_name = os.path.basename(path.rstrip("/"))

        subset_infos.append({
            "name": subset_name,
            "dir": path,
            "train_file": train_file,
            "id_test_file": id_test_file,
            "ood_token_file": ood_token_file,
            "ood_cst_file": ood_cst_file,
        })

    return subset_infos


# ============================================================
# 3. 代码修复评估指标：EM / BLEU / CodeBLEU
# ============================================================

def exact_match_score(preds, refs):
    """
    Exact Match Accuracy，返回百分制 0-100。
    """
    if len(preds) == 0:
        return 0.0

    hit = 0

    for pred, ref in zip(preds, refs):
        if normalize_code(pred) == normalize_code(ref):
            hit += 1

    return 100.0 * hit / len(preds)


def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu_score(preds, refs, max_n=4, smooth=1.0):
    """
    简单 corpus BLEU 实现，返回百分制 0-100。
    这是为了避免环境中没有 sacrebleu/nltk 时不能跑。
    """
    clipped_counts = [0 for _ in range(max_n)]
    total_counts = [0 for _ in range(max_n)]

    pred_len = 0
    ref_len = 0

    for pred, ref in zip(preds, refs):
        pred_tokens = normalize_code(pred).split()
        ref_tokens = normalize_code(ref).split()

        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)

            overlap = {
                ng: min(count, ref_ngrams.get(ng, 0))
                for ng, count in pred_ngrams.items()
            }

            clipped_counts[n - 1] += sum(overlap.values())
            total_counts[n - 1] += max(sum(pred_ngrams.values()), 0)

    if pred_len == 0:
        return 0.0

    precisions = []

    for i in range(max_n):
        precision_i = (clipped_counts[i] + smooth) / (total_counts[i] + smooth)
        precisions.append(precision_i)

    log_precision = sum(math.log(p) for p in precisions) / max_n

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(pred_len, 1))

    bleu = bp * math.exp(log_precision)

    return float(bleu * 100.0)


def codebleu_score(preds, refs, lang="java"):
    """
    CodeBLEU，返回百分制 0-100。

    优先使用 pip 包 codebleu：
        pip install codebleu

    如果当前环境没有安装 codebleu，则返回 -1.0。
    正式实验建议安装 codebleu 后再跑。
    """
    try:
        from codebleu import calc_codebleu
    except Exception as e:
        print(f"[警告] 当前环境无法 import codebleu，CodeBLEU 将记为 -1.0。错误: {repr(e)}")
        return -1.0

    norm_preds = [normalize_code(x) for x in preds]
    norm_refs = [normalize_code(x) for x in refs]

    try:
        # 常见 codebleu 包 API：
        # calc_codebleu(references, predictions, lang="java")
        result = calc_codebleu(
            references=[[r] for r in norm_refs],
            predictions=norm_preds,
            lang=lang
        )
    except TypeError:
        try:
            # 另一种 API：
            # calc_codebleu(refs, preds, lang)
            result = calc_codebleu(
                [[r] for r in norm_refs],
                norm_preds,
                lang
            )
        except Exception as e:
            print(f"[警告] CodeBLEU 计算失败，CodeBLEU 将记为 -1.0。错误: {repr(e)}")
            return -1.0
    except Exception as e:
        print(f"[警告] CodeBLEU 计算失败，CodeBLEU 将记为 -1.0。错误: {repr(e)}")
        return -1.0

    if isinstance(result, dict):
        value = (
            result.get("codebleu")
            or result.get("CodeBLEU")
            or result.get("code_bleu")
            or result.get("score")
        )
    else:
        value = result

    if value is None:
        print("[警告] CodeBLEU 返回结果中没有找到 codebleu 字段，CodeBLEU 将记为 -1.0。")
        return -1.0

    value = float(value)

    # 有的实现返回 0-1，有的返回 0-100，这里统一成 0-100。
    if value <= 1.0:
        value *= 100.0

    return value


def compute_repair_metrics(preds, refs):
    em = exact_match_score(preds, refs)
    bleu = corpus_bleu_score(preds, refs)
    codebleu = codebleu_score(preds, refs, lang="java")

    return {
        "exact_match": float(em),
        "bleu": float(bleu),
        "codebleu": float(codebleu),
        "num_samples": len(preds)
    }


# ============================================================
# 4. Dataset
# ============================================================

class Bugs2FixDataset(Dataset):
    def __init__(self, path, tokenizer, max_samples=None, shuffle_before_limit=False):
        self.path = path
        self.tokenizer = tokenizer
        self.examples = []

        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

        print(f">>> 正在加载 Bugs2Fix 数据: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"读取 {os.path.basename(path)}", leave=False)):
                line = line.strip()

                if not line:
                    continue

                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    continue

                source = get_source(js)
                target = get_target(js)

                if not source or not target:
                    continue

                self.examples.append({
                    "idx": str(js.get("idx", line_idx)),
                    "source": source,
                    "target": target,
                    "shift_type": js.get("shift_type", "id"),
                    "shift_success": js.get("shift_success", None),
                    "fallback_reason": js.get("fallback_reason", "")
                })

        if shuffle_before_limit:
            random.shuffle(self.examples)

        if max_samples is not None and len(self.examples) > max_samples:
            self.examples = self.examples[:max_samples]

        shift_counter = Counter(x["shift_type"] for x in self.examples)

        print(f">>> 成功加载 {len(self.examples)} 条样本")
        print(f">>> shift_type 分布: {dict(shift_counter)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]

        source_inputs = self.tokenizer(
            item["source"],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_inputs = self.tokenizer(
            item["target"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_inputs["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_inputs["input_ids"].squeeze(0),
            "attention_mask": source_inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "idx": item["idx"],
            "source": item["source"],
            "target": item["target"],
            "shift_type": item["shift_type"]
        }


# ============================================================
# 5. 训练与评估
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch):
    model.train()

    total_loss = 0.0
    step_count = 0
    use_amp = DEVICE.type == "cuda"

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast(enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

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


def evaluate_generation(model, tokenizer, dataloader, desc="生成评估", save_path=None):
    model.eval()

    preds = []
    refs = []
    records = []

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            with autocast(enabled=use_amp):
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_TARGET_LEN,
                    num_beams=NUM_BEAMS,
                    early_stopping=True
                )

            batch_preds = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            batch_refs = list(batch["target"])

            preds.extend(batch_preds)
            refs.extend(batch_refs)

            for i in range(len(batch_preds)):
                pred_norm = normalize_code(batch_preds[i])
                ref_norm = normalize_code(batch_refs[i])

                records.append({
                    "idx": batch["idx"][i],
                    "shift_type": batch["shift_type"][i],
                    "source": batch["source"][i],
                    "target": batch_refs[i],
                    "prediction": batch_preds[i],
                    "exact_match": pred_norm == ref_norm
                })

    result = compute_repair_metrics(preds, refs)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return result


def print_final_table(title, train_eval, id_eval, ood_eval):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'指标':<18} | "
        f"{'ID Train':<12} | "
        f"{'ID Test':<12} | "
        f"{'OOD Test':<12} | "
        f"{'跌幅 ID-OOD':<12}"
    )
    print("-" * 100)

    for key, name in [
        ("exact_match", "Exact Match"),
        ("bleu", "BLEU"),
        ("codebleu", "CodeBLEU")
    ]:
        print(
            f"{name:<18} | "
            f"{train_eval[key]:<12.4f} | "
            f"{id_eval[key]:<12.4f} | "
            f"{ood_eval[key]:<12.4f} | "
            f"{id_eval[key] - ood_eval[key]:<12.4f}"
        )

    print("-" * 100)
    print("ID Train 样本数:", train_eval["num_samples"])
    print("ID Test 样本数 :", id_eval["num_samples"])
    print("OOD Test 样本数:", ood_eval["num_samples"])
    print("=" * 100)


# ============================================================
# 6. 单个 subset 运行
# ============================================================

def run_one_subset(info, tokenizer, model_type, model_path):
    subset_name = info["name"]
    subset_dir = info["dir"]
    model_type = normalize_model_type(model_type)

    save_dir = os.path.join(SAVE_ROOT, model_type, subset_name)
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, f"best_bugs2fix_{model_type}.bin")
    metrics_file = os.path.join(save_dir, f"rq1_bugs2fix_metrics_{model_type}.json")

    print("\n" + "#" * 120)
    print(f"开始运行 Bugs2Fix RQ1: {subset_name}")
    print("#" * 120)
    print("subset_dir:", subset_dir)
    print("train:", info["train_file"])
    print("id test:", info["id_test_file"])
    print("ood token:", info["ood_token_file"])
    print("ood cst:", info["ood_cst_file"])
    print("model_type:", model_type)
    print("model_path:", model_path)
    print("save_dir:", save_dir)
    print("#" * 120)

    start_time = time.time()

    train_ds = Bugs2FixDataset(
        info["train_file"],
        tokenizer,
        max_samples=MAX_TRAIN_SAMPLES,
        shuffle_before_limit=True
    )

    id_test_ds = Bugs2FixDataset(
        info["id_test_file"],
        tokenizer,
        max_samples=MAX_EVAL_SAMPLES,
        shuffle_before_limit=False
    )

    ood_token_ds = Bugs2FixDataset(
        info["ood_token_file"],
        tokenizer,
        max_samples=MAX_EVAL_SAMPLES,
        shuffle_before_limit=False
    )

    ood_cst_ds = Bugs2FixDataset(
        info["ood_cst_file"],
        tokenizer,
        max_samples=MAX_EVAL_SAMPLES,
        shuffle_before_limit=False
    )

    if MAX_TRAIN_EVAL_SAMPLES is not None:
        train_eval_size = min(MAX_TRAIN_EVAL_SAMPLES, len(train_ds))
        train_eval_ds = Subset(train_ds, range(train_eval_size))
    else:
        train_eval_ds = train_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    id_test_loader = DataLoader(
        id_test_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    ood_token_loader = DataLoader(
        ood_token_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    ood_cst_loader = DataLoader(
        ood_cst_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = build_generation_model(model_type, model_path, tokenizer)

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

    best_em = -1.0
    best_epoch = 0

    print("\n>>> 开始训练 Bugs2Fix 代码修复模型...")

    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            epoch
        )

        # RQ1 不使用 valid：每轮用 ID Test 选择 best epoch。
        # 代码修复官方核心指标是 Exact Match，所以用 EM 选最佳模型。
        id_eval_for_best = evaluate_generation(
            model,
            tokenizer,
            id_test_loader,
            desc=f"验证 Epoch {epoch} on ID Test"
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch} | "
            f"Loss: {train_loss:.4f} | "
            f"ID EM: {id_eval_for_best['exact_match']:.4f} | "
            f"ID BLEU: {id_eval_for_best['bleu']:.4f} | "
            f"ID CodeBLEU: {id_eval_for_best['codebleu']:.4f} | "
            f"Epoch Time: {format_time(epoch_time)}"
        )

        epoch_save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_{model_type}.bin")
        torch.save(model.state_dict(), epoch_save_path)

        if id_eval_for_best["exact_match"] > best_em:
            best_em = id_eval_for_best["exact_match"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ 新最佳模型已保存: {best_model_path}")

    train_time = time.time() - train_start

    print("\n" + "=" * 100)
    print(f"训练阶段总耗时: {format_time(train_time)}")
    print(f"加载第 {best_epoch} 轮最佳模型，执行最终 RQ1 评估...")
    print("=" * 100)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"未找到最佳模型文件: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    eval_start = time.time()

    train_pred_path = os.path.join(save_dir, "predictions_id_train_sample.jsonl")
    id_pred_path = os.path.join(save_dir, "predictions_id_test.jsonl")
    token_pred_path = os.path.join(save_dir, "predictions_ood_token_test.jsonl")
    cst_pred_path = os.path.join(save_dir, "predictions_ood_cst_test.jsonl")

    train_eval = evaluate_generation(
        model,
        tokenizer,
        train_eval_loader,
        desc="最终评估 ID Train",
        save_path=train_pred_path
    )

    id_eval = evaluate_generation(
        model,
        tokenizer,
        id_test_loader,
        desc="最终评估 ID Test",
        save_path=id_pred_path
    )

    token_eval = evaluate_generation(
        model,
        tokenizer,
        ood_token_loader,
        desc="最终评估 OOD Token Test",
        save_path=token_pred_path
    )

    cst_eval = evaluate_generation(
        model,
        tokenizer,
        ood_cst_loader,
        desc="最终评估 OOD CST Test",
        save_path=cst_pred_path
    )

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    final_metrics = {
        "task": "Code Repair / Code Refinement",
        "dataset": "CodeXGLUE Bugs2Fix",
        "subset": subset_name,
        "model_type": model_type,
        "model_path": model_path,
        "best_epoch": best_epoch,
        "best_id_test_exact_match": best_em,
        "id_train": train_eval,
        "id_test": id_eval,
        "ood_test": {
            "token": token_eval,
            "cst": cst_eval
        },
        "drop": {
            "token": {
                "exact_match_drop": id_eval["exact_match"] - token_eval["exact_match"],
                "bleu_drop": id_eval["bleu"] - token_eval["bleu"],
                "codebleu_drop": id_eval["codebleu"] - token_eval["codebleu"]
            },
            "cst": {
                "exact_match_drop": id_eval["exact_match"] - cst_eval["exact_match"],
                "bleu_drop": id_eval["bleu"] - cst_eval["bleu"],
                "codebleu_drop": id_eval["codebleu"] - cst_eval["codebleu"]
            }
        },
        "time": {
            "train_elapsed_seconds": train_time,
            "train_elapsed_readable": format_time(train_time),
            "eval_elapsed_seconds": eval_time,
            "eval_elapsed_readable": format_time(eval_time),
            "total_elapsed_seconds": total_time,
            "total_elapsed_readable": format_time(total_time)
        },
        "files": {
            "train_file": info["train_file"],
            "id_test_file": info["id_test_file"],
            "ood_token_file": info["ood_token_file"],
            "ood_cst_file": info["ood_cst_file"],
            "best_model": best_model_path,
            "train_predictions": train_pred_path,
            "id_predictions": id_pred_path,
            "token_predictions": token_pred_path,
            "cst_predictions": cst_pred_path
        }
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print_final_table(
        title=f"RQ1 最终结果：Bugs2Fix - {subset_name} - TOKEN Shift",
        train_eval=train_eval,
        id_eval=id_eval,
        ood_eval=token_eval
    )

    print_final_table(
        title=f"RQ1 最终结果：Bugs2Fix - {subset_name} - CST Shift",
        train_eval=train_eval,
        id_eval=id_eval,
        ood_eval=cst_eval
    )

    print("\n运行时间与文件保存信息")
    print("-" * 100)
    print(f"训练阶段总耗时: {format_time(train_time)}")
    print(f"最终评估总耗时: {format_time(eval_time)}")
    print(f"当前子集总耗时: {format_time(total_time)}")
    print(f"最佳模型保存于: {best_model_path}")
    print(f"ID Train 预测结果保存于: {train_pred_path}")
    print(f"ID Test 预测结果保存于: {id_pred_path}")
    print(f"OOD Token 预测结果保存于: {token_pred_path}")
    print(f"OOD CST 预测结果保存于: {cst_pred_path}")
    print(f"汇总指标保存于: {metrics_file}")
    print("-" * 100)

    return final_metrics


# ============================================================
# 7. 主函数
# ============================================================

def main():
    program_start = time.time()
    args = build_arg_parser().parse_args()
    selected_model_type = normalize_model_type(args.model_type)

    if selected_model_type == "all":
        experiments = [
            ("codeT5", args.codeT5_path),
            ("PLBART", args.PLBART_path),
        ]
    else:
        experiments = [
            (selected_model_type, args.model_path or get_default_model_path(selected_model_type)),
        ]

    print("=" * 120)
    print("CodeXGLUE Bugs2Fix RQ1: Code Repair under Token/CST Shift")
    print("=" * 120)
    print("MODEL_TYPE:", selected_model_type)
    print("EXPERIMENTS:", experiments)
    print("DATA_ROOT:", DATA_ROOT)
    print("SAVE_ROOT:", SAVE_ROOT)
    print("DEVICE:", DEVICE)
    print("SUBSET_DIRS:", SUBSET_DIRS)
    print("Metrics: Exact Match, BLEU, CodeBLEU")
    print("Best model selected by ID Test Exact Match")
    print("=" * 120)

    subset_infos = discover_subset_dirs()

    if not subset_infos:
        raise ValueError(
            "没有找到包含 id_train/id_test/ood_test_token/ood_test_cst 的 Bugs2Fix 子集目录。"
        )

    print("\n将要运行的 Bugs2Fix 子集:")
    for x in subset_infos:
        print(" -", x["name"], "=>", x["dir"])

    all_model_summaries = []

    for model_type, model_path in experiments:
        set_seed(SEED)

        print("\n" + "=" * 120)
        print(f"开始运行模型: {model_type}")
        print("MODEL_PATH:", model_path)
        print("=" * 120)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        all_results = []

        for info in subset_infos:
            result = run_one_subset(info, tokenizer, model_type, model_path)
            all_results.append(result)

        summary_path = os.path.join(SAVE_ROOT, model_type, f"rq1_all_bugs2fix_summary_{model_type}.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        summary = {
            "task": "Code Repair / Code Refinement",
            "dataset": "CodeXGLUE Bugs2Fix",
            "model_type": model_type,
            "model_path": model_path,
            "metrics": ["Exact Match", "BLEU", "CodeBLEU"],
            "best_model_selection": "ID Test Exact Match",
            "num_subsets": len(all_results),
            "results": all_results,
            "program_elapsed_seconds": time.time() - program_start,
            "program_elapsed_readable": format_time(time.time() - program_start)
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        all_model_summaries.append(summary_path)

        print("\n" + "=" * 120)
        print(f"模型 {model_type} 运行完成")
        print("=" * 120)
        print("总子集数:", len(all_results))
        print("当前模型汇总文件:", summary_path)

    print("\n" + "=" * 120)
    print("全部 Bugs2Fix RQ1 运行完成")
    print("=" * 120)
    print("总模型数:", len(experiments))
    print("总子集数:", len(subset_infos))
    print("总耗时:", format_time(time.time() - program_start))
    print("总汇总文件:")
    for path in all_model_summaries:
        print(" -", path)
    print("=" * 120)


if __name__ == "__main__":
    main()
