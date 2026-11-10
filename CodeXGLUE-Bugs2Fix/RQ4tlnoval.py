# -*- coding: utf-8 -*-
"""
RQ4 Transfer Learning for CodeXGLUE Bugs2Fix / Code Refinement.

Small-CST setting only:
  - source checkpoint: RQ1 best model trained on ID train
  - transfer train : small/ood_train_cst(.jsonl)
  - final test     : small/id_test.jsonl and small/ood_test_cst_new(.jsonl)

This is a no-validation protocol, aligned with the existing RQ4 scripts: the
best transfer checkpoint is selected by OOD test Exact Match.
"""

import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


# ============================================================
# 1. Config
# ============================================================

MODEL_TYPE = "PLBART"  # options: "codeT5", "PLBART"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codeT5", "PLBART"]

PROJECT_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SUBSET_NAME = "small"
SUBSET_DIR = os.path.join(DATA_ROOT, SUBSET_NAME)
SAVE_ROOT = os.path.join(DATA_ROOT, "rq4_tl_small_cst_results")

ID_TEST_FILE = os.path.join(SUBSET_DIR, "id_test.jsonl")
OOD_TRAIN_FILE_CANDIDATES = [
    os.path.join(SUBSET_DIR, "ood_train_cst.jsonl"),
    os.path.join(SUBSET_DIR, "ood_train_cst"),
]
OOD_TEST_FILE_CANDIDATES = [
    os.path.join(SUBSET_DIR, "ood_test_cst_new.jsonl"),
    os.path.join(SUBSET_DIR, "ood_test_cst_new"),
]

RQ1_OUTPUT_ROOT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "rq1_outputs"),
    os.path.join(DATA_ROOT, "rq1_outputs"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
EPOCHS = 8
LR_FT = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
NUM_BEAMS = 5
SEED = 42
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

os.makedirs(SAVE_ROOT, exist_ok=True)


def normalize_model_type(model_type):
    lower = str(model_type).lower()
    for name in MODEL_PATHS:
        if lower == name.lower():
            return name
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {list(MODEL_PATHS.keys())}")


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, SAVE_DIR, RQ1_BEST_MODEL, FINAL_MODEL_PATH, METRICS_FILE
    MODEL_TYPE = normalize_model_type(model_type)
    MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
    SAVE_DIR = os.path.join(SAVE_ROOT, MODEL_TYPE, SUBSET_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    rq1_candidates = [
        os.path.join(root, MODEL_TYPE, SUBSET_NAME, f"best_bugs2fix_{MODEL_TYPE}.bin")
        for root in RQ1_OUTPUT_ROOT_CANDIDATES
    ]
    RQ1_BEST_MODEL = next((path for path in rq1_candidates if os.path.exists(path)), rq1_candidates[0])
    FINAL_MODEL_PATH = os.path.join(SAVE_DIR, f"final_tl_bugs2fix_{MODEL_TYPE}_{SUBSET_NAME}_cst.bin")
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq4_tl_bugs2fix_{MODEL_TYPE}_{SUBSET_NAME}_cst.json")


configure_model_runtime(MODEL_TYPE)


# ============================================================
# 2. Utilities / Metrics
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


def resolve_first_existing(candidates, role):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find {role}. Tried:\n" + "\n".join(f"  - {p}" for p in candidates))


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_code(text):
    if text is None:
        return ""
    return " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").strip().split())


def get_source(js):
    return js.get("source") or js.get("buggy") or js.get("src") or js.get("input") or ""


def get_target(js):
    return js.get("target") or js.get("fixed") or js.get("tgt") or js.get("output") or ""


def exact_match_score(preds, refs):
    if not preds:
        return 0.0
    return 100.0 * sum(normalize_code(p) == normalize_code(r) for p, r in zip(preds, refs)) / len(preds)


def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu_score(preds, refs, max_n=4, smooth=1.0):
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
            clipped_counts[n - 1] += sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in pred_ngrams.items())
            total_counts[n - 1] += max(sum(pred_ngrams.values()), 0)
    if pred_len == 0:
        return 0.0
    precisions = [(clipped_counts[i] + smooth) / (total_counts[i] + smooth) for i in range(max_n)]
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n) * 100.0)


def codebleu_score(preds, refs, lang="java"):
    try:
        from codebleu import calc_codebleu
    except Exception as exc:
        print(f"[Warning] Cannot import codebleu; CodeBLEU=-1.0. {repr(exc)}")
        return -1.0
    norm_preds = [normalize_code(x) for x in preds]
    norm_refs = [normalize_code(x) for x in refs]
    try:
        result = calc_codebleu(references=[[r] for r in norm_refs], predictions=norm_preds, lang=lang)
    except TypeError:
        result = calc_codebleu([[r] for r in norm_refs], norm_preds, lang)
    except Exception as exc:
        print(f"[Warning] CodeBLEU failed; CodeBLEU=-1.0. {repr(exc)}")
        return -1.0
    value = result.get("codebleu") if isinstance(result, dict) else result
    if value is None:
        return -1.0
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def compute_repair_metrics(preds, refs):
    return {
        "exact_match": float(exact_match_score(preds, refs)),
        "bleu": float(corpus_bleu_score(preds, refs)),
        "codebleu": float(codebleu_score(preds, refs, lang="java")),
        "num_samples": len(preds),
    }


# ============================================================
# 3. Dataset / Model
# ============================================================

class Bugs2FixDataset(Dataset):
    def __init__(self, path, tokenizer, max_samples=None, shuffle_before_limit=False):
        self.path = path
        self.tokenizer = tokenizer
        self.examples = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Reading {os.path.basename(path)}", leave=False)):
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
                })
        if shuffle_before_limit:
            random.shuffle(self.examples)
        if max_samples is not None and len(self.examples) > max_samples:
            self.examples = self.examples[:max_samples]
        print(f">>> Loaded {len(self.examples)} Bugs2Fix samples from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        source_inputs = self.tokenizer(
            item["source"], max_length=MAX_SOURCE_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_inputs = self.tokenizer(
            item["target"], max_length=MAX_TARGET_LEN, padding="max_length", truncation=True, return_tensors="pt"
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
            "shift_type": item["shift_type"],
        }


def build_generation_model(model_type, model_path):
    model_type = normalize_model_type(model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model.to(DEVICE)


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        tried = [
            os.path.join(root, MODEL_TYPE, SUBSET_NAME, f"best_bugs2fix_{MODEL_TYPE}.bin")
            for root in RQ1_OUTPUT_ROOT_CANDIDATES
        ]
        raise FileNotFoundError("Cannot find RQ1 best model. Tried:\n" + "\n".join(f"  - {path}" for path in tried))
    model = build_generation_model(MODEL_TYPE, MODEL_PATH)
    model.load_state_dict(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    print(">>> Loaded RQ1 checkpoint:", RQ1_BEST_MODEL)
    return model


# ============================================================
# 4. Train / Eval
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    steps = 0
    use_amp = DEVICE.type == "cuda"
    pbar = tqdm(dataloader, desc=f"TL Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        with autocast(enabled=use_amp):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if torch.isnan(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.item())
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_generation(model, tokenizer, dataloader, desc="Evaluating", save_path=None):
    model.eval()
    preds, refs, records = [], [], []
    use_amp = DEVICE.type == "cuda"
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with autocast(enabled=use_amp):
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        batch_refs = list(batch["target"])
        preds.extend(batch_preds)
        refs.extend(batch_refs)
        for i, pred in enumerate(batch_preds):
            records.append({
                "idx": batch["idx"][i],
                "shift_type": batch["shift_type"][i],
                "source": batch["source"][i],
                "target": batch_refs[i],
                "prediction": pred,
                "exact_match": normalize_code(pred) == normalize_code(batch_refs[i]),
            })
    result = compute_repair_metrics(preds, refs)
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def print_final_table(title, id_eval, ood_eval):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'Metric':<18} | {'ID Test':<12} | {'OOD Test':<12} | {'Drop ID-OOD':<12}")
    print("-" * 100)
    for key, name in [("exact_match", "Exact Match"), ("bleu", "BLEU"), ("codebleu", "CodeBLEU")]:
        print(f"{name:<18} | {id_eval[key]:<12.4f} | {ood_eval[key]:<12.4f} | {id_eval[key] - ood_eval[key]:<12.4f}")
    print("-" * 100)
    print("ID samples :", id_eval["num_samples"])
    print("OOD samples:", ood_eval["num_samples"])
    print("=" * 100)


def run_one_model(model_type=None):
    configure_model_runtime(model_type or MODEL_TYPE)
    set_seed(SEED)
    start_time = time.time()

    ood_train_file = resolve_first_existing(OOD_TRAIN_FILE_CANDIDATES, "OOD CST train file")
    ood_test_file = resolve_first_existing(OOD_TEST_FILE_CANDIDATES, "OOD CST test file")

    print("=" * 120)
    print("Bugs2Fix RQ4 Transfer Learning: small-cst")
    print("=" * 120)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TRAIN_FILE:", ood_train_file)
    print("OOD_TEST_FILE:", ood_test_file)
    print("SAVE_DIR:", SAVE_DIR)
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    id_test_ds = Bugs2FixDataset(ID_TEST_FILE, tokenizer, max_samples=MAX_EVAL_SAMPLES)
    ood_train_ds = Bugs2FixDataset(ood_train_file, tokenizer, max_samples=MAX_TRAIN_SAMPLES, shuffle_before_limit=True)
    ood_test_ds = Bugs2FixDataset(ood_test_file, tokenizer, max_samples=MAX_EVAL_SAMPLES)

    id_loader = DataLoader(id_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    ood_train_loader = DataLoader(ood_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    ood_loader = DataLoader(ood_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    model = load_rq1_model()
    optimizer = AdamW(model.parameters(), lr=LR_FT, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(ood_train_loader) * EPOCHS, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    best_epoch = 0
    best_ood_em = -1.0
    epoch_metrics = []
    train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, ood_train_loader, optimizer, scheduler, scaler, epoch)
        epoch_id_eval = evaluate_generation(model, tokenizer, id_loader, desc=f"Epoch {epoch} ID Test")
        epoch_ood_eval = evaluate_generation(model, tokenizer, ood_loader, desc=f"Epoch {epoch} OOD CST Test")
        is_best = epoch_ood_eval["exact_match"] > best_ood_em
        if is_best:
            best_epoch = epoch
            best_ood_em = float(epoch_ood_eval["exact_match"])
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f">>> New best TL checkpoint by OOD EM: epoch={epoch}, ood_em={best_ood_em:.4f}")
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "id_test": epoch_id_eval,
            "ood_test": epoch_ood_eval,
            "is_best": is_best,
            "epoch_seconds": time.time() - epoch_start,
            "epoch_readable": format_time(time.time() - epoch_start),
        })
        print(
            f"Epoch {epoch} | loss={train_loss:.4f} | ID EM={epoch_id_eval['exact_match']:.4f} | "
            f"OOD EM={epoch_ood_eval['exact_match']:.4f} | time={format_time(time.time() - epoch_start)}"
        )

    train_time = time.time() - train_start
    model.load_state_dict(safe_torch_load(FINAL_MODEL_PATH, map_location=DEVICE))
    id_pred_path = os.path.join(SAVE_DIR, f"pred_tl_id_{MODEL_TYPE}_{SUBSET_NAME}_cst.jsonl")
    ood_pred_path = os.path.join(SAVE_DIR, f"pred_tl_ood_{MODEL_TYPE}_{SUBSET_NAME}_cst.jsonl")
    final_id_eval = evaluate_generation(model, tokenizer, id_loader, desc="Best TL ID Test", save_path=id_pred_path)
    final_ood_eval = evaluate_generation(model, tokenizer, ood_loader, desc="Best TL OOD CST Test", save_path=ood_pred_path)

    metrics = {
        "task": "Code Repair / Code Refinement",
        "method": "RQ4 transfer learning on small-cst OOD train split",
        "dataset": "CodeXGLUE Bugs2Fix",
        "subset": SUBSET_NAME,
        "shift_type": "cst",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "rq1_best_model": RQ1_BEST_MODEL,
        "final_transfer_model": FINAL_MODEL_PATH,
        "best_epoch": best_epoch,
        "best_ood_exact_match": best_ood_em,
        "id_test": final_id_eval,
        "ood_test": final_ood_eval,
        "drop_id_minus_ood": {
            "exact_match_drop": final_id_eval["exact_match"] - final_ood_eval["exact_match"],
            "bleu_drop": final_id_eval["bleu"] - final_ood_eval["bleu"],
            "codebleu_drop": final_id_eval["codebleu"] - final_ood_eval["codebleu"],
        },
        "epoch_metrics": epoch_metrics,
        "data": {
            "id_test_file": ID_TEST_FILE,
            "ood_train_file": ood_train_file,
            "ood_test_file": ood_test_file,
        },
        "files": {
            "metrics": METRICS_FILE,
            "final_model": FINAL_MODEL_PATH,
            "id_predictions": id_pred_path,
            "ood_predictions": ood_pred_path,
        },
        "time": {
            "train_seconds": train_time,
            "train_readable": format_time(train_time),
            "total_seconds": time.time() - start_time,
            "total_readable": format_time(time.time() - start_time),
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print_final_table(
        f"RQ4 Final Results: Bugs2Fix {SUBSET_NAME.upper()} CST Transfer - {MODEL_TYPE}",
        final_id_eval,
        final_ood_eval,
    )
    print("Metrics saved:", METRICS_FILE)
    return metrics


def main():
    models_to_run = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    all_results = []
    for model_type in models_to_run:
        try:
            all_results.append(run_one_model(model_type))
        except Exception as exc:
            if not RUN_ALL_MODELS:
                raise
            all_results.append({"status": "failed", "model_type": model_type, "error": repr(exc)})
            print(f"[Error] RQ4 TL failed for {model_type}: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if RUN_ALL_MODELS:
        path = os.path.join(SAVE_ROOT, f"rq4_tl_bugs2fix_{SUBSET_NAME}_cst_all_models.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"models": models_to_run, "results": all_results}, f, ensure_ascii=False, indent=2)
        print("All-model metrics saved:", path)


if __name__ == "__main__":
    main()
