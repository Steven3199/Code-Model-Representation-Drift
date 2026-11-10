# -*- coding: utf-8 -*-
"""
CodeXGLUE CodeTrans RQ4 Transfer Learning.

ID direction from RQ1:
    C# -> Java

OOD language-direction shift:
    Java -> C#

RQ4 TL protocol:
    load RQ1 best checkpoint (C# -> Java)
    -> fine-tune on randomly split OOD train (Java -> C#)
    -> evaluate ID test (C# -> Java) and OOD test (Java -> C#)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


# ============================================================
# 1. Configuration
# ============================================================

MODEL_TYPE = "PLBART"  # choices: "codeT5", "PLBART", "all"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
MODEL_TYPES_TO_RUN = ["codeT5", "PLBART"]

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/data"
OOD_SPLIT_DIR = os.path.join(DATA_DIR, "ood_split_java_to_cs_80_20")
RQ1_SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq1_language_shift_outputs_cs_to_java_id"
SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq4_tl_language_shift_outputs_cs_to_java_id"

ID_TEST_SRC_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.cs")
ID_TEST_TGT_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.java")

OOD_TRAIN_SRC_FILE = os.path.join(OOD_SPLIT_DIR, "ood_train.java-cs.txt.java")
OOD_TRAIN_TGT_FILE = os.path.join(OOD_SPLIT_DIR, "ood_train.java-cs.txt.cs")
OOD_TEST_SRC_FILE = os.path.join(OOD_SPLIT_DIR, "ood_test.java-cs.txt.java")
OOD_TEST_TGT_FILE = os.path.join(OOD_SPLIT_DIR, "ood_test.java-cs.txt.cs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 8
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
NUM_BEAMS = 5
SEED = 42
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
SOURCE_PREFIX = "translate code:"
ENABLE_CODEBLEU = True
ID_CODEBLEU_LANG = "java"
OOD_CODEBLEU_LANG = "c_sharp"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.2f}s"


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_model_type(model_type):
    lower = str(model_type).lower()
    if lower == "all":
        return "all"
    aliases = {"codet5": "codeT5", "codet5": "codeT5", "plbart": "PLBART"}
    if model_type in MODEL_PATHS:
        return model_type
    if lower in aliases:
        return aliases[lower]
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from codeT5, PLBART, all")


def resolve_model_path(model_type, model_path=None):
    return model_path or MODEL_PATHS[normalize_model_type(model_type)]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE CodeTrans RQ4 Transfer Learning")
    parser.add_argument("--model_type", default=MODEL_TYPE, help="codeT5, PLBART, or all")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--codeT5_path", default=MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=MODEL_PATHS["PLBART"])
    return parser


def get_rq1_best_model(model_type):
    return os.path.join(
        RQ1_SAVE_DIR,
        model_type,
        f"best_codetrans_cs_to_java_id_{model_type}.bin",
    )


def get_output_paths(model_type):
    run_save_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(run_save_dir, exist_ok=True)
    prefix = f"rq4_tl_codetrans_cs_to_java_id_java_to_cs_ood_{model_type}"
    return {
        "save_dir": run_save_dir,
        "best_model": os.path.join(run_save_dir, f"best_{prefix}.bin"),
        "metrics": os.path.join(run_save_dir, f"{prefix}_metrics.json"),
        "id_predictions": os.path.join(run_save_dir, f"pred_tl_id_test_cs_to_java_{model_type}.jsonl"),
        "ood_predictions": os.path.join(run_save_dir, f"pred_tl_ood_test_java_to_cs_{model_type}.jsonl"),
    }


def normalize_code(text):
    return " ".join(str(text).strip().split())


def read_parallel_files(src_path, tgt_path, max_samples=None):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Target file not found: {tgt_path}")
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        src_lines = [line.rstrip("\n") for line in f]
    with open(tgt_path, "r", encoding="utf-8", errors="ignore") as f:
        tgt_lines = [line.rstrip("\n") for line in f]
    n = min(len(src_lines), len(tgt_lines))
    if max_samples is not None:
        n = min(n, int(max_samples))
    data = []
    for i in range(n):
        src = src_lines[i].strip()
        tgt = tgt_lines[i].strip()
        if not src or not tgt:
            continue
        data.append({"idx": str(i), "source": src, "target": tgt})
    print(f">>> Loaded {len(data)} pairs: {src_path} -> {tgt_path}")
    return data


def configure_special_tokens(tokenizer, model):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None:
        if tokenizer.bos_token_id is not None:
            model.config.decoder_start_token_id = tokenizer.bos_token_id
        elif tokenizer.eos_token_id is not None:
            model.config.decoder_start_token_id = tokenizer.eos_token_id
        else:
            model.config.decoder_start_token_id = tokenizer.pad_token_id


# ============================================================
# 3. Metrics
# ============================================================

def exact_match(preds, refs):
    if not preds:
        return 0.0
    hits = sum(1 for pred, ref in zip(preds, refs) if normalize_code(pred) == normalize_code(ref))
    return 100.0 * hits / len(preds)


def compute_bleu(preds, refs):
    if not preds:
        return 0.0
    smoothie = SmoothingFunction().method4
    references = [[normalize_code(ref).split()] for ref in refs]
    hypotheses = [normalize_code(pred).split() for pred in preds]
    try:
        return float(corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100.0)
    except Exception as exc:
        print(f"[BLEU Warning] failed: {repr(exc)}")
        return 0.0


def _normalise_codebleu_score(raw_score):
    if raw_score is None:
        return None
    score = float(raw_score)
    return score * 100.0 if score <= 1.0 else score


def compute_codebleu_with_package(preds, refs, lang):
    if not ENABLE_CODEBLEU or not preds or len(preds) != len(refs):
        return None
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(
            references=refs,
            predictions=preds,
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None,
        )
        if isinstance(result, dict):
            score = result.get("codebleu", result.get("CodeBLEU"))
            return _normalise_codebleu_score(score)
        if isinstance(result, (int, float)):
            return _normalise_codebleu_score(result)
    except Exception as exc:
        print(f"[CodeBLEU Warning] failed for lang={lang}: {repr(exc)}")
    return None


def compute_codebleu_safe(preds, refs, lang):
    if not ENABLE_CODEBLEU:
        return None
    lang = str(lang).strip()
    if lang.lower() in {"c_sharp", "csharp", "cs", "c#"}:
        candidates = [lang, "c_sharp", "csharp", "cs"]
    elif lang.lower() == "java":
        candidates = ["java"]
    else:
        candidates = [lang]
    seen = set()
    for candidate in [x for x in candidates if not (x in seen or seen.add(x))]:
        score = compute_codebleu_with_package(preds, refs, candidate)
        if score is not None:
            return score
    return None


def compute_generation_metrics(preds, refs, lang):
    codebleu = compute_codebleu_safe(preds, refs, lang)
    return {
        "exact_match": float(exact_match(preds, refs)),
        "bleu": float(compute_bleu(preds, refs)),
        "codebleu": None if codebleu is None else float(codebleu),
        "num_samples": len(preds),
    }


def id_ood_drop(id_eval, ood_eval):
    return {
        "exact_match_drop": id_eval["exact_match"] - ood_eval["exact_match"],
        "bleu_drop": id_eval["bleu"] - ood_eval["bleu"],
        "codebleu_drop": None if id_eval["codebleu"] is None or ood_eval["codebleu"] is None else id_eval["codebleu"] - ood_eval["codebleu"],
    }


def fmt_metric(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


# ============================================================
# 4. Dataset / Model
# ============================================================

class CodeTransDataset(Dataset):
    def __init__(self, data, tokenizer, source_prefix=SOURCE_PREFIX):
        self.examples = []
        for item in data:
            src_text = f"{source_prefix} {item['source']}"
            source_enc = tokenizer(
                src_text,
                max_length=MAX_SOURCE_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            target_enc = tokenizer(
                item["target"],
                max_length=MAX_TARGET_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = target_enc["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100
            self.examples.append({
                "idx": item["idx"],
                "source": item["source"],
                "target": item["target"],
                "input_ids": source_enc["input_ids"].squeeze(0),
                "attention_mask": source_enc["attention_mask"].squeeze(0),
                "labels": labels,
            })
        print(f">>> Built CodeTransDataset: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def load_rq1_model(model_type, model_path, tokenizer):
    best_model_path = get_rq1_best_model(model_type)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"RQ1 best model not found: {best_model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    configure_special_tokens(tokenizer, model)
    model.load_state_dict(safe_torch_load(best_model_path, map_location=DEVICE))
    print(">>> Loaded RQ1 checkpoint:", best_model_path)
    return model, best_model_path


# ============================================================
# 5. Train / Eval
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch):
    model.train()
    use_amp = DEVICE.type == "cuda"
    total_loss = 0.0
    steps = 0
    pbar = tqdm(dataloader, desc=f"TL Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
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
def generate_predictions(model, dataloader, tokenizer, desc="Generating", save_path=None):
    model.eval()
    preds, refs, records = [], [], []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
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
                "source": batch["source"][i],
                "reference": batch_refs[i],
                "prediction": pred,
                "exact_match": normalize_code(pred) == normalize_code(batch_refs[i]),
            })
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return preds, refs


def evaluate_generation(model, dataloader, tokenizer, lang, desc, save_path=None):
    preds, refs = generate_predictions(model, dataloader, tokenizer, desc=desc, save_path=save_path)
    return compute_generation_metrics(preds, refs, lang)


def print_final_table(title, id_eval, ood_eval):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'Metric':<18} | {'ID Test':<12} | {'OOD Test':<12} | {'Drop ID-OOD':<12}")
    print("-" * 110)
    for key, name in [("exact_match", "Exact Match"), ("bleu", "BLEU"), ("codebleu", "CodeBLEU")]:
        id_value = id_eval.get(key)
        ood_value = ood_eval.get(key)
        drop = None if id_value is None or ood_value is None else id_value - ood_value
        print(f"{name:<18} | {fmt_metric(id_value):<12} | {fmt_metric(ood_value):<12} | {fmt_metric(drop):<12}")
    print("-" * 110)
    print("ID samples :", id_eval["num_samples"])
    print("OOD samples:", ood_eval["num_samples"])
    print("=" * 110)


# ============================================================
# 6. Run
# ============================================================

def run_experiment(model_type, model_path):
    set_seed(SEED)
    model_type = normalize_model_type(model_type)
    model_path = resolve_model_path(model_type, model_path)
    output_paths = get_output_paths(model_type)
    start_time = time.time()

    print("=" * 120)
    print("CodeXGLUE-CodeTrans RQ4 Transfer Learning: C#->Java ID, Java->C# OOD split")
    print("=" * 120)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("RQ1_BEST_MODEL:", get_rq1_best_model(model_type))
    print("ID Test :", ID_TEST_SRC_FILE, "->", ID_TEST_TGT_FILE)
    print("OOD Train:", OOD_TRAIN_SRC_FILE, "->", OOD_TRAIN_TGT_FILE)
    print("OOD Test :", OOD_TEST_SRC_FILE, "->", OOD_TEST_TGT_FILE)
    print("SAVE_DIR:", output_paths["save_dir"])
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id_test_data = read_parallel_files(ID_TEST_SRC_FILE, ID_TEST_TGT_FILE, max_samples=MAX_EVAL_SAMPLES)
    ood_train_data = read_parallel_files(OOD_TRAIN_SRC_FILE, OOD_TRAIN_TGT_FILE, max_samples=MAX_TRAIN_SAMPLES)
    ood_test_data = read_parallel_files(OOD_TEST_SRC_FILE, OOD_TEST_TGT_FILE, max_samples=MAX_EVAL_SAMPLES)

    id_test_ds = CodeTransDataset(id_test_data, tokenizer)
    ood_train_ds = CodeTransDataset(ood_train_data, tokenizer)
    ood_test_ds = CodeTransDataset(ood_test_data, tokenizer)
    id_loader = DataLoader(id_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    ood_train_loader = DataLoader(ood_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    ood_loader = DataLoader(ood_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model, rq1_best_model = load_rq1_model(model_type, model_path, tokenizer)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        id_eval_epoch = evaluate_generation(model, id_loader, tokenizer, ID_CODEBLEU_LANG, f"Epoch {epoch} ID Test")
        ood_eval_epoch = evaluate_generation(model, ood_loader, tokenizer, OOD_CODEBLEU_LANG, f"Epoch {epoch} OOD Test")
        is_best = ood_eval_epoch["exact_match"] > best_ood_em
        if is_best:
            best_epoch = epoch
            best_ood_em = float(ood_eval_epoch["exact_match"])
            torch.save(model.state_dict(), output_paths["best_model"])
            print(f">>> New best TL checkpoint by OOD EM: epoch={epoch}, ood_em={best_ood_em:.4f}")
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "id_test": id_eval_epoch,
            "ood_test": ood_eval_epoch,
            "is_best": is_best,
            "epoch_seconds": time.time() - epoch_start,
            "epoch_readable": format_time(time.time() - epoch_start),
        })
        print(
            f"Epoch {epoch} | loss={train_loss:.4f} | ID EM={id_eval_epoch['exact_match']:.4f} | "
            f"OOD EM={ood_eval_epoch['exact_match']:.4f} | time={format_time(time.time() - epoch_start)}"
        )

    train_time = time.time() - train_start
    model.load_state_dict(safe_torch_load(output_paths["best_model"], map_location=DEVICE))
    id_eval = evaluate_generation(model, id_loader, tokenizer, ID_CODEBLEU_LANG, "Best TL ID Test", output_paths["id_predictions"])
    ood_eval = evaluate_generation(model, ood_loader, tokenizer, OOD_CODEBLEU_LANG, "Best TL OOD Test", output_paths["ood_predictions"])

    metrics = {
        "task": "Code-to-Code Translation",
        "dataset": "CodeXGLUE-CodeTrans",
        "method": "RQ4 transfer learning on random 80/20 Java->C# OOD split",
        "model_type": model_type,
        "model_path": model_path,
        "rq1_best_model": rq1_best_model,
        "best_epoch": best_epoch,
        "best_ood_exact_match": best_ood_em,
        "id_direction": "C# -> Java",
        "ood_direction": "Java -> C#",
        "id_test": id_eval,
        "ood_test": ood_eval,
        "drop_id_minus_ood": id_ood_drop(id_eval, ood_eval),
        "epoch_metrics": epoch_metrics,
        "files": {
            "metrics": output_paths["metrics"],
            "final_model": output_paths["best_model"],
            "id_predictions": output_paths["id_predictions"],
            "ood_predictions": output_paths["ood_predictions"],
            "id_test_src": ID_TEST_SRC_FILE,
            "id_test_tgt": ID_TEST_TGT_FILE,
            "ood_train_src": OOD_TRAIN_SRC_FILE,
            "ood_train_tgt": OOD_TRAIN_TGT_FILE,
            "ood_test_src": OOD_TEST_SRC_FILE,
            "ood_test_tgt": OOD_TEST_TGT_FILE,
        },
        "time": {
            "train_seconds": train_time,
            "train_readable": format_time(train_time),
            "total_seconds": time.time() - start_time,
            "total_readable": format_time(time.time() - start_time),
        },
    }
    with open(output_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print_final_table(f"RQ4 TL Final Results: CodeTrans {model_type}", id_eval, ood_eval)
    print("Metrics saved:", output_paths["metrics"])
    return metrics


def main():
    args = build_arg_parser().parse_args()
    MODEL_PATHS["codeT5"] = args.codeT5_path
    MODEL_PATHS["PLBART"] = args.PLBART_path
    selected_model_type = normalize_model_type(args.model_type)
    if selected_model_type == "all":
        experiments = [("codeT5", args.codeT5_path), ("PLBART", args.PLBART_path)]
    else:
        experiments = [(selected_model_type, args.model_path)]
    all_results = []
    for model_type, model_path in experiments:
        try:
            all_results.append(run_experiment(model_type, model_path))
        except Exception as exc:
            if selected_model_type != "all":
                raise
            all_results.append({"status": "failed", "model_type": model_type, "error": repr(exc)})
            print(f"[Error] RQ4 TL failed for {model_type}: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if selected_model_type == "all":
        summary_path = os.path.join(SAVE_DIR, "rq4_tl_codetrans_cs_to_java_id_all_models.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)
        print("All-model summary saved:", summary_path)


if __name__ == "__main__":
    main()
