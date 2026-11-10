# -*- coding: utf-8 -*-
"""
CodeXGLUE-CodeTrans RQ1: Language-direction Shift

ID direction:
    Java -> C#

OOD direction:
    C# -> Java

Data files:
    train.java-cs.txt.java  -> train.java-cs.txt.cs
    test.java-cs.txt.java   -> test.java-cs.txt.cs

OOD construction:
    Reverse the same test parallel pairs:
    test.java-cs.txt.cs     -> test.java-cs.txt.java

RQ1 logic:
    1. Fine-tune codeT5 or PLBART on Java -> C# training pairs.
    2. Select the best checkpoint by ID Test Exact Match.
    3. Final evaluation:
        - ID Test:  Java -> C#
        - OOD Test: C# -> Java

Fixes in this version:
    - Corrected CodeBLEU call: references=refs, not references=[refs].
    - Added visible CodeBLEU error logs instead of silently returning N/A.
    - Added multiple target-language aliases for C#.
    - Added a helper to recompute metrics from saved prediction JSONL files without retraining.
"""

import os
import json
import time
import random
import argparse

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


# ============================================================
# 1. Configuration
# ============================================================

MODEL_TYPE = "PLBART"          # choices: "codeT5", "PLBART", "all"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
MODEL_PATH = MODEL_PATHS["PLBART"]

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/data"

# ============================================================
# ID direction: Java -> C#
# ============================================================

ID_TRAIN_SRC_FILE = os.path.join(DATA_DIR, "train.java-cs.txt.java")
ID_TRAIN_TGT_FILE = os.path.join(DATA_DIR, "train.java-cs.txt.cs")

ID_TEST_SRC_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.java")
ID_TEST_TGT_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.cs")

# ============================================================
# OOD language-direction shift: C# -> Java
# ============================================================

OOD_TEST_SRC_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.cs")
OOD_TEST_TGT_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.java")

SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq1_language_shift_outputs_java_to_cs_id"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

BATCH_SIZE = 8
EPOCHS = 8
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0

NUM_BEAMS = 5
SEED = 42

# Set to smaller values for debugging; use None for formal experiments.
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

# CodeBLEU target languages.
# ID: Java -> C#, so target language is C#.
# OOD: C# -> Java, so target language is Java.
ID_CODEBLEU_LANG = "c_sharp"
OOD_CODEBLEU_LANG = "java"

# If True, try to calculate CodeBLEU. If the package/parser fails, the script keeps running.
ENABLE_CODEBLEU = True

# If True, do not train; only recompute metrics from existing prediction JSONL files.
# Useful when you already have ID_PRED_FILE and OOD_PRED_FILE but CodeBLEU was previously N/A.
RECOMPUTE_FROM_SAVED_PREDICTIONS_ONLY = False


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
    return f"{int(h)} 小时 {int(m)} 分钟 {s:.2f} 秒"


def normalize_model_type(model_type):
    model_type_lower = model_type.lower()
    if model_type_lower == "all":
        return "all"

    aliases = {
        "codet5": "codeT5",
        "code-t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
    }
    if model_type_lower in aliases:
        return aliases[model_type_lower]

    for name in MODEL_PATHS:
        if model_type_lower == name.lower():
            return name

    raise ValueError(f"Unsupported model_type: {model_type}")


def get_default_model_path(model_type):
    model_type = normalize_model_type(model_type)
    if model_type == "all":
        raise ValueError("model_type='all' does not have a single default path")
    return MODEL_PATHS[model_type]


def resolve_model_path(model_type, model_path):
    if os.path.exists(model_path):
        return model_path

    fallback_paths = {
        "codeT5": ["/opt/data/private/codet5", "/opt/data/private/CodeT5"],
        "PLBART": ["/opt/data/private/plbart", "/opt/data/private/PLBART"],
    }

    for fallback_path in fallback_paths.get(model_type, []):
        if os.path.exists(fallback_path):
            print(f"[Path Warning] {model_path} not found, using {fallback_path}")
            return fallback_path

    return model_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE CodeTrans RQ1")
    parser.add_argument(
        "--model_type",
        default=MODEL_TYPE,
        help="Run codeT5/codet5, PLBART/plbart, or both with all.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local model path for a single-model run.",
    )
    parser.add_argument("--codeT5_path", default=MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=MODEL_PATHS["PLBART"])
    return parser


def get_output_paths(model_type):
    run_save_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(run_save_dir, exist_ok=True)

    return {
        "save_dir": run_save_dir,
        "best_model": os.path.join(run_save_dir, f"best_codetrans_java_to_cs_id_{model_type}.bin"),
        "metrics": os.path.join(run_save_dir, f"rq1_codetrans_java_to_cs_id_metrics_{model_type}.json"),
        "id_predictions": os.path.join(run_save_dir, f"predictions_id_test_java_to_cs_{model_type}.jsonl"),
        "ood_predictions": os.path.join(run_save_dir, f"predictions_ood_test_cs_to_java_{model_type}.jsonl"),
    }


def normalize_code(text):
    return " ".join(str(text).strip().split())


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_parallel_files(src_path, tgt_path, max_samples=None):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"找不到源文件: {src_path}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"找不到目标文件: {tgt_path}")

    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        src_lines = [x.rstrip("\n") for x in f]

    with open(tgt_path, "r", encoding="utf-8", errors="ignore") as f:
        tgt_lines = [x.rstrip("\n") for x in f]

    n = min(len(src_lines), len(tgt_lines))
    if max_samples is not None:
        n = min(n, max_samples)

    data = []
    for i in range(n):
        src = src_lines[i].strip()
        tgt = tgt_lines[i].strip()
        if not src or not tgt:
            continue
        data.append({
            "idx": str(i),
            "source": src,
            "target": tgt,
        })

    return data


# ============================================================
# 3. Generation metrics
# ============================================================

def exact_match(preds, refs):
    if len(preds) == 0:
        return 0.0

    hit = 0
    for pred, ref in zip(preds, refs):
        if normalize_code(pred) == normalize_code(ref):
            hit += 1
    return 100.0 * hit / len(preds)


def compute_bleu(preds, refs):
    if len(preds) == 0:
        return 0.0

    smoothie = SmoothingFunction().method4
    references = [[normalize_code(ref).split()] for ref in refs]
    hypotheses = [normalize_code(pred).split() for pred in preds]

    try:
        bleu = corpus_bleu(
            references,
            hypotheses,
            smoothing_function=smoothie,
        )
        return float(bleu * 100.0)
    except Exception as exc:
        print(f"[BLEU Warning] failed: {repr(exc)}")
        return 0.0


def _normalise_codebleu_score(raw_score):
    """Convert CodeBLEU score to percentage scale."""
    if raw_score is None:
        return None
    score = float(raw_score)
    # Most versions return [0, 1]. If a version already returns [0, 100], keep it.
    return score * 100.0 if score <= 1.0 else score


def compute_codebleu_with_package(preds, refs, lang):
    """
    Compute CodeBLEU with the pip package `codebleu`.

    Important fix:
        Use references=refs, not references=[refs].

    For codebleu==0.4.0, refs and preds should usually both be List[str]
    with the same length. Some old wrappers use a nested reference format, but the
    pip package interface generally expects flat lists for single-reference data.
    """
    if not ENABLE_CODEBLEU:
        return None

    if len(preds) == 0 or len(refs) == 0:
        print(f"[CodeBLEU Warning] empty predictions/references for lang={lang}")
        return None

    if len(preds) != len(refs):
        print(
            f"[CodeBLEU Warning] length mismatch: "
            f"preds={len(preds)}, refs={len(refs)}, lang={lang}"
        )
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

        print(f"[CodeBLEU] lang={lang}, raw_result={result}")

        if isinstance(result, dict):
            score = result.get("codebleu", None)
            if score is None:
                score = result.get("CodeBLEU", None)
            if score is None:
                print(f"[CodeBLEU Warning] no codebleu key in result for lang={lang}: {result}")
                return None
            return _normalise_codebleu_score(score)

        if isinstance(result, (int, float)):
            return _normalise_codebleu_score(result)

        print(f"[CodeBLEU Warning] unsupported result type for lang={lang}: {type(result)}")
        return None

    except Exception as exc:
        print(
            f"[CodeBLEU Warning] failed for lang={lang}: "
            f"preds={len(preds)}, refs={len(refs)}, error={repr(exc)}"
        )
        return None


def compute_codebleu_safe(preds, refs, lang):
    """
    Try multiple language aliases for C#.

    Typical languages:
        Java target: "java"
        C# target:  "c_sharp"; if unavailable, try "csharp" and "cs".
    """
    if not ENABLE_CODEBLEU:
        return None

    if len(preds) == 0:
        return None

    lang = str(lang).strip()

    if lang.lower() in {"c_sharp", "csharp", "cs", "c#"}:
        candidates = [lang, "c_sharp", "csharp", "cs"]
    elif lang.lower() in {"java"}:
        candidates = ["java"]
    else:
        candidates = [lang]

    # Deduplicate while preserving order.
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    for candidate_lang in candidates:
        score = compute_codebleu_with_package(preds, refs, candidate_lang)
        if score is not None:
            return score

    print(f"[CodeBLEU Warning] all language candidates failed: {candidates}")
    return None


def compute_generation_metrics(preds, refs, lang):
    em = exact_match(preds, refs)
    bleu = compute_bleu(preds, refs)
    codebleu = compute_codebleu_safe(preds, refs, lang)

    return {
        "exact_match": float(em),
        "bleu": float(bleu),
        "codebleu": None if codebleu is None else float(codebleu),
        "num_samples": len(preds),
    }


def load_predictions_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到预测文件: {path}")

    preds, refs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            preds.append(item.get("prediction", ""))
            refs.append(item.get("reference", ""))

    return preds, refs


# ============================================================
# 4. Dataset
# ============================================================

class CodeTransDataset(Dataset):
    def __init__(self, data, tokenizer, source_prefix):
        self.examples = []
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix

        for item in data:
            src = item["source"]
            tgt = item["target"]
            src_text = f"{source_prefix} {src}"

            source_enc = tokenizer(
                src_text,
                max_length=MAX_SOURCE_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            target_enc = tokenizer(
                tgt,
                max_length=MAX_TARGET_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            labels = target_enc["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100

            self.examples.append({
                "idx": item["idx"],
                "source": src,
                "target": tgt,
                "input_ids": source_enc["input_ids"].squeeze(0),
                "attention_mask": source_enc["attention_mask"].squeeze(0),
                "labels": labels,
            })

        print(f">>> Loaded {len(self.examples)} examples, prefix={source_prefix}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# ============================================================
# 5. Training and evaluation
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    step_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        step_count += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(step_count, 1)


@torch.no_grad()
def generate_predictions(model, dataloader, tokenizer, desc="Generating", save_path=None):
    model.eval()

    all_preds = []
    all_refs = []
    records = []

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

        preds = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        refs = batch["target"]
        sources = batch["source"]
        idxs = batch["idx"]

        for i in range(len(preds)):
            pred = preds[i]
            ref = refs[i]

            all_preds.append(pred)
            all_refs.append(ref)

            records.append({
                "idx": idxs[i],
                "source": sources[i],
                "reference": ref,
                "prediction": pred,
                "exact_match": normalize_code(pred) == normalize_code(ref),
            })

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return all_preds, all_refs


def evaluate_generation(model, dataloader, tokenizer, lang, desc, save_path=None):
    preds, refs = generate_predictions(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        desc=desc,
        save_path=save_path,
    )

    metrics = compute_generation_metrics(
        preds=preds,
        refs=refs,
        lang=lang,
    )

    return metrics


def print_result_table(title, id_eval, ood_eval):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'指标':<20} | "
        f"{'ID Test':<12} | "
        f"{'OOD Test':<12} | "
        f"{'跌幅 ID-OOD':<12}"
    )
    print("-" * 100)

    for key, name in [
        ("exact_match", "Exact Match"),
        ("bleu", "BLEU"),
        ("codebleu", "CodeBLEU"),
    ]:
        id_value = id_eval.get(key)
        ood_value = ood_eval.get(key)

        if id_value is None or ood_value is None:
            print(
                f"{name:<20} | "
                f"{'N/A':<12} | "
                f"{'N/A':<12} | "
                f"{'N/A':<12}"
            )
        else:
            print(
                f"{name:<20} | "
                f"{id_value:<12.4f} | "
                f"{ood_value:<12.4f} | "
                f"{id_value - ood_value:<12.4f}"
            )

    print("-" * 100)
    print("ID samples :", id_eval["num_samples"])
    print("OOD samples:", ood_eval["num_samples"])
    print("=" * 100)


# ============================================================
# 6. Main process
# ============================================================

def recompute_from_saved_predictions(model_type=MODEL_TYPE):
    model_type = normalize_model_type(model_type)
    output_paths = get_output_paths(model_type)

    print("=" * 100)
    print("Recompute CodeTrans metrics from saved prediction JSONL files")
    print("=" * 100)
    print("MODEL_TYPE:", model_type)
    print("ID_PRED_FILE:", output_paths["id_predictions"])
    print("OOD_PRED_FILE:", output_paths["ood_predictions"])
    print("ID_CODEBLEU_LANG:", ID_CODEBLEU_LANG)
    print("OOD_CODEBLEU_LANG:", OOD_CODEBLEU_LANG)
    print("=" * 100)

    id_preds, id_refs = load_predictions_jsonl(output_paths["id_predictions"])
    ood_preds, ood_refs = load_predictions_jsonl(output_paths["ood_predictions"])

    id_eval = compute_generation_metrics(id_preds, id_refs, ID_CODEBLEU_LANG)
    ood_eval = compute_generation_metrics(ood_preds, ood_refs, OOD_CODEBLEU_LANG)

    print_result_table(
        title="RQ1 重新计算结果：CodeTrans Language-direction Shift",
        id_eval=id_eval,
        ood_eval=ood_eval,
    )

    recomputed_path = os.path.join(output_paths["save_dir"], f"rq1_codetrans_java_to_cs_id_metrics_recomputed_{model_type}.json")
    output = {
        "task": "Code-to-Code Translation",
        "dataset": "CodeXGLUE-CodeTrans",
        "model_type": model_type,
        "shift_type": "language_direction_shift",
        "id_direction": "Java -> C#",
        "ood_direction": "C# -> Java",
        "id_test": id_eval,
        "ood_language_test": ood_eval,
        "drop": {
            "exact_match_drop": id_eval["exact_match"] - ood_eval["exact_match"],
            "bleu_drop": id_eval["bleu"] - ood_eval["bleu"],
            "codebleu_drop": None if id_eval["codebleu"] is None or ood_eval["codebleu"] is None else id_eval["codebleu"] - ood_eval["codebleu"],
        },
        "files": {
            "id_predictions": output_paths["id_predictions"],
            "ood_predictions": output_paths["ood_predictions"],
            "recomputed_metrics": recomputed_path,
        },
    }

    with open(recomputed_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("重新计算的指标保存于:", recomputed_path)


def run_experiment(model_type, model_path):
    set_seed(SEED)
    model_type = normalize_model_type(model_type)
    model_path = resolve_model_path(model_type, model_path)
    output_paths = get_output_paths(model_type)

    program_start = time.time()

    print("=" * 120)
    print("CodeXGLUE-CodeTrans RQ1: Java->C# ID, C#->Java OOD Language-direction Shift")
    print("=" * 120)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("DATA_DIR:", DATA_DIR)
    print("ID Train:", ID_TRAIN_SRC_FILE, "->", ID_TRAIN_TGT_FILE)
    print("ID Test :", ID_TEST_SRC_FILE, "->", ID_TEST_TGT_FILE)
    print("OOD Test:", OOD_TEST_SRC_FILE, "->", OOD_TEST_TGT_FILE)
    print("SAVE_DIR:", output_paths["save_dir"])
    print("DEVICE:", DEVICE)
    print("MAX_TRAIN_SAMPLES:", MAX_TRAIN_SAMPLES)
    print("MAX_EVAL_SAMPLES:", MAX_EVAL_SAMPLES)
    print("ENABLE_CODEBLEU:", ENABLE_CODEBLEU)
    print("ID_CODEBLEU_LANG:", ID_CODEBLEU_LANG)
    print("OOD_CODEBLEU_LANG:", OOD_CODEBLEU_LANG)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)

    train_data = read_parallel_files(
        ID_TRAIN_SRC_FILE,
        ID_TRAIN_TGT_FILE,
        max_samples=MAX_TRAIN_SAMPLES,
    )
    id_test_data = read_parallel_files(
        ID_TEST_SRC_FILE,
        ID_TEST_TGT_FILE,
        max_samples=MAX_EVAL_SAMPLES,
    )
    ood_test_data = read_parallel_files(
        OOD_TEST_SRC_FILE,
        OOD_TEST_TGT_FILE,
        max_samples=MAX_EVAL_SAMPLES,
    )

    print("\n数据规模:")
    print("ID Train:", len(train_data))
    print("ID Test :", len(id_test_data))
    print("OOD Test:", len(ood_test_data))

    # CodeT5 often accepts a simple instruction prefix. Keep the same prefix for both directions
    # to ensure the OOD setting isolates language-direction shift.
    train_ds = CodeTransDataset(train_data, tokenizer, source_prefix="translate code:")
    id_test_ds = CodeTransDataset(id_test_data, tokenizer, source_prefix="translate code:")
    ood_test_ds = CodeTransDataset(ood_test_data, tokenizer, source_prefix="translate code:")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_test_loader = DataLoader(ood_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_id_em = -1.0
    best_epoch = 0

    print("\n>>> 开始训练 CodeTrans Java -> C# 模型...")
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

        id_eval_epoch = evaluate_generation(
            model=model,
            dataloader=id_test_loader,
            tokenizer=tokenizer,
            lang=ID_CODEBLEU_LANG,
            desc=f"ID Test Eval Epoch {epoch}",
            save_path=None,
        )

        epoch_time = time.time() - epoch_start

        codebleu_show = "N/A" if id_eval_epoch["codebleu"] is None else f"{id_eval_epoch['codebleu']:.4f}"
        print(
            f"Epoch {epoch} | "
            f"Loss: {train_loss:.4f} | "
            f"ID EM: {id_eval_epoch['exact_match']:.4f} | "
            f"ID BLEU: {id_eval_epoch['bleu']:.4f} | "
            f"ID CodeBLEU: {codebleu_show} | "
            f"Time: {format_time(epoch_time)}"
        )

        ckpt_path = os.path.join(output_paths["save_dir"], f"checkpoint_epoch_{epoch}_{model_type}.bin")
        torch.save(model.state_dict(), ckpt_path)

        # RQ1: select the best model by ID Test Exact Match.
        if id_eval_epoch["exact_match"] > best_id_em:
            best_id_em = id_eval_epoch["exact_match"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_paths["best_model"])
            print(f">>> 新最佳模型已保存: {output_paths['best_model']}")

    train_time = time.time() - train_start

    print("\n" + "=" * 100)
    print(f"训练完成，最佳 epoch={best_epoch}, best ID EM={best_id_em:.4f}")
    print("加载最佳模型进行最终评估...")
    print("=" * 100)

    model.load_state_dict(safe_torch_load(output_paths["best_model"], map_location=DEVICE))

    eval_start = time.time()

    id_eval = evaluate_generation(
        model=model,
        dataloader=id_test_loader,
        tokenizer=tokenizer,
        lang=ID_CODEBLEU_LANG,
        desc="Final ID Test: Java -> C#",
        save_path=output_paths["id_predictions"],
    )

    ood_eval = evaluate_generation(
        model=model,
        dataloader=ood_test_loader,
        tokenizer=tokenizer,
        lang=OOD_CODEBLEU_LANG,
        desc="Final OOD Test: C# -> Java",
        save_path=output_paths["ood_predictions"],
    )

    eval_time = time.time() - eval_start
    total_time = time.time() - program_start

    print_result_table(
        title="RQ1 最终结果：CodeTrans Language-direction Shift",
        id_eval=id_eval,
        ood_eval=ood_eval,
    )

    final_metrics = {
        "task": "Code-to-Code Translation",
        "dataset": "CodeXGLUE-CodeTrans",
        "model_type": model_type,
        "model_path": model_path,
        "shift_type": "language_direction_shift",
        "id_direction": "Java -> C#",
        "ood_direction": "C# -> Java",
        "codebleu": {
            "enabled": ENABLE_CODEBLEU,
            "id_target_lang": ID_CODEBLEU_LANG,
            "ood_target_lang": OOD_CODEBLEU_LANG,
        },
        "files": {
            "id_train_src": ID_TRAIN_SRC_FILE,
            "id_train_tgt": ID_TRAIN_TGT_FILE,
            "id_test_src": ID_TEST_SRC_FILE,
            "id_test_tgt": ID_TEST_TGT_FILE,
            "ood_test_src": OOD_TEST_SRC_FILE,
            "ood_test_tgt": OOD_TEST_TGT_FILE,
            "best_model": output_paths["best_model"],
            "id_predictions": output_paths["id_predictions"],
            "ood_predictions": output_paths["ood_predictions"],
            "metrics": output_paths["metrics"],
        },
        "best_epoch": best_epoch,
        "best_id_exact_match": best_id_em,
        "id_test": id_eval,
        "ood_language_test": ood_eval,
        "drop": {
            "exact_match_drop": id_eval["exact_match"] - ood_eval["exact_match"],
            "bleu_drop": id_eval["bleu"] - ood_eval["bleu"],
            "codebleu_drop": None if id_eval["codebleu"] is None or ood_eval["codebleu"] is None else id_eval["codebleu"] - ood_eval["codebleu"],
        },
        "time": {
            "train_seconds": train_time,
            "train_readable": format_time(train_time),
            "eval_seconds": eval_time,
            "eval_readable": format_time(eval_time),
            "total_seconds": total_time,
            "total_readable": format_time(total_time),
        },
    }

    with open(output_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("\n运行时间与文件保存信息")
    print("-" * 100)
    print("训练耗时:", format_time(train_time))
    print("评估耗时:", format_time(eval_time))
    print("总耗时:", format_time(total_time))
    print("最佳模型:", output_paths["best_model"])
    print("ID 预测:", output_paths["id_predictions"])
    print("OOD 预测:", output_paths["ood_predictions"])
    print("指标文件:", output_paths["metrics"])
    print("-" * 100)

    return final_metrics


def main():
    args = build_arg_parser().parse_args()
    selected_model_type = normalize_model_type(args.model_type)

    if RECOMPUTE_FROM_SAVED_PREDICTIONS_ONLY:
        if selected_model_type == "all":
            for model_type in ["codeT5", "PLBART"]:
                recompute_from_saved_predictions(model_type)
        else:
            recompute_from_saved_predictions(selected_model_type)
        return

    if selected_model_type == "all":
        experiments = [
            ("codeT5", args.codeT5_path),
            ("PLBART", args.PLBART_path),
        ]
    else:
        experiments = [
            (selected_model_type, args.model_path or get_default_model_path(selected_model_type)),
        ]

    all_results = []
    for model_type, model_path in experiments:
        all_results.append(run_experiment(model_type, model_path))

    if len(all_results) > 1:
        summary_path = os.path.join(SAVE_DIR, "rq1_codetrans_java_to_cs_id_metrics_all_models.json")
        summary = {
            "task": "Code-to-Code Translation",
            "dataset": "CodeXGLUE-CodeTrans",
            "models": [item["model_type"] for item in all_results],
            "shift_type": "language_direction_shift",
            "id_direction": "Java -> C#",
            "ood_direction": "C# -> Java",
            "results": all_results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("全部模型汇总文件:", summary_path)


if __name__ == "__main__":
    main()
