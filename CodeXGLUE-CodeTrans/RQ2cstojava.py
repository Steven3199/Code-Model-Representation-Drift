# -*- coding: utf-8 -*-
"""
CodeXGLUE-CodeTrans RQ2: generation-oriented OOD detection.

RQ1 trains a code translation model on the ID language direction:
    C# -> Java

and evaluates OOD language-direction shift:
    Java -> C#

This RQ2 script loads the best RQ1 checkpoint and detects OOD samples with
three generation-specific detectors:

1. EncoderKNN:
   Distance from the source-code encoder representation to the ID training
   source representation space.

2. SequenceNLL:
   Average target-token negative log-likelihood under teacher forcing.

3. TokenEntropy:
   Average target-token predictive entropy under teacher forcing.

All scores follow the same direction: larger score means more OOD.
Metrics: AUROC, AUPR, TNR@95.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ============================================================
# 1. Configuration
# ============================================================

MODEL_TYPE = "all"          # choices: "codeT5", "PLBART", "all"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/data"

# RQ1 direction: C# -> Java as ID, Java -> C# as OOD.
ID_TRAIN_SRC_FILE = os.path.join(DATA_DIR, "train.java-cs.txt.cs")
ID_TRAIN_TGT_FILE = os.path.join(DATA_DIR, "train.java-cs.txt.java")

ID_TEST_SRC_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.cs")
ID_TEST_TGT_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.java")

OOD_TEST_SRC_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.java")
OOD_TEST_TGT_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.cs")

RQ1_SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq1_language_shift_outputs_cs_to_java_id"
SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq2_language_shift_outputs_cs_to_java_id"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42

MAX_REF_SAMPLES = 3000
MAX_EVAL_SAMPLES = None
KNN_K = 10

SOURCE_PREFIX = "translate code:"

DETECTOR_NAMES = [
    "EncoderKNN",
    "SequenceNLL",
    "TokenEntropy",
]


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    model_type_lower = str(model_type).lower()
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

    raise ValueError(f"Unsupported model_type={model_type}. Choose from codeT5, PLBART, all.")


def get_default_model_path(model_type):
    model_type = normalize_model_type(model_type)
    if model_type == "all":
        raise ValueError("model_type='all' does not have a single default model path")
    return MODEL_PATHS[model_type]


def resolve_model_path(model_type, model_path):
    if model_path and os.path.exists(model_path):
        return model_path

    fallback_paths = {
        "codeT5": ["/opt/data/private/codet5", "/opt/data/private/CodeT5"],
        "PLBART": ["/opt/data/private/plbart", "/opt/data/private/PLBART"],
    }
    default_path = get_default_model_path(model_type)

    for fallback_path in [default_path] + fallback_paths.get(model_type, []):
        if os.path.exists(fallback_path):
            if model_path and fallback_path != model_path:
                print(f"[Path Warning] {model_path} not found, using {fallback_path}")
            return fallback_path

    return model_path or default_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE CodeTrans RQ2 OOD detection")
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


def read_parallel_files(src_path, tgt_path, max_samples=None):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Cannot find source file: {src_path}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Cannot find target file: {tgt_path}")

    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        src_lines = [line.rstrip("\n") for line in f]
    with open(tgt_path, "r", encoding="utf-8", errors="ignore") as f:
        tgt_lines = [line.rstrip("\n") for line in f]

    n = min(len(src_lines), len(tgt_lines))
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

    if max_samples is not None:
        data = data[:max_samples]

    print(f">>> Loaded {len(data)} pairs from {src_path} -> {tgt_path}")
    return data


def get_output_paths(model_type):
    run_save_dir = os.path.join(RQ1_SAVE_DIR, model_type)
    return {
        "save_dir": run_save_dir,
        "best_model": os.path.join(run_save_dir, f"best_codetrans_cs_to_java_id_{model_type}.bin"),
    }


def get_rq2_output_paths(model_type):
    run_save_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(run_save_dir, exist_ok=True)
    return {
        "save_dir": run_save_dir,
        "metrics": os.path.join(run_save_dir, f"rq2_codetrans_cs_to_java_id_ood_metrics_{model_type}.json"),
        "score_file": os.path.join(run_save_dir, f"rq2_codetrans_cs_to_java_id_scores_{model_type}.jsonl"),
    }


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


def load_rq1_best_model(model_type, model_path):
    output_paths = get_output_paths(model_type)
    best_model_path = output_paths["best_model"]
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"RQ1 best model not found: {best_model_path}. "
            "Please run RQ1 first for this model."
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    state = safe_torch_load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f">>> Loaded RQ1 best model: {best_model_path}")
    return model, best_model_path


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


set_seed(SEED)


# ============================================================
# 3. Dataset
# ============================================================

class CodeTransOODDataset(Dataset):
    def __init__(self, data, tokenizer, source_prefix=SOURCE_PREFIX, domain="id"):
        self.examples = []
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.domain = domain

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
                "domain": domain,
            })

        print(f">>> Loaded {len(self.examples)} {domain} examples, prefix={source_prefix}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


# ============================================================
# 4. Detector scores
# ============================================================

def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


@torch.no_grad()
def extract_encoder_features(model, dataloader, desc="Extracting encoder features"):
    model.eval()
    encoder = model.get_encoder()
    features = []
    indices = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = F.normalize(pooled, p=2, dim=1)

        features.append(pooled.detach().cpu())
        indices.extend(batch["idx"])

    if not features:
        raise ValueError(f"{desc}: empty dataloader")

    return {
        "features": torch.cat(features, dim=0),
        "idx": indices,
    }


def compute_knn_scores(ref_features, eval_features, k=10):
    ref = F.normalize(ref_features, p=2, dim=1)
    eva = F.normalize(eval_features, p=2, dim=1)
    distances = torch.cdist(eva, ref, p=2)
    k = min(k, distances.size(1))
    topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
    return to_numpy(topk_dist.mean(dim=1))


@torch.no_grad()
def compute_generation_uncertainty_scores(model, dataloader, desc="Computing generation scores"):
    model.eval()
    nll_scores = []
    entropy_scores = []
    indices = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        logits = outputs.logits
        vocab_size = logits.size(-1)
        valid_mask = labels.ne(-100)
        denom = valid_mask.sum(dim=1).clamp(min=1)

        token_nll = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size())
        seq_nll = (token_nll * valid_mask).sum(dim=1) / denom

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        seq_entropy = (token_entropy * valid_mask).sum(dim=1) / denom

        nll_scores.extend(to_numpy(seq_nll).tolist())
        entropy_scores.extend(to_numpy(seq_entropy).tolist())
        indices.extend(batch["idx"])

    return {
        "SequenceNLL": np.asarray(nll_scores, dtype=np.float64),
        "TokenEntropy": np.asarray(entropy_scores, dtype=np.float64),
        "idx": indices,
    }


def compute_detector_scores(model, ref_features, eval_loader, eval_name):
    eval_features = extract_encoder_features(
        model,
        eval_loader,
        desc=f"Extract encoder features [{eval_name}]",
    )
    generation_scores = compute_generation_uncertainty_scores(
        model,
        eval_loader,
        desc=f"Compute generation scores [{eval_name}]",
    )

    scores = {
        "EncoderKNN": compute_knn_scores(ref_features, eval_features["features"], k=KNN_K),
        "SequenceNLL": generation_scores["SequenceNLL"],
        "TokenEntropy": generation_scores["TokenEntropy"],
    }

    return {
        "scores": {name: np.asarray(scores[name], dtype=np.float64) for name in DETECTOR_NAMES},
        "idx": eval_features["idx"],
    }


# ============================================================
# 5. Metrics / output
# ============================================================

def compute_ood_metrics(id_scores, ood_scores):
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)

    y_true = np.concatenate([
        np.zeros_like(id_scores, dtype=np.int64),
        np.ones_like(ood_scores, dtype=np.int64),
    ])
    y_score = np.concatenate([id_scores, ood_scores])

    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = 0.0

    try:
        aupr = average_precision_score(y_true, y_score)
    except Exception:
        aupr = 0.0

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        idx = np.where(tpr >= 0.95)[0]
        fpr95 = 1.0 if len(idx) == 0 else float(fpr[idx[0]])
        tnr95 = 1.0 - fpr95
    except Exception:
        tnr95 = 0.0

    return {
        "AUROC": float(auroc),
        "AUPR": float(aupr),
        "TNR@95": float(tnr95),
    }


def print_result_table(title, results):
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    print(f"{'OOD Detector':<18} | {'AUROC':<10} | {'AUPR':<10} | {'TNR@95':<10}")
    print("-" * 86)
    for name in DETECTOR_NAMES:
        m = results[name]
        print(f"{name:<18} | {m['AUROC']:<10.4f} | {m['AUPR']:<10.4f} | {m['TNR@95']:<10.4f}")
    print("=" * 86)


def write_score_file(path, id_result, ood_result):
    id_scores = id_result["scores"]
    ood_scores = ood_result["scores"]

    with open(path, "w", encoding="utf-8") as f:
        n_id = len(next(iter(id_scores.values())))
        n_ood = len(next(iter(ood_scores.values())))

        for i in range(n_id):
            rec = {
                "domain": "id",
                "label": 0,
                "idx": str(id_result["idx"][i]),
                "scores": {name: float(id_scores[name][i]) for name in DETECTOR_NAMES},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for i in range(n_ood):
            rec = {
                "domain": "ood",
                "label": 1,
                "idx": str(ood_result["idx"][i]),
                "scores": {name: float(ood_scores[name][i]) for name in DETECTOR_NAMES},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 6. Main experiment
# ============================================================

def run_experiment(model_type, model_path):
    set_seed(SEED)
    model_type = normalize_model_type(model_type)
    model_path = resolve_model_path(model_type, model_path)
    rq1_paths = get_output_paths(model_type)
    rq2_paths = get_rq2_output_paths(model_type)
    start_time = time.time()

    print("=" * 120)
    print("CodeXGLUE-CodeTrans RQ2: Generation OOD Detection for Language-direction Shift")
    print("=" * 120)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("RQ1_BEST_MODEL:", rq1_paths["best_model"])
    print("ID Train:", ID_TRAIN_SRC_FILE, "->", ID_TRAIN_TGT_FILE)
    print("ID Test :", ID_TEST_SRC_FILE, "->", ID_TEST_TGT_FILE)
    print("OOD Test:", OOD_TEST_SRC_FILE, "->", OOD_TEST_TGT_FILE)
    print("SAVE_DIR:", rq2_paths["save_dir"])
    print("DEVICE:", DEVICE)
    print("DETECTORS:", DETECTOR_NAMES)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model, best_model_path = load_rq1_best_model(model_type, model_path)
    configure_special_tokens(tokenizer, model)

    train_data = read_parallel_files(
        ID_TRAIN_SRC_FILE,
        ID_TRAIN_TGT_FILE,
        max_samples=MAX_REF_SAMPLES,
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

    train_ref_ds = CodeTransOODDataset(train_data, tokenizer, source_prefix=SOURCE_PREFIX, domain="id_train_ref")
    id_test_ds = CodeTransOODDataset(id_test_data, tokenizer, source_prefix=SOURCE_PREFIX, domain="id_test")
    ood_test_ds = CodeTransOODDataset(ood_test_data, tokenizer, source_prefix=SOURCE_PREFIX, domain="ood_test")

    if len(train_ref_ds) == 0:
        raise ValueError("ID train reference set is empty.")
    if len(id_test_ds) == 0:
        raise ValueError("ID test set is empty.")
    if len(ood_test_ds) == 0:
        raise ValueError("OOD test set is empty.")

    ref_loader = make_loader(train_ref_ds, shuffle=False)
    id_loader = make_loader(id_test_ds, shuffle=False)
    ood_loader = make_loader(ood_test_ds, shuffle=False)

    print("\n>>> Extracting ID reference encoder space...")
    ref = extract_encoder_features(model, ref_loader, desc="Extract ID train reference")
    ref_features = ref["features"]

    print("\n>>> Computing ID detector scores...")
    id_result = compute_detector_scores(model, ref_features, id_loader, "id_test")

    print("\n>>> Computing OOD detector scores...")
    ood_result = compute_detector_scores(model, ref_features, ood_loader, "ood_test")

    results = {}
    for detector_name in DETECTOR_NAMES:
        results[detector_name] = compute_ood_metrics(
            id_result["scores"][detector_name],
            ood_result["scores"][detector_name],
        )

    print_result_table(
        title=f"RQ2 Results: CodeTrans C#->Java ID vs Java->C# OOD - {model_type}",
        results=results,
    )

    write_score_file(rq2_paths["score_file"], id_result, ood_result)

    total_time = time.time() - start_time
    final_metrics = {
        "task": "Code-to-Code Translation",
        "dataset": "CodeXGLUE-CodeTrans",
        "model_type": model_type,
        "model_path": model_path,
        "rq1_best_model": best_model_path,
        "shift_type": "language_direction_shift",
        "id_direction": "C# -> Java",
        "ood_direction": "Java -> C#",
        "method": "generation-oriented OOD detection with three detectors",
        "detectors": DETECTOR_NAMES,
        "detector_notes": {
            "EncoderKNN": "mean distance from source encoder embedding to nearest ID train source embeddings",
            "SequenceNLL": "average target-token negative log-likelihood under teacher forcing",
            "TokenEntropy": "average target-token predictive entropy under teacher forcing",
        },
        "score_direction": "larger score means more OOD",
        "metrics": ["AUROC", "AUPR", "TNR@95"],
        "max_ref_samples": MAX_REF_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "knn_k": KNN_K,
        "results": results,
        "files": {
            "id_train_src": ID_TRAIN_SRC_FILE,
            "id_train_tgt": ID_TRAIN_TGT_FILE,
            "id_test_src": ID_TEST_SRC_FILE,
            "id_test_tgt": ID_TEST_TGT_FILE,
            "ood_test_src": OOD_TEST_SRC_FILE,
            "ood_test_tgt": OOD_TEST_TGT_FILE,
            "score_file": rq2_paths["score_file"],
            "metrics_file": rq2_paths["metrics"],
        },
        "time": {
            "total_seconds": total_time,
            "total_readable": format_time(total_time),
        },
    }

    with open(rq2_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("\nRQ2 score file:", rq2_paths["score_file"])
    print("RQ2 metrics file:", rq2_paths["metrics"])
    print("Total time:", format_time(total_time))
    return final_metrics


def main():
    args = build_arg_parser().parse_args()
    MODEL_PATHS["codeT5"] = args.codeT5_path
    MODEL_PATHS["PLBART"] = args.PLBART_path

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

    all_results = []
    for model_type, model_path in experiments:
        all_results.append(run_experiment(model_type, model_path))
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        summary_path = os.path.join(SAVE_DIR, "rq2_codetrans_cs_to_java_id_ood_metrics_all_models.json")
        summary = {
            "task": "Code-to-Code Translation",
            "dataset": "CodeXGLUE-CodeTrans",
            "models": [item["model_type"] for item in all_results],
            "shift_type": "language_direction_shift",
            "id_direction": "C# -> Java",
            "ood_direction": "Java -> C#",
            "detectors": DETECTOR_NAMES,
            "metrics": ["AUROC", "AUPR", "TNR@95"],
            "results": all_results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("All-model RQ2 summary saved:", summary_path)


if __name__ == "__main__":
    main()
