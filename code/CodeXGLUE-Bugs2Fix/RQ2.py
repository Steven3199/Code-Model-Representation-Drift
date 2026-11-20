# -*- coding: utf-8 -*-
"""
CodeXGLUE Bugs2Fix RQ2: generation-oriented OOD detection.

RQ1 evaluates code repair quality under token/CST shifts. RQ2 uses the best
RQ1 generation model and detects whether a test sample is OOD with three
generation-specific signals:

1. EncoderKNN:
   Mean distance from the source-code encoder representation to nearest ID
   training representations. Larger distance means farther from the ID input
   space.

2. SequenceNLL:
   Average token negative log-likelihood of the reference fix under teacher
   forcing. Larger NLL means lower generation confidence.

3. TokenEntropy:
   Average predictive entropy over target tokens under teacher forcing. Larger
   entropy means more uncertainty in the generation distribution.

All detector scores use the same direction: larger score means more OOD.
Evaluation metrics: AUROC, AUPR, TNR@95.
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ============================================================
# 1. Config
# ============================================================

MODEL_TYPE = "all"             # choices: "codeT5", "PLBART", "all"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}

DATA_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement"
RQ1_SAVE_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement/rq1_outputs"
SAVE_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement/rq2_outputs"

# Set to None to scan all subsets under DATA_ROOT. Or use e.g. ["small"].
SUBSET_DIRS = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42

MAX_REF_SAMPLES = 3000
MAX_EVAL_SAMPLES = None
KNN_K = 10

DETECTOR_NAMES = [
    "EncoderKNN",
    "SequenceNLL",
    "TokenEntropy",
]

os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def normalize_model_type(model_type):
    model_type_lower = str(model_type).lower()
    if model_type_lower == "all":
        return "all"
    for name in MODEL_PATHS:
        if model_type_lower == name.lower():
            return name
    raise ValueError(f"Unsupported model_type={model_type}. Choose from ['codeT5', 'PLBART', 'all']")


def get_default_model_path(model_type):
    model_type = normalize_model_type(model_type)
    if model_type == "all":
        raise ValueError("model_type='all' does not have a single default model path")
    return MODEL_PATHS[model_type]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE Bugs2Fix RQ2 OOD detection")
    parser.add_argument(
        "--model_type",
        default=MODEL_TYPE,
        choices=["codeT5", "PLBART", "all"],
        help="Model to evaluate. Use 'all' to run codeT5 and PLBART sequentially.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local model path for a single-model run. Defaults to MODEL_PATHS[model_type].",
    )
    parser.add_argument("--codeT5_path", default=MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=MODEL_PATHS["PLBART"])
    return parser


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
    subset_infos = []

    if SUBSET_DIRS is not None:
        candidates = [os.path.join(DATA_ROOT, name) for name in SUBSET_DIRS]
    else:
        candidates = []
        required = {
            "id_train.jsonl",
            "id_test.jsonl",
            "ood_test_token.jsonl",
            "ood_test_cst.jsonl",
        }
        for root, _, files in os.walk(DATA_ROOT):
            if required.issubset(set(files)):
                candidates.append(root)

    for path in sorted(set(candidates)):
        train_file = os.path.join(path, "id_train.jsonl")
        id_test_file = os.path.join(path, "id_test.jsonl")
        ood_token_file = os.path.join(path, "ood_test_token.jsonl")
        ood_cst_file = os.path.join(path, "ood_test_cst.jsonl")

        if not all(os.path.exists(x) for x in [train_file, id_test_file, ood_token_file, ood_cst_file]):
            continue

        subset_infos.append({
            "name": os.path.basename(path.rstrip("/")),
            "dir": path,
            "train_file": train_file,
            "id_test_file": id_test_file,
            "ood_token_file": ood_token_file,
            "ood_cst_file": ood_cst_file,
        })

    return subset_infos


def build_rq1_best_model_path(model_type, subset_name):
    return os.path.join(
        RQ1_SAVE_ROOT,
        model_type,
        subset_name,
        f"best_bugs2fix_{model_type}.bin",
    )


def build_generation_model(model_type, model_path):
    model_type = normalize_model_type(model_type)
    if model_type not in ["codeT5", "PLBART"]:
        raise ValueError(f"Unsupported model_type={model_type}")
    return AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)


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


def load_rq1_best_model(model_type, model_path, subset_name):
    best_model_path = build_rq1_best_model_path(model_type, subset_name)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"RQ1 best model not found: {best_model_path}. "
            "Run RQ1 first for this model/subset."
        )

    model = build_generation_model(model_type, model_path)
    state = safe_torch_load(best_model_path, map_location=DEVICE)
    state = unwrap_checkpoint(state)
    model.load_state_dict(state)
    model.eval()

    print(f">>> Loaded RQ1 best model: {best_model_path}")
    return model, best_model_path


def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ============================================================
# 3. Dataset
# ============================================================

class Bugs2FixDataset(Dataset):
    def __init__(self, path, tokenizer, name="dataset", max_samples=None, shuffle_before_limit=False):
        self.path = path
        self.tokenizer = tokenizer
        self.examples = []

        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find file: {path}")

        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Loading {name}", leave=False)):
                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                source = get_source(js)
                target = get_target(js)
                if not source or not target:
                    skipped += 1
                    continue

                self.examples.append({
                    "idx": str(js.get("idx", js.get("id", line_idx))),
                    "source": source,
                    "target": target,
                    "shift_type": js.get("shift_type", "id"),
                    "shift_success": js.get("shift_success", None),
                    "fallback_reason": js.get("fallback_reason", ""),
                })

        if shuffle_before_limit:
            random.shuffle(self.examples)
        if max_samples is not None and len(self.examples) > max_samples:
            self.examples = self.examples[:max_samples]

        shift_counter = Counter(x["shift_type"] for x in self.examples)
        print(f">>> Loaded {len(self.examples)} samples from {path}")
        print(f">>> Skipped samples: {skipped}")
        print(f">>> shift_type distribution: {dict(shift_counter)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]

        source_inputs = self.tokenizer(
            item["source"],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_inputs = self.tokenizer(
            item["target"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
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
    shift_types = []

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
        shift_types.extend(batch["shift_type"])

    if not features:
        raise ValueError(f"{desc}: empty dataloader")

    return {
        "features": torch.cat(features, dim=0),
        "idx": indices,
        "shift_type": shift_types,
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
    shift_types = []

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
        shift_types.extend(batch["shift_type"])

    return {
        "SequenceNLL": np.asarray(nll_scores, dtype=np.float64),
        "TokenEntropy": np.asarray(entropy_scores, dtype=np.float64),
        "idx": indices,
        "shift_type": shift_types,
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
        "shift_type": eval_features["shift_type"],
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


def fmt_metric(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{float(x):.{digits}f}"


def build_metric_rows(subset_name, model_type, shift_name, shift_results):
    rows = []
    for detector_name in DETECTOR_NAMES:
        metrics = shift_results[detector_name]
        rows.append({
            "Subset": subset_name,
            "Model": model_type,
            "Shift": shift_name,
            "Detector": detector_name,
            "AUROC": metrics["AUROC"],
            "AUPR": metrics["AUPR"],
            "TNR@95": metrics["TNR@95"],
        })
    return rows


def write_metrics_tsv(path, rows):
    header = ["Subset", "Model", "Shift", "Detector", "AUROC", "AUPR", "TNR@95"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join([
                str(row["Subset"]),
                str(row["Model"]),
                str(row["Shift"]),
                str(row["Detector"]),
                fmt_metric(row["AUROC"], 4),
                fmt_metric(row["AUPR"], 4),
                fmt_metric(row["TNR@95"], 4),
            ]) + "\n")


def print_excel_copy_metrics(title, rows):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header = ["Subset", "Model", "Shift", "Detector", "AUROC", "AUPR", "TNR@95"]
    print("\t".join(header))
    for row in rows:
        print("\t".join([
            str(row["Subset"]),
            str(row["Model"]),
            str(row["Shift"]),
            str(row["Detector"]),
            fmt_metric(row["AUROC"], 4),
            fmt_metric(row["AUPR"], 4),
            fmt_metric(row["TNR@95"], 4),
        ]))
    print("=" * 100)


def flatten_final_metrics(final_metrics):
    rows = []
    subset_name = final_metrics["subset"]
    model_type = final_metrics["model_type"]
    for shift_name, shift_results in final_metrics["results"].items():
        rows.extend(build_metric_rows(subset_name, model_type, shift_name, shift_results))
    return rows


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
                "shift_type": str(id_result["shift_type"][i]),
                "scores": {name: float(id_scores[name][i]) for name in DETECTOR_NAMES},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for i in range(n_ood):
            rec = {
                "domain": "ood",
                "label": 1,
                "idx": str(ood_result["idx"][i]),
                "shift_type": str(ood_result["shift_type"][i]),
                "scores": {name: float(ood_scores[name][i]) for name in DETECTOR_NAMES},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 6. Run one subset/model
# ============================================================

def run_one_subset(info, model_type, model_path):
    set_seed(SEED)
    subset_name = info["name"]
    model_type = normalize_model_type(model_type)

    save_dir = os.path.join(SAVE_ROOT, model_type, subset_name)
    os.makedirs(save_dir, exist_ok=True)

    metrics_file = os.path.join(save_dir, f"rq2_bugs2fix_ood_metrics_{model_type}.json")
    metrics_tsv_file = os.path.join(save_dir, f"rq2_bugs2fix_ood_metrics_{model_type}_excel.tsv")

    print("\n" + "#" * 120)
    print(f"Running Bugs2Fix RQ2 OOD Detection: subset={subset_name}, model={model_type}")
    print("#" * 120)
    print("train:", info["train_file"])
    print("id test:", info["id_test_file"])
    print("ood token:", info["ood_token_file"])
    print("ood cst:", info["ood_cst_file"])
    print("model_path:", model_path)
    print("save_dir:", save_dir)
    print("#" * 120)

    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model, best_model_path = load_rq1_best_model(model_type, model_path, subset_name)
    configure_special_tokens(tokenizer, model)

    train_ds_full = Bugs2FixDataset(
        info["train_file"],
        tokenizer,
        name="id_train_ref",
        max_samples=MAX_REF_SAMPLES,
        shuffle_before_limit=True,
    )
    id_test_ds = Bugs2FixDataset(
        info["id_test_file"],
        tokenizer,
        name="id_test",
        max_samples=MAX_EVAL_SAMPLES,
    )
    ood_token_ds = Bugs2FixDataset(
        info["ood_token_file"],
        tokenizer,
        name="ood_token",
        max_samples=MAX_EVAL_SAMPLES,
    )
    ood_cst_ds = Bugs2FixDataset(
        info["ood_cst_file"],
        tokenizer,
        name="ood_cst",
        max_samples=MAX_EVAL_SAMPLES,
    )

    if len(train_ds_full) == 0:
        raise ValueError("ID reference train set is empty.")
    if len(id_test_ds) == 0:
        raise ValueError("ID test set is empty.")

    ref_loader = make_loader(train_ds_full, shuffle=False)
    id_loader = make_loader(id_test_ds, shuffle=False)
    token_loader = make_loader(ood_token_ds, shuffle=False)
    cst_loader = make_loader(ood_cst_ds, shuffle=False)

    print("\n>>> Extracting ID reference encoder space...")
    ref = extract_encoder_features(model, ref_loader, desc="Extract ID train reference")
    ref_features = ref["features"]

    print("\n>>> Computing ID detector scores...")
    id_result = compute_detector_scores(model, ref_features, id_loader, "id_test")

    all_results = {}
    all_score_files = {}
    excel_rows = []

    for shift_name, loader in [
        ("token", token_loader),
        ("cst", cst_loader),
    ]:
        print("\n" + "=" * 100)
        print(f"Evaluating OOD Detection for {shift_name.upper()} shift")
        print("=" * 100)

        ood_result = compute_detector_scores(model, ref_features, loader, f"ood_{shift_name}")

        shift_results = {}
        for detector_name in DETECTOR_NAMES:
            shift_results[detector_name] = compute_ood_metrics(
                id_result["scores"][detector_name],
                ood_result["scores"][detector_name],
            )

        all_results[shift_name] = shift_results
        excel_rows.extend(build_metric_rows(subset_name, model_type, shift_name, shift_results))
        print_result_table(
            title=f"RQ2 Results: Bugs2Fix {subset_name} {shift_name.upper()} Shift - {model_type}",
            results=shift_results,
        )

        score_file = os.path.join(save_dir, f"rq2_scores_{model_type}_{subset_name}_{shift_name}.jsonl")
        write_score_file(score_file, id_result, ood_result)
        all_score_files[shift_name] = score_file
        print("Score file saved:", score_file)

    total_time = time.time() - start_time
    final_metrics = {
        "task": "Code Repair / Code Refinement",
        "dataset": "CodeXGLUE Bugs2Fix",
        "subset": subset_name,
        "model_type": model_type,
        "model_path": model_path,
        "rq1_best_model": best_model_path,
        "method": "generation-oriented OOD detection with three detectors",
        "detectors": DETECTOR_NAMES,
        "detector_notes": {
            "EncoderKNN": "mean distance from source encoder embedding to nearest ID train embeddings",
            "SequenceNLL": "average target-token negative log-likelihood under teacher forcing",
            "TokenEntropy": "average target-token predictive entropy under teacher forcing",
        },
        "score_direction": "larger score means more OOD",
        "metrics": ["AUROC", "AUPR", "TNR@95"],
        "max_ref_samples": MAX_REF_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "knn_k": KNN_K,
        "results": all_results,
        "files": {
            "train_file": info["train_file"],
            "id_test_file": info["id_test_file"],
            "ood_token_file": info["ood_token_file"],
            "ood_cst_file": info["ood_cst_file"],
            "score_files": all_score_files,
            "metrics_file": metrics_file,
            "metrics_tsv_file": metrics_tsv_file,
        },
        "time": {
            "total_seconds": total_time,
            "total_readable": format_time(total_time),
        },
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    write_metrics_tsv(metrics_tsv_file, excel_rows)
    print_excel_copy_metrics(
        title=f"EXCEL_COPY_METRICS\tCodeXGLUE Bugs2Fix\t{subset_name}\t{model_type}",
        rows=excel_rows,
    )

    print("\nRQ2 metrics saved:", metrics_file)
    print("RQ2 metrics TSV saved:", metrics_tsv_file)
    print("Total time:", format_time(total_time))
    return final_metrics


# ============================================================
# 7. Main
# ============================================================

def main():
    program_start = time.time()
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

    print("=" * 120)
    print("CodeXGLUE Bugs2Fix RQ2: Generation OOD Detection under Token/CST Shift")
    print("=" * 120)
    print("MODEL_TYPE:", selected_model_type)
    print("EXPERIMENTS:", experiments)
    print("DATA_ROOT:", DATA_ROOT)
    print("RQ1_SAVE_ROOT:", RQ1_SAVE_ROOT)
    print("SAVE_ROOT:", SAVE_ROOT)
    print("DEVICE:", DEVICE)
    print("SUBSET_DIRS:", SUBSET_DIRS)
    print("DETECTORS:", DETECTOR_NAMES)
    print("METRICS: AUROC, AUPR, TNR@95")
    print("=" * 120)

    subset_infos = discover_subset_dirs()
    if not subset_infos:
        raise ValueError("No Bugs2Fix subset directories found with required JSONL files.")

    print("\nSubsets to run:")
    for x in subset_infos:
        print(" -", x["name"], "=>", x["dir"])

    all_model_summaries = []
    all_excel_rows = []
    for model_type, model_path in experiments:
        model_results = []
        for info in subset_infos:
            result = run_one_subset(info, model_type, model_path)
            model_results.append(result)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        summary_path = os.path.join(SAVE_ROOT, model_type, f"rq2_all_bugs2fix_ood_summary_{model_type}.json")
        summary_tsv_path = os.path.join(SAVE_ROOT, model_type, f"rq2_all_bugs2fix_ood_summary_{model_type}_excel.tsv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_rows = []
        for result in model_results:
            summary_rows.extend(flatten_final_metrics(result))
        all_excel_rows.extend(summary_rows)
        summary = {
            "task": "Code Repair / Code Refinement",
            "dataset": "CodeXGLUE Bugs2Fix",
            "model_type": model_type,
            "model_path": model_path,
            "detectors": DETECTOR_NAMES,
            "metrics": ["AUROC", "AUPR", "TNR@95"],
            "num_subsets": len(model_results),
            "results": model_results,
            "program_elapsed_seconds": time.time() - program_start,
            "program_elapsed_readable": format_time(time.time() - program_start),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        write_metrics_tsv(summary_tsv_path, summary_rows)
        print_excel_copy_metrics(
            title=f"EXCEL_COPY_MODEL_SUMMARY\tCodeXGLUE Bugs2Fix\t{model_type}",
            rows=summary_rows,
        )

        all_model_summaries.append(summary_path)
        print("Model summary saved:", summary_path)
        print("Model summary TSV saved:", summary_tsv_path)

    all_models_tsv_path = os.path.join(SAVE_ROOT, "rq2_all_models_bugs2fix_ood_summary_excel.tsv")
    write_metrics_tsv(all_models_tsv_path, all_excel_rows)
    print_excel_copy_metrics(
        title="EXCEL_COPY_ALL_MODELS\tCodeXGLUE Bugs2Fix\tRQ2",
        rows=all_excel_rows,
    )

    print("\n" + "=" * 120)
    print("All RQ2 experiments finished")
    print("=" * 120)
    print("Total models:", len(experiments))
    print("Total subsets:", len(subset_infos))
    print("Total time:", format_time(time.time() - program_start))
    print("Summary files:")
    for path in all_model_summaries:
        print(" -", path)
    print("All-model TSV:", all_models_tsv_path)
    print("=" * 120)


if __name__ == "__main__":
    main()
