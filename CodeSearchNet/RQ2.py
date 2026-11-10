import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, T5EncoderModel


"""
CodeSearchNet RQ2: retrieval-oriented OOD detection.

Semantic code search is a retrieval task, not a fixed-label classification task.
The model produces relevance scores between natural-language queries and code
snippets instead of class probabilities. Therefore this script evaluates three
retrieval-specific OOD detectors:

1. KNN:                distance from the query-code pair representation to ID pairs.
2. MatchingConfidence: negative matched-pair relevance score.
3. RankingEntropy:     entropy of the query ranking distribution over a fixed ID code bank.

All detector scores follow the same direction: larger score means more OOD.
"""


# ============================================================
# 1. Config
# ============================================================

#LANGUAGE = "python"
LANGUAGE = "python"

SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

# True runs all five models sequentially. Set False and edit MODEL_TYPE for one model.
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_TYPE = "RoBERTa"

DATA_ROOT = "/opt/data/private/CodeSearchNet"
DATA_DIR = os.path.join(DATA_ROOT, LANGUAGE)

# RQ1 in this folder may use id_train_full.jsonl. This script will use the first
# existing file from this list as the ID reference set.
ID_TRAIN_CANDIDATES = [
    os.path.join(DATA_DIR, "id_train.jsonl"),
    os.path.join(DATA_DIR, "id_train_full.jsonl"),
]
ID_TEST_FILE = os.path.join(DATA_DIR, "id_test.jsonl")

OOD_TEST_FILES = {
    "token": os.path.join(DATA_DIR, "ood_test_token.jsonl"),
    "cst": os.path.join(DATA_DIR, "ood_test_cst.jsonl"),
}

SAVE_DIR = os.path.join(DATA_DIR, "rq2_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

MAX_EVAL_SAMPLES = 1000
MAX_REF_SAMPLES = 3000

KNN_K = 10
LOGIT_SCALE = 20.0
RANKING_TOPK = 100

DETECTOR_NAMES = [
    "KNN",
    "MatchingConfidence",
    "RankingEntropy",
]


def canonical_model_type(model_type):
    for name in SUPPORTED_MODEL_TYPES:
        if str(model_type).lower() == name.lower():
            return name
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {SUPPORTED_MODEL_TYPES}")


def resolve_model_path(model_type):
    return MODEL_PATHS[canonical_model_type(model_type)]


def resolve_id_train_file():
    for path in ID_TRAIN_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No ID train file found from candidates: {ID_TRAIN_CANDIDATES}")


def build_rq1_best_model_path(model_type):
    model_type = canonical_model_type(model_type)
    return os.path.join(DATA_DIR, "rq1_outputs", f"best_codesearch_{LANGUAGE}_{model_type}.bin")


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, RQ1_BEST_MODEL, METRICS_FILE

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = resolve_model_path(MODEL_TYPE)
    RQ1_BEST_MODEL = build_rq1_best_model_path(MODEL_TYPE)
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq2_codesearch_metrics_{LANGUAGE}_{MODEL_TYPE}.json")


configure_model_runtime(MODEL_TYPE)
ALL_MODELS_METRICS_FILE = os.path.join(SAVE_DIR, f"rq2_codesearch_metrics_{LANGUAGE}_all_models.json")


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


def unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    return checkpoint


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def zscore_from_ref(values, ref_values):
    values = np.asarray(values, dtype=np.float64)
    ref_values = np.asarray(ref_values, dtype=np.float64)
    return (values - np.mean(ref_values)) / (np.std(ref_values) + 1e-12)


set_seed(SEED)


# ============================================================
# 3. Dataset
# ============================================================

class CodeSearchDataset(Dataset):
    def __init__(self, file_path, tokenizer, name="dataset"):
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find file: {file_path}")

        skipped = 0
        shift_counter = Counter()

        with open(file_path, "r", encoding="utf-8") as f:
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

                nl_txt = self.get_nl_text(js)
                code_txt = self.get_code_text(js)

                if not nl_txt or not code_txt:
                    skipped += 1
                    continue

                if nl_txt in code_txt:
                    code_txt = code_txt.replace(nl_txt, "")
                    code_txt = code_txt.replace('""""""', "").replace("''''''", "")

                shift_type = js.get("shift_type", "id")
                self.examples.append({
                    "nl": nl_txt,
                    "code": code_txt,
                    "idx": str(js.get("idx", js.get("id", line_idx))),
                    "shift_type": shift_type,
                })
                shift_counter[shift_type] += 1

        print(f">>> Loaded {len(self.examples)} samples from {file_path}")
        print(f">>> Skipped samples: {skipped}")
        print(f">>> shift_type distribution: {dict(shift_counter)}")

    @staticmethod
    def get_nl_text(js):
        if js.get("docstring"):
            return js["docstring"]
        if js.get("doc"):
            return js["doc"]
        if js.get("nl"):
            return js["nl"]
        if js.get("query"):
            return js["query"]
        if "docstring_tokens" in js:
            return " ".join(js["docstring_tokens"])
        return ""

    @staticmethod
    def get_code_text(js):
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
            return_tensors="pt",
        )
        code_inputs = self.tokenizer(
            item["code"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "nl_input_ids": nl_inputs["input_ids"].squeeze(0),
            "nl_attention_mask": nl_inputs["attention_mask"].squeeze(0),
            "pl_input_ids": code_inputs["input_ids"].squeeze(0),
            "pl_attention_mask": code_inputs["attention_mask"].squeeze(0),
            "idx": item["idx"],
            "shift_type": item["shift_type"],
        }


# ============================================================
# 4. Dual encoder
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
            attention_mask=attention_mask,
            return_dict=True,
        )
        if "t5" in self.model_type or "plbart" in self.model_type:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            vec = torch.sum(outputs.last_hidden_state * mask, dim=1)
            vec = vec / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            vec = outputs.last_hidden_state[:, 0, :]
        return vec

    def forward(self, nl_ids, nl_mask, pl_ids, pl_mask):
        nl_vec = self.get_vec(nl_ids, nl_mask)
        pl_vec = self.get_vec(pl_ids, pl_mask)
        return F.normalize(nl_vec, p=2, dim=1), F.normalize(pl_vec, p=2, dim=1)


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        raise FileNotFoundError(f"RQ1 best model not found: {RQ1_BEST_MODEL}")

    model = DualEncoderModel(MODEL_PATH, MODEL_TYPE).to(DEVICE)
    state = unwrap_checkpoint(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    if not isinstance(state, dict):
        raise RuntimeError("Loaded RQ1 checkpoint is not a state_dict-like object.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    matched = [key for key in state.keys() if key in model.state_dict()]
    encoder_matched = [key for key in matched if key.startswith("encoder.")]

    print(f">>> Loaded RQ1 best model: {RQ1_BEST_MODEL}")
    print(
        f">>> model_type={MODEL_TYPE}, matched={len(matched)}, "
        f"encoder_matched={len(encoder_matched)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        print("Missing preview:", missing[:10])
    if unexpected:
        print("Unexpected preview:", unexpected[:10])
    if len(encoder_matched) == 0:
        raise RuntimeError(
            "No encoder parameters were loaded from the RQ1 checkpoint. "
            "Check MODEL_TYPE, MODEL_PATH, and RQ1_BEST_MODEL."
        )
    return model


# ============================================================
# 5. Vector extraction
# ============================================================

@torch.no_grad()
def extract_vectors(model, dataloader, desc="Extracting"):
    model.eval()
    all_nl_vecs = []
    all_pl_vecs = []
    all_indices = []
    all_shift_types = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        nl_vec, pl_vec = model(
            batch["nl_input_ids"].to(DEVICE),
            batch["nl_attention_mask"].to(DEVICE),
            batch["pl_input_ids"].to(DEVICE),
            batch["pl_attention_mask"].to(DEVICE),
        )
        all_nl_vecs.append(nl_vec.detach().cpu())
        all_pl_vecs.append(pl_vec.detach().cpu())
        all_indices.extend(batch["idx"])
        all_shift_types.extend(batch["shift_type"])

    if not all_nl_vecs:
        raise ValueError(f"{desc}: empty dataloader")

    return {
        "nl_vecs": torch.cat(all_nl_vecs, dim=0),
        "pl_vecs": torch.cat(all_pl_vecs, dim=0),
        "idx": all_indices,
        "shift_type": all_shift_types,
    }


def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ============================================================
# 6. Detector scores
# ============================================================

def build_pair_features(nl_vecs, pl_vecs):
    nl_vecs = F.normalize(nl_vecs, p=2, dim=1)
    pl_vecs = F.normalize(pl_vecs, p=2, dim=1)
    abs_diff = torch.abs(nl_vecs - pl_vecs)
    prod = nl_vecs * pl_vecs
    return torch.cat([nl_vecs, pl_vecs, abs_diff, prod], dim=1)


def compute_knn_scores(ref_pair_features, eval_pair_features, k=10):
    ref = F.normalize(ref_pair_features, p=2, dim=1)
    eva = F.normalize(eval_pair_features, p=2, dim=1)
    distances = torch.cdist(eva, ref, p=2)
    k = min(k, distances.size(1))
    topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
    return to_numpy(topk_dist.mean(dim=1))


def compute_matching_confidence_scores(nl_vecs, pl_vecs):
    nl_vecs = F.normalize(nl_vecs, p=2, dim=1)
    pl_vecs = F.normalize(pl_vecs, p=2, dim=1)
    matched_logits = torch.sum(nl_vecs * pl_vecs, dim=1) * LOGIT_SCALE
    return to_numpy(-matched_logits)


def compute_ranking_entropy_scores(nl_vecs, code_bank_vecs, topk=100):
    nl_vecs = F.normalize(nl_vecs, p=2, dim=1)
    code_bank_vecs = F.normalize(code_bank_vecs, p=2, dim=1)

    logits = torch.matmul(nl_vecs, code_bank_vecs.T) * LOGIT_SCALE
    if topk is not None and topk > 0 and topk < logits.size(1):
        logits, _ = torch.topk(logits, k=topk, dim=1, largest=True)

    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    normalizer = np.log(max(2, logits.size(1)))
    return to_numpy(entropy / normalizer)


def compute_detector_scores(ref_pair_features, eval_vecs, code_bank_vecs):
    eval_pair_features = build_pair_features(eval_vecs["nl_vecs"], eval_vecs["pl_vecs"])
    return {
        "KNN": compute_knn_scores(ref_pair_features, eval_pair_features, k=KNN_K),
        "MatchingConfidence": compute_matching_confidence_scores(
            eval_vecs["nl_vecs"],
            eval_vecs["pl_vecs"],
        ),
        "RankingEntropy": compute_ranking_entropy_scores(
            eval_vecs["nl_vecs"],
            code_bank_vecs,
            topk=RANKING_TOPK,
        ),
    }


def public_detector_scores(score_dict):
    return {name: np.asarray(score_dict[name], dtype=np.float64) for name in DETECTOR_NAMES}


# ============================================================
# 7. Metrics / output
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
    print(f"{'OOD Detector':<16} | {'AUROC':<10} | {'AUPR':<10} | {'TNR@95':<10}")
    print("-" * 86)
    for name in DETECTOR_NAMES:
        m = results[name]
        print(f"{name:<16} | {m['AUROC']:<10.4f} | {m['AUPR']:<10.4f} | {m['TNR@95']:<10.4f}")
    print("=" * 86)


def write_score_file(path, id_scores, ood_scores, id_vecs, ood_vecs):
    with open(path, "w", encoding="utf-8") as f:
        n_id = len(next(iter(id_scores.values())))
        n_ood = len(next(iter(ood_scores.values())))

        for i in range(n_id):
            rec = {
                "domain": "id",
                "label": 0,
                "idx": str(id_vecs["idx"][i]),
                "shift_type": str(id_vecs["shift_type"][i]),
                "scores": {name: float(scores[i]) for name, scores in id_scores.items()},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for i in range(n_ood):
            rec = {
                "domain": "ood",
                "label": 1,
                "idx": str(ood_vecs["idx"][i]),
                "shift_type": str(ood_vecs["shift_type"][i]),
                "scores": {name: float(scores[i]) for name, scores in ood_scores.items()},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 8. Main
# ============================================================

def main(model_type=None):
    if model_type is not None:
        configure_model_runtime(model_type)
    else:
        configure_model_runtime(MODEL_TYPE)
    set_seed(SEED)
    start_time = time.time()

    train_file = resolve_id_train_file()
    existing_ood_files = {shift: path for shift, path in OOD_TEST_FILES.items() if os.path.exists(path)}
    if not existing_ood_files:
        raise FileNotFoundError(f"No OOD test files found from: {OOD_TEST_FILES}")

    print("=" * 120)
    print("CodeSearchNet RQ2: retrieval-based OOD detection")
    print("=" * 120)
    print("LANGUAGE:", LANGUAGE)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("SUPPORTED_MODEL_TYPES:", SUPPORTED_MODEL_TYPES)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("ID_TRAIN_FILE:", train_file)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", existing_ood_files)
    print("DETECTORS:", DETECTOR_NAMES)
    print("MAX_EVAL_SAMPLES:", MAX_EVAL_SAMPLES)
    print("MAX_REF_SAMPLES:", MAX_REF_SAMPLES)
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("\n>>> Loading datasets...")
    train_ds_full = CodeSearchDataset(train_file, tokenizer, name="id_train")
    id_test_ds_full = CodeSearchDataset(ID_TEST_FILE, tokenizer, name="id_test")
    ood_test_ds_full = {
        shift: CodeSearchDataset(path, tokenizer, name=f"ood_test_{shift}")
        for shift, path in existing_ood_files.items()
    }

    if len(train_ds_full) == 0:
        raise ValueError("id_train is empty.")
    if len(id_test_ds_full) == 0:
        raise ValueError("id_test is empty.")

    ref_size = min(MAX_REF_SAMPLES, len(train_ds_full))
    ref_ds = Subset(train_ds_full, range(ref_size))

    if MAX_EVAL_SAMPLES is not None:
        id_eval_size = min(MAX_EVAL_SAMPLES, len(id_test_ds_full))
        id_eval_ds = Subset(id_test_ds_full, range(id_eval_size))
    else:
        id_eval_ds = id_test_ds_full

    ood_eval_ds = {}
    for shift, ds in ood_test_ds_full.items():
        if MAX_EVAL_SAMPLES is not None:
            size = min(MAX_EVAL_SAMPLES, len(ds))
            ood_eval_ds[shift] = Subset(ds, range(size))
        else:
            ood_eval_ds[shift] = ds

    print("\n[Sanity Check]")
    print("ID train full size:", len(train_ds_full))
    print("ID reference size:", len(ref_ds))
    print("ID eval size:", len(id_eval_ds))
    for shift, ds in ood_eval_ds.items():
        print(f"OOD eval size [{shift}]:", len(ds))

    ref_loader = make_loader(ref_ds, shuffle=False)
    id_loader = make_loader(id_eval_ds, shuffle=False)
    ood_loaders = {shift: make_loader(ds, shuffle=False) for shift, ds in ood_eval_ds.items()}

    print("\n>>> Loading RQ1 best model...")
    model = load_rq1_model()

    print("\n>>> Extracting ID reference vectors...")
    ref_vecs = extract_vectors(model, ref_loader, desc="Extract ID train reference")

    print("\n>>> Extracting ID test vectors...")
    id_vecs = extract_vectors(model, id_loader, desc="Extract ID test")

    print("\n>>> Building ID reference pair space and fixed ID code bank...")
    ref_pair_features = build_pair_features(ref_vecs["nl_vecs"], ref_vecs["pl_vecs"])
    code_bank_vecs = ref_vecs["pl_vecs"]

    print("\n>>> Computing ID detector scores...")
    id_all_scores = public_detector_scores(compute_detector_scores(
        ref_pair_features,
        id_vecs,
        code_bank_vecs,
    ))

    all_results = {}
    all_score_files = {}

    for shift, loader in ood_loaders.items():
        print("\n" + "=" * 100)
        print(f"Evaluating OOD Detection for {shift.upper()} shift")
        print("=" * 100)

        ood_vecs = extract_vectors(model, loader, desc=f"Extract OOD test {shift}")
        ood_all_scores = public_detector_scores(compute_detector_scores(
            ref_pair_features,
            ood_vecs,
            code_bank_vecs,
        ))

        shift_results = {}
        for detector_name in DETECTOR_NAMES:
            shift_results[detector_name] = compute_ood_metrics(
                id_all_scores[detector_name],
                ood_all_scores[detector_name],
            )

        all_results[shift] = shift_results
        print_result_table(
            title=f"RQ2 Results: CodeSearchNet-{LANGUAGE} {shift.upper()} Shift - {MODEL_TYPE}",
            results=shift_results,
        )

        score_file = os.path.join(SAVE_DIR, f"rq2_scores_{LANGUAGE}_{MODEL_TYPE}_{shift}.jsonl")
        write_score_file(score_file, id_all_scores, ood_all_scores, id_vecs, ood_vecs)
        all_score_files[shift] = score_file
        print("Score file saved:", score_file)

    total_time = time.time() - start_time
    final_metrics = {
        "task": "Code Search",
        "dataset": f"CodeSearchNet-{LANGUAGE}",
        "method": "RQ2 retrieval-oriented OOD detection with three retrieval-specific detectors",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "supported_model_types": SUPPORTED_MODEL_TYPES,
        "rq1_best_model": RQ1_BEST_MODEL,
        "language": LANGUAGE,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "max_ref_samples": MAX_REF_SAMPLES,
        "knn_k": KNN_K,
        "logit_scale": LOGIT_SCALE,
        "ranking_topk": RANKING_TOPK,
        "code_bank": "ID reference code vectors from the same fixed reference set used for all ID/OOD ranking scores",
        "metrics": ["AUROC", "AUPR", "TNR@95"],
        "detectors": DETECTOR_NAMES,
        "detector_notes": {
            "KNN": "mean distance from query-code pair embedding to nearest ID reference pair embeddings",
            "MatchingConfidence": "negative matched query-code relevance score; larger means lower matching confidence",
            "RankingEntropy": "normalized top-k entropy of query relevance scores over the fixed ID code bank",
        },
        "results": all_results,
        "files": {
            "train_file": train_file,
            "id_test_file": ID_TEST_FILE,
            "ood_test_files": existing_ood_files,
            "score_files": all_score_files,
            "metrics_file": METRICS_FILE,
        },
        "time": {
            "total_seconds": total_time,
            "total_readable": format_time(total_time),
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 120)
    print("RQ2 finished")
    print("=" * 120)
    print("Metrics file:", METRICS_FILE)
    print("Total time:", format_time(total_time))
    print("=" * 120)
    return final_metrics


if __name__ == "__main__":
    if RUN_ALL_MODELS:
        all_model_results = []
        for run_model_type in MODEL_TYPES_TO_RUN:
            print("\n\n" + "#" * 120)
            print(f"Running CodeSearchNet RQ2 for model: {run_model_type}")
            print("#" * 120)
            all_model_results.append(main(run_model_type))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        with open(ALL_MODELS_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": "Code Search",
                    "dataset": f"CodeSearchNet-{LANGUAGE}",
                    "models": MODEL_TYPES_TO_RUN,
                    "detectors": DETECTOR_NAMES,
                    "results": all_model_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("All-model metrics file:", ALL_MODELS_METRICS_FILE)
    else:
        main(MODEL_TYPE)
