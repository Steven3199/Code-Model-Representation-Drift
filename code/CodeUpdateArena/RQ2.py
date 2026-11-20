# -*- coding: utf-8 -*-
"""
CodeUpdateArena RQ2: Generation-oriented OOD Detection under API Evolution Shift.

RQ1 trains a seq2seq code generation model on ID data and evaluates the
performance drop on OOD data (API evolution shift).

This RQ2 script loads the best RQ1 checkpoint and detects OOD samples with
three generation-specific detectors:

1. EncoderKNN:
   Distance from the prompt encoder representation to the ID training
   prompt representation space.

2. SequenceNLL:
   Average target-token negative log-likelihood under teacher forcing.

3. TokenEntropy:
   Average target-token predictive entropy under teacher forcing.

All scores follow the same direction: larger score means more OOD.
Metrics: AUROC, AUPR, TNR@95.

Supported models: CodeT5, PLBART.
"""

import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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

DATA_DIR = "/opt/data/private/CodeUpdateArena/generation_splits"

ID_TRAIN_FILE = os.path.join(DATA_DIR, "arena_gen_id_train.jsonl")
ID_TEST_FILE  = os.path.join(DATA_DIR, "arena_gen_id_test.jsonl")
OOD_TEST_FILE = os.path.join(DATA_DIR, "arena_gen_ood_test.jsonl")

RQ1_SAVE_DIR = "/opt/data/private/CodeUpdateArena/rq1_generation_outputs"
SAVE_DIR     = "/opt/data/private/CodeUpdateArena/rq2_ood_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512
BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42

MAX_REF_SAMPLES = 3000
MAX_EVAL_SAMPLES = None
KNN_K = 10

# PLBART language configuration.
PLBART_SOURCE_LANG = "en_XX"
PLBART_TARGET_LANG = "python"

# PLBART format tokens (must match RQ1 training).
USE_FORMAT_TOKENS_FOR_PLBART = True
FORMAT_NEWLINE_TOKEN = "<NL>"
MAX_INDENT_LEVEL = 12
FORMAT_INDENT_TOKENS = [f"<INDENT_{i}>" for i in range(MAX_INDENT_LEVEL + 1)]
FORMAT_SPECIAL_TOKENS = [FORMAT_NEWLINE_TOKEN] + FORMAT_INDENT_TOKENS

PROMPT_FIELD_CANDIDATES = [
    "prompt", "question", "instruction", "problem", "description",
    "task", "input", "context", "nl", "docstring",
]
REFERENCE_FIELD_CANDIDATES = [
    "target", "ref_solution", "reference", "solution", "canonical_solution",
    "answer", "output", "completion", "code",
]
FIELD_INFERENCE_SAMPLES = 500

USE_METADATA_IN_PROMPT = True

DETECTOR_NAMES = [
    "EncoderKNN",
    "SequenceNLL",
    "TokenEntropy",
]


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def format_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.2f}s"


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_model_type(model_type: str) -> str:
    model_type_lower = str(model_type).lower()
    if model_type_lower == "all":
        return "all"
    aliases = {
        "codet5": "codeT5", "code-t5": "codeT5",
        "plbart": "PLBART", "pl-bart": "PLBART",
    }
    if model_type_lower in aliases:
        return aliases[model_type_lower]
    for name in MODEL_PATHS:
        if model_type_lower == name.lower():
            return name
    raise ValueError(
        f"Unsupported model_type={model_type}. "
        f"Choose from codeT5, PLBART, all."
    )


def get_default_model_path(model_type: str) -> str:
    model_type = normalize_model_type(model_type)
    if model_type == "all":
        raise ValueError("model_type='all' does not have a single default model path")
    return MODEL_PATHS[model_type]


def resolve_model_path(model_type: str, model_path: Optional[str]) -> str:
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
    parser = argparse.ArgumentParser(
        description="CodeUpdateArena RQ2: generation-oriented OOD detection."
    )
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


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")
    data = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad > 0:
        print(f"[Warning] {path} bad json lines: {bad}")
    return data


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(stringify_value(x) for x in value if x is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def clean_code_fence(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    fence = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        text = fence.group(1).strip()
    for prefix in ["Answer:", "Solution:", "Code:", "```python", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip()


# ---- Field inference (same logic as RQ1) ----

def field_statistics(
    items: List[Dict[str, Any]],
    candidates: List[str],
    max_samples: int,
) -> List[Dict[str, Any]]:
    subset = items[:max_samples]
    stats = []
    for order, field in enumerate(candidates):
        values = [item.get(field) for item in subset]
        nonempty = [v for v in values if is_nonempty(v)]
        lengths = [len(stringify_value(v)) for v in nonempty]
        coverage = len(nonempty) / max(len(subset), 1)
        avg_chars = float(np.mean(lengths)) if lengths else 0.0
        stats.append({
            "field": field,
            "candidate_order": order,
            "nonempty_count": len(nonempty),
            "coverage": coverage,
            "avg_chars": avg_chars,
        })
    stats.sort(key=lambda x: (x["coverage"], -x["candidate_order"]), reverse=True)
    return stats


def infer_field_name(
    items: List[Dict[str, Any]],
    candidates: List[str],
    override: Optional[str],
    role: str,
    required: bool,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if override is not None:
        missing_count = sum(
            1 for item in items if not is_nonempty(item.get(override))
        )
        if required and missing_count == len(items):
            raise ValueError(
                f"{role} field override '{override}' is empty in every sample."
            )
        print(
            f">>> {role} field uses explicit override: {override} "
            f"(missing={missing_count}/{len(items)})"
        )
        return override, []

    stats = field_statistics(items, candidates, FIELD_INFERENCE_SAMPLES)
    valid = [x for x in stats if x["nonempty_count"] > 0]
    if not valid:
        if required:
            raise ValueError(
                f"Cannot infer a non-empty {role} field from candidates={candidates}."
            )
        print(f"[Warning] No {role} field found; this field will be treated as empty.")
        return None, stats

    selected = valid[0]
    ties = [
        x for x in valid
        if abs(x["coverage"] - selected["coverage"]) < 1e-12
    ]
    if len(ties) > 1:
        print(
            f"[Warning] Ambiguous {role} fields: {[x['field'] for x in ties]}. "
            f"Selected '{selected['field']}' by candidate priority."
        )
    print(
        f">>> Auto-selected {role} field: {selected['field']} | "
        f"coverage={selected['coverage']:.4f} | "
        f"avg_chars={selected['avg_chars']:.2f}"
    )
    return selected["field"], stats


def build_prompt(item: Dict[str, Any], prompt_field: str) -> str:
    prompt = stringify_value(item.get(prompt_field)).strip()
    if USE_METADATA_IN_PROMPT:
        metadata_lines = []
        for key in ["package", "api_path", "update_type", "update_id"]:
            value = item.get(key)
            if is_nonempty(value):
                metadata_lines.append(f"{key}: {stringify_value(value)}")
        for key in [
            "update_doc", "api_update", "update_description",
            "doc", "documentation", "new_doc",
        ]:
            value = item.get(key)
            if is_nonempty(value):
                metadata_lines.append(f"{key}: {stringify_value(value)}")
                break
        if metadata_lines:
            prompt = (
                "API update information:\n"
                + "\n".join(metadata_lines)
                + "\n\nTask:\n"
                + prompt
            )
    return prompt.strip()


# ---- PLBART format tokens ----

def should_use_format_tokens() -> bool:
    return MODEL_TYPE == "PLBART" and USE_FORMAT_TOKENS_FOR_PLBART


def encode_code_with_format_tokens(code: str) -> str:
    code = clean_code_fence(code)
    lines = code.splitlines()
    encoded_lines = []
    for line in lines:
        raw = line.rstrip()
        if raw.strip() == "":
            encoded_lines.append("<INDENT_0>")
            continue
        stripped = raw.lstrip(" ")
        indent_spaces = len(raw) - len(stripped)
        indent_level = min(indent_spaces // 4, MAX_INDENT_LEVEL)
        encoded_lines.append(f"<INDENT_{indent_level}> {stripped}")
    return f" {FORMAT_NEWLINE_TOKEN} ".join(encoded_lines).strip()


def get_plbart_language_id(tokenizer, lang_name: str) -> int:
    lang_map = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_map, dict):
        if lang_name in lang_map:
            return int(lang_map[lang_name])
    candidates_to_try = [lang_name]
    if lang_name == "python":
        candidates_to_try += ["__python__", "<python>", "python3"]
    elif lang_name == "en_XX":
        candidates_to_try += ["__en_XX__", "en", "english"]
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token in candidates_to_try:
        if token in vocab:
            return int(vocab[token])
    for token in candidates_to_try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    available = list(lang_map.keys()) if isinstance(lang_map, dict) else None
    raise RuntimeError(
        f"Cannot find PLBART language id for '{lang_name}'. "
        f"Available language codes: {available}"
    )


def configure_plbart_tokenizer(tokenizer):
    """Set PLBART language-aware tokenization on src/tgt sides."""
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = PLBART_SOURCE_LANG
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = PLBART_TARGET_LANG
    print(">>> PLBART tokenizer language configured")
    print("    src_lang:", getattr(tokenizer, "src_lang", None))
    print("    tgt_lang:", getattr(tokenizer, "tgt_lang", None))


# ---- Path resolution ----

def get_rq1_model_dir(model_type: str) -> str:
    """Best model directory saved by RQ1."""
    model_type = normalize_model_type(model_type)
    return os.path.join(
        RQ1_SAVE_DIR, model_type, f"best_generation_{model_type}"
    )


def get_rq2_output_paths(model_type: str) -> Dict[str, str]:
    model_type = normalize_model_type(model_type)
    run_save_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(run_save_dir, exist_ok=True)
    return {
        "save_dir": run_save_dir,
        "metrics": os.path.join(
            run_save_dir,
            f"rq2_ood_metrics_{model_type}.json",
        ),
        "score_file": os.path.join(
            run_save_dir,
            f"rq2_ood_scores_{model_type}.jsonl",
        ),
    }


# ============================================================
# 3. Dataset
# ============================================================

class ArenaOODDataset(Dataset):
    """Loads prompt+reference pairs from a CodeUpdateArena JSONL split."""

    def __init__(
        self,
        path: str,
        tokenizer,
        name: str,
        max_samples: Optional[int] = None,
    ):
        self.path = path
        self.name = name
        self.tokenizer = tokenizer
        self.examples: List[Dict[str, Any]] = []

        raw_items = read_jsonl(path)
        if max_samples is not None:
            raw_items = raw_items[:max_samples]

        self.prompt_field, _ = infer_field_name(
            raw_items,
            PROMPT_FIELD_CANDIDATES,
            override=None,
            role=f"{name}.prompt",
            required=True,
        )
        self.reference_field, _ = infer_field_name(
            raw_items,
            REFERENCE_FIELD_CANDIDATES,
            override=None,
            role=f"{name}.reference",
            required=True,
        )

        bad_prompt = 0
        bad_ref = 0

        for idx, item in enumerate(
            tqdm(raw_items, desc=f"Loading {name}", leave=False)
        ):
            prompt = build_prompt(item, self.prompt_field)
            reference = clean_code_fence(
                stringify_value(item.get(self.reference_field))
            )

            if not prompt:
                bad_prompt += 1
                continue
            if not reference:
                bad_ref += 1
                continue

            self.examples.append({
                "idx": str(item.get("id") or item.get("idx") or idx),
                "prompt": prompt,
                "reference": reference,
            })

        print("=" * 100)
        print(f"ArenaOODDataset loaded: {name}")
        print("path:", path)
        print("num_samples:", len(self.examples))
        print("prompt_field:", self.prompt_field)
        print("reference_field:", self.reference_field)
        print("bad_prompt:", bad_prompt)
        print("bad_reference:", bad_ref)
        print("=" * 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


# ============================================================
# 4. Detector score computation
# ============================================================

def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def encode_target_for_loss(tokenizer, targets: List[str]) -> torch.Tensor:
    """Tokenize target/reference strings into label ids.

    Handles PLBART format tokens and the text_target / as_target_tokenizer
    API differences between CodeT5 and PLBART.
    """
    if should_use_format_tokens():
        targets = [encode_code_with_format_tokens(t) for t in targets]

    try:
        encoded = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    except TypeError:
        if not hasattr(tokenizer, "as_target_tokenizer"):
            raise
        with tokenizer.as_target_tokenizer():
            encoded = tokenizer(
                targets,
                max_length=MAX_TARGET_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

    labels = encoded["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return labels


@torch.no_grad()
def extract_encoder_features(
    model,
    tokenizer,
    dataset: ArenaOODDataset,
    desc: str = "Extracting encoder features",
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    model.eval()
    encoder = model.get_encoder()
    features_list = []
    indices = []

    examples = dataset.examples

    for start in tqdm(range(0, len(examples), batch_size), desc=desc, leave=False):
        batch = examples[start:start + batch_size]
        prompts = [ex["prompt"] for ex in batch]

        source_inputs = tokenizer(
            prompts,
            max_length=MAX_SOURCE_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = encoder(
            input_ids=source_inputs["input_ids"],
            attention_mask=source_inputs["attention_mask"],
            return_dict=True,
        )
        pooled = masked_mean_pool(
            outputs.last_hidden_state,
            source_inputs["attention_mask"],
        )
        pooled = F.normalize(pooled, p=2, dim=1)

        features_list.append(pooled.detach().cpu())
        indices.extend([ex["idx"] for ex in batch])

    if not features_list:
        raise ValueError(f"{desc}: empty dataset")

    return {
        "features": torch.cat(features_list, dim=0),
        "idx": indices,
    }


def compute_knn_scores(
    ref_features: torch.Tensor,
    eval_features: torch.Tensor,
    k: int = KNN_K,
) -> np.ndarray:
    ref = F.normalize(ref_features, p=2, dim=1)
    eva = F.normalize(eval_features, p=2, dim=1)
    distances = torch.cdist(eva, ref, p=2)
    k = min(k, distances.size(1))
    topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
    return topk_dist.mean(dim=1).numpy()


@torch.no_grad()
def compute_generation_uncertainty_scores(
    model,
    tokenizer,
    dataset: ArenaOODDataset,
    desc: str = "Computing generation scores",
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    model.eval()
    nll_scores = []
    entropy_scores = []
    indices = []

    examples = dataset.examples

    for start in tqdm(range(0, len(examples), batch_size), desc=desc, leave=False):
        batch = examples[start:start + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        refs = [ex["reference"] for ex in batch]

        source_inputs = tokenizer(
            prompts,
            max_length=MAX_SOURCE_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        labels = encode_target_for_loss(tokenizer, refs).to(DEVICE)

        outputs = model(
            input_ids=source_inputs["input_ids"],
            attention_mask=source_inputs["attention_mask"],
            labels=labels,
            return_dict=True,
        )

        logits = outputs.logits       # (B, seq_len, vocab)
        vocab_size = logits.size(-1)
        valid_mask = labels.ne(-100)  # (B, seq_len)
        denom = valid_mask.sum(dim=1).clamp(min=1)  # (B,)

        # SequenceNLL: per-token cross-entropy averaged over valid tokens.
        token_nll = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size())
        seq_nll = (token_nll * valid_mask).sum(dim=1) / denom

        # TokenEntropy: predictive entropy averaged over valid tokens.
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, seq_len)
        seq_entropy = (token_entropy * valid_mask).sum(dim=1) / denom

        nll_scores.extend(seq_nll.cpu().numpy().tolist())
        entropy_scores.extend(seq_entropy.cpu().numpy().tolist())
        indices.extend([ex["idx"] for ex in batch])

    return {
        "SequenceNLL": np.asarray(nll_scores, dtype=np.float64),
        "TokenEntropy": np.asarray(entropy_scores, dtype=np.float64),
        "idx": indices,
    }


def compute_all_detector_scores(
    model,
    tokenizer,
    ref_features: torch.Tensor,
    eval_dataset: ArenaOODDataset,
    eval_name: str,
) -> Dict[str, Any]:
    eval_enc = extract_encoder_features(
        model, tokenizer, eval_dataset,
        desc=f"Extract encoder features [{eval_name}]",
    )
    gen_scores = compute_generation_uncertainty_scores(
        model, tokenizer, eval_dataset,
        desc=f"Compute generation scores [{eval_name}]",
    )

    scores = {
        "EncoderKNN": compute_knn_scores(ref_features, eval_enc["features"], k=KNN_K),
        "SequenceNLL": gen_scores["SequenceNLL"],
        "TokenEntropy": gen_scores["TokenEntropy"],
    }

    return {
        "scores": {name: np.asarray(scores[name], dtype=np.float64) for name in DETECTOR_NAMES},
        "idx": eval_enc["idx"],
    }


# ============================================================
# 5. Metrics / output
# ============================================================

def compute_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Dict[str, float]:
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


def print_result_table(title: str, results: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    print(f"{'OOD Detector':<18} | {'AUROC':<10} | {'AUPR':<10} | {'TNR@95':<10}")
    print("-" * 86)
    for name in DETECTOR_NAMES:
        m = results[name]
        print(f"{name:<18} | {m['AUROC']:<10.4f} | {m['AUPR']:<10.4f} | {m['TNR@95']:<10.4f}")
    print("=" * 86)


def write_score_file(
    path: str,
    id_result: Dict[str, Any],
    ood_result: Dict[str, Any],
):
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

def run_experiment(model_type: str, model_path: Optional[str] = None):
    global MODEL_TYPE
    MODEL_TYPE = normalize_model_type(model_type)
    model_path = resolve_model_path(MODEL_TYPE, model_path)

    rq1_model_dir = get_rq1_model_dir(MODEL_TYPE)
    rq2_paths = get_rq2_output_paths(MODEL_TYPE)
    start_time = time.time()

    print("=" * 120)
    print("CodeUpdateArena RQ2: Generation-Oriented OOD Detection (API Evolution Shift)")
    print("=" * 120)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH (base):", model_path)
    print("RQ1 model dir:", rq1_model_dir)
    print("ID Train:", ID_TRAIN_FILE)
    print("ID Test :", ID_TEST_FILE)
    print("OOD Test:", OOD_TEST_FILE)
    print("SAVE_DIR:", rq2_paths["save_dir"])
    print("DEVICE:", DEVICE)
    print("DETECTORS:", DETECTOR_NAMES)
    print("MAX_REF_SAMPLES:", MAX_REF_SAMPLES)
    print("KNN_K:", KNN_K)
    print("USE_FORMAT_TOKENS_FOR_PLBART:", should_use_format_tokens())
    print("=" * 120)

    # ---- Load RQ1 best model ----
    if not os.path.exists(rq1_model_dir):
        raise FileNotFoundError(
            f"RQ1 best model directory not found: {rq1_model_dir}. "
            "Please run RQ1 first for this model."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(rq1_model_dir).to(DEVICE)

    # Add format tokens if PLBART (the saved tokenizer may already have them,
    # but we loaded from base path, so add them now).
    if should_use_format_tokens():
        existing_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        tokens_to_add = [tok for tok in FORMAT_SPECIAL_TOKENS if tok not in existing_vocab]
        if tokens_to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            if len(tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
            print(f">>> Added PLBART format tokens: {tokens_to_add}")

    # Configure special tokens.
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

    # PLBART language-aware tokenization.
    if MODEL_TYPE == "PLBART":
        configure_plbart_tokenizer(tokenizer)

    model.eval()
    print(">>> RQ1 model loaded from:", rq1_model_dir)

    # ---- Load datasets ----
    train_data = read_jsonl(ID_TRAIN_FILE)
    if MAX_REF_SAMPLES is not None:
        train_data = train_data[:MAX_REF_SAMPLES]
    id_test_data = read_jsonl(ID_TEST_FILE)
    ood_test_data = read_jsonl(OOD_TEST_FILE)

    # Build simple dataset wrappers. We build them manually here because the
    # reference set only needs prompts (no teacher forcing on targets).
    # We reuse the field inference from the train split for all splits.
    train_ds = ArenaOODDataset(ID_TRAIN_FILE, tokenizer, name="id_train_ref",
                                max_samples=MAX_REF_SAMPLES)
    id_test_ds = ArenaOODDataset(ID_TEST_FILE, tokenizer, name="id_test",
                                  max_samples=MAX_EVAL_SAMPLES)
    ood_test_ds = ArenaOODDataset(OOD_TEST_FILE, tokenizer, name="ood_test",
                                   max_samples=MAX_EVAL_SAMPLES)

    if len(train_ds) == 0:
        raise ValueError("ID train reference set is empty.")
    if len(id_test_ds) == 0:
        raise ValueError("ID test set is empty.")
    if len(ood_test_ds) == 0:
        raise ValueError("OOD test set is empty.")

    # ---- Extract ID reference encoder space ----
    print("\n>>> Extracting ID reference encoder space...")
    ref = extract_encoder_features(
        model, tokenizer, train_ds,
        desc="Extract ID train reference",
    )
    ref_features = ref["features"]

    # ---- Compute detector scores ----
    print("\n>>> Computing ID detector scores...")
    id_result = compute_all_detector_scores(
        model, tokenizer, ref_features, id_test_ds, "id_test",
    )

    print("\n>>> Computing OOD detector scores...")
    ood_result = compute_all_detector_scores(
        model, tokenizer, ref_features, ood_test_ds, "ood_test",
    )

    # ---- OOD detection metrics ----
    results = {}
    for detector_name in DETECTOR_NAMES:
        results[detector_name] = compute_ood_metrics(
            id_result["scores"][detector_name],
            ood_result["scores"][detector_name],
        )

    print_result_table(
        title=f"RQ2 Results: CodeUpdateArena API Evolution Shift - {MODEL_TYPE}",
        results=results,
    )

    write_score_file(rq2_paths["score_file"], id_result, ood_result)

    total_time = time.time() - start_time

    final_metrics = {
        "task": "Program Synthesis / Code Generation",
        "dataset": "CodeUpdateArena",
        "model_type": MODEL_TYPE,
        "model_path": model_path,
        "rq1_model_dir": rq1_model_dir,
        "shift_type": "API_evolution_shift",
        "method": "generation-oriented OOD detection with three detectors",
        "detectors": DETECTOR_NAMES,
        "detector_notes": {
            "EncoderKNN": (
                "Mean L2 distance from prompt encoder embedding to "
                "K nearest ID training prompt embeddings"
            ),
            "SequenceNLL": (
                "Average target-token negative log-likelihood under "
                "teacher forcing with the reference solution"
            ),
            "TokenEntropy": (
                "Average target-token predictive entropy under "
                "teacher forcing with the reference solution"
            ),
        },
        "score_direction": "larger score means more OOD",
        "metrics": ["AUROC", "AUPR", "TNR@95"],
        "max_ref_samples": MAX_REF_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "knn_k": KNN_K,
        "plbart_format_tokens": should_use_format_tokens(),
        "results": results,
        "files": {
            "id_train": ID_TRAIN_FILE,
            "id_test": ID_TEST_FILE,
            "ood_test": OOD_TEST_FILE,
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


# ============================================================
# 7. CLI
# ============================================================

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
    for run_mt, run_mp in experiments:
        try:
            all_results.append(run_experiment(run_mt, run_mp))
        except Exception as exc:
            all_results.append({
                "model_type": run_mt,
                "status": "failed",
                "error": repr(exc),
            })
            print(f"[Error] Model {run_mt} failed: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if len(all_results) > 1:
        summary_path = os.path.join(SAVE_DIR, "rq2_ood_metrics_all_models.json")
        summary = {
            "task": "Program Synthesis / Code Generation",
            "dataset": "CodeUpdateArena",
            "models": [item.get("model_type", "unknown") for item in all_results],
            "shift_type": "API_evolution_shift",
            "detectors": DETECTOR_NAMES,
            "metrics": ["AUROC", "AUPR", "TNR@95"],
            "results": all_results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("All-model RQ2 summary saved:", summary_path)


if __name__ == "__main__":
    main()
