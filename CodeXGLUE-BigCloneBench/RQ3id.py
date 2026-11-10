# -*- coding: utf-8 -*-
"""
RQ3: Layer-wise Representation-Space Diagnostics for Clone Detection

Dataset:
    CodeXGLUE BigCloneBench clone detection

Models:
    CodeBERT / GraphCodeBERT / RoBERTa: AutoModel + CLS pooling
    CodeT5 / PLBART: encoder + mean pooling

Metrics:
1. Alignment（ID-OOD 跨域对齐）
   Pair-level ID-OOD cosine alignment for token/cst shifts.
2. Anisotropy（仅 ID 样本）
   Average off-diagonal pairwise cosine similarity over ID representations.
3. Effective Rank（仅 ID 样本）
   exp(entropy(singular value distribution)) after centering the ID matrix.
4. Entropy（仅 ID 样本）
   Fixed-head probing entropy over ID samples using the fine-tuned clone classifier head.

Only Alignment is an ID-vs-OOD metric; the other three are ID-only.

This script loads RQ1 best checkpoints:
    /opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/rq1_outputs/best_clone_{MODEL_TYPE}.bin
"""

import csv
import io
import json
import os
import random
import sys
import time
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, T5EncoderModel


# ============================================================
# 0. Environment
# ============================================================

if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

START_TIME = time.time()


# ============================================================
# 1. Config
# ============================================================

TASK_NAME = "Clone Detection"
DATASET_NAME = "CodeXGLUE BigCloneBench"

SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

RUN_ALL_MODELS = True
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_TYPE = "PLBART"
MODEL_PATH = MODEL_PATHS[MODEL_TYPE]

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset"
RQ1_SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/rq1_outputs"
SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/rq3_results"

ID_TEST_FILE = os.path.join(DATA_DIR, "id_test.jsonl")
OOD_TEST_FILES = {
    "token": os.path.join(DATA_DIR, "ood_test_token.jsonl"),
    "cst": os.path.join(DATA_DIR, "ood_test_cst.jsonl"),
}

# "token", "cst", or "all". Clone shifts are expected to be line-aligned.
SHIFT_TYPE = "cst"

# 后三个指标（anisotropy / effective rank / entropy）所用的样本集。
# 只作为输出文件名后缀和 meta 记录使用，保证基于 ID 的结果不会覆盖此前基于 OOD 的结果。
METRIC_SPLIT_TAG = "id"

NUM_LABELS = 2
MAX_LEN = 512
BATCH_SIZE = 8
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ANALYSIS_SAMPLES = None
STRICT_PAIR_ALIGNMENT = True
MAX_ANISO_SAMPLES = None
MAX_RANK_SAMPLES = None
ANISO_CHUNK_SIZE = 1024
ENCODER_POOLING = "auto"  # auto / cls / mean

os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_{METRIC_SPLIT_TAG}.json")
RESULT_CSV = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_{METRIC_SPLIT_TAG}.csv")
RESULT_RAW_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_{METRIC_SPLIT_TAG}_raw_paste_table.tsv")
RESULT_NORMALIZED_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_{METRIC_SPLIT_TAG}_normalized_paste_table.tsv")
ALL_MODELS_RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_all_models_{SHIFT_TYPE}_{METRIC_SPLIT_TAG}.json")


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
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds"


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def canonical_model_type(model_type: str) -> str:
    aliases = {
        "codebert": "codebert",
        "code-bert": "codebert",
        "graphcodebert": "GraphCodeBERT",
        "graph-codebert": "GraphCodeBERT",
        "codet5": "codeT5",
        "code_t5": "codeT5",
        "code-t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
        "roberta": "RoBERTa",
        "roberta-base": "RoBERTa",
    }
    key = str(model_type).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Supported: {SUPPORTED_MODEL_TYPES}")
    return aliases[key]


def configure_model_runtime(model_type: str, shift_type: str):
    global MODEL_TYPE, MODEL_PATH, RESULT_JSON, RESULT_CSV, RESULT_RAW_PASTE_TSV, RESULT_NORMALIZED_PASTE_TSV

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
    RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{shift_type}_{METRIC_SPLIT_TAG}.json")
    RESULT_CSV = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{shift_type}_{METRIC_SPLIT_TAG}.csv")
    RESULT_RAW_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{shift_type}_{METRIC_SPLIT_TAG}_raw_paste_table.tsv")
    RESULT_NORMALIZED_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_clone_layer_geometry_{MODEL_TYPE}_{shift_type}_{METRIC_SPLIT_TAG}_normalized_paste_table.tsv")


def get_best_model_path(model_type: str) -> str:
    model_type = canonical_model_type(model_type)
    return os.path.join(RQ1_SAVE_DIR, f"best_clone_{model_type}.bin")


def resolve_pooling_strategy(model_type: str, pooling: str = "auto") -> str:
    pooling = pooling.lower()
    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING={pooling}")
    mt = model_type.lower()
    return "mean" if ("t5" in mt or "plbart" in mt) else "cls"


def get_label(js: dict) -> Optional[int]:
    label = js.get("label")
    if label is None:
        label = js.get("target")
    if label is None:
        return None
    return int(label)


def get_pair_id(js: dict, fallback: int) -> str:
    idx1 = js.get("idx1", "")
    idx2 = js.get("idx2", "")
    if idx1 != "" or idx2 != "":
        return f"{idx1}::{idx2}"
    idx = js.get("idx")
    if idx is not None:
        return str(idx)
    return str(fallback)


def deterministic_subset_indices(n: int, max_n: Optional[int], seed: int = 42) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def tensor_mean_or_nan(tensors: List[torch.Tensor]) -> float:
    if not tensors:
        return float("nan")
    return float(torch.cat(tensors, dim=0).float().mean().item())


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


# ============================================================
# 3. Dataset
# ============================================================

class ClonePairDataset(Dataset):
    def __init__(
        self,
        id_file: str,
        ood_file: str,
        tokenizer,
        max_samples: Optional[int] = None,
        strict_alignment: bool = True,
        shift_name: str = "shift",
    ):
        self.pairs = []
        self.tokenizer = tokenizer
        self.shift_name = shift_name

        if not os.path.exists(id_file):
            raise FileNotFoundError(f"Cannot find ID file: {id_file}")
        if not os.path.exists(ood_file):
            raise FileNotFoundError(f"Cannot find OOD file: {ood_file}")

        with open(id_file, "r", encoding="utf-8") as f_id:
            id_lines = [line.strip() for line in f_id if line.strip()]
        with open(ood_file, "r", encoding="utf-8") as f_ood:
            ood_lines = [line.strip() for line in f_ood if line.strip()]

        n = min(len(id_lines), len(ood_lines))
        if max_samples is not None:
            n = min(n, max_samples)

        bad_lines = 0
        pair_id_mismatch = 0
        label_mismatch = 0

        for i in range(n):
            try:
                js_id = json.loads(id_lines[i])
                js_ood = json.loads(ood_lines[i])
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            id_code1 = js_id.get("code1", "")
            id_code2 = js_id.get("code2", "")
            ood_code1 = js_ood.get("code1", "")
            ood_code2 = js_ood.get("code2", "")
            id_label = get_label(js_id)
            ood_label = get_label(js_ood)

            if not id_code1 or not id_code2 or not ood_code1 or not ood_code2:
                bad_lines += 1
                continue
            if id_label is None or ood_label is None:
                bad_lines += 1
                continue

            id_pair_id = get_pair_id(js_id, i)
            ood_pair_id = get_pair_id(js_ood, i)
            if id_pair_id != ood_pair_id:
                pair_id_mismatch += 1
            if int(id_label) != int(ood_label):
                label_mismatch += 1

            if strict_alignment and (id_pair_id != ood_pair_id or int(id_label) != int(ood_label)):
                raise ValueError(
                    f"Pair alignment failed at line {i}: "
                    f"id_pair={id_pair_id}, ood_pair={ood_pair_id}, "
                    f"id_label={id_label}, ood_label={ood_label}. "
                    "For token/cst clone shifts, ID/OOD files should be line-aligned."
                )

            self.pairs.append({
                "pair_id": id_pair_id,
                "idx1": js_id.get("idx1", ""),
                "idx2": js_id.get("idx2", ""),
                "label": int(id_label),
                "id_code1": id_code1,
                "id_code2": id_code2,
                "ood_code1": ood_code1,
                "ood_code2": ood_code2,
                "shift_type": js_ood.get("shift_type", shift_name),
            })

        print("=" * 100)
        print("ClonePairDataset loaded")
        print("shift:", shift_name)
        print("ID file :", id_file)
        print("OOD file:", ood_file)
        print("num_pairs:", len(self.pairs))
        print("bad_lines:", bad_lines)
        print("pair_id_mismatch:", pair_id_mismatch)
        print("label_mismatch:", label_mismatch)
        print("label_distribution:", dict(Counter(x["label"] for x in self.pairs)))
        print("=" * 100)

    def __len__(self):
        return len(self.pairs)

    def _encode_pair(self, code1: str, code2: str):
        return self.tokenizer(
            code1,
            code2,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, index: int):
        item = self.pairs[index]
        id_inputs = self._encode_pair(item["id_code1"], item["id_code2"])
        ood_inputs = self._encode_pair(item["ood_code1"], item["ood_code2"])

        batch = {
            "pair_id": item["pair_id"],
            "idx1": item["idx1"],
            "idx2": item["idx2"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "shift_type": item["shift_type"],
            "id_input_ids": id_inputs["input_ids"].squeeze(0),
            "id_attention_mask": id_inputs["attention_mask"].squeeze(0),
            "ood_input_ids": ood_inputs["input_ids"].squeeze(0),
            "ood_attention_mask": ood_inputs["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in id_inputs:
            batch["id_token_type_ids"] = id_inputs["token_type_ids"].squeeze(0)
        if "token_type_ids" in ood_inputs:
            batch["ood_token_type_ids"] = ood_inputs["token_type_ids"].squeeze(0)
        return batch


# ============================================================
# 4. Model
# ============================================================

class CloneLayerModel(nn.Module):
    def __init__(self, model_path: str, model_type: str, num_labels: int, pooling: str = "auto"):
        super().__init__()
        self.model_type = canonical_model_type(model_type).lower()
        self.pooling_strategy = resolve_pooling_strategy(model_type, pooling)

        if "t5" in self.model_type:
            self.encoder = T5EncoderModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            base_model = AutoModel.from_pretrained(model_path)
            if hasattr(base_model, "get_encoder"):
                self.encoder = base_model.get_encoder()
            elif hasattr(base_model, "encoder"):
                self.encoder = base_model.encoder
            else:
                raise ValueError("PLBART requires a model with get_encoder() or encoder.")
            hidden_size = getattr(base_model.config, "d_model", None)
            if hidden_size is None:
                hidden_size = getattr(base_model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot infer hidden size from PLBART config.")
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _use_token_type_ids(self):
        return "t5" not in self.model_type and "plbart" not in self.model_type

    def forward_hidden_states(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if token_type_ids is not None and self._use_token_type_ids():
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        return outputs.hidden_states

    def pool_layer_hidden(self, hidden_state, attention_mask):
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
            summed = torch.sum(hidden_state * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / denom
        return hidden_state[:, 0, :]

    def classify_pooled_features(self, pooled_features):
        return self.classifier(self.dropout(pooled_features))

    def load_ft_weights(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find fine-tuned weights: {path}")

        print(f">>> Loading fine-tuned weights: {path}")
        state = safe_torch_load(path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        matched = [key for key in state.keys() if key in self.state_dict()]
        encoder_matched = [key for key in matched if key.startswith("encoder.")]
        classifier_matched = [key for key in matched if key.startswith("classifier.")]

        print(
            f">>> Weights loaded: matched={len(matched)}, "
            f"encoder_matched={len(encoder_matched)}, classifier_matched={len(classifier_matched)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, pooling={self.pooling_strategy}"
        )
        if len(classifier_matched) == 0:
            raise RuntimeError("No classifier parameters were loaded from the RQ1 checkpoint.")
        for key in ["classifier.weight", "classifier.bias"]:
            if key not in matched:
                raise RuntimeError(f"Final classifier key was not loaded: {key}")

        return {
            "matched": len(matched),
            "encoder_matched": len(encoder_matched),
            "classifier_matched": len(classifier_matched),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "pooling_strategy": self.pooling_strategy,
        }


# ============================================================
# 5. Metrics
# ============================================================

def get_alignment_pair(id_emb: torch.Tensor, ood_emb: torch.Tensor) -> float:
    if id_emb.size(0) == 0:
        return float("nan")
    id_norm = F.normalize(id_emb.float(), p=2, dim=1)
    ood_norm = F.normalize(ood_emb.float(), p=2, dim=1)
    return F.cosine_similarity(id_norm, ood_norm, dim=1).mean().item()


def get_anisotropy(emb: torch.Tensor, max_samples: Optional[int] = MAX_ANISO_SAMPLES) -> float:
    if emb.size(0) < 2:
        return float("nan")
    if max_samples is not None and emb.size(0) > max_samples:
        idx = deterministic_subset_indices(emb.size(0), max_samples, seed=SEED)
        emb = emb[idx]
    z = emb.detach().float().cpu()
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = F.normalize(z, p=2, dim=1)
    n = z.size(0)
    total = 0.0
    for start in range(0, n, ANISO_CHUNK_SIZE):
        end = min(start + ANISO_CHUNK_SIZE, n)
        sim = torch.matmul(z[start:end], z.T)
        total += float(sim.sum().item())
    total -= float(n)
    return total / float(n * (n - 1))


def get_effective_rank(emb: torch.Tensor, max_samples: Optional[int] = MAX_RANK_SAMPLES) -> float:
    if emb.size(0) < 2:
        return float("nan")
    if max_samples is not None and emb.size(0) > max_samples:
        idx = deterministic_subset_indices(emb.size(0), max_samples, seed=SEED + 1)
        emb = emb[idx]
    x = emb.detach().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(dim=0, keepdim=True)
    x = x.cpu()
    try:
        singular_values = torch.linalg.svdvals(x)
    except RuntimeError:
        singular_values = torch.svd(x)[1]
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel() == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return float(torch.exp(entropy).item())


def get_entropy_values_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)


def pct_delta(base: float, ft: float) -> float:
    if base is None or np.isnan(base) or abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (ft - base) / abs(base)


def minmax_normalize(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if finite.sum() == 0:
        return out.tolist()
    vals = arr[finite]
    lo = float(vals.min())
    hi = float(vals.max())
    if abs(hi - lo) < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (vals - lo) / (hi - lo)
    return out.tolist()


def add_normalized_columns(rows: List[dict]) -> List[dict]:
    metric_pairs = [
        ("alignment", "alignment_base", "alignment_ft"),
        ("anisotropy", "anisotropy_base_id", "anisotropy_ft_id"),
        ("rank", "effective_rank_base_id", "effective_rank_ft_id"),
        ("entropy", "entropy_base_id", "entropy_ft_id"),
    ]
    shifts = sorted(set(row["shift"] for row in rows))
    for shift in shifts:
        group = [row for row in rows if row["shift"] == shift]
        for metric_name, base_key, ft_key in metric_pairs:
            combined = [row[base_key] for row in group] + [row[ft_key] for row in group]
            normalized = minmax_normalize(combined)
            n = len(group)
            for i, row in enumerate(group):
                row[f"{metric_name}_base_n"] = float(normalized[i])
                row[f"{metric_name}_ft_n"] = float(normalized[n + i])
                row[f"{metric_name}_delta_n_ft_minus_base"] = float(normalized[n + i] - normalized[i])
    return rows


# ============================================================
# 6. Extraction
# ============================================================

def init_models():
    ft_weights = get_best_model_path(MODEL_TYPE)
    if not os.path.exists(ft_weights):
        raise FileNotFoundError(f"Cannot find RQ1 best clone model: {ft_weights}")

    print("=" * 100)
    print("Model initialization")
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("FT_WEIGHTS:", ft_weights)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("=" * 100)

    base_model = CloneLayerModel(MODEL_PATH, MODEL_TYPE, NUM_LABELS, pooling=ENCODER_POOLING).to(DEVICE)
    ft_model = CloneLayerModel(MODEL_PATH, MODEL_TYPE, NUM_LABELS, pooling=ENCODER_POOLING).to(DEVICE)
    load_info = ft_model.load_ft_weights(ft_weights)

    base_model.eval()
    ft_model.eval()
    return base_model, ft_model, load_info


@torch.no_grad()
def extract_features_pair(base_model, ft_model, dataloader):
    stats = None
    label_counter = Counter()
    total_examples = 0

    def init_layer_stats(num_layers, hidden_size):
        layers = []
        for _ in range(num_layers):
            layers.append({
                "n": 0,
                "alignment_base_sum": 0.0,
                "alignment_ft_sum": 0.0,
                "entropy_base_id_sum": 0.0,
                "entropy_ft_id_sum": 0.0,
                "base_id_norm_sum": torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE),
                "ft_id_norm_sum": torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE),
                "base_id_sum": torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE),
                "ft_id_sum": torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE),
                "base_id_xtx": torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=DEVICE),
                "ft_id_xtx": torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=DEVICE),
            })
        return layers

    def update_representation_stats(layer_stats, prefix, pooled):
        x = pooled.detach().float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z = F.normalize(x, p=2, dim=1)
        layer_stats[f"{prefix}_norm_sum"] += z.sum(dim=0)
        layer_stats[f"{prefix}_sum"] += x.sum(dim=0)
        layer_stats[f"{prefix}_xtx"] += torch.matmul(x.T, x)

    for batch in tqdm(dataloader, desc="Extracting layer-wise clone features", leave=False):
        id_ids = batch["id_input_ids"].to(DEVICE)
        id_mask = batch["id_attention_mask"].to(DEVICE)
        ood_ids = batch["ood_input_ids"].to(DEVICE)
        ood_mask = batch["ood_attention_mask"].to(DEVICE)
        batch_labels = batch["label"].cpu().tolist()
        label_counter.update(int(x) for x in batch_labels)
        total_examples += len(batch_labels)

        id_token_type_ids = batch.get("id_token_type_ids")
        ood_token_type_ids = batch.get("ood_token_type_ids")
        if id_token_type_ids is not None:
            id_token_type_ids = id_token_type_ids.to(DEVICE)
        if ood_token_type_ids is not None:
            ood_token_type_ids = ood_token_type_ids.to(DEVICE)

        hs_base_id = base_model.forward_hidden_states(id_ids, id_mask, token_type_ids=id_token_type_ids)
        hs_base_ood = base_model.forward_hidden_states(ood_ids, ood_mask, token_type_ids=ood_token_type_ids)
        hs_ft_id = ft_model.forward_hidden_states(id_ids, id_mask, token_type_ids=id_token_type_ids)
        hs_ft_ood = ft_model.forward_hidden_states(ood_ids, ood_mask, token_type_ids=ood_token_type_ids)

        num_layers = min(len(hs_base_id), len(hs_base_ood), len(hs_ft_id), len(hs_ft_ood))

        for layer in range(num_layers):
            b_id_dev = base_model.pool_layer_hidden(hs_base_id[layer], id_mask)
            b_ood_dev = base_model.pool_layer_hidden(hs_base_ood[layer], ood_mask)
            f_id_dev = ft_model.pool_layer_hidden(hs_ft_id[layer], id_mask)
            f_ood_dev = ft_model.pool_layer_hidden(hs_ft_ood[layer], ood_mask)

            if stats is None:
                stats = init_layer_stats(num_layers, int(b_ood_dev.size(1)))

            layer_stats = stats[layer]
            layer_stats["n"] += int(b_ood_dev.size(0))
            layer_stats["alignment_base_sum"] += float(F.cosine_similarity(
                F.normalize(b_id_dev.float(), p=2, dim=1),
                F.normalize(b_ood_dev.float(), p=2, dim=1),
                dim=1,
            ).sum().item())
            layer_stats["alignment_ft_sum"] += float(F.cosine_similarity(
                F.normalize(f_id_dev.float(), p=2, dim=1),
                F.normalize(f_ood_dev.float(), p=2, dim=1),
                dim=1,
            ).sum().item())

            # 后三个指标只用 ID 样本；熵用 FT 分类头对 ID 表征做 fixed-head probe。
            logits_base_id = ft_model.classify_pooled_features(b_id_dev)
            logits_ft_id = ft_model.classify_pooled_features(f_id_dev)
            layer_stats["entropy_base_id_sum"] += float(get_entropy_values_from_logits(logits_base_id).sum().item())
            layer_stats["entropy_ft_id_sum"] += float(get_entropy_values_from_logits(logits_ft_id).sum().item())

            update_representation_stats(layer_stats, "base_id", b_id_dev)
            update_representation_stats(layer_stats, "ft_id", f_id_dev)

        del hs_base_id, hs_base_ood, hs_ft_id, hs_ft_ood

    if stats is None:
        raise ValueError("No features were extracted. Check dataset and filters.")
    return stats, label_counter, total_examples


def anisotropy_from_norm_sum(n: int, norm_sum: torch.Tensor) -> float:
    if n < 2:
        return float("nan")
    total = float(torch.dot(norm_sum, norm_sum).item()) - float(n)
    return total / float(n * (n - 1))


def effective_rank_from_moments(n: int, sum_vec: torch.Tensor, xtx: torch.Tensor) -> float:
    if n < 2:
        return float("nan")
    sum_vec = sum_vec.detach().double().cpu()
    xtx = xtx.detach().double().cpu()
    centered_xtx = xtx - torch.outer(sum_vec, sum_vec) / float(n)
    centered_xtx = 0.5 * (centered_xtx + centered_xtx.T)
    try:
        eigvals = torch.linalg.eigvalsh(centered_xtx)
    except RuntimeError:
        eigvals = torch.from_numpy(np.linalg.eigvalsh(centered_xtx.numpy()))
    singular_values = torch.sqrt(torch.clamp(eigvals, min=0.0))
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel() == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return float(torch.exp(entropy).item())


# ============================================================
# 7. Results
# ============================================================

def build_result_row(layer, shift_name, layer_stats, label_counter):
    n = int(layer_stats["n"])
    alignment_base = layer_stats["alignment_base_sum"] / n if n else float("nan")
    alignment_ft = layer_stats["alignment_ft_sum"] / n if n else float("nan")
    aniso_base_id = anisotropy_from_norm_sum(n, layer_stats["base_id_norm_sum"])
    aniso_ft_id = anisotropy_from_norm_sum(n, layer_stats["ft_id_norm_sum"])
    rank_base_id = effective_rank_from_moments(n, layer_stats["base_id_sum"], layer_stats["base_id_xtx"])
    rank_ft_id = effective_rank_from_moments(n, layer_stats["ft_id_sum"], layer_stats["ft_id_xtx"])
    entropy_base_id = layer_stats["entropy_base_id_sum"] / n if n else float("nan")
    entropy_ft_id = layer_stats["entropy_ft_id_sum"] / n if n else float("nan")

    return {
        "shift": shift_name,
        "layer": int(layer),
        "alignment_mode": "pair",
        "alignment_units": n,
        "label_distribution": {str(k): int(v) for k, v in sorted(label_counter.items())},
        "alignment_base": alignment_base,
        "alignment_ft": alignment_ft,
        "alignment_delta_ft_minus_base": alignment_ft - alignment_base,
        "alignment_delta_percent": pct_delta(alignment_base, alignment_ft),
        "anisotropy_base_id": aniso_base_id,
        "anisotropy_ft_id": aniso_ft_id,
        "anisotropy_delta_id_ft_minus_base": aniso_ft_id - aniso_base_id,
        "anisotropy_delta_percent": pct_delta(aniso_base_id, aniso_ft_id),
        "effective_rank_base_id": rank_base_id,
        "effective_rank_ft_id": rank_ft_id,
        "effective_rank_delta_id_ft_minus_base": rank_ft_id - rank_base_id,
        "effective_rank_delta_percent": pct_delta(rank_base_id, rank_ft_id),
        "entropy_base_id": entropy_base_id,
        "entropy_ft_id": entropy_ft_id,
        "entropy_delta_id_ft_minus_base": entropy_ft_id - entropy_base_id,
        "entropy_delta_percent": pct_delta(entropy_base_id, entropy_ft_id),
    }


def aggregate_results_pair(shift_name: str, stats, label_counter):
    results = []
    for layer, layer_stats in enumerate(stats):
        results.append(build_result_row(
            layer=layer,
            shift_name=shift_name,
            layer_stats=layer_stats,
            label_counter=label_counter,
        ))
    return results


def print_result_table(results):
    print("\n" + "=" * 172)
    print(f"RQ3 Clone Layer-wise Diagnostics: {MODEL_TYPE.upper()}".center(172))
    print("=" * 172)
    print(f"{'Shift':<8} | {'Layer':<7} | {'Align(ID-OOD) B/FT/Delta':<28} | {'Aniso(ID) B/FT/Delta':<31} | {'EffRank(ID) B/FT/Delta':<34} | {'Entropy(ID) B/FT/Delta':<31}")
    print("-" * 172)
    for row in results:
        print(
            f"{row['shift']:<8} | "
            f"L{row['layer']:02d}     | "
            f"{row['alignment_base']:.4f} / {row['alignment_ft']:.4f} / {row['alignment_delta_ft_minus_base']:+.4f} | "
            f"{row['anisotropy_base_id']:.4f} / {row['anisotropy_ft_id']:.4f} / {row['anisotropy_delta_id_ft_minus_base']:+.4f} | "
            f"{row['effective_rank_base_id']:<8.2f} / {row['effective_rank_ft_id']:<8.2f} / {row['effective_rank_delta_id_ft_minus_base']:+.2f} | "
            f"{row['entropy_base_id']:.4f} / {row['entropy_ft_id']:.4f} / {row['entropy_delta_id_ft_minus_base']:+.4f}"
        )
    print("=" * 172)


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{float(x):.{digits}f}"


def fmt_percent(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{float(x):.{digits}f}%"


def build_excel_raw_rows(results: List[dict]):
    rows = [[
        "Shift",
        "Layer",
        "Align_base",
        "Align_ft",
        "Aniso_base",
        "Aniso_ft",
        "Rank_base",
        "Rank_ft",
        "Entropy_base",
        "Entropy_ft",
    ]]
    for row in results:
        rows.append([
            row["shift"],
            f"L{int(row['layer']):02d}",
            fmt_value(row["alignment_base"], 4),
            fmt_value(row["alignment_ft"], 4),
            fmt_value(row["anisotropy_base_id"], 4),
            fmt_value(row["anisotropy_ft_id"], 4),
            fmt_value(row["effective_rank_base_id"], 4),
            fmt_value(row["effective_rank_ft_id"], 4),
            fmt_value(row["entropy_base_id"], 4),
            fmt_value(row["entropy_ft_id"], 4),
        ])
    return rows


def build_excel_normalized_rows(results: List[dict]):
    rows = [[
        "Shift",
        "Layer",
        "Align_base_n",
        "Align_ft_n",
        "Aniso_base_n",
        "Aniso_ft_n",
        "Rank_base_n",
        "Rank_ft_n",
        "Entropy_base_n",
        "Entropy_ft_n",
    ]]
    for row in results:
        rows.append([
            row["shift"],
            f"L{int(row['layer']):02d}",
            fmt_value(row["alignment_base_n"], 4),
            fmt_value(row["alignment_ft_n"], 4),
            fmt_value(row["anisotropy_base_n"], 4),
            fmt_value(row["anisotropy_ft_n"], 4),
            fmt_value(row["rank_base_n"], 4),
            fmt_value(row["rank_ft_n"], 4),
            fmt_value(row["entropy_base_n"], 4),
            fmt_value(row["entropy_ft_n"], 4),
        ])
    return rows


def write_tsv(path: str, rows: List[List[str]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")


def print_excel_paste_table(results: List[dict]):
    print("\n" + "=" * 172)
    print(f"EXCEL_COPY_RAW\t{DATASET_NAME}\t{SHIFT_TYPE}\t{MODEL_TYPE}")
    print("=" * 172)
    for row in build_excel_raw_rows(results):
        print("\t".join(row))
    print("=" * 172)

    print("\n" + "=" * 172)
    print(f"EXCEL_COPY_NORMALIZED\t{DATASET_NAME}\t{SHIFT_TYPE}\t{MODEL_TYPE}")
    print("=" * 172)
    for row in build_excel_normalized_rows(results):
        print("\t".join(row))
    print("=" * 172)


def save_results(results: List[dict], load_info: dict, dataset_info: dict):
    meta = {
        "task": TASK_NAME,
        "dataset": DATASET_NAME,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "ft_model_weights": get_best_model_path(MODEL_TYPE),
        "encoder_pooling": load_info.get("pooling_strategy"),
        "load_info": {
            "matched": load_info.get("matched"),
            "encoder_matched": load_info.get("encoder_matched"),
            "classifier_matched": load_info.get("classifier_matched"),
            "missing_count": len(load_info.get("missing", [])),
            "unexpected_count": len(load_info.get("unexpected", [])),
        },
        "shift_type": SHIFT_TYPE,
        "metric_split": {
            "alignment": "id_vs_ood",
            "anisotropy": "id_only",
            "effective_rank": "id_only",
            "entropy": "id_only",
        },
        "num_labels": NUM_LABELS,
        "max_len": MAX_LEN,
        "max_analysis_samples": MAX_ANALYSIS_SAMPLES,
        "strict_pair_alignment": STRICT_PAIR_ALIGNMENT,
        "max_aniso_samples": MAX_ANISO_SAMPLES,
        "max_rank_samples": MAX_RANK_SAMPLES,
        "aniso_chunk_size": ANISO_CHUNK_SIZE,
        "dataset_info": dataset_info,
        "metrics": ["alignment", "anisotropy", "effective_rank", "entropy"],
        "normalization": {
            "method": "min-max",
            "scope": "per model + shift + metric, using base and fine-tuned values from all analyzed layers together",
            "columns": "raw columns are preserved in CSV/JSON/raw TSV; *_n columns are normalized to [0, 1] for plotting",
        },
        "notes": [
            "Clone inputs are tokenized as code1/code2 pairs, matching RQ1.",
            "CodeT5 and PLBART use mean pooling under ENCODER_POOLING=auto; CodeBERT/GraphCodeBERT/RoBERTa use CLS pooling.",
            "Alignment is pair-level because token/cst shifts are expected to preserve sample order and labels.",
            "Alignment is the only ID-vs-OOD metric; anisotropy, effective rank, and entropy are computed on ID samples only.",
            "Anisotropy and Effective Rank use streaming full-dataset moment statistics over ID representations, avoiding full embedding storage.",
            "Entropy is fixed-head probing entropy over ID samples using the fine-tuned clone classifier head.",
        ],
        "results": results,
        "elapsed_seconds": time.time() - START_TIME,
        "elapsed_readable": format_time(time.time() - START_TIME),
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    write_csv(RESULT_CSV, results)
    write_tsv(RESULT_RAW_PASTE_TSV, build_excel_raw_rows(results))
    write_tsv(RESULT_NORMALIZED_PASTE_TSV, build_excel_normalized_rows(results))

    print("\nResults saved:")
    print("JSON          :", RESULT_JSON)
    print("CSV           :", RESULT_CSV)
    print("Raw TSV       :", RESULT_RAW_PASTE_TSV)
    print("Normalized TSV:", RESULT_NORMALIZED_PASTE_TSV)


# ============================================================
# 8. Main
# ============================================================

def run_one_model(model_type: str):
    configure_model_runtime(model_type, SHIFT_TYPE)
    model_start = time.time()

    print("=" * 120)
    print("RQ3: Layer-wise Representation-Space Diagnostics for Clone Detection")
    print("=" * 120)
    print("TASK_NAME:", TASK_NAME)
    print("DATASET_NAME:", DATASET_NAME)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("FT_MODEL_WEIGHTS:", get_best_model_path(MODEL_TYPE))
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", OOD_TEST_FILES)
    print("SHIFT_TYPE:", SHIFT_TYPE)
    print("DEVICE:", DEVICE)
    print("MAX_ANALYSIS_SAMPLES:", MAX_ANALYSIS_SAMPLES)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model, ft_model, load_info = init_models()

    shifts_to_run = list(OOD_TEST_FILES.keys()) if SHIFT_TYPE == "all" else [SHIFT_TYPE]
    all_results = []
    dataset_info = {}

    for shift_name in shifts_to_run:
        if shift_name not in OOD_TEST_FILES:
            raise ValueError(f"Unsupported SHIFT_TYPE={shift_name}. Choose from token, cst, all.")

        dataset = ClonePairDataset(
            ID_TEST_FILE,
            OOD_TEST_FILES[shift_name],
            tokenizer,
            max_samples=MAX_ANALYSIS_SAMPLES,
            strict_alignment=STRICT_PAIR_ALIGNMENT,
            shift_name=shift_name,
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        stats, label_counter, total_examples = extract_features_pair(base_model, ft_model, loader)
        shift_results = aggregate_results_pair(shift_name, stats, label_counter)
        all_results.extend(shift_results)
        dataset_info[shift_name] = {
            "num_pairs": len(dataset),
            "processed_pairs": int(total_examples),
            "id_file": ID_TEST_FILE,
            "ood_file": OOD_TEST_FILES[shift_name],
            "label_distribution": {str(k): int(v) for k, v in sorted(label_counter.items())},
        }

    all_results = add_normalized_columns(all_results)
    print_result_table(all_results)
    save_results(all_results, load_info=load_info, dataset_info=dataset_info)
    print_excel_paste_table(all_results)

    print("\nFinished.")
    print("Elapsed:", format_time(time.time() - model_start))

    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "ft_model_weights": get_best_model_path(MODEL_TYPE),
        "pooling_strategy": load_info.get("pooling_strategy"),
        "num_rows": len(all_results),
        "result_json": RESULT_JSON,
        "result_csv": RESULT_CSV,
        "result_raw_tsv": RESULT_RAW_PASTE_TSV,
        "result_normalized_tsv": RESULT_NORMALIZED_PASTE_TSV,
        "elapsed_seconds": time.time() - model_start,
        "elapsed_readable": format_time(time.time() - model_start),
    }


def main():
    models_to_run = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    summaries = []

    for run_model_type in models_to_run:
        try:
            summaries.append(run_one_model(run_model_type))
        except Exception as exc:
            if not RUN_ALL_MODELS:
                raise
            summaries.append({
                "status": "failed",
                "model_type": run_model_type,
                "error": repr(exc),
            })
            print("\n" + "!" * 120)
            print(f"[Error] RQ3 failed for model={run_model_type}: {repr(exc)}")
            print("Continuing with the next model because RUN_ALL_MODELS=True.")
            print("!" * 120)
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if RUN_ALL_MODELS:
        all_summary = {
            "task": TASK_NAME,
            "dataset": DATASET_NAME,
            "shift_type": SHIFT_TYPE,
            "models_requested": models_to_run,
            "summaries": summaries,
            "elapsed_seconds": time.time() - START_TIME,
            "elapsed_readable": format_time(time.time() - START_TIME),
        }
        with open(ALL_MODELS_RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_summary, f, ensure_ascii=False, indent=2)
        print("\nAll-model RQ3 summary saved:", ALL_MODELS_RESULT_JSON)
        print("Total elapsed:", format_time(time.time() - START_TIME))


if __name__ == "__main__":
    main()
