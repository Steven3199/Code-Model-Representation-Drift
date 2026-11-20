# -*- coding: utf-8 -*-
"""
RQ3: Layer-wise Representation-Space Diagnostics for Code Classification

适配：
- CodeBERT / GraphCodeBERT / RoBERTa: AutoModel + CLS pooling
- CodeT5: T5EncoderModel + mean pooling
- PLBART: encoder + mean pooling
- 标签默认 0-based，不做 label-1
- 分类头自动识别：single / mlp_seq / mlp_named

指标：
1. Alignment
   - token/cst: pair-level ID-OOD cosine alignment
   - time/user: label-centroid ID-OOD cosine alignment
2. Anisotropy
   - average off-diagonal pairwise cosine similarity
3. Effective Rank
   - raw exp(entropy(singular value distribution)) over the centered OOD matrix
4. Entropy
   - raw predictive entropy averaged over OOD samples
"""

import os
import json
import csv
import time
import sys
import io
import random
from collections import Counter
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5EncoderModel, T5Config
from tqdm import tqdm


# ============================================================
# 0. Environment
# ============================================================

if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

START_TIME = time.time()


# ============================================================
# 1. Config
# ============================================================

TASK_NAME = "Code Classification"
DATASET_NAME = "CodeS"

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
MODEL_TYPE = "GraphCodeBERT"
MODEL_PATH = MODEL_PATHS[MODEL_TYPE]

DATA_DIR = "/opt/data/private/codes/Python800/Python800/cst/pre-trained"
FT_MODEL_WEIGHT_TEMPLATE = "/opt/data/private/codes/Python800/Python800/cst/best_model_{model_type}_Python800.bin"
FT_MODEL_WEIGHTS = FT_MODEL_WEIGHT_TEMPLATE.format(model_type=MODEL_TYPE)
SAVE_DIR = "/opt/data/private/codes/Python800/Python800/cst/RQ3_results"

ID_TEST_FILE = "id_test.jsonl"
OOD_TEST_FILE = "ood_test.jsonl"

# token/cst 用逐样本配对对齐；time/user 用类别中心对齐。
SHIFT_TYPE = "cst"  # choices: "token", "cst", "time", "user"

NUM_LABELS = 800 #一定要记得改

LABEL_OFFSET = 0  # 你的 CodeS 文件标签是 0-based，因此保持 0。
MAX_LEN = 256
BATCH_SIZE = 16
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ANALYSIS_SAMPLES = None

CLASSIFIER_HEAD_TYPE = "auto"  # auto / single / mlp_seq / mlp_named
ENCODER_POOLING = "auto"       # auto / cls / mean；auto: CodeT5/PLBART=mean, CodeBERT/GraphCodeBERT=cls

# Pair-level token/cst RQ3 needs the same sample order. Keep idx strict by default.
# Labels in generated/transformed OOD files may occasionally drift; warn but do not stop.
STRICT_PAIR_ALIGNMENT = True
STRICT_IDX_ALIGNMENT = True
STRICT_LABEL_ALIGNMENT = False
MAX_ANISO_SAMPLES = None
MAX_RANK_SAMPLES = None
ANISO_CHUNK_SIZE = 1024

os.makedirs(SAVE_DIR, exist_ok=True)
RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}.json")
RESULT_CSV = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}.csv")
RESULT_RAW_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_raw_paste_table.tsv")
RESULT_NORMALIZED_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_normalized_paste_table.tsv")
ALL_MODELS_RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_layer_geometry_all_models_{SHIFT_TYPE}.json")


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


def canonical_model_type(model_type: str) -> str:
    model_type_lower = str(model_type).lower()
    aliases = {
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
    if model_type_lower in aliases:
        return aliases[model_type_lower]

    for name in SUPPORTED_MODEL_TYPES:
        if model_type_lower == name.lower():
            return name

    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {SUPPORTED_MODEL_TYPES}")


def configure_model_runtime(model_type: str):
    global MODEL_TYPE, MODEL_PATH, FT_MODEL_WEIGHTS, RESULT_JSON, RESULT_CSV, RESULT_RAW_PASTE_TSV, RESULT_NORMALIZED_PASTE_TSV

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
    FT_MODEL_WEIGHTS = FT_MODEL_WEIGHT_TEMPLATE.format(model_type=MODEL_TYPE)
    RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}.json")
    RESULT_CSV = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}.csv")
    RESULT_RAW_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_raw_paste_table.tsv")
    RESULT_NORMALIZED_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_normalized_paste_table.tsv")


def safe_torch_load(path: str, map_location="cpu"):
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


def strip_prefix(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key.replace(prefix, "", 1)
    return key


def infer_head_type_from_state_dict(state_dict) -> str:
    keys = [strip_prefix(key) for key in state_dict.keys()]
    if any(key.startswith("classification_head.dense.") or key.startswith("classification_head.out_proj.") for key in keys):
        return "mlp_named"
    if any(key.startswith("classifier.0.") or key.startswith("classifier.3.") for key in keys):
        return "mlp_seq"
    if any(key.startswith("classifier.dense.") or key.startswith("classifier.out_proj.") for key in keys):
        return "mlp_named"
    if any(key in ("classifier.weight", "classifier.bias", "score.weight", "score.bias") for key in keys):
        return "single"
    return "single"


def adapt_state_dict_keys(state_dict, head_type: str):
    new_dict = {}
    for key, value in state_dict.items():
        new_key = strip_prefix(key)

        if new_key.startswith("roberta."):
            new_key = new_key.replace("roberta.", "encoder.", 1)
        elif new_key.startswith("bert."):
            new_key = new_key.replace("bert.", "encoder.", 1)
        elif new_key.startswith("transformer."):
            new_key = new_key.replace("transformer.", "encoder.", 1)
        elif new_key.startswith("encoder.encoder."):
            new_key = new_key.replace("encoder.encoder.", "encoder.", 1)

        if new_key.startswith("classification_head.dense."):
            new_key = new_key.replace("classification_head.dense.", "classifier.dense.", 1)
        elif new_key.startswith("classification_head.out_proj."):
            new_key = new_key.replace("classification_head.out_proj.", "classifier.out_proj.", 1)

        if new_key == "score.weight":
            new_key = "classifier.weight"
        elif new_key == "score.bias":
            new_key = "classifier.bias"

        if head_type == "single":
            if new_key.startswith("classifier.out_proj."):
                new_key = new_key.replace("classifier.out_proj.", "classifier.", 1)
            elif new_key.startswith("classifier.3."):
                new_key = new_key.replace("classifier.3.", "classifier.", 1)
        elif head_type == "mlp_seq":
            if new_key.startswith("classifier.dense."):
                new_key = new_key.replace("classifier.dense.", "classifier.0.", 1)
            elif new_key.startswith("classifier.out_proj."):
                new_key = new_key.replace("classifier.out_proj.", "classifier.3.", 1)
        elif head_type == "mlp_named":
            if new_key.startswith("classifier.0."):
                new_key = new_key.replace("classifier.0.", "classifier.dense.", 1)
            elif new_key.startswith("classifier.3."):
                new_key = new_key.replace("classifier.3.", "classifier.out_proj.", 1)

        new_dict[new_key] = value
    return new_dict


def filter_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model", tuple(value.shape) if hasattr(value, "shape") else None, None))
            continue
        if hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append((key, "shape_mismatch", tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible[key] = value
    return compatible, skipped


def get_required_final_classifier_keys(head_type: str) -> List[str]:
    if head_type == "single":
        return ["classifier.weight", "classifier.bias"]
    if head_type == "mlp_named":
        return ["classifier.out_proj.weight", "classifier.out_proj.bias"]
    if head_type == "mlp_seq":
        return ["classifier.3.weight", "classifier.3.bias"]
    raise ValueError(f"Unsupported head_type: {head_type}")


def resolve_pooling_strategy(model_type: str, pooling: str = "auto") -> str:
    pooling = pooling.lower()
    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")
    model_type_lower = model_type.lower()
    return "mean" if ("t5" in model_type_lower or "plbart" in model_type_lower) else "cls"


def get_code(js: dict) -> str:
    return js.get("func") or js.get("code") or js.get("function") or js.get("source") or ""


def get_label(js: dict) -> Optional[int]:
    label = js.get("label")
    if label is None:
        label = js.get("target")
    if label is None:
        return None
    return int(label) - LABEL_OFFSET


def get_raw_label(js: dict) -> Optional[int]:
    label = js.get("label")
    if label is None:
        label = js.get("target")
    if label is None:
        return None
    return int(label)


def get_idx(js: dict, fallback: int) -> str:
    idx = js.get("idx")
    if idx is None:
        idx = js.get("id")
    if idx is None:
        idx = fallback
    return str(idx)


def resolve_data_path(file_name: str) -> str:
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(DATA_DIR, file_name)


def deterministic_subset_indices(n: int, max_n: Optional[int], seed: int = 42) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def tensor_mean_or_nan(tensors: List[torch.Tensor]) -> float:
    if not tensors:
        return float("nan")
    return float(torch.cat(tensors, dim=0).float().mean().item())


# ============================================================
# 3. Dataset
# ============================================================

class CodeDataset(Dataset):
    def __init__(self, file_name: str, tokenizer, max_samples: Optional[int] = None, name: str = "data"):
        self.examples = []
        self.path = resolve_data_path(file_name)
        self.name = name
        self.tokenizer = tokenizer

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Cannot find data file: {self.path}")

        bad_lines = 0
        skipped_label = 0
        skipped_range = 0

        with open(self.path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if max_samples is not None:
            lines = lines[:max_samples]

        for line_idx, line in enumerate(tqdm(lines, desc=f"Loading {self.name}", leave=False)):
            try:
                js = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            code = get_code(js)
            label = get_label(js)
            raw_label = get_raw_label(js)
            if not code:
                bad_lines += 1
                continue
            if label is None:
                skipped_label += 1
                continue
            if label < 0 or label >= NUM_LABELS:
                skipped_range += 1
                continue

            self.examples.append({
                "idx": get_idx(js, line_idx),
                "code": code,
                "label": int(label),
                "raw_label": int(raw_label),
            })

        print("=" * 100)
        print(f"CodeDataset loaded: {self.name}")
        print("path:", self.path)
        print("num_samples:", len(self.examples))
        print("bad_lines:", bad_lines)
        print("skipped_label:", skipped_label)
        print("skipped_label_out_of_range:", skipped_range)
        print("0-based label check: expected=[0,", NUM_LABELS - 1, "]")
        print("label_distribution_top10:", dict(Counter(x["label"] for x in self.examples).most_common(10)))
        print("=" * 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        item = self.examples[index]
        inputs = self.tokenizer(
            item["code"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "idx": item["idx"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "raw_label": torch.tensor(item["raw_label"], dtype=torch.long),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


class AlignedPairDataset(Dataset):
    def __init__(self, id_file: str, ood_file: str, tokenizer, max_samples: Optional[int] = None):
        self.pairs = []
        self.tokenizer = tokenizer
        id_path = resolve_data_path(id_file)
        ood_path = resolve_data_path(ood_file)

        if not os.path.exists(id_path):
            raise FileNotFoundError(f"Cannot find ID file: {id_path}")
        if not os.path.exists(ood_path):
            raise FileNotFoundError(f"Cannot find OOD file: {ood_path}")

        with open(id_path, "r", encoding="utf-8") as f_id:
            id_lines = [line.strip() for line in f_id if line.strip()]
        with open(ood_path, "r", encoding="utf-8") as f_ood:
            ood_lines = [line.strip() for line in f_ood if line.strip()]

        n = min(len(id_lines), len(ood_lines))
        if max_samples is not None:
            n = min(n, max_samples)

        bad_lines = 0
        idx_mismatch = 0
        label_mismatch = 0
        skipped_range = 0

        for i in range(n):
            try:
                js_id = json.loads(id_lines[i])
                js_ood = json.loads(ood_lines[i])
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            id_code = get_code(js_id)
            ood_code = get_code(js_ood)
            if not id_code or not ood_code:
                bad_lines += 1
                continue

            id_idx = get_idx(js_id, i)
            ood_idx = get_idx(js_ood, i)
            id_label = get_label(js_id)
            ood_label = get_label(js_ood)
            id_raw_label = get_raw_label(js_id)
            if id_label is None or ood_label is None:
                bad_lines += 1
                continue
            if id_label < 0 or id_label >= NUM_LABELS or ood_label < 0 or ood_label >= NUM_LABELS:
                skipped_range += 1
                continue

            if id_idx != ood_idx:
                idx_mismatch += 1
            if id_label != ood_label:
                label_mismatch += 1

            if STRICT_PAIR_ALIGNMENT and STRICT_IDX_ALIGNMENT and id_idx != ood_idx:
                raise ValueError(
                    f"Pair idx alignment failed at line {i}: "
                    f"id_idx={id_idx}, ood_idx={ood_idx}, "
                    f"id_label={id_label}, ood_label={ood_label}. "
                    f"For token/cst shifts, ID and OOD files should describe the same sample order."
                )

            if STRICT_PAIR_ALIGNMENT and STRICT_LABEL_ALIGNMENT and id_label != ood_label:
                raise ValueError(
                    f"Pair label alignment failed at line {i}: "
                    f"id_idx={id_idx}, ood_idx={ood_idx}, "
                    f"id_label={id_label}, ood_label={ood_label}. "
                    f"Set STRICT_LABEL_ALIGNMENT=False to keep paired samples and only warn."
                )

            self.pairs.append({
                "idx": id_idx,
                "id_code": id_code,
                "ood_code": ood_code,
                "label": int(id_label),
                "raw_label": int(id_raw_label),
            })

        print("=" * 100)
        print("AlignedPairDataset loaded")
        print("ID file :", id_path)
        print("OOD file:", ood_path)
        print("num_pairs:", len(self.pairs))
        print("bad_lines:", bad_lines)
        print("idx_mismatch:", idx_mismatch)
        print("label_mismatch:", label_mismatch)
        if label_mismatch > 0 and not STRICT_LABEL_ALIGNMENT:
            print(
                "[Warning] Label mismatches were found but kept because "
                "STRICT_LABEL_ALIGNMENT=False. Pair-level RQ3 will use ID labels "
                "for reporting and keep ID/OOD samples paired by idx/order."
            )
        print("skipped_label_out_of_range:", skipped_range)
        print("label_distribution_top10:", dict(Counter(x["label"] for x in self.pairs).most_common(10)))
        print("=" * 100)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        item = self.pairs[index]
        id_inputs = self.tokenizer(item["id_code"], max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        ood_inputs = self.tokenizer(item["ood_code"], max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "idx": item["idx"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "raw_label": torch.tensor(item["raw_label"], dtype=torch.long),
            "id_input_ids": id_inputs["input_ids"].squeeze(0),
            "id_attention_mask": id_inputs["attention_mask"].squeeze(0),
            "ood_input_ids": ood_inputs["input_ids"].squeeze(0),
            "ood_attention_mask": ood_inputs["attention_mask"].squeeze(0),
        }


# ============================================================
# 4. Model
# ============================================================

class NamedMLPHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        nn.init.eye_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def prepare_final_features(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x

    def forward(self, features):
        return self.out_proj(self.prepare_final_features(features))


class CodeClassificationModel(nn.Module):
    def __init__(self, model_path: str, model_type: str, num_labels: int, head_type: str = "single", pooling: str = "auto"):
        super().__init__()
        self.model_type = model_type.lower()
        self.head_type = head_type
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
                raise ValueError(
                    "PLBART model should expose get_encoder() or encoder; "
                    f"please check MODEL_PATH={model_path}"
                )
            hidden_size = getattr(base_model.config, "d_model", None)
            if hidden_size is None:
                hidden_size = getattr(base_model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot infer PLBART hidden size from model config.")
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.hidden_size

        self.hidden_size = hidden_size

        if head_type == "single":
            self.classifier = nn.Linear(hidden_size, num_labels)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        elif head_type == "mlp_named":
            self.classifier = NamedMLPHead(hidden_size, num_labels)
        else:
            self.head_type = "mlp_seq"
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_labels),
            )
            nn.init.eye_(self.classifier[0].weight)
            nn.init.zeros_(self.classifier[0].bias)
            nn.init.xavier_uniform_(self.classifier[3].weight)
            nn.init.zeros_(self.classifier[3].bias)

    def actual_encoder_layer_count(self):
        candidate_paths = [
            ("encoder.block", getattr(getattr(self.encoder, "encoder", None), "block", None)),
            ("encoder.layers", getattr(getattr(self.encoder, "encoder", None), "layers", None)),
            ("encoder.layer", getattr(getattr(self.encoder, "encoder", None), "layer", None)),
            ("block", getattr(self.encoder, "block", None)),
            ("layers", getattr(self.encoder, "layers", None)),
            ("layer", getattr(self.encoder, "layer", None)),
        ]
        for name, layers in candidate_paths:
            if layers is not None and hasattr(layers, "__len__"):
                return name, len(layers)
        return None, None

    def architecture_summary(self):
        config = getattr(self.encoder, "config", None)
        if config is None:
            return {"model_type": self.model_type, "hidden_size": self.hidden_size}

        actual_layer_attr, actual_layer_count = self.actual_encoder_layer_count()
        summary = {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "pooling_strategy": self.pooling_strategy,
            "actual_encoder_layer_attr": actual_layer_attr,
            "actual_encoder_layer_count": actual_layer_count,
        }

        for key in [
            "num_hidden_layers",
            "num_layers",
            "encoder_layers",
            "decoder_layers",
            "num_decoder_layers",
            "d_model",
            "hidden_size",
        ]:
            if hasattr(config, key):
                summary[key] = getattr(config, key)

        if "plbart" in self.model_type or "t5" in self.model_type:
            summary["rq3_layer_scope"] = (
                "encoder_only; decoder layers are not used for this classification "
                "representation analysis"
            )
        else:
            summary["rq3_layer_scope"] = "encoder/backbone hidden states"

        return summary

    def forward_hidden_states(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states

    def pool_layer_hidden(self, hidden_state, attention_mask):
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            summed = torch.sum(hidden_state * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / denom
        return hidden_state[:, 0, :]

    def prepare_classifier_features(self, features):
        if self.head_type == "single":
            return features
        if self.head_type == "mlp_named":
            return self.classifier.prepare_final_features(features)
        x = self.classifier[0](features)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x

    def classify_final_features(self, final_features):
        if self.head_type == "single":
            return self.classifier(final_features)
        if self.head_type == "mlp_named":
            return self.classifier.out_proj(final_features)
        return self.classifier[3](final_features)

    def classify_pooled_features(self, pooled_features):
        return self.classify_final_features(self.prepare_classifier_features(pooled_features))

    # 兼容旧函数名。
    def classify_cls_features(self, cls_features):
        return self.classify_pooled_features(cls_features)

    def load_ft_weights(self, path: str, head_type: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find fine-tuned weights: {path}")

        print(f">>> Loading fine-tuned weights: {path}")
        checkpoint = safe_torch_load(path, map_location="cpu")
        raw_state = unwrap_checkpoint(checkpoint)
        adapted_state = adapt_state_dict_keys(raw_state, head_type=head_type)
        compatible_state, skipped = filter_compatible_state_dict(self, adapted_state)

        missing, unexpected = self.load_state_dict(compatible_state, strict=False)
        own_keys = set(self.state_dict().keys())
        matched = [key for key in compatible_state.keys() if key in own_keys]
        encoder_matched = [key for key in matched if key.startswith("encoder.")]
        classifier_matched = [key for key in matched if key.startswith("classifier.")]

        print(
            f">>> Weights loaded: head_type={head_type}, pooling={self.pooling_strategy}, "
            f"matched={len(matched)}, encoder_matched={len(encoder_matched)}, "
            f"classifier_matched={len(classifier_matched)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
        )

        skipped_classifier = [item for item in skipped if item[0].startswith("classifier.")]
        if skipped_classifier:
            print("[Warning] Skipped classifier keys:", skipped_classifier[:50])
        missing_classifier = [key for key in missing if key.startswith("classifier.")]
        if missing_classifier:
            print("[Warning] Missing classifier keys:", missing_classifier[:50])
        if unexpected:
            print("[Warning] Unexpected keys first 50:", unexpected[:50])

        if len(classifier_matched) == 0:
            raise ValueError("No classifier parameters were matched. Please check CLASSIFIER_HEAD_TYPE and checkpoint keys.")

        required_final_keys = get_required_final_classifier_keys(self.head_type)
        missing_final_keys = [key for key in required_final_keys if key not in matched]
        print(">>> Required final classifier keys:", required_final_keys)
        if missing_final_keys:
            raise RuntimeError(
                "Final classifier output layer was not fully loaded from the fine-tuned checkpoint. "
                f"Missing final classifier keys: {missing_final_keys}."
            )
        print(">>> Final classifier output layer fully loaded.")

        return {
            "matched": len(matched),
            "encoder_matched": len(encoder_matched),
            "classifier_matched": len(classifier_matched),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "skipped": skipped,
            "required_final_classifier_keys": required_final_keys,
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


def get_alignment_label_centroid(id_emb: torch.Tensor, ood_emb: torch.Tensor, id_labels: np.ndarray, ood_labels: np.ndarray) -> Tuple[float, int]:
    if id_emb.size(0) == 0 or ood_emb.size(0) == 0:
        return float("nan"), 0
    id_emb = id_emb.float()
    ood_emb = ood_emb.float()
    common_labels = sorted(
        set(int(x) for x in id_labels.tolist() if int(x) >= 0).intersection(
            set(int(x) for x in ood_labels.tolist() if int(x) >= 0)
        )
    )
    sims = []
    for label in common_labels:
        id_mask = torch.tensor(id_labels == label, dtype=torch.bool)
        ood_mask = torch.tensor(ood_labels == label, dtype=torch.bool)
        if id_mask.sum().item() == 0 or ood_mask.sum().item() == 0:
            continue
        id_centroid = id_emb[id_mask].mean(dim=0, keepdim=True)
        ood_centroid = ood_emb[ood_mask].mean(dim=0, keepdim=True)
        sim = F.cosine_similarity(F.normalize(id_centroid, p=2, dim=1), F.normalize(ood_centroid, p=2, dim=1), dim=1).item()
        sims.append(sim)
    return mean_or_nan(sims), len(sims)


def get_anisotropy(embeddings: torch.Tensor, max_samples: Optional[int] = MAX_ANISO_SAMPLES) -> float:
    n = embeddings.size(0)
    if n <= 1:
        return float("nan")
    idx = deterministic_subset_indices(n, max_samples, seed=SEED)
    x = embeddings[idx].detach().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = F.normalize(x, p=2, dim=1)
    m = x.size(0)
    total = 0.0
    for start in range(0, m, ANISO_CHUNK_SIZE):
        end = min(start + ANISO_CHUNK_SIZE, m)
        sim = torch.matmul(x[start:end], x.T)
        total += float(sim.sum().item())
    total -= float(m)
    return total / float(m * (m - 1))


def get_effective_rank(embeddings: torch.Tensor, max_samples: Optional[int] = MAX_RANK_SAMPLES) -> float:
    """Return raw effective rank of the centered representation matrix."""
    n = embeddings.size(0)
    if n <= 1:
        return float("nan")
    idx = deterministic_subset_indices(n, max_samples, seed=SEED)
    x = embeddings[idx].detach().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(dim=0, keepdim=True)
    x = x.cpu()
    try:
        s = torch.linalg.svdvals(x)
        s = torch.clamp(s, min=1e-12)
        p = s / (s.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return float(torch.exp(entropy).item())
    except Exception as exc:
        print(f"[Warning] Effective rank failed: {repr(exc)}")
        return float("nan")


def get_entropy_values_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return raw predictive entropy for each sample."""
    probs = F.softmax(logits.float(), dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)


# ============================================================
# 6. Feature extraction
# ============================================================

def init_models():
    if not os.path.exists(FT_MODEL_WEIGHTS):
        raise FileNotFoundError(f"Cannot find fine-tuned model weights: {FT_MODEL_WEIGHTS}")

    checkpoint = safe_torch_load(FT_MODEL_WEIGHTS, map_location="cpu")
    raw_state = unwrap_checkpoint(checkpoint)
    head_type = infer_head_type_from_state_dict(raw_state) if CLASSIFIER_HEAD_TYPE == "auto" else CLASSIFIER_HEAD_TYPE

    print("=" * 100)
    print("Model initialization")
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("FT_MODEL_WEIGHTS:", FT_MODEL_WEIGHTS)
    print("inferred/head_type:", head_type)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("=" * 100)

    base_model = CodeClassificationModel(MODEL_PATH, MODEL_TYPE, NUM_LABELS, head_type=head_type, pooling=ENCODER_POOLING).to(DEVICE)
    ft_model = CodeClassificationModel(MODEL_PATH, MODEL_TYPE, NUM_LABELS, head_type=head_type, pooling=ENCODER_POOLING).to(DEVICE)

    print("Model architecture summary:")
    arch_summary = base_model.architecture_summary()
    print(json.dumps(arch_summary, ensure_ascii=False, indent=2))
    actual_layers = arch_summary.get("actual_encoder_layer_count")
    if MODEL_TYPE in {"codeT5", "PLBART"} and actual_layers is not None and actual_layers != 12:
        print(
            "[Warning] The loaded model has "
            f"{actual_layers} actual encoder layers, not 12. "
            f"Please check whether MODEL_PATH={MODEL_PATH} points to the same "
            "12-layer checkpoint/directory used by your previous RQ3 experiment."
        )

    load_info = ft_model.load_ft_weights(FT_MODEL_WEIGHTS, head_type=head_type)

    base_model.eval()
    ft_model.eval()
    return base_model, ft_model, head_type, load_info


@torch.no_grad()
def extract_features_pair(base_model, ft_model, dataloader):
    features = None
    entropy_stats = None
    labels_all = []
    printed_layer_count = False

    for batch in tqdm(dataloader, desc="Extract pair-level features"):
        id_ids = batch["id_input_ids"].to(DEVICE)
        id_mask = batch["id_attention_mask"].to(DEVICE)
        ood_ids = batch["ood_input_ids"].to(DEVICE)
        ood_mask = batch["ood_attention_mask"].to(DEVICE)
        labels_all.append(batch["label"].cpu().numpy())

        hs_base_id = base_model.forward_hidden_states(id_ids, id_mask)
        hs_base_ood = base_model.forward_hidden_states(ood_ids, ood_mask)
        hs_ft_id = ft_model.forward_hidden_states(id_ids, id_mask)
        hs_ft_ood = ft_model.forward_hidden_states(ood_ids, ood_mask)
        num_layers = min(len(hs_base_id), len(hs_base_ood), len(hs_ft_id), len(hs_ft_ood))

        if not printed_layer_count:
            print(
                "[Layer count] hidden_state_entries:",
                {
                    "base_id": len(hs_base_id),
                    "base_ood": len(hs_base_ood),
                    "ft_id": len(hs_ft_id),
                    "ft_ood": len(hs_ft_ood),
                    "used_by_rq3": num_layers,
                },
            )
            print(
                "[Layer count note] Transformers hidden_states may include the embedding output. "
                "For CodeT5/PLBART this script analyzes encoder hidden states only; "
                "decoder layers are not part of the classification representation path."
            )
            printed_layer_count = True

        if features is None:
            features = {"base_id": [[] for _ in range(num_layers)], "base_ood": [[] for _ in range(num_layers)], "ft_id": [[] for _ in range(num_layers)], "ft_ood": [[] for _ in range(num_layers)]}
            entropy_stats = {"entropy_base_ood": [[] for _ in range(num_layers)], "entropy_ft_ood": [[] for _ in range(num_layers)]}

        for layer in range(num_layers):
            b_id = base_model.pool_layer_hidden(hs_base_id[layer], id_mask).detach().cpu()
            b_ood = base_model.pool_layer_hidden(hs_base_ood[layer], ood_mask).detach().cpu()
            f_id = ft_model.pool_layer_hidden(hs_ft_id[layer], id_mask).detach().cpu()
            f_ood = ft_model.pool_layer_hidden(hs_ft_ood[layer], ood_mask).detach().cpu()

            features["base_id"][layer].append(b_id)
            features["base_ood"][layer].append(b_ood)
            features["ft_id"][layer].append(f_id)
            features["ft_ood"][layer].append(f_ood)

            logits_base_ood = ft_model.classify_pooled_features(base_model.pool_layer_hidden(hs_base_ood[layer], ood_mask))
            logits_ft_ood = ft_model.classify_pooled_features(ft_model.pool_layer_hidden(hs_ft_ood[layer], ood_mask))
            entropy_stats["entropy_base_ood"][layer].append(get_entropy_values_from_logits(logits_base_ood).detach().cpu())
            entropy_stats["entropy_ft_ood"][layer].append(get_entropy_values_from_logits(logits_ft_ood).detach().cpu())

    labels = np.concatenate(labels_all, axis=0) if labels_all else np.array([])
    for key in features:
        for layer in range(len(features[key])):
            features[key][layer] = torch.cat(features[key][layer], dim=0)
    return features, entropy_stats, labels


@torch.no_grad()
def extract_features_single(model, ft_head_model, dataloader, split_name: str):
    features = None
    entropy_stats = None
    labels_all = []
    printed_layer_count = False

    for batch in tqdm(dataloader, desc=f"Extract {split_name} features"):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels_all.append(batch["label"].cpu().numpy())
        hidden_states = model.forward_hidden_states(ids, mask)
        num_layers = len(hidden_states)

        if not printed_layer_count:
            print(
                "[Layer count] hidden_state_entries:",
                {
                    "split": split_name,
                    "current_model": len(hidden_states),
                    "used_by_rq3": num_layers,
                },
            )
            print(
                "[Layer count note] Transformers hidden_states may include the embedding output. "
                "For CodeT5/PLBART this script analyzes encoder hidden states only; "
                "decoder layers are not part of the classification representation path."
            )
            printed_layer_count = True

        if features is None:
            features = [[] for _ in range(num_layers)]
            entropy_stats = [[] for _ in range(num_layers)]

        for layer in range(num_layers):
            pooled_feat = model.pool_layer_hidden(hidden_states[layer], mask)
            features[layer].append(pooled_feat.detach().cpu())
            logits = ft_head_model.classify_pooled_features(pooled_feat)
            entropy_stats[layer].append(get_entropy_values_from_logits(logits).detach().cpu())

    labels = np.concatenate(labels_all, axis=0) if labels_all else np.array([])
    for layer in range(len(features)):
        features[layer] = torch.cat(features[layer], dim=0)
    return features, entropy_stats, labels


def extract_features_label_level(base_model, ft_model, id_loader, ood_loader):
    base_id, _, id_labels = extract_features_single(base_model, ft_model, id_loader, "base-id")
    base_ood, ent_base_ood, ood_labels = extract_features_single(base_model, ft_model, ood_loader, "base-ood")
    ft_id, _, id_labels_ft = extract_features_single(ft_model, ft_model, id_loader, "ft-id")
    ft_ood, ent_ft_ood, ood_labels_ft = extract_features_single(ft_model, ft_model, ood_loader, "ft-ood")

    if not np.array_equal(id_labels, id_labels_ft):
        print("[Warning] ID labels differ between base and FT extraction. This should not happen.")
    if not np.array_equal(ood_labels, ood_labels_ft):
        print("[Warning] OOD labels differ between base and FT extraction. This should not happen.")

    return {
        "base_id": base_id,
        "base_ood": base_ood,
        "ft_id": ft_id,
        "ft_ood": ft_ood,
    }, {"entropy_base_ood": ent_base_ood, "entropy_ft_ood": ent_ft_ood}, id_labels, ood_labels


# ============================================================
# 7. Aggregation
# ============================================================

def build_result_row(layer, alignment_mode, alignment_units, alignment_base, alignment_ft, anisotropy_base_id, anisotropy_base_ood, anisotropy_ft_id, anisotropy_ft_ood, rank_base_id, rank_base_ood, rank_ft_id, rank_ft_ood, entropy_base_ood, entropy_ft_ood):
    return {
        "layer": layer,
        "alignment_mode": alignment_mode,
        "alignment_units": alignment_units,
        "alignment_base": alignment_base,
        "alignment_ft": alignment_ft,
        "alignment_delta_ft_minus_base": alignment_ft - alignment_base,
        "anisotropy_base_ood": anisotropy_base_ood,
        "anisotropy_ft_ood": anisotropy_ft_ood,
        "anisotropy_delta_ood_ft_minus_base": anisotropy_ft_ood - anisotropy_base_ood,
        "effective_rank_base_ood": rank_base_ood,
        "effective_rank_ft_ood": rank_ft_ood,
        "effective_rank_delta_ood_ft_minus_base": rank_ft_ood - rank_base_ood,
        "entropy_base_ood": entropy_base_ood,
        "entropy_ft_ood": entropy_ft_ood,
        "entropy_delta_ood_ft_minus_base": entropy_ft_ood - entropy_base_ood,
        "anisotropy_base_id": anisotropy_base_id,
        "anisotropy_ft_id": anisotropy_ft_id,
        "anisotropy_gap_base_ood_minus_id": anisotropy_base_ood - anisotropy_base_id,
        "anisotropy_gap_ft_ood_minus_id": anisotropy_ft_ood - anisotropy_ft_id,
        "effective_rank_base_id": rank_base_id,
        "effective_rank_ft_id": rank_ft_id,
        "effective_rank_gap_base_ood_minus_id": rank_base_ood - rank_base_id,
        "effective_rank_gap_ft_ood_minus_id": rank_ft_ood - rank_ft_id,
    }


def aggregate_results_pair(features, entropy_stats, labels):
    num_layers = len(features["base_id"])
    results = []
    for layer in range(num_layers):
        b_id = features["base_id"][layer]
        b_ood = features["base_ood"][layer]
        f_id = features["ft_id"][layer]
        f_ood = features["ft_ood"][layer]
        results.append(build_result_row(
            layer=layer,
            alignment_mode="pair",
            alignment_units=int(b_id.size(0)),
            alignment_base=get_alignment_pair(b_id, b_ood),
            alignment_ft=get_alignment_pair(f_id, f_ood),
            anisotropy_base_id=get_anisotropy(b_id),
            anisotropy_base_ood=get_anisotropy(b_ood),
            anisotropy_ft_id=get_anisotropy(f_id),
            anisotropy_ft_ood=get_anisotropy(f_ood),
            rank_base_id=get_effective_rank(b_id),
            rank_base_ood=get_effective_rank(b_ood),
            rank_ft_id=get_effective_rank(f_id),
            rank_ft_ood=get_effective_rank(f_ood),
            entropy_base_ood=tensor_mean_or_nan(entropy_stats["entropy_base_ood"][layer]),
            entropy_ft_ood=tensor_mean_or_nan(entropy_stats["entropy_ft_ood"][layer]),
        ))
    return results


def aggregate_results_label_level(features, entropy_stats, id_labels, ood_labels):
    num_layers = len(features["base_id"])
    results = []
    for layer in range(num_layers):
        b_id = features["base_id"][layer]
        b_ood = features["base_ood"][layer]
        f_id = features["ft_id"][layer]
        f_ood = features["ft_ood"][layer]
        alignment_base, units_base = get_alignment_label_centroid(b_id, b_ood, id_labels, ood_labels)
        alignment_ft, units_ft = get_alignment_label_centroid(f_id, f_ood, id_labels, ood_labels)
        results.append(build_result_row(
            layer=layer,
            alignment_mode="label_centroid",
            alignment_units=int(min(units_base, units_ft)),
            alignment_base=alignment_base,
            alignment_ft=alignment_ft,
            anisotropy_base_id=get_anisotropy(b_id),
            anisotropy_base_ood=get_anisotropy(b_ood),
            anisotropy_ft_id=get_anisotropy(f_id),
            anisotropy_ft_ood=get_anisotropy(f_ood),
            rank_base_id=get_effective_rank(b_id),
            rank_base_ood=get_effective_rank(b_ood),
            rank_ft_id=get_effective_rank(f_id),
            rank_ft_ood=get_effective_rank(f_ood),
            entropy_base_ood=tensor_mean_or_nan(entropy_stats["entropy_base_ood"][layer]),
            entropy_ft_ood=tensor_mean_or_nan(entropy_stats["entropy_ft_ood"][layer]),
        ))
    return results


# ============================================================
# 8. Reporting and saving
# ============================================================

def print_result_table(results):
    print("\n" + "=" * 160)
    print(f"RQ3 Layer-wise Representation Diagnostics: {MODEL_TYPE.upper()} | shift={SHIFT_TYPE}".center(160))
    print("=" * 160)
    print(f"{'Layer':<7} | {'Align B/FT/Δ ↑':<28} | {'Aniso-OOD B/FT/Δ ↓':<31} | {'NormEffRank-OOD B/FT/Δ ↑':<34} | {'NormEntropy-OOD B/FT/Δ':<31}")
    print("-" * 160)
    for row in results:
        print(
            f"L{row['layer']:02d}     | "
            f"{row['alignment_base']:.4f} / {row['alignment_ft']:.4f} / {row['alignment_delta_ft_minus_base']:+.4f} | "
            f"{row['anisotropy_base_ood']:.4f} / {row['anisotropy_ft_ood']:.4f} / {row['anisotropy_delta_ood_ft_minus_base']:+.4f} | "
            f"{row['effective_rank_base_ood']:<8.4f} / {row['effective_rank_ft_ood']:<8.4f} / {row['effective_rank_delta_ood_ft_minus_base']:+.4f} | "
            f"{row['entropy_base_ood']:.4f} / {row['entropy_ft_ood']:.4f} / {row['entropy_delta_ood_ft_minus_base']:+.4f}"
        )
    print("=" * 160)


def write_csv(path: str, results: List[dict]):
    fieldnames = list(results[0].keys()) if results else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)



def is_nan_value(value) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except Exception:
        return True


def pct_delta(base, ft):
    """Return percentage change: (FT - Base) / |Base| * 100."""
    try:
        base = float(base)
        ft = float(ft)
    except Exception:
        return float("nan")
    if np.isnan(base) or np.isnan(ft) or abs(base) < 1e-12:
        return float("nan")
    return (ft - base) / abs(base) * 100.0


def fmt_value(value, digits=4):
    if is_nan_value(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def fmt_percent(value, digits=4):
    if is_nan_value(value):
        return "N/A"
    return f"{float(value):.{digits}f}%"


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


def add_plot_normalized_columns(results: List[dict]) -> List[dict]:
    """
    Add CodeSearchNet-style min-max columns for Excel plotting.

    This does not change the metric definitions. Rank and entropy are already
    theoretically normalized metrics; the *_n columns only normalize each
    base/FT curve pair to [0, 1] for plotting within one model+shift.
    """
    metric_pairs = [
        ("alignment", "alignment_base", "alignment_ft"),
        ("anisotropy", "anisotropy_base_ood", "anisotropy_ft_ood"),
        ("rank", "effective_rank_base_ood", "effective_rank_ft_ood"),
        ("entropy", "entropy_base_ood", "entropy_ft_ood"),
    ]
    for metric_name, base_key, ft_key in metric_pairs:
        combined = [row[base_key] for row in results] + [row[ft_key] for row in results]
        normalized = minmax_normalize(combined)
        n = len(results)
        for i, row in enumerate(results):
            row[f"{metric_name}_base_n"] = float(normalized[i])
            row[f"{metric_name}_ft_n"] = float(normalized[n + i])
            row[f"{metric_name}_delta_n_ft_minus_base"] = float(normalized[n + i] - normalized[i])
    return results


def build_excel_raw_rows(results: List[dict]):
    rows = [[
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
            f"L{int(row['layer']):02d}",
            fmt_value(row["alignment_base"], 4),
            fmt_value(row["alignment_ft"], 4),
            fmt_value(row["anisotropy_base_ood"], 4),
            fmt_value(row["anisotropy_ft_ood"], 4),
            fmt_value(row["effective_rank_base_ood"], 4),
            fmt_value(row["effective_rank_ft_ood"], 4),
            fmt_value(row["entropy_base_ood"], 4),
            fmt_value(row["entropy_ft_ood"], 4),
        ])
    return rows


def build_excel_normalized_rows(results: List[dict]):
    rows = [[
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
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)


def print_excel_paste_table(results: List[dict]):
    print("\n" + "=" * 160)
    print(f"EXCEL_COPY_RAW\t{DATASET_NAME}\t{SHIFT_TYPE}\t{MODEL_TYPE}")
    print("=" * 160)
    for row in build_excel_raw_rows(results):
        print("\t".join(row))
    print("=" * 160)

    print("\n" + "=" * 160)
    print(f"EXCEL_COPY_NORMALIZED\t{DATASET_NAME}\t{SHIFT_TYPE}\t{MODEL_TYPE}")
    print("=" * 160)
    for row in build_excel_normalized_rows(results):
        print("\t".join(row))
    print("=" * 160)


def save_results(results: List[dict], head_type: str, load_info: dict, dataset_info: dict):
    meta = {
        "task": TASK_NAME,
        "dataset": DATASET_NAME,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "ft_model_weights": FT_MODEL_WEIGHTS,
        "head_type": head_type,
        "encoder_pooling": load_info.get("pooling_strategy"),
        "load_info": {
            "matched": load_info.get("matched"),
            "encoder_matched": load_info.get("encoder_matched"),
            "classifier_matched": load_info.get("classifier_matched"),
            "num_missing": len(load_info.get("missing", [])),
            "num_unexpected": len(load_info.get("unexpected", [])),
            "num_skipped": len(load_info.get("skipped", [])),
            "required_final_classifier_keys": load_info.get("required_final_classifier_keys"),
        },
        "data_dir": DATA_DIR,
        "id_test_file": ID_TEST_FILE,
        "ood_test_file": OOD_TEST_FILE,
        "shift_type": SHIFT_TYPE,
        "alignment_definition": "pair-level" if SHIFT_TYPE.lower() in {"token", "cst"} else "label-level centroid",
        "num_labels": NUM_LABELS,
        "label_offset": LABEL_OFFSET,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "max_analysis_samples": MAX_ANALYSIS_SAMPLES,
        "max_aniso_samples": MAX_ANISO_SAMPLES,
        "max_rank_samples": MAX_RANK_SAMPLES,
        "dataset_info": dataset_info,
        "metrics": ["alignment", "anisotropy", "effective_rank", "entropy"],
        "metric_definitions": {
            "alignment": "mean one-to-one ID/OOD cosine similarity for token/cst shifts; label-centroid cosine for time/user shifts",
            "anisotropy": "mean off-diagonal pairwise cosine similarity over all OOD representations",
            "effective_rank": "raw exp(entropy(singular-value distribution)) of the centered OOD representation matrix",
            "entropy": "raw predictive entropy averaged over OOD samples",
        },
        "normalization": {
            "raw_metrics": "No theoretical normalization is applied to effective rank or entropy in CSV/JSON/raw TSV.",
            "plot_tsv_min_max": (
                "The *_normalized_paste_table.tsv file additionally min-max normalizes "
                "alignment, anisotropy, effective rank, and entropy "
                "over base+FT layer values for this model+shift only."
            ),
        },
        "notes": [
            "Labels are treated as 0-based by default because LABEL_OFFSET=0.",
            "CodeT5 and PLBART use mean pooling under ENCODER_POOLING=auto; CodeBERT/GraphCodeBERT use CLS pooling.",
            "Alignment is pair-level for token/cst shifts and label-centroid-level for time/user shifts.",
            "Anisotropy is average off-diagonal pairwise cosine similarity.",
            "Effective Rank is the raw effective rank of the centered OOD representation matrix.",
            "Entropy is raw fixed-head probing entropy, so its maximum is log(num_labels) for classification.",
            "Raw CSV/JSON metrics do not use theoretical or per-dataset min-max scaling.",
            "The normalized paste TSV is only for Excel plotting and mirrors the CodeSearchNet RQ3 output style.",
        ],
        "results": results,
        "elapsed_seconds": time.time() - START_TIME,
        "elapsed_readable": format_time(time.time() - START_TIME),
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if results:
        write_csv(RESULT_CSV, results)
        write_tsv(RESULT_RAW_PASTE_TSV, build_excel_raw_rows(results))
        write_tsv(RESULT_NORMALIZED_PASTE_TSV, build_excel_normalized_rows(results))
    print("\nResults saved:")
    print("JSON          :", RESULT_JSON)
    print("CSV           :", RESULT_CSV)
    print("Raw TSV       :", RESULT_RAW_PASTE_TSV)
    print("Normalized TSV:", RESULT_NORMALIZED_PASTE_TSV)


# ============================================================
# 9. Main
# ============================================================

def run_one_model(model_type: str):
    configure_model_runtime(model_type)
    model_start = time.time()

    print("=" * 120)
    print("RQ3: Layer-wise Representation-Space Diagnostics for Code Classification")
    print("=" * 120)
    print("TASK_NAME:", TASK_NAME)
    print("DATASET_NAME:", DATASET_NAME)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("FT_MODEL_WEIGHTS:", FT_MODEL_WEIGHTS)
    print("DATA_DIR:", DATA_DIR)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILE:", OOD_TEST_FILE)
    print("SHIFT_TYPE:", SHIFT_TYPE)
    print("NUM_LABELS:", NUM_LABELS)
    print("LABEL_OFFSET:", LABEL_OFFSET)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model, ft_model, head_type, load_info = init_models()

    shift = SHIFT_TYPE.lower().strip()
    if shift in {"token", "cst"}:
        dataset = AlignedPairDataset(ID_TEST_FILE, OOD_TEST_FILE, tokenizer, max_samples=MAX_ANALYSIS_SAMPLES)
        dataset_info = {
            "mode": "pair",
            "num_pairs": len(dataset),
            "label_distribution_top10": dict(Counter(x["label"] for x in dataset.pairs).most_common(10)),
        }
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        features, entropy_stats, labels = extract_features_pair(base_model, ft_model, loader)
        results = aggregate_results_pair(features, entropy_stats, labels)
    elif shift in {"time", "user"}:
        id_dataset = CodeDataset(ID_TEST_FILE, tokenizer, max_samples=MAX_ANALYSIS_SAMPLES, name="id_test")
        ood_dataset = CodeDataset(OOD_TEST_FILE, tokenizer, max_samples=MAX_ANALYSIS_SAMPLES, name="ood_test")
        dataset_info = {
            "mode": "label_centroid",
            "num_id": len(id_dataset),
            "num_ood": len(ood_dataset),
            "id_label_distribution_top10": dict(Counter(x["label"] for x in id_dataset.examples).most_common(10)),
            "ood_label_distribution_top10": dict(Counter(x["label"] for x in ood_dataset.examples).most_common(10)),
        }
        id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False)
        ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False)
        features, entropy_stats, id_labels, ood_labels = extract_features_label_level(base_model, ft_model, id_loader, ood_loader)
        results = aggregate_results_label_level(features, entropy_stats, id_labels, ood_labels)
    else:
        raise ValueError(f"Unsupported SHIFT_TYPE={SHIFT_TYPE}. Use one of: token, cst, time, user.")

    results = add_plot_normalized_columns(results)
    print_result_table(results)
    print_excel_paste_table(results)
    save_results(results, head_type=head_type, load_info=load_info, dataset_info=dataset_info)
    print("\nFinished.")
    print("Elapsed:", format_time(time.time() - model_start))

    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "ft_model_weights": FT_MODEL_WEIGHTS,
        "head_type": head_type,
        "pooling_strategy": load_info.get("pooling_strategy"),
        "num_layers": len(results),
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
            error_summary = {
                "status": "failed",
                "model_type": run_model_type,
                "error": repr(exc),
            }
            summaries.append(error_summary)
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
