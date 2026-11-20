# -*- coding: utf-8 -*-
"""
CodeSearchNet RQ3: layer-wise retrieval geometry analysis.

This script evaluates five retrieval models with four RQ3 metrics:
  1. Alignment: cosine similarity between NL query and paired OOD code.
  2. Anisotropy: average pairwise cosine similarity among OOD code embeddings.
  3. Effective Rank: centered SVD effective rank of OOD code embeddings.
  4. Contrastive Entropy: raw retrieval distribution entropy over the full OOD code bank.

The model/pooling/checkpoint logic is aligned with CodeSearchNet RQ1/RQ4:
  - CodeBERT / GraphCodeBERT / RoBERTa: AutoModel + CLS pooling
  - CodeT5 / PLBART: encoder + mean pooling
  - RQ1 checkpoint: rq1_outputs/best_codesearch_{language}_{model_type}.bin
"""

import csv
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5Config, T5EncoderModel


# ============================================================
# 1. Config
# ============================================================

LANGUAGE = "python"  # "python" or "java"
SHIFT_TYPE = "cst"  # "token" or "cst"
RUN_ALL_SHIFTS = True
SHIFT_TYPES_TO_RUN = ["token", "cst"]

MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

RUN_ALL_MODELS = True
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
#MODEL_TYPE = "RoBERTa"

DATA_ROOT = "/opt/data/private/CodeSearchNet"
DATA_DIR = os.path.join(DATA_ROOT, LANGUAGE)
RQ1_OUTPUT_DIR = os.path.join(DATA_DIR, "rq1_outputs")
SAVE_DIR = os.path.join(DATA_DIR, "rq3_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

ID_TEST_FILE = os.path.join(DATA_DIR, "id_test.jsonl")
OOD_TEST_FILES = {
    "token": os.path.join(DATA_DIR, "ood_test_token.jsonl"),
    "cst": os.path.join(DATA_DIR, "ood_test_cst.jsonl"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_ANALYSIS_SAMPLES = None
TEMPERATURE_SCALE = 20.0
SEED = 42
ANISO_CHUNK_SIZE = 1024
ENTROPY_QUERY_CHUNK_SIZE = 64

# False gives the 12 encoder blocks for 12-layer encoders. True adds hidden_states[0].
INCLUDE_EMBEDDING_LAYER = True


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
    return f"{int(h)}h {int(m)}m {s:.1f}s"


def canonical_model_type(model_type):
    aliases = {
        "codebert": "codebert",
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
    key = str(model_type).strip()
    if key in MODEL_PATHS:
        return key
    lower = key.lower()
    if lower in aliases:
        return aliases[lower]
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {list(MODEL_PATHS.keys())}")


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


def strip_prefix(key):
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key.replace(prefix, "", 1)
    return key


def adapt_state_dict_keys(state_dict):
    adapted = {}
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
        adapted[new_key] = value
    return adapted


def filter_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue
        compatible[key] = value
    return compatible, skipped


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def get_nl_text(js):
    return (
        js.get("docstring")
        or js.get("doc")
        or js.get("nl")
        or js.get("query")
        or (" ".join(js["docstring_tokens"]) if "docstring_tokens" in js else "")
    )


def get_code_text(js):
    code = js.get("func") or js.get("code") or js.get("function") or ""
    if not code and "code_tokens" in js:
        code = " ".join(js["code_tokens"])
    return code


def build_rq1_best_model_path(model_type):
    model_type = canonical_model_type(model_type)
    return os.path.join(RQ1_OUTPUT_DIR, f"best_codesearch_{LANGUAGE}_{model_type}.bin")


def layer_label(hidden_index):
    if INCLUDE_EMBEDDING_LAYER:
        return f"L{hidden_index:02d}"
    return f"L{hidden_index:02d}"


# ============================================================
# 3. Metrics
# ============================================================

def get_anisotropy(embeddings):
    n = embeddings.size(0)
    if n <= 1:
        return 0.0
    x = embeddings.detach().float().cpu()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = F.normalize(x, p=2, dim=1)
    total = 0.0
    for start in range(0, n, ANISO_CHUNK_SIZE):
        end = min(start + ANISO_CHUNK_SIZE, n)
        sim = torch.matmul(x[start:end], x.T)
        total += float(sim.sum().item())
    total -= float(n)
    return total / float(n * (n - 1))


def get_effective_rank(embeddings):
    n = embeddings.size(0)
    if n <= 1:
        return 1.0
    x = embeddings.detach().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(dim=0, keepdim=True)
    x = x.cpu()
    try:
        s = torch.linalg.svdvals(x)
    except Exception:
        s = torch.svd(x)[1]
    s = torch.clamp(s, min=1e-12)
    p = s / (s.sum() + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    return float(torch.exp(entropy).item())


def get_contrastive_entropy(nl_emb, code_emb, temp=20.0):
    n = nl_emb.size(0)
    if n <= 1:
        return torch.zeros(n, device=nl_emb.device)
    nl = nl_emb.detach().float().cpu()
    code = code_emb.detach().float().cpu()
    nl = torch.nan_to_num(nl, nan=0.0, posinf=0.0, neginf=0.0)
    code = torch.nan_to_num(code, nan=0.0, posinf=0.0, neginf=0.0)
    nl = F.normalize(nl, p=2, dim=1)
    code = F.normalize(code, p=2, dim=1)
    entropies = []
    for start in range(0, n, ENTROPY_QUERY_CHUNK_SIZE):
        end = min(start + ENTROPY_QUERY_CHUNK_SIZE, n)
        logits = torch.matmul(nl[start:end], code.T) * temp
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        entropies.append(entropy)
    return torch.cat(entropies, dim=0)


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(np.mean(values))


# ============================================================
# 4. Dataset
# ============================================================

class PairedRetrievalDataset(Dataset):
    def __init__(self, id_path, ood_path, tokenizer, max_samples=None):
        self.examples = []
        self.tokenizer = tokenizer

        id_raw = read_jsonl(id_path)
        ood_raw = read_jsonl(ood_path)
        n = min(len(id_raw), len(ood_raw))
        if max_samples is not None:
            n = min(n, int(max_samples))

        mismatch_count = 0
        skipped = 0
        for i in range(n):
            js_id = id_raw[i]
            js_ood = ood_raw[i]
            id_idx = str(js_id.get("idx", js_id.get("id", i)))
            ood_idx = str(js_ood.get("idx", js_ood.get("id", i)))
            if id_idx != ood_idx:
                mismatch_count += 1

            nl = get_nl_text(js_id)
            code_id = get_code_text(js_id)
            code_ood = get_code_text(js_ood)
            if not nl or not code_ood:
                skipped += 1
                continue
            if nl in code_ood:
                code_ood = code_ood.replace(nl, "").replace('""""""', "").replace("''''''", "")
            if nl in code_id:
                code_id = code_id.replace(nl, "").replace('""""""', "").replace("''''''", "")

            self.examples.append({
                "idx": id_idx,
                "nl": nl,
                "code_id": code_id,
                "code_ood": code_ood,
                "shift_type": js_ood.get("shift_type", "ood"),
            })

        print(f">>> Loaded paired RQ3 samples: {len(self.examples)} from {id_path} + {ood_path}")
        print(f">>> Skipped: {skipped}; idx mismatches: {mismatch_count}")
        if mismatch_count > 0:
            print("[Warning] ID/OOD files are not perfectly line-aligned by idx. Metrics still run by line order.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        nl = self.tokenizer(
            item["nl"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        code_id = self.tokenizer(
            item["code_id"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        code_ood = self.tokenizer(
            item["code_ood"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "nl_input_ids": nl["input_ids"].squeeze(0),
            "nl_attention_mask": nl["attention_mask"].squeeze(0),
            "id_input_ids": code_id["input_ids"].squeeze(0),
            "id_attention_mask": code_id["attention_mask"].squeeze(0),
            "ood_input_ids": code_ood["input_ids"].squeeze(0),
            "ood_attention_mask": code_ood["attention_mask"].squeeze(0),
            "idx": item["idx"],
            "shift_type": item["shift_type"],
        }


# ============================================================
# 5. Model
# ============================================================

class UnifiedRetrievalEncoder(nn.Module):
    def __init__(self, model_path, model_type, load_pretrained=True):
        super().__init__()
        self.model_type = model_type.lower()

        if "t5" in self.model_type:
            if load_pretrained:
                self.encoder = T5EncoderModel.from_pretrained(model_path)
            else:
                self.encoder = T5EncoderModel(T5Config.from_pretrained(model_path))
            self.hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            if load_pretrained:
                base_model = AutoModel.from_pretrained(model_path)
            else:
                base_model = AutoModel.from_config(AutoConfig.from_pretrained(model_path))
            if hasattr(base_model, "get_encoder"):
                self.encoder = base_model.get_encoder()
            elif hasattr(base_model, "encoder"):
                self.encoder = base_model.encoder
            else:
                raise ValueError("PLBART model should expose get_encoder() or encoder.")
            self.hidden_size = getattr(base_model.config, "d_model", None) or getattr(base_model.config, "hidden_size", None)
        else:
            if load_pretrained:
                self.encoder = AutoModel.from_pretrained(model_path)
            else:
                self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_path))
            self.hidden_size = self.encoder.config.hidden_size

    def forward_hidden_states(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states

    def pool_layer(self, layer_hidden, attention_mask):
        if "t5" in self.model_type or "plbart" in self.model_type:
            mask = attention_mask.unsqueeze(-1).to(layer_hidden.dtype)
            return torch.sum(layer_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return layer_hidden[:, 0, :]


def load_finetuned_model(model_type, model_path, checkpoint_path):
    model = UnifiedRetrievalEncoder(model_path, model_type, load_pretrained=False).to(DEVICE)
    raw_state = unwrap_checkpoint(safe_torch_load(checkpoint_path, map_location=DEVICE))
    adapted_state = adapt_state_dict_keys(raw_state)
    compatible_state, skipped = filter_compatible_state_dict(model, adapted_state)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    matched = [key for key in compatible_state if key in model.state_dict()]
    encoder_matched = [key for key in matched if key.startswith("encoder.")]
    print(
        f">>> Loaded fine-tuned weights: {checkpoint_path}\n"
        f">>> matched={len(matched)}, encoder_matched={len(encoder_matched)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
    )
    if len(encoder_matched) == 0:
        raise RuntimeError("No encoder parameters loaded from RQ1 checkpoint.")
    return model.eval()


# ============================================================
# 6. Analysis
# ============================================================

def init_stats(layer_indices):
    return {
        int(i): {
            "alignment": [],
            "nl_vecs": [],
            "ood_vecs": [],
        }
        for i in layer_indices
    }


@torch.no_grad()
def compute_layer_metrics(model, dataloader, desc):
    model.eval()
    stats = None
    layer_indices = None

    for batch in tqdm(dataloader, desc=desc, leave=False):
        nl_ids = batch["nl_input_ids"].to(DEVICE)
        nl_mask = batch["nl_attention_mask"].to(DEVICE)
        ood_ids = batch["ood_input_ids"].to(DEVICE)
        ood_mask = batch["ood_attention_mask"].to(DEVICE)

        hs_nl = model.forward_hidden_states(nl_ids, nl_mask)
        hs_ood = model.forward_hidden_states(ood_ids, ood_mask)
        n_hidden = min(len(hs_nl), len(hs_ood))

        if layer_indices is None:
            start = 0 if INCLUDE_EMBEDDING_LAYER else 1
            layer_indices = list(range(start, n_hidden))
            stats = init_stats(layer_indices)
            print(f">>> {desc}: hidden_states={n_hidden}, analyzed_layers={len(layer_indices)}, include_embedding={INCLUDE_EMBEDDING_LAYER}")

        for layer_id in layer_indices:
            v_nl = model.pool_layer(hs_nl[layer_id], nl_mask)
            v_ood = model.pool_layer(hs_ood[layer_id], ood_mask)
            alignment = F.cosine_similarity(
                F.normalize(v_nl.float(), p=2, dim=1),
                F.normalize(v_ood.float(), p=2, dim=1),
                dim=1,
            )

            stats[layer_id]["alignment"].extend(alignment.detach().cpu().numpy().tolist())
            stats[layer_id]["nl_vecs"].append(v_nl.detach().cpu())
            stats[layer_id]["ood_vecs"].append(v_ood.detach().cpu())

    if stats is None:
        raise ValueError(f"{desc}: no batches to analyze.")

    summary = {}
    for layer_id, values in stats.items():
        nl_all = torch.cat(values["nl_vecs"], dim=0)
        ood_all = torch.cat(values["ood_vecs"], dim=0)
        entropy = get_contrastive_entropy(nl_all, ood_all, temp=TEMPERATURE_SCALE)
        summary[layer_id] = {
            "alignment": mean_or_zero(values["alignment"]),
            "anisotropy": get_anisotropy(ood_all),
            "effective_rank": get_effective_rank(ood_all),
            "contrastive_entropy": float(entropy.mean().item()) if entropy.numel() else 0.0,
            "num_alignment_samples": len(values["alignment"]),
            "num_batches": len(values["ood_vecs"]),
        }
    return summary


def build_rows(model_type, shift_type, base_metrics, ft_metrics):
    rows = []
    for layer_id in sorted(base_metrics.keys()):
        b = base_metrics[layer_id]
        f = ft_metrics[layer_id]
        row = {
            "model_type": model_type,
            "shift_type": shift_type,
            "layer_index": int(layer_id),
            "layer": layer_label(layer_id),
            "alignment_base": b["alignment"],
            "alignment_finetuned": f["alignment"],
            "alignment_delta_ft_minus_base": f["alignment"] - b["alignment"],
            "anisotropy_base": b["anisotropy"],
            "anisotropy_finetuned": f["anisotropy"],
            "anisotropy_delta_ft_minus_base": f["anisotropy"] - b["anisotropy"],
            "effective_rank_base": b["effective_rank"],
            "effective_rank_finetuned": f["effective_rank"],
            "effective_rank_delta_ft_minus_base": f["effective_rank"] - b["effective_rank"],
            "contrastive_entropy_base": b["contrastive_entropy"],
            "contrastive_entropy_finetuned": f["contrastive_entropy"],
            "contrastive_entropy_delta_ft_minus_base": f["contrastive_entropy"] - b["contrastive_entropy"],
            "num_samples": f["num_alignment_samples"],
            "num_batches": f["num_batches"],
        }
        rows.append(row)
    return add_normalized_columns(rows)


def minmax_normalize(values):
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


def add_normalized_columns(rows):
    """
    Add min-max normalized columns for plotting.

    Normalization is done per model+shift+metric over all layer values from both
    base and fine-tuned models, so base and fine-tuned curves are directly
    comparable within the same plot.
    """
    metrics = ["alignment", "anisotropy", "effective_rank", "contrastive_entropy"]
    for metric in metrics:
        base_key = f"{metric}_base"
        ft_key = f"{metric}_finetuned"
        combined = [row[base_key] for row in rows] + [row[ft_key] for row in rows]
        normalized = minmax_normalize(combined)
        n = len(rows)
        base_norm = normalized[:n]
        ft_norm = normalized[n:]
        for i, row in enumerate(rows):
            row[f"{base_key}_norm"] = float(base_norm[i])
            row[f"{ft_key}_norm"] = float(ft_norm[i])
            row[f"{metric}_delta_norm_ft_minus_base"] = float(ft_norm[i] - base_norm[i])
    return rows


def print_table(model_type, shift_type, rows):
    print("\n" + "=" * 150)
    print(f"RQ3 Results: CodeSearchNet-{LANGUAGE} {shift_type.upper()} Shift - {model_type}")
    print("=" * 150)

    print_excel_tables(model_type, shift_type, rows)


def print_excel_tables(model_type, shift_type, rows):
    print("\n" + "=" * 150)
    print(f"EXCEL_COPY_RAW\tCodeSearchNet-{LANGUAGE}\t{shift_type}\t{model_type}")
    print("=" * 150)
    raw_headers = [
        "dataset",
        "language",
        "model_type",
        "shift_type",
        "layer",
        "layer_index",
        "alignment_base",
        "alignment_finetuned",
        "alignment_delta_ft_minus_base",
        "anisotropy_base",
        "anisotropy_finetuned",
        "anisotropy_delta_ft_minus_base",
        "effective_rank_base",
        "effective_rank_finetuned",
        "effective_rank_delta_ft_minus_base",
        "contrastive_entropy_base",
        "contrastive_entropy_finetuned",
        "contrastive_entropy_delta_ft_minus_base",
        "num_samples",
        "num_batches",
    ]
    print("\t".join(raw_headers))
    for row in rows:
        print("\t".join([
            f"CodeSearchNet-{LANGUAGE}",
            LANGUAGE,
            str(row["model_type"]),
            str(row["shift_type"]),
            str(row["layer"]),
            str(row["layer_index"]),
            f"{row['alignment_base']:.6f}",
            f"{row['alignment_finetuned']:.6f}",
            f"{row['alignment_delta_ft_minus_base']:.6f}",
            f"{row['anisotropy_base']:.6f}",
            f"{row['anisotropy_finetuned']:.6f}",
            f"{row['anisotropy_delta_ft_minus_base']:.6f}",
            f"{row['effective_rank_base']:.6f}",
            f"{row['effective_rank_finetuned']:.6f}",
            f"{row['effective_rank_delta_ft_minus_base']:.6f}",
            f"{row['contrastive_entropy_base']:.6f}",
            f"{row['contrastive_entropy_finetuned']:.6f}",
            f"{row['contrastive_entropy_delta_ft_minus_base']:.6f}",
            str(row["num_samples"]),
            str(row["num_batches"]),
        ]))
    print("=" * 150)

    print("\n" + "=" * 150)
    print(f"EXCEL_COPY_NORMALIZED\tCodeSearchNet-{LANGUAGE}\t{shift_type}\t{model_type}")
    print("=" * 150)
    norm_headers = [
        "dataset",
        "language",
        "model_type",
        "shift_type",
        "layer",
        "layer_index",
        "alignment_base_norm",
        "alignment_finetuned_norm",
        "alignment_delta_norm_ft_minus_base",
        "anisotropy_base_norm",
        "anisotropy_finetuned_norm",
        "anisotropy_delta_norm_ft_minus_base",
        "effective_rank_base_norm",
        "effective_rank_finetuned_norm",
        "effective_rank_delta_norm_ft_minus_base",
        "contrastive_entropy_base_norm",
        "contrastive_entropy_finetuned_norm",
        "contrastive_entropy_delta_norm_ft_minus_base",
        "num_samples",
        "num_batches",
    ]
    print("\t".join(norm_headers))
    for row in rows:
        print("\t".join([
            f"CodeSearchNet-{LANGUAGE}",
            LANGUAGE,
            str(row["model_type"]),
            str(row["shift_type"]),
            str(row["layer"]),
            str(row["layer_index"]),
            f"{row['alignment_base_norm']:.6f}",
            f"{row['alignment_finetuned_norm']:.6f}",
            f"{row['alignment_delta_norm_ft_minus_base']:.6f}",
            f"{row['anisotropy_base_norm']:.6f}",
            f"{row['anisotropy_finetuned_norm']:.6f}",
            f"{row['anisotropy_delta_norm_ft_minus_base']:.6f}",
            f"{row['effective_rank_base_norm']:.6f}",
            f"{row['effective_rank_finetuned_norm']:.6f}",
            f"{row['effective_rank_delta_norm_ft_minus_base']:.6f}",
            f"{row['contrastive_entropy_base_norm']:.6f}",
            f"{row['contrastive_entropy_finetuned_norm']:.6f}",
            f"{row['contrastive_entropy_delta_norm_ft_minus_base']:.6f}",
            str(row["num_samples"]),
            str(row["num_batches"]),
        ]))
    print("=" * 150)
    headers = [
        "Layer",
        "Align_base", "Align_ft",
        "Aniso_base", "Aniso_ft",
        "Rank_base", "Rank_ft",
        "Entropy_base", "Entropy_ft",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join([
            row["layer"],
            f"{row['alignment_base']:.4f}", f"{row['alignment_finetuned']:.4f}",
            f"{row['anisotropy_base']:.4f}", f"{row['anisotropy_finetuned']:.4f}",
            f"{row['effective_rank_base']:.2f}", f"{row['effective_rank_finetuned']:.2f}",
            f"{row['contrastive_entropy_base']:.4f}", f"{row['contrastive_entropy_finetuned']:.4f}",
        ]))
    print("=" * 150)

    print("\n" + "=" * 150)
    print(f"RQ3 Normalized Results: CodeSearchNet-{LANGUAGE} {shift_type.upper()} Shift - {model_type}")
    print("=" * 150)
    norm_headers = [
        "Layer",
        "Align_base_n", "Align_ft_n",
        "Aniso_base_n", "Aniso_ft_n",
        "Rank_base_n", "Rank_ft_n",
        "Entropy_base_n", "Entropy_ft_n",
    ]
    print("\t".join(norm_headers))
    for row in rows:
        print("\t".join([
            row["layer"],
            f"{row['alignment_base_norm']:.4f}", f"{row['alignment_finetuned_norm']:.4f}",
            f"{row['anisotropy_base_norm']:.4f}", f"{row['anisotropy_finetuned_norm']:.4f}",
            f"{row['effective_rank_base_norm']:.4f}", f"{row['effective_rank_finetuned_norm']:.4f}",
            f"{row['contrastive_entropy_base_norm']:.4f}", f"{row['contrastive_entropy_finetuned_norm']:.4f}",
        ]))
    print("=" * 150)


def build_excel_raw_rows(rows):
    table = [[
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
    for row in rows:
        table.append([
            str(row["layer"]),
            f"{row['alignment_base']:.4f}",
            f"{row['alignment_finetuned']:.4f}",
            f"{row['anisotropy_base']:.4f}",
            f"{row['anisotropy_finetuned']:.4f}",
            f"{row['effective_rank_base']:.4f}",
            f"{row['effective_rank_finetuned']:.4f}",
            f"{row['contrastive_entropy_base']:.4f}",
            f"{row['contrastive_entropy_finetuned']:.4f}",
        ])
    return table


def build_excel_normalized_rows(rows):
    table = [[
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
    for row in rows:
        table.append([
            str(row["layer"]),
            f"{row['alignment_base_norm']:.4f}",
            f"{row['alignment_finetuned_norm']:.4f}",
            f"{row['anisotropy_base_norm']:.4f}",
            f"{row['anisotropy_finetuned_norm']:.4f}",
            f"{row['effective_rank_base_norm']:.4f}",
            f"{row['effective_rank_finetuned_norm']:.4f}",
            f"{row['contrastive_entropy_base_norm']:.4f}",
            f"{row['contrastive_entropy_finetuned_norm']:.4f}",
        ])
    return table


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path, table_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(table_rows)


def run_one_model_shift(model_type, shift_type):
    model_type = canonical_model_type(model_type)
    model_path = MODEL_PATHS[model_type]
    ood_file = OOD_TEST_FILES[shift_type]
    checkpoint_path = build_rq1_best_model_path(model_type)

    if not os.path.exists(ID_TEST_FILE):
        raise FileNotFoundError(f"ID test file not found: {ID_TEST_FILE}")
    if not os.path.exists(ood_file):
        raise FileNotFoundError(f"OOD test file not found: {ood_file}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"RQ1 checkpoint not found: {checkpoint_path}")

    print("\n" + "#" * 120)
    print(f"Running RQ3: model={model_type}, shift={shift_type}")
    print("MODEL_PATH:", model_path)
    print("RQ1_CHECKPOINT:", checkpoint_path)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILE:", ood_file)
    print("#" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PairedRetrievalDataset(ID_TEST_FILE, ood_file, tokenizer, max_samples=MAX_ANALYSIS_SAMPLES)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    base_model = UnifiedRetrievalEncoder(model_path, model_type, load_pretrained=True).to(DEVICE).eval()
    base_metrics = compute_layer_metrics(base_model, dataloader, desc=f"{model_type}-{shift_type} base")
    del base_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    ft_model = load_finetuned_model(model_type, model_path, checkpoint_path)
    ft_metrics = compute_layer_metrics(ft_model, dataloader, desc=f"{model_type}-{shift_type} finetuned")
    del ft_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    rows = build_rows(model_type, shift_type, base_metrics, ft_metrics)
    print_table(model_type, shift_type, rows)

    csv_path = os.path.join(SAVE_DIR, f"rq3_codesearchnet_{LANGUAGE}_{shift_type}_{model_type}_layers.csv")
    raw_tsv_path = os.path.join(SAVE_DIR, f"rq3_codesearchnet_{LANGUAGE}_{shift_type}_{model_type}_raw_paste_table.tsv")
    norm_tsv_path = os.path.join(SAVE_DIR, f"rq3_codesearchnet_{LANGUAGE}_{shift_type}_{model_type}_normalized_paste_table.tsv")
    json_path = os.path.join(SAVE_DIR, f"rq3_codesearchnet_{LANGUAGE}_{shift_type}_{model_type}_layers.json")
    write_csv(csv_path, rows)
    write_tsv(raw_tsv_path, build_excel_raw_rows(rows))
    write_tsv(norm_tsv_path, build_excel_normalized_rows(rows))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Code Search",
            "dataset": f"CodeSearchNet-{LANGUAGE}",
            "model_type": model_type,
            "shift_type": shift_type,
            "model_path": model_path,
            "rq1_checkpoint": checkpoint_path,
            "metrics": ["alignment", "anisotropy", "effective_rank", "contrastive_entropy"],
            "metric_definitions": {
                "alignment": "mean one-to-one cosine similarity between each OOD query and its paired OOD code representation",
                "anisotropy": "mean off-diagonal pairwise cosine similarity over all OOD code representations",
                "effective_rank": "raw exp(entropy(singular-value distribution)) of the centered full OOD code representation matrix",
                "contrastive_entropy": "raw retrieval entropy for each OOD query over the full OOD code bank, averaged over OOD queries",
            },
            "normalization": {
                "method": "min-max",
                "scope": "per model + shift + metric, using base and fine-tuned values from all analyzed layers together",
                "columns": "raw columns are preserved in CSV/JSON/raw TSV; *_norm columns are normalized to [0, 1] for plotting; *_delta_norm_ft_minus_base is finetuned_norm - base_norm",
            },
            "include_embedding_layer": INCLUDE_EMBEDDING_LAYER,
            "pooling": "mean for CodeT5/PLBART; CLS for CodeBERT/GraphCodeBERT/RoBERTa",
            "files": {
                "id_test_file": ID_TEST_FILE,
                "ood_test_file": ood_file,
                "csv": csv_path,
                "raw_paste_tsv": raw_tsv_path,
                "normalized_paste_tsv": norm_tsv_path,
                "json": json_path,
            },
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print("CSV saved:", csv_path)
    print("Raw paste TSV saved:", raw_tsv_path)
    print("Normalized paste TSV saved:", norm_tsv_path)
    print("JSON saved:", json_path)
    return rows


def main():
    set_seed(SEED)
    start = time.time()

    models = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    shifts = SHIFT_TYPES_TO_RUN if RUN_ALL_SHIFTS else [SHIFT_TYPE]
    all_results = []

    print("=" * 120)
    print("CodeSearchNet RQ3: five-model layer-wise retrieval geometry analysis")
    print("=" * 120)
    print("LANGUAGE:", LANGUAGE)
    print("MODELS:", models)
    print("SHIFTS:", shifts)
    print("DEVICE:", DEVICE)
    print("INCLUDE_EMBEDDING_LAYER:", INCLUDE_EMBEDDING_LAYER)
    print("MAX_ANALYSIS_SAMPLES:", MAX_ANALYSIS_SAMPLES)
    print("=" * 120)

    for model_type in models:
        for shift_type in shifts:
            try:
                rows = run_one_model_shift(model_type, shift_type)
                all_results.append({
                    "status": "ok",
                    "model_type": canonical_model_type(model_type),
                    "shift_type": shift_type,
                    "rows": rows,
                })
            except Exception as exc:
                if not RUN_ALL_MODELS and not RUN_ALL_SHIFTS:
                    raise
                print(f"[Error] RQ3 failed for model={model_type}, shift={shift_type}: {repr(exc)}")
                all_results.append({
                    "status": "failed",
                    "model_type": str(model_type),
                    "shift_type": str(shift_type),
                    "error": repr(exc),
                })
            finally:
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    summary_path = os.path.join(SAVE_DIR, f"rq3_codesearchnet_{LANGUAGE}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Code Search",
            "dataset": f"CodeSearchNet-{LANGUAGE}",
            "models": models,
            "shifts": shifts,
            "metrics": ["alignment", "anisotropy", "effective_rank", "contrastive_entropy"],
            "metric_definitions": {
                "alignment": "mean one-to-one cosine similarity between each OOD query and its paired OOD code representation",
                "anisotropy": "mean off-diagonal pairwise cosine similarity over all OOD code representations",
                "effective_rank": "raw exp(entropy(singular-value distribution)) of the centered full OOD code representation matrix",
                "contrastive_entropy": "raw retrieval entropy for each OOD query over the full OOD code bank, averaged over OOD queries",
            },
            "normalization": {
                "method": "min-max",
                "scope": "per model + shift + metric, using base and fine-tuned values from all analyzed layers together",
                "columns": "raw columns are preserved in CSV/JSON/raw TSV; *_norm columns are normalized to [0, 1] for plotting; *_delta_norm_ft_minus_base is finetuned_norm - base_norm",
            },
            "include_embedding_layer": INCLUDE_EMBEDDING_LAYER,
            "results": all_results,
            "time_seconds": time.time() - start,
            "time_readable": format_time(time.time() - start),
        }, f, ensure_ascii=False, indent=2)

    print("\nAll RQ3 experiments finished.")
    print("Summary saved:", summary_path)
    print("Total time:", format_time(time.time() - start))


if __name__ == "__main__":
    main()
