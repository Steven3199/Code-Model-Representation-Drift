# -*- coding: utf-8 -*-
"""
CodeUpdateArena RQ3: Layer-wise Representation-Space Diagnostics
for Seq2Seq Program Synthesis under API Evolution Shift.

Adapted from CodeRefinement RQ3 for the seq2seq generation setting.

Four indicators:
  1. ID/OOD Source Alignment:
     for each OOD prompt representation, take its maximum cosine similarity
     to all ID prompt representations, then average over OOD samples.
  2. Source Anisotropy:
     average pairwise cosine similarity among OOD prompt encoder embeddings.
  3. Source Effective Rank:
     centered SVD effective rank of OOD prompt encoder embeddings.
  4. Decoder Token Entropy:
     per-OOD-sample token-level predictive entropy under shifted teacher forcing.

Models:
    CodeT5 / PLBART (AutoModelForSeq2SeqLM)

RQ1 best model is loaded from the saved HuggingFace directory:
    {SAVE_DIR}/{MODEL_TYPE}/best_generation_{MODEL_TYPE}/

Raw and normalized columns are both exported for plotting.
"""

import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ============================================================
# 1. Config
# ============================================================

MODEL_TYPE = "codeT5"  # "codeT5" or "PLBART"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}

RUN_ALL_MODELS = True
MODEL_TYPES_TO_RUN = ["codeT5", "PLBART"]

DATA_DIR = "/opt/data/private/CodeUpdateArena/generation_splits"
ID_TEST_FILE = os.path.join(DATA_DIR, "arena_gen_id_test.jsonl")
OOD_TEST_FILE = os.path.join(DATA_DIR, "arena_gen_ood_test.jsonl")

RQ1_SAVE_DIR = "/opt/data/private/CodeUpdateArena/rq1_generation_outputs"
SAVE_DIR = "/opt/data/private/CodeUpdateArena/rq3_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512
BATCH_SIZE = 4
NUM_WORKERS = 0
MAX_ANALYSIS_SAMPLES = None
ANISO_CHUNK_SIZE = 1024
MIN_ID_OOD_PAIR_COVERAGE = 0.80

INCLUDE_EMBEDDING_LAYER = True

# PLBART settings (aligned with RQ1).
PLBART_SOURCE_LANG = "en_XX"
PLBART_TARGET_LANG = "python"
USE_FORMAT_TOKENS_FOR_PLBART = True
FORMAT_NEWLINE_TOKEN = "<NL>"
MAX_INDENT_LEVEL = 12
FORMAT_INDENT_TOKENS = [f"<INDENT_{i}>" for i in range(MAX_INDENT_LEVEL + 1)]
FORMAT_SPECIAL_TOKENS = [FORMAT_NEWLINE_TOKEN] + FORMAT_INDENT_TOKENS

# Prompt construction (mirrors RQ1).
USE_METADATA_IN_PROMPT = True
PROMPT_FIELD_CANDIDATES = [
    "prompt", "question", "instruction", "problem", "description",
    "task", "input", "context", "nl", "docstring",
]
REFERENCE_FIELD_CANDIDATES = [
    "target", "ref_solution", "reference", "solution", "canonical_solution",
    "answer", "output", "completion", "code",
]

RESULT_JSON_TEMPLATE = "rq3_arena_layer_geometry_{model_type}.json"
RESULT_CSV_TEMPLATE = "rq3_arena_layer_geometry_{model_type}.csv"
RESULT_RAW_PASTE_TSV_TEMPLATE = "rq3_arena_layer_geometry_{model_type}_raw_paste_table.tsv"
RESULT_NORMALIZED_PASTE_TSV_TEMPLATE = "rq3_arena_layer_geometry_{model_type}_normalized_paste_table.tsv"


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed=42):
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


def normalize_model_type(model_type: str) -> str:
    aliases = {
        "codet5": "codeT5",
        "code_t5": "codeT5",
        "code-t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
    }
    key = str(model_type).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Supported: codeT5, PLBART")
    return aliases[key]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")
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


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        items = [stringify_value(v) for v in value if stringify_value(v)]
        return "\n".join(items)
    return str(value).strip()


def clean_code_fence(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    pattern = r"^\s*```(?:\w+)?\s*\n?(.*?)\n?\s*```\s*$"
    match = re.match(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def get_sample_id(item: Dict[str, Any], fallback: int) -> str:
    for key in ("sample_id", "id", "idx", "index"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return str(fallback)


def is_nonempty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (list, dict)):
        return bool(value)
    return bool(value)


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


def build_prompt(item: Dict[str, Any], prompt_field: str) -> str:
    prompt = stringify_value(item.get(prompt_field)).strip()
    if USE_METADATA_IN_PROMPT:
        metadata_lines = []
        for key in ["package", "api_path", "update_type", "update_id"]:
            value = item.get(key)
            if is_nonempty(value):
                metadata_lines.append(f"{key}: {stringify_value(value)}")
        for key in ["update_doc", "api_update", "update_description", "doc", "documentation", "new_doc"]:
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


def infer_field_name(raw_items: List[Dict], candidates: List[str], role: str) -> str:
    field_stats: Dict[str, int] = {}
    for item in raw_items[:min(len(raw_items), 200)]:
        for key in candidates:
            if key in item and stringify_value(item.get(key)):
                field_stats[key] = field_stats.get(key, 0) + 1

    if not field_stats:
        raise ValueError(f"Cannot infer field for {role}. None of {candidates} found.")
    best = max(field_stats, key=field_stats.get)
    return best


# ============================================================
# 3. Data Loading
# ============================================================

def load_arena_examples(path: str, prompt_field: str, reference_field: str, max_samples=None):
    raw_items = read_jsonl(path)
    if max_samples is not None:
        raw_items = raw_items[:max_samples]

    examples = []
    bad = 0
    for idx, item in enumerate(raw_items):
        prompt = build_prompt(item, prompt_field)
        reference = clean_code_fence(stringify_value(item.get(reference_field)))
        if not prompt or not reference:
            bad += 1
            continue
        examples.append({
            "idx": get_sample_id(item, idx),
            "prompt": prompt,
            "reference": reference,
            "raw": item,
        })

    print(f">>> Loaded {len(examples)} valid examples from {path} (bad={bad})")
    return examples


def load_unpaired_id_ood(id_file: str, ood_file: str, max_samples=None):
    raw_id = read_jsonl(id_file)
    raw_ood = read_jsonl(ood_file)

    prompt_field = infer_field_name(raw_id + raw_ood, PROMPT_FIELD_CANDIDATES, "prompt")
    reference_field = infer_field_name(raw_id + raw_ood, REFERENCE_FIELD_CANDIDATES, "reference")
    print(f">>> Inferred fields: prompt={prompt_field}, reference={reference_field}")

    id_examples = load_arena_examples(id_file, prompt_field, reference_field, max_samples=None)
    ood_examples = load_arena_examples(ood_file, prompt_field, reference_field, max_samples=None)

    if max_samples is not None:
        id_examples = id_examples[:max_samples]
        ood_examples = ood_examples[:max_samples]

    print("=" * 100)
    print("CodeUpdateArena unpaired ID/OOD report")
    print("ID file:", id_file)
    print("OOD file:", ood_file)
    print("ID samples used:", len(id_examples))
    print("OOD samples used:", len(ood_examples))
    print("Prompt field:", prompt_field)
    print("Reference field:", reference_field)
    print("Alignment: mean OOD-to-nearest-ID cosine similarity")
    print("=" * 100)

    if not id_examples or not ood_examples:
        raise ValueError("ID and OOD datasets must both contain at least one valid sample.")

    return id_examples, ood_examples, prompt_field, reference_field


# ============================================================
# 4. Paired Dataset
# ============================================================

class ArenaDataset(Dataset):
    def __init__(self, examples, tokenizer, include_target=False):
        self.tokenizer = tokenizer
        self.examples = examples
        self.include_target = include_target

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        source = self.tokenizer(
            item["prompt"],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "idx": item["idx"],
        }

        if not self.include_target:
            return result

        if should_use_format_tokens():
            reference = encode_code_with_format_tokens(item["reference"])
        else:
            reference = item["reference"]

        target = self.tokenizer(
            reference,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        result["labels"] = labels
        return result


# ============================================================
# 5. Metrics
# ============================================================

def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


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


def get_nearest_id_alignment(id_embeddings, ood_embeddings):
    id_x = torch.nan_to_num(
        id_embeddings.detach().float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ood_x = torch.nan_to_num(
        ood_embeddings.detach().float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    id_x = F.normalize(id_x, p=2, dim=1)
    ood_x = F.normalize(ood_x, p=2, dim=1)

    nearest = []
    for start in range(0, ood_x.size(0), ANISO_CHUNK_SIZE):
        end = min(start + ANISO_CHUNK_SIZE, ood_x.size(0))
        similarities = torch.matmul(ood_x[start:end], id_x.T)
        nearest.append(similarities.max(dim=1).values)

    return float(torch.cat(nearest, dim=0).mean().item())


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


def project_decoder_hidden_to_logits(model, hidden):
    sequence_output = hidden

    # Apply decoder final layer norm for T5 and PLBART/BART.
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
    if decoder is not None:
        for attr_name in ("final_layer_norm", "layer_norm"):
            final_ln = getattr(decoder, attr_name, None)
            if final_ln is not None:
                sequence_output = final_ln(sequence_output)
                break

    # T5 applies this scale before the tied LM head. BART/PLBART does not.
    model_family = str(getattr(model.config, "model_type", "")).lower()
    if "t5" in model_family and getattr(model.config, "tie_word_embeddings", False):
        d_model = getattr(model.config, "d_model", None)
        if d_model is not None:
            sequence_output = sequence_output * (float(d_model) ** -0.5)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("model has no lm_head attribute; cannot project decoder hidden states to logits")
    logits = lm_head(sequence_output)
    final_bias = getattr(model, "final_logits_bias", None)
    if final_bias is not None and logits.shape[-1] == final_bias.shape[-1]:
        logits = logits + final_bias
    return logits


def build_shifted_decoder_inputs(model, labels):
    pad_token_id = model.config.pad_token_id
    clean_labels = labels.clone()
    clean_labels[clean_labels == -100] = pad_token_id

    if hasattr(model, "prepare_decoder_input_ids_from_labels"):
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=clean_labels)
    else:
        start_id = model.config.decoder_start_token_id
        if start_id is None:
            start_id = pad_token_id
        decoder_input_ids = clean_labels.new_full(clean_labels.shape, pad_token_id)
        decoder_input_ids[:, 0] = start_id
        decoder_input_ids[:, 1:] = clean_labels[:, :-1]

    decoder_attention_mask = decoder_input_ids.ne(pad_token_id).long()
    return decoder_input_ids, decoder_attention_mask


def token_entropy_from_decoder_layer(model, decoder_hidden, labels):
    decoder_hidden = decoder_hidden.float()
    if not torch.isfinite(decoder_hidden).all():
        raise FloatingPointError(
            "Non-finite decoder hidden states detected. "
            "CodeT5 must run without FP16 autocast for entropy extraction."
        )

    logits = project_decoder_hidden_to_logits(model, decoder_hidden).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError(
            "Non-finite vocabulary logits detected during entropy extraction."
        )

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    token_entropy = -(probs * log_probs).sum(dim=-1)
    if not torch.isfinite(token_entropy).all():
        raise FloatingPointError(
            "Non-finite token entropy detected; refusing to replace it with zero."
        )

    valid_mask = labels.ne(-100)
    denom = valid_mask.sum(dim=1).clamp(min=1)
    return (token_entropy * valid_mask).sum(dim=1) / denom


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(np.mean(values))


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
    metrics = ["alignment", "anisotropy", "effective_rank", "token_entropy"]
    for metric in metrics:
        base_key = f"{metric}_base"
        ft_key = f"{metric}_finetuned"
        combined = [row[base_key] for row in rows] + [row[ft_key] for row in rows]
        normalized = minmax_normalize(combined)
        n = len(rows)
        for i, row in enumerate(rows):
            row[f"{base_key}_norm"] = float(normalized[i])
            row[f"{ft_key}_norm"] = float(normalized[n + i])
            row[f"{metric}_delta_norm_ft_minus_base"] = float(normalized[n + i] - normalized[i])
    return rows


# ============================================================
# 6. Layer-wise Metric Extraction
# ============================================================

def layer_label(hidden_index):
    return f"L{hidden_index:02d}"


def init_stats(layer_indices):
    return {
        int(i): {
            "id_vecs": [],
            "ood_vecs": [],
            "token_entropy": [],
        }
        for i in layer_indices
    }


@torch.no_grad()
def compute_layer_metrics(model, id_dataloader, ood_dataloader, desc):
    model.eval()
    stats = None
    layer_indices = None
    model_family = str(getattr(model.config, "model_type", "")).lower()
    # T5 pre-norm decoder states can overflow under fp16 autocast. Keep CodeT5
    # in fp32 so entropy is not silently converted to zero.
    use_amp = DEVICE.type == "cuda" and "t5" not in model_family
    encoder = model.get_encoder()

    for batch in tqdm(id_dataloader, desc=f"{desc} ID", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            id_encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        id_enc_states = id_encoder_outputs.hidden_states
        if layer_indices is None:
            start = 0 if INCLUDE_EMBEDDING_LAYER else 1
            layer_indices = list(range(start, len(id_enc_states)))
            stats = init_stats(layer_indices)
            print(f">>> {desc}: encoder_hidden_states={len(id_enc_states)}, "
                  f"layers_analyzed={len(layer_indices)}, "
                  f"include_embedding={INCLUDE_EMBEDDING_LAYER}")

        for layer_id in layer_indices:
            id_vec = masked_mean(id_enc_states[layer_id], attention_mask)
            stats[layer_id]["id_vecs"].append(id_vec.detach().cpu())

    for batch in tqdm(ood_dataloader, desc=f"{desc} OOD", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_input_ids, decoder_attention_mask = build_shifted_decoder_inputs(model, labels)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        ood_enc_states = outputs.encoder_hidden_states
        dec_states = outputs.decoder_hidden_states
        n_hidden = min(len(ood_enc_states), len(dec_states))

        for layer_id in layer_indices:
            if layer_id >= n_hidden:
                continue
            ood_vec = masked_mean(ood_enc_states[layer_id], attention_mask)
            entropy = token_entropy_from_decoder_layer(model, dec_states[layer_id], labels)

            stats[layer_id]["token_entropy"].extend(entropy.detach().cpu().numpy().tolist())
            stats[layer_id]["ood_vecs"].append(ood_vec.detach().cpu())

    if stats is None:
        raise ValueError(f"{desc}: no batches to analyze.")

    summary = {}
    for layer_id, values in stats.items():
        id_all = torch.cat(values["id_vecs"], dim=0)
        ood_all = torch.cat(values["ood_vecs"], dim=0)
        summary[layer_id] = {
            "alignment": get_nearest_id_alignment(id_all, ood_all),
            "anisotropy": get_anisotropy(ood_all),
            "effective_rank": get_effective_rank(ood_all),
            "token_entropy": mean_or_zero(values["token_entropy"]),
            "num_samples": int(ood_all.size(0)),
            "num_batches": len(values["ood_vecs"]),
        }

    entropy_values = [
        layer_summary["token_entropy"]
        for layer_summary in summary.values()
    ]
    if entropy_values and max(abs(value) for value in entropy_values) < 1e-10:
        raise RuntimeError(
            f"{desc}: token entropy is zero at every layer. "
            "Check model logits and target labels before using these results."
        )

    return summary


# ============================================================
# 7. Build Results
# ============================================================

def build_rows(model_type, base_metrics, ft_metrics):
    rows = []
    for layer_id in sorted(base_metrics.keys()):
        b = base_metrics[layer_id]
        f = ft_metrics[layer_id]
        rows.append({
            "model_type": model_type,
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
            "token_entropy_base": b["token_entropy"],
            "token_entropy_finetuned": f["token_entropy"],
            "token_entropy_delta_ft_minus_base": f["token_entropy"] - b["token_entropy"],
            "num_samples": f["num_samples"],
            "num_batches": f["num_batches"],
        })
    return add_normalized_columns(rows)


def print_table(title, rows):
    print("\n" + "=" * 150)
    print(title)
    print("=" * 150)
    headers = [
        "Layer", "Align_base", "Align_ft",
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
            f"{row['token_entropy_base']:.4f}", f"{row['token_entropy_finetuned']:.4f}",
        ]))
    print("-" * 150)
    print("Normalized values for plotting")
    print("\t".join([
        "Layer", "Align_base_n", "Align_ft_n",
        "Aniso_base_n", "Aniso_ft_n",
        "Rank_base_n", "Rank_ft_n",
        "Entropy_base_n", "Entropy_ft_n",
    ]))
    for row in rows:
        print("\t".join([
            row["layer"],
            f"{row['alignment_base_norm']:.4f}", f"{row['alignment_finetuned_norm']:.4f}",
            f"{row['anisotropy_base_norm']:.4f}", f"{row['anisotropy_finetuned_norm']:.4f}",
            f"{row['effective_rank_base_norm']:.4f}", f"{row['effective_rank_finetuned_norm']:.4f}",
            f"{row['token_entropy_base_norm']:.4f}", f"{row['token_entropy_finetuned_norm']:.4f}",
        ]))
    print("=" * 150)


def build_excel_raw_rows(rows):
    table = [[
        "Layer", "Align_base", "Align_ft",
        "Aniso_base", "Aniso_ft",
        "Rank_base", "Rank_ft",
        "Entropy_base", "Entropy_ft",
    ]]
    for row in rows:
        table.append([
            str(row["layer"]),
            f"{row['alignment_base']:.4f}", f"{row['alignment_finetuned']:.4f}",
            f"{row['anisotropy_base']:.4f}", f"{row['anisotropy_finetuned']:.4f}",
            f"{row['effective_rank_base']:.4f}", f"{row['effective_rank_finetuned']:.4f}",
            f"{row['token_entropy_base']:.4f}", f"{row['token_entropy_finetuned']:.4f}",
        ])
    return table


def build_excel_normalized_rows(rows):
    table = [[
        "Layer", "Align_base_n", "Align_ft_n",
        "Aniso_base_n", "Aniso_ft_n",
        "Rank_base_n", "Rank_ft_n",
        "Entropy_base_n", "Entropy_ft_n",
    ]]
    for row in rows:
        table.append([
            str(row["layer"]),
            f"{row['alignment_base_norm']:.4f}", f"{row['alignment_finetuned_norm']:.4f}",
            f"{row['anisotropy_base_norm']:.4f}", f"{row['anisotropy_finetuned_norm']:.4f}",
            f"{row['effective_rank_base_norm']:.4f}", f"{row['effective_rank_finetuned_norm']:.4f}",
            f"{row['token_entropy_base_norm']:.4f}", f"{row['token_entropy_finetuned_norm']:.4f}",
        ])
    return table


def print_excel_tables(dataset_name, model_type, rows):
    print("\n" + "=" * 150)
    print(f"EXCEL_COPY_RAW\t{dataset_name}\t{model_type}")
    print("=" * 150)
    for row in build_excel_raw_rows(rows):
        print("\t".join(row))
    print("=" * 150)

    print("\n" + "=" * 150)
    print(f"EXCEL_COPY_NORMALIZED\t{dataset_name}\t{model_type}")
    print("=" * 150)
    for row in build_excel_normalized_rows(rows):
        print("\t".join(row))
    print("=" * 150)


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


# ============================================================
# 8. Model Loading
# ============================================================

def get_rq1_best_model_dir(model_type: str) -> str:
    model_type = normalize_model_type(model_type)
    return os.path.join(RQ1_SAVE_DIR, model_type, f"best_generation_{model_type}")


def configure_plbart(tokenizer, model):
    if MODEL_TYPE != "PLBART":
        return
    try:
        tokenizer.src_lang = PLBART_SOURCE_LANG
    except Exception:
        pass
    try:
        tokenizer.tgt_lang = PLBART_TARGET_LANG
    except Exception:
        pass

    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(PLBART_TARGET_LANG)
    model.config.forced_bos_token_id = None


def init_tokenizer_and_base_model(model_type, model_path):
    # Tokenizer from the base model path.
    if model_type == "PLBART":
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                src_lang=PLBART_SOURCE_LANG,
                tgt_lang=PLBART_TARGET_LANG,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add PLBART format tokens if needed.
    added_format_tokens = False
    if should_use_format_tokens():
        existing_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        tokens_to_add = [tok for tok in FORMAT_SPECIAL_TOKENS if tok not in existing_vocab]
        if tokens_to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            added_format_tokens = True
            print(f">>> Added PLBART format tokens: {tokens_to_add}")
        else:
            print(">>> PLBART format tokens already exist in tokenizer.")

    # Ensure pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Resize if format tokens were added.
    if added_format_tokens:
        base_model.resize_token_embeddings(len(tokenizer))

    if model_type == "PLBART":
        configure_plbart(tokenizer, base_model)
    else:
        if base_model.config.decoder_start_token_id is None:
            base_model.config.decoder_start_token_id = tokenizer.pad_token_id

    base_model.eval()

    print("=" * 100)
    print("Base model / tokenizer initialization")
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("Tokenizer class:", type(tokenizer).__name__)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("decoder_start_token_id:", base_model.config.decoder_start_token_id)
    print("=" * 100)

    return tokenizer, base_model


def load_ft_model(model_type, best_model_dir, tokenizer):
    if not os.path.exists(best_model_dir):
        raise FileNotFoundError(f"RQ1 best model directory not found: {best_model_dir}")

    ft_model = AutoModelForSeq2SeqLM.from_pretrained(best_model_dir).to(DEVICE)

    # Resize if format tokens were added to tokenizer.
    if should_use_format_tokens():
        ft_model.resize_token_embeddings(len(tokenizer))

    if ft_model.config.pad_token_id is None:
        ft_model.config.pad_token_id = tokenizer.pad_token_id

    if model_type == "PLBART":
        configure_plbart(tokenizer, ft_model)
    else:
        if ft_model.config.decoder_start_token_id is None:
            ft_model.config.decoder_start_token_id = tokenizer.pad_token_id

    ft_model.eval()

    print("=" * 100)
    print("Fine-tuned model loaded")
    print("RQ1 best model dir:", best_model_dir)
    print("decoder_start_token_id:", ft_model.config.decoder_start_token_id)
    print("=" * 100)

    return ft_model


# ============================================================
# 9. Run
# ============================================================

def run_one_model(model_type):
    model_type = normalize_model_type(model_type)
    model_path = MODEL_PATHS[model_type]
    rq1_best_dir = get_rq1_best_model_dir(model_type)

    save_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(save_dir, exist_ok=True)

    result_json = os.path.join(save_dir, RESULT_JSON_TEMPLATE.format(model_type=model_type))
    result_csv = os.path.join(save_dir, RESULT_CSV_TEMPLATE.format(model_type=model_type))
    result_raw_tsv = os.path.join(save_dir, RESULT_RAW_PASTE_TSV_TEMPLATE.format(model_type=model_type))
    result_norm_tsv = os.path.join(save_dir, RESULT_NORMALIZED_PASTE_TSV_TEMPLATE.format(model_type=model_type))

    start_time = time.time()

    print("\n" + "#" * 120)
    print(f"CodeUpdateArena RQ3: model={model_type}")
    print("MODEL_PATH:", model_path)
    print("RQ1_BEST_MODEL_DIR:", rq1_best_dir)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILE:", OOD_TEST_FILE)
    print("SAVE_DIR:", save_dir)
    print("#" * 120)

    tokenizer, base_model = init_tokenizer_and_base_model(model_type, model_path)

    id_examples, ood_examples, prompt_field, reference_field = load_unpaired_id_ood(
        ID_TEST_FILE,
        OOD_TEST_FILE,
        max_samples=MAX_ANALYSIS_SAMPLES,
    )
    id_dataset = ArenaDataset(id_examples, tokenizer, include_target=False)
    ood_dataset = ArenaDataset(ood_examples, tokenizer, include_target=True)
    id_dataloader = DataLoader(
        id_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    ood_dataloader = DataLoader(
        ood_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Base model metrics.
    base_metrics = compute_layer_metrics(
        base_model, id_dataloader, ood_dataloader,
        desc=f"{model_type} base",
    )
    del base_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Fine-tuned model metrics.
    ft_model = load_ft_model(model_type, rq1_best_dir, tokenizer)
    ft_metrics = compute_layer_metrics(
        ft_model, id_dataloader, ood_dataloader,
        desc=f"{model_type} finetuned",
    )
    del ft_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    rows = build_rows(model_type, base_metrics, ft_metrics)

    print_table(f"RQ3 Results: CodeUpdateArena API Evolution Shift - {model_type}", rows)
    print_excel_tables("CodeUpdateArena", model_type, rows)

    write_csv(result_csv, rows)
    write_tsv(result_raw_tsv, build_excel_raw_rows(rows))
    write_tsv(result_norm_tsv, build_excel_normalized_rows(rows))

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Program Synthesis under API Evolution",
            "dataset": "CodeUpdateArena",
            "model_type": model_type,
            "model_path": model_path,
            "rq1_best_model_dir": rq1_best_dir,
            "id_test_file": ID_TEST_FILE,
            "ood_test_file": OOD_TEST_FILE,
            "metrics": ["alignment", "anisotropy", "effective_rank", "token_entropy"],
            "metric_definitions": {
                "alignment": "for each OOD prompt encoder representation, maximum cosine similarity to all ID prompt encoder representations, averaged over OOD samples",
                "anisotropy": "average pairwise cosine similarity among OOD prompt encoder representations",
                "effective_rank": "centered SVD effective rank of OOD prompt encoder representations",
                "token_entropy": "per-OOD-sample decoder token predictive entropy under shifted teacher forcing, averaged over valid target tokens and then over OOD samples",
            },
            "normalization": {
                "method": "min-max",
                "scope": "per model + metric, using base and fine-tuned values from all analyzed layers together",
                "columns": "raw columns are preserved in CSV/JSON/raw TSV; *_norm columns normalized to [0, 1]",
            },
            "include_embedding_layer": INCLUDE_EMBEDDING_LAYER,
            "prompt_field": prompt_field,
            "reference_field": reference_field,
            "num_id_samples": len(id_dataset),
            "num_ood_samples": len(ood_dataset),
            "files": {
                "csv": result_csv,
                "raw_paste_tsv": result_raw_tsv,
                "normalized_paste_tsv": result_norm_tsv,
                "json": result_json,
            },
            "rows": rows,
            "elapsed_seconds": time.time() - start_time,
            "elapsed_readable": format_time(time.time() - start_time),
        }, f, ensure_ascii=False, indent=2)

    print("\nResults saved:")
    print("  JSON:", result_json)
    print("  CSV:", result_csv)
    print("  Raw TSV:", result_raw_tsv)
    print("  Normalized TSV:", result_norm_tsv)
    print("  Elapsed:", format_time(time.time() - start_time))

    return rows


def main():
    global MODEL_TYPE, USE_FORMAT_TOKENS_FOR_PLBART

    start = time.time()
    models = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [normalize_model_type(MODEL_TYPE)]

    for mt in models:
        # Temporarily set MODEL_TYPE so should_use_format_tokens() works correctly.
        saved_model_type = MODEL_TYPE
        MODEL_TYPE = normalize_model_type(mt)
        try:
            run_one_model(mt)
        finally:
            MODEL_TYPE = saved_model_type
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 120)
    print(f"RQ3 CodeUpdateArena all models finished. Total elapsed: {format_time(time.time() - start)}")
    print("=" * 120)


if __name__ == "__main__":
    main()
