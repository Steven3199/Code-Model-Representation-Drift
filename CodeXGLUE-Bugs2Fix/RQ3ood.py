# -*- coding: utf-8 -*-
"""
CodeRefinement / Bugs2Fix RQ3: generation-oriented layer-wise geometry analysis.

This adapts the four RQ3 indicators to seq2seq code repair:
  1. ID/OOD Source Alignment:
     one-to-one cosine similarity between ID buggy/source representation and
     semantically corresponding OOD buggy/source representation.
  2. Source Anisotropy:
     average pairwise cosine similarity among OOD source encoder embeddings.
  3. Source Effective Rank:
     centered SVD effective rank of OOD source encoder embeddings.
  4. Decoder Token Entropy:
     per-OOD-sample token-level predictive entropy under shifted teacher forcing.

Raw and normalized columns are both exported for plotting.
"""

import csv
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ============================================================
# 1. Config
# ============================================================

MODEL_TYPE = "codeT5"  # choices: "codeT5", "PLBART", "all"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
RUN_ALL_MODELS = True
MODEL_TYPES_TO_RUN = ["codeT5", "PLBART"]

DATA_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement"
RQ1_OUTPUT_ROOT = os.path.join(DATA_ROOT, "rq1_outputs")
SAVE_ROOT = os.path.join(DATA_ROOT, "rq3_outputs")

# None scans all Bugs2Fix subsets with required files. Or use ["small"].
SUBSET_DIRS = None
SHIFT_TYPE = "token"  # "token" or "cst"
RUN_ALL_SHIFTS = True
SHIFT_TYPES_TO_RUN = ["token", "cst"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
NUM_WORKERS = 0
MAX_ANALYSIS_SAMPLES = None
SEED = 42
ANISO_CHUNK_SIZE = 1024
MIN_ID_OOD_PAIR_COVERAGE = 0.95

# False gives encoder/decoder block layers only; True includes embedding layer.
INCLUDE_EMBEDDING_LAYER = True

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
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.1f}s"


def normalize_model_type(model_type):
    lower = str(model_type).lower()
    if lower == "all":
        return "all"
    for name in MODEL_PATHS:
        if lower == name.lower():
            return name
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {list(MODEL_PATHS.keys()) + ['all']}")


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_source(js):
    return js.get("source") or js.get("buggy") or js.get("src") or js.get("input") or ""


def get_target(js):
    return js.get("target") or js.get("fixed") or js.get("tgt") or js.get("output") or ""


def get_sample_key(js, fallback_index):
    for field in ("idx", "id", "sample_id", "example_id"):
        value = js.get(field)
        if value is not None and str(value) != "":
            return str(value)
    return f"position_{fallback_index}"


def discover_subset_dirs():
    subset_infos = []
    if SUBSET_DIRS is not None:
        candidates = [os.path.join(DATA_ROOT, name) for name in SUBSET_DIRS]
    else:
        candidates = []
        required = {"id_test.jsonl", "ood_test_token.jsonl", "ood_test_cst.jsonl"}
        for root, _, files in os.walk(DATA_ROOT):
            if required.issubset(set(files)):
                candidates.append(root)

    for path in sorted(set(candidates)):
        id_test_file = os.path.join(path, "id_test.jsonl")
        token_file = os.path.join(path, "ood_test_token.jsonl")
        cst_file = os.path.join(path, "ood_test_cst.jsonl")
        if not all(os.path.exists(x) for x in [id_test_file, token_file, cst_file]):
            continue
        subset_infos.append({
            "name": os.path.basename(path.rstrip("/")),
            "dir": path,
            "id_test_file": id_test_file,
            "ood_files": {"token": token_file, "cst": cst_file},
        })
    return subset_infos


def build_rq1_best_model_path(model_type, subset_name):
    return os.path.join(RQ1_OUTPUT_ROOT, model_type, subset_name, f"best_bugs2fix_{model_type}.bin")


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


def load_seq2seq_model(model_type, model_path, checkpoint_path=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RQ1 checkpoint not found: {checkpoint_path}")
        model.load_state_dict(safe_torch_load(checkpoint_path, map_location=DEVICE))
        print(">>> Loaded RQ1 checkpoint:", checkpoint_path)
    return model.eval()


def layer_label(hidden_index):
    return f"L{hidden_index:02d}"


# ============================================================
# 3. Dataset
# ============================================================

def read_bugs2fix_examples(path, max_samples=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")

    examples = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc=f"Reading {os.path.basename(path)}", leave=False)):
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
            examples.append({
                "idx": get_sample_key(js, line_idx),
                "source": source,
                "target": target,
                "shift_type": js.get("shift_type", "id"),
                "raw": js,
            })

    if max_samples is not None and len(examples) > max_samples:
        examples = examples[:max_samples]
    print(f">>> Loaded {len(examples)} valid Bugs2Fix examples from {path}; skipped={skipped}")
    return examples


def align_id_ood_examples(id_file, ood_file, max_samples=None):
    id_examples = read_bugs2fix_examples(id_file, max_samples=None)
    ood_examples = read_bugs2fix_examples(ood_file, max_samples=None)
    id_map = {}
    duplicate_id_keys = 0

    for ex in id_examples:
        if ex["idx"] in id_map:
            duplicate_id_keys += 1
            continue
        id_map[ex["idx"]] = ex

    pairs = []
    unmatched = []
    for ood_ex in ood_examples:
        key = ood_ex["idx"]
        if key not in id_map:
            unmatched.append(key)
            continue
        pairs.append((id_map[key], ood_ex))

    coverage = len(pairs) / max(1, len(ood_examples))
    print("=" * 100)
    print("CodeRefinement ID/OOD pairing report")
    print("ID file:", id_file)
    print("OOD file:", ood_file)
    print("ID valid samples:", len(id_examples))
    print("OOD valid samples:", len(ood_examples))
    print("Aligned pairs:", len(pairs))
    print("Pair coverage:", f"{coverage:.2%}")
    print("Duplicate ID keys:", duplicate_id_keys)
    print("Unmatched OOD keys:", len(unmatched))

    if coverage < MIN_ID_OOD_PAIR_COVERAGE and len(id_examples) == len(ood_examples):
        print("[Warning] Key coverage is low; using positional ID/OOD alignment because valid sample counts match.")
        pairs = list(zip(id_examples, ood_examples))
        coverage = len(pairs) / max(1, len(ood_examples))
        print("Positional pair coverage:", f"{coverage:.2%}")

    if coverage < MIN_ID_OOD_PAIR_COVERAGE:
        raise RuntimeError(
            f"ID/OOD pair coverage is only {coverage:.2%}, lower than required {MIN_ID_OOD_PAIR_COVERAGE:.2%}."
        )

    if max_samples is not None and len(pairs) > max_samples:
        pairs = pairs[:max_samples]
    print("Pairs used for RQ3:", len(pairs))
    print("=" * 100)
    return pairs


class PairedBugs2FixDataset(Dataset):
    def __init__(self, id_file, ood_file, tokenizer, max_samples=None):
        self.tokenizer = tokenizer
        self.pairs = align_id_ood_examples(id_file, ood_file, max_samples=max_samples)
        shift_counter = Counter(ood_ex["shift_type"] for _, ood_ex in self.pairs)
        print(f">>> Paired Bugs2FixDataset loaded: {len(self.pairs)} pairs")
        print(f">>> OOD shift_type distribution: {dict(shift_counter)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        id_item, ood_item = self.pairs[i]
        id_source = self.tokenizer(
            id_item["source"],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ood_source = self.tokenizer(
            ood_item["source"],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ood_target = self.tokenizer(
            ood_item["target"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = ood_target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "id_input_ids": id_source["input_ids"].squeeze(0),
            "id_attention_mask": id_source["attention_mask"].squeeze(0),
            "ood_input_ids": ood_source["input_ids"].squeeze(0),
            "ood_attention_mask": ood_source["attention_mask"].squeeze(0),
            "labels": labels,
            "idx": ood_item["idx"],
            "shift_type": ood_item["shift_type"],
        }


# ============================================================
# 4. Metrics
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

    # Apply decoder final layer norm for T5/codeT5 (and PLBART/BART if present).
    # hidden_states from output_hidden_states=True are pre-layer-norm, but
    # lm_head expects post-layer-norm inputs. Without this step codeT5
    # entropy values will be incorrect / NaN.
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
    if decoder is not None:
        for attr_name in ("final_layer_norm", "layer_norm"):
            final_ln = getattr(decoder, attr_name, None)
            if final_ln is not None:
                sequence_output = final_ln(sequence_output)
                break

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
            "Non-finite decoder hidden states detected during entropy extraction."
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


def init_stats(layer_indices):
    return {
        int(i): {
            "alignment": [],
            "enc_vecs": [],
            "token_entropy": [],
        }
        for i in layer_indices
    }


@torch.no_grad()
def compute_layer_metrics(model, dataloader, desc):
    model.eval()
    stats = None
    layer_indices = None
    encoder = model.get_encoder()

    for batch in tqdm(dataloader, desc=desc, leave=False):
        id_input_ids = batch["id_input_ids"].to(DEVICE)
        id_attention_mask = batch["id_attention_mask"].to(DEVICE)
        ood_input_ids = batch["ood_input_ids"].to(DEVICE)
        ood_attention_mask = batch["ood_attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_input_ids, decoder_attention_mask = build_shifted_decoder_inputs(model, labels)

        id_encoder_outputs = encoder(
            input_ids=id_input_ids,
            attention_mask=id_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        outputs = model(
            input_ids=ood_input_ids,
            attention_mask=ood_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        id_enc_states = id_encoder_outputs.hidden_states
        ood_enc_states = outputs.encoder_hidden_states
        dec_states = outputs.decoder_hidden_states
        n_hidden = min(len(id_enc_states), len(ood_enc_states), len(dec_states))

        if layer_indices is None:
            start = 0 if INCLUDE_EMBEDDING_LAYER else 1
            layer_indices = list(range(start, n_hidden))
            stats = init_stats(layer_indices)
            print(f">>> {desc}: paired ID/OOD source hidden_states={n_hidden}, analyzed_layers={len(layer_indices)}, include_embedding={INCLUDE_EMBEDDING_LAYER}")

        for layer_id in layer_indices:
            id_vec = masked_mean(id_enc_states[layer_id], id_attention_mask)
            ood_vec = masked_mean(ood_enc_states[layer_id], ood_attention_mask)
            align = F.cosine_similarity(
                F.normalize(id_vec.float(), p=2, dim=1),
                F.normalize(ood_vec.float(), p=2, dim=1),
                dim=1,
            )
            entropy = token_entropy_from_decoder_layer(model, dec_states[layer_id], labels)

            stats[layer_id]["alignment"].extend(align.detach().cpu().numpy().tolist())
            stats[layer_id]["token_entropy"].extend(entropy.detach().cpu().numpy().tolist())
            stats[layer_id]["enc_vecs"].append(ood_vec.detach().cpu())

    if stats is None:
        raise ValueError(f"{desc}: no batches to analyze.")

    summary = {}
    for layer_id, values in stats.items():
        enc_all = torch.cat(values["enc_vecs"], dim=0)
        summary[layer_id] = {
            "alignment": mean_or_zero(values["alignment"]),
            "anisotropy": get_anisotropy(enc_all),
            "effective_rank": get_effective_rank(enc_all),
            "token_entropy": mean_or_zero(values["token_entropy"]),
            "num_samples": len(values["alignment"]),
            "num_batches": len(values["enc_vecs"]),
        }

    entropy_values = [
        layer_summary["token_entropy"]
        for layer_summary in summary.values()
    ]
    if entropy_values and max(abs(value) for value in entropy_values) < 1e-10:
        raise RuntimeError(
            f"{desc}: token entropy is zero at every layer. "
            "Check decoder logits and target labels before using these results."
        )

    return summary


def build_rows(model_type, subset_name, shift_type, base_metrics, ft_metrics):
    rows = []
    for layer_id in sorted(base_metrics.keys()):
        b = base_metrics[layer_id]
        f = ft_metrics[layer_id]
        rows.append({
            "model_type": model_type,
            "subset": subset_name,
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
            f"{row['token_entropy_base']:.4f}", f"{row['token_entropy_finetuned']:.4f}",
        ]))
    print("-" * 150)
    print("Normalized values for plotting")
    print("\t".join([
        "Layer",
        "Align_base_n", "Align_ft_n",
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
            f"{row['token_entropy_base']:.4f}",
            f"{row['token_entropy_finetuned']:.4f}",
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
            f"{row['token_entropy_base_norm']:.4f}",
            f"{row['token_entropy_finetuned_norm']:.4f}",
        ])
    return table


def print_excel_tables(dataset_name, shift_type, model_type, rows):
    print("\n" + "=" * 150)
    print(f"EXCEL_COPY_RAW\t{dataset_name}\t{shift_type}\t{model_type}")
    print("=" * 150)
    for row in build_excel_raw_rows(rows):
        print("\t".join(row))
    print("=" * 150)

    print("\n" + "=" * 150)
    print(f"EXCEL_COPY_NORMALIZED\t{dataset_name}\t{shift_type}\t{model_type}")
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
# 5. Run
# ============================================================

def run_one(model_type, subset_info, shift_type):
    model_type = normalize_model_type(model_type)
    subset_name = subset_info["name"]
    model_path = MODEL_PATHS[model_type]
    id_file = subset_info["id_test_file"]
    ood_file = subset_info["ood_files"][shift_type]
    checkpoint = build_rq1_best_model_path(model_type, subset_name)
    save_dir = os.path.join(SAVE_ROOT, model_type, subset_name)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "#" * 120)
    print(f"Running CodeRefinement RQ3: model={model_type}, subset={subset_name}, shift={shift_type}")
    print("MODEL_PATH:", model_path)
    print("RQ1_CHECKPOINT:", checkpoint)
    print("ID_FILE:", id_file)
    print("OOD_FILE:", ood_file)
    print("SAVE_DIR:", save_dir)
    print("#" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = PairedBugs2FixDataset(id_file, ood_file, tokenizer, max_samples=MAX_ANALYSIS_SAMPLES)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    base_model = load_seq2seq_model(model_type, model_path, checkpoint_path=None)
    configure_special_tokens(tokenizer, base_model)
    base_metrics = compute_layer_metrics(base_model, dataloader, desc=f"{model_type}-{subset_name}-{shift_type} base")
    del base_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    ft_model = load_seq2seq_model(model_type, model_path, checkpoint_path=checkpoint)
    configure_special_tokens(tokenizer, ft_model)
    ft_metrics = compute_layer_metrics(ft_model, dataloader, desc=f"{model_type}-{subset_name}-{shift_type} finetuned")
    del ft_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    rows = build_rows(model_type, subset_name, shift_type, base_metrics, ft_metrics)
    print_table(f"RQ3 Results: CodeRefinement {subset_name.upper()} {shift_type.upper()} - {model_type}", rows)
    print_excel_tables(f"CodeRefinement-{subset_name}", shift_type, model_type, rows)

    csv_path = os.path.join(save_dir, f"rq3_coderefinement_{subset_name}_{shift_type}_{model_type}_layers.csv")
    raw_tsv_path = os.path.join(save_dir, f"rq3_coderefinement_{subset_name}_{shift_type}_{model_type}_raw_paste_table.tsv")
    norm_tsv_path = os.path.join(save_dir, f"rq3_coderefinement_{subset_name}_{shift_type}_{model_type}_normalized_paste_table.tsv")
    json_path = os.path.join(save_dir, f"rq3_coderefinement_{subset_name}_{shift_type}_{model_type}_layers.json")
    write_csv(csv_path, rows)
    write_tsv(raw_tsv_path, build_excel_raw_rows(rows))
    write_tsv(norm_tsv_path, build_excel_normalized_rows(rows))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Code Repair / Code Refinement",
            "dataset": "CodeXGLUE Bugs2Fix",
            "subset": subset_name,
            "model_type": model_type,
            "shift_type": shift_type,
            "model_path": model_path,
            "rq1_checkpoint": checkpoint,
            "metrics": ["alignment", "anisotropy", "effective_rank", "token_entropy"],
            "metric_definitions": {
                "alignment": "one-to-one cosine similarity between paired ID source encoder representation and OOD source encoder representation",
                "anisotropy": "average pairwise cosine similarity among OOD source encoder representations",
                "effective_rank": "centered SVD effective rank of OOD source encoder representations",
                "token_entropy": "per-OOD-sample decoder token predictive entropy under shifted teacher forcing, averaged over valid target tokens and then over OOD samples",
            },
            "normalization": {
                "method": "min-max",
                "scope": "per model + subset + shift + metric, using base and fine-tuned values from all analyzed layers together",
                "columns": "raw columns are preserved in CSV/JSON/raw TSV; *_norm columns are normalized to [0, 1] for plotting; *_delta_norm_ft_minus_base is finetuned_norm - base_norm",
            },
            "include_embedding_layer": INCLUDE_EMBEDDING_LAYER,
            "files": {
                "id_file": id_file,
                "ood_file": ood_file,
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
    models = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [normalize_model_type(MODEL_TYPE)]
    shifts = SHIFT_TYPES_TO_RUN if RUN_ALL_SHIFTS else [SHIFT_TYPE]
    subsets = discover_subset_dirs()
    if not subsets:
        raise ValueError("No CodeRefinement subsets found with id_test/ood_test_token/ood_test_cst files.")

    print("=" * 120)
    print("CodeRefinement RQ3: generation-oriented layer-wise geometry analysis")
    print("=" * 120)
    print("MODELS:", models)
    print("SHIFTS:", shifts)
    print("SUBSETS:", [x["name"] for x in subsets])
    print("DEVICE:", DEVICE)
    print("MAX_ANALYSIS_SAMPLES:", MAX_ANALYSIS_SAMPLES)
    print("INCLUDE_EMBEDDING_LAYER:", INCLUDE_EMBEDDING_LAYER)
    print("=" * 120)

    all_results = []
    for model_type in models:
        for subset_info in subsets:
            for shift_type in shifts:
                try:
                    rows = run_one(model_type, subset_info, shift_type)
                    all_results.append({
                        "status": "ok",
                        "model_type": model_type,
                        "subset": subset_info["name"],
                        "shift_type": shift_type,
                        "rows": rows,
                    })
                except Exception as exc:
                    if not RUN_ALL_MODELS and len(subsets) == 1 and not RUN_ALL_SHIFTS:
                        raise
                    print(f"[Error] RQ3 failed for model={model_type}, subset={subset_info['name']}, shift={shift_type}: {repr(exc)}")
                    all_results.append({
                        "status": "failed",
                        "model_type": model_type,
                        "subset": subset_info["name"],
                        "shift_type": shift_type,
                        "error": repr(exc),
                    })
                finally:
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

    summary_path = os.path.join(SAVE_ROOT, "rq3_coderefinement_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Code Repair / Code Refinement",
            "dataset": "CodeXGLUE Bugs2Fix",
            "models": models,
            "shifts": shifts,
            "subsets": [x["name"] for x in subsets],
            "metrics": ["alignment", "anisotropy", "effective_rank", "token_entropy"],
            "metric_definitions": {
                "alignment": "one-to-one cosine similarity between paired ID source encoder representation and OOD source encoder representation",
                "anisotropy": "average pairwise cosine similarity among all OOD source encoder representations",
                "effective_rank": "centered SVD effective rank of all OOD source encoder representations",
                "token_entropy": "per-OOD-sample decoder token predictive entropy under shifted teacher forcing, averaged over valid target tokens and then over OOD samples",
            },
            "normalization": "min-max per model+subset+shift+metric over base and fine-tuned layer values; raw TSV and normalized TSV are both exported",
            "include_embedding_layer": INCLUDE_EMBEDDING_LAYER,
            "results": all_results,
            "time_seconds": time.time() - start,
            "time_readable": format_time(time.time() - start),
        }, f, ensure_ascii=False, indent=2)

    print("\nAll CodeRefinement RQ3 experiments finished.")
    print("Summary saved:", summary_path)
    print("Total time:", format_time(time.time() - start))


if __name__ == "__main__":
    main()
