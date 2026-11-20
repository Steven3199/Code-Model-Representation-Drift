# -*- coding: utf-8 -*-
"""
CodeXGLUE CodeTrans RQ3: layer-wise representation diagnosis.

Generation-task RQ3 metrics adapted to code translation:
  1. Alignment:
     one-to-one cosine similarity between paired ID source and OOD source
     encoder representations. For CodeTrans Java<->C#, line i is the same
     semantic pair in opposite directions.

  2. Anisotropy:
     average pairwise cosine similarity among all OOD source encoder
     representations.

  3. Effective Rank:
     centered SVD effective rank of the full OOD source representation matrix.

  4. Entropy:
     per-OOD-sample decoder token predictive entropy under shifted teacher
     forcing, averaged over valid target tokens and then over OOD samples.

Both raw and normalized TSV tables are exported for direct Excel copying.
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ============================================================
# 1. Configuration
# ============================================================

MODEL_TYPE = "PLBART"  # choices: "codeT5", "PLBART", "all"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
MODEL_TYPES_TO_RUN = ["codeT5", "PLBART"]

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/data"

# Which language direction is treated as ID.
#   java_to_cs_id: Java -> C# as ID, C# -> Java as OOD.
#   cs_to_java_id: C# -> Java as ID, Java -> C# as OOD.
ID_DIRECTION = "cs_to_java_id"

DIRECTION_CONFIGS = {
    "java_to_cs_id": {
        "shift_name": "java_to_cs_id__cs_to_java_ood",
        "id_label": "Java -> C#",
        "ood_label": "C# -> Java",
        "id_src": "test.java-cs.txt.java",
        "id_tgt": "test.java-cs.txt.cs",
        "ood_src": "test.java-cs.txt.cs",
        "ood_tgt": "test.java-cs.txt.java",
        "rq1_dir": "rq1_language_shift_outputs_java_to_cs_id",
        "rq3_dir": "rq3_language_shift_outputs_java_to_cs_id",
        "best_prefix": "best_codetrans_java_to_cs_id",
        "rq3_prefix": "rq3_codetrans_java_to_cs_id",
    },
    "cs_to_java_id": {
        "shift_name": "cs_to_java_id__java_to_cs_ood",
        "id_label": "C# -> Java",
        "ood_label": "Java -> C#",
        "id_src": "test.java-cs.txt.cs",
        "id_tgt": "test.java-cs.txt.java",
        "ood_src": "test.java-cs.txt.java",
        "ood_tgt": "test.java-cs.txt.cs",
        "rq1_dir": "rq1_language_shift_outputs_cs_to_java_id",
        "rq3_dir": "rq3_language_shift_outputs_cs_to_java_id",
        "best_prefix": "best_codetrans_cs_to_java_id",
        "rq3_prefix": "rq3_codetrans_cs_to_java_id",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
NUM_WORKERS = 0
MAX_ANALYSIS_SAMPLES = None
SEED = 42

SOURCE_PREFIX = "translate code:"
INCLUDE_EMBEDDING_LAYER = True


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    lower = str(model_type).lower()
    if lower == "all":
        return "all"
    aliases = {
        "codet5": "codeT5",
        "codeT5": "codeT5",
        "plbart": "PLBART",
        "PLBART": "PLBART",
    }
    if model_type in aliases:
        return aliases[model_type]
    if lower in aliases:
        return aliases[lower]
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from codeT5, PLBART, all")


def resolve_model_path(model_type, model_path=None):
    return model_path or MODEL_PATHS[normalize_model_type(model_type)]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE CodeTrans RQ3 layer-wise representation diagnosis")
    parser.add_argument("--model_type", default=MODEL_TYPE, choices=["codeT5", "PLBART", "all"])
    parser.add_argument("--id_direction", default=ID_DIRECTION, choices=sorted(DIRECTION_CONFIGS.keys()))
    parser.add_argument("--codeT5_path", default=MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=MODEL_PATHS["PLBART"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_samples", type=int, default=MAX_ANALYSIS_SAMPLES)
    parser.add_argument("--include_embedding", action="store_true", default=INCLUDE_EMBEDDING_LAYER)
    parser.add_argument("--exclude_embedding", action="store_false", dest="include_embedding")
    return parser


def get_direction_config(id_direction):
    if id_direction not in DIRECTION_CONFIGS:
        raise ValueError(f"Unsupported id_direction={id_direction}. Choose from {list(DIRECTION_CONFIGS.keys())}")
    cfg = dict(DIRECTION_CONFIGS[id_direction])
    cfg["id_test_src_file"] = os.path.join(DATA_DIR, cfg["id_src"])
    cfg["id_test_tgt_file"] = os.path.join(DATA_DIR, cfg["id_tgt"])
    cfg["ood_test_src_file"] = os.path.join(DATA_DIR, cfg["ood_src"])
    cfg["ood_test_tgt_file"] = os.path.join(DATA_DIR, cfg["ood_tgt"])
    cfg["rq1_save_dir"] = os.path.join(
        "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans",
        cfg["rq1_dir"],
    )
    cfg["rq3_save_dir"] = os.path.join(
        "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans",
        cfg["rq3_dir"],
    )
    return cfg


def get_output_paths(model_type, direction_cfg):
    run_save_dir = os.path.join(direction_cfg["rq1_save_dir"], model_type)
    return {
        "save_dir": run_save_dir,
        "best_model": os.path.join(run_save_dir, f"{direction_cfg['best_prefix']}_{model_type}.bin"),
    }


def get_rq3_paths(model_type, direction_cfg):
    run_save_dir = os.path.join(direction_cfg["rq3_save_dir"], model_type)
    os.makedirs(run_save_dir, exist_ok=True)
    prefix = f"{direction_cfg['rq3_prefix']}_{model_type}"
    return {
        "save_dir": run_save_dir,
        "csv": os.path.join(run_save_dir, f"{prefix}.csv"),
        "json": os.path.join(run_save_dir, f"{prefix}.json"),
        "raw_tsv": os.path.join(run_save_dir, f"{prefix}_raw_paste_table.tsv"),
        "normalized_tsv": os.path.join(run_save_dir, f"{prefix}_normalized_paste_table.tsv"),
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


def load_seq2seq_model(model_type, model_path, checkpoint_path=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RQ1 best model not found: {checkpoint_path}")
        model.load_state_dict(safe_torch_load(checkpoint_path, map_location=DEVICE))
        print(">>> Loaded fine-tuned RQ1 checkpoint:", checkpoint_path)
    model.eval()
    return model


def layer_label(hidden_index):
    return f"L{hidden_index:02d}"


def to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


set_seed(SEED)


# ============================================================
# 3. Data
# ============================================================

def read_parallel_files(src_path, tgt_path, max_samples=None):
    with open(src_path, "r", encoding="utf-8") as f_src:
        src_lines = [line.rstrip("\n") for line in f_src]
    with open(tgt_path, "r", encoding="utf-8") as f_tgt:
        tgt_lines = [line.rstrip("\n") for line in f_tgt]
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Parallel file length mismatch: {src_path}={len(src_lines)}, {tgt_path}={len(tgt_lines)}")
    data = []
    for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        if src.strip() == "" and tgt.strip() == "":
            continue
        data.append({"idx": idx, "source": src, "target": tgt})
    if max_samples is not None:
        data = data[:max_samples]
    print(f">>> Loaded {len(data)} pairs from {src_path} -> {tgt_path}")
    return data


def align_id_ood_pairs(id_data, ood_data, max_samples=None):
    n = min(len(id_data), len(ood_data))
    if n == 0:
        raise ValueError("Empty ID/OOD paired data.")
    if len(id_data) != len(ood_data):
        print(f"[Warning] ID/OOD size mismatch, using first {n} pairs: id={len(id_data)}, ood={len(ood_data)}")
    if max_samples is not None:
        n = min(n, int(max_samples))
    pairs = []
    for i in range(n):
        pairs.append({
            "idx": i,
            "id_source": id_data[i]["source"],
            "id_target": id_data[i]["target"],
            "ood_source": ood_data[i]["source"],
            "ood_target": ood_data[i]["target"],
        })
    print(f">>> Aligned {len(pairs)} ID/OOD CodeTrans pairs line-by-line")
    return pairs


class PairedCodeTransDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer
        for item in pairs:
            id_source = f"{SOURCE_PREFIX} {item['id_source']}"
            ood_source = f"{SOURCE_PREFIX} {item['ood_source']}"
            id_enc = tokenizer(
                id_source,
                max_length=MAX_SOURCE_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ood_enc = tokenizer(
                ood_source,
                max_length=MAX_SOURCE_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ood_target_enc = tokenizer(
                item["ood_target"],
                max_length=MAX_TARGET_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = ood_target_enc["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100
            self.examples.append({
                "idx": int(item["idx"]),
                "id_input_ids": id_enc["input_ids"].squeeze(0),
                "id_attention_mask": id_enc["attention_mask"].squeeze(0),
                "ood_input_ids": ood_enc["input_ids"].squeeze(0),
                "ood_attention_mask": ood_enc["attention_mask"].squeeze(0),
                "labels": labels,
            })
        print(f">>> Built PairedCodeTransDataset: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# ============================================================
# 4. RQ3 metric helpers
# ============================================================

def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def get_anisotropy(embeddings):
    if embeddings.size(0) <= 1:
        return 0.0
    x = F.normalize(embeddings.float(), p=2, dim=1)
    n = x.size(0)
    total = 0.0
    count = 0
    chunk_size = 1024
    for start in range(0, n, chunk_size):
        chunk = x[start:start + chunk_size]
        sim = torch.matmul(chunk, x.t())
        total += float(sim.sum().item())
        count += int(sim.numel())
    return (total - n) / max(count - n, 1)


def get_effective_rank(embeddings):
    if embeddings.size(0) <= 1:
        return 0.0
    x = embeddings.float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x)
    except Exception:
        s = torch.svd(x)[1]
    s = torch.clamp(s, min=1e-12)
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


def project_decoder_hidden_to_logits(model, hidden):
    sequence_output = hidden
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "project_out"):
        project_out = model.model.decoder.project_out
        if project_out is not None:
            sequence_output = project_out(sequence_output)
    if hasattr(model, "lm_head"):
        logits = model.lm_head(sequence_output)
        if getattr(model.config, "tie_word_embeddings", False):
            logits = logits / math.sqrt(getattr(model.config, "d_model", sequence_output.size(-1)))
        if hasattr(model, "final_logits_bias"):
            logits = logits + model.final_logits_bias.to(logits.device)
        return logits
    raise AttributeError("Cannot project decoder hidden states to logits: missing lm_head.")


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
    logits = project_decoder_hidden_to_logits(model, decoder_hidden)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    token_entropy = -(probs * log_probs).sum(dim=-1)
    valid_mask = labels.ne(-100).float()
    denom = valid_mask.sum(dim=1).clamp(min=1.0)
    seq_entropy = (token_entropy * valid_mask).sum(dim=1) / denom
    return seq_entropy.detach().cpu()


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def minmax_normalize(values):
    arr = np.asarray([to_float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return arr
    low = float(np.min(arr))
    high = float(np.max(arr))
    if abs(high - low) < 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def add_normalized_columns(rows):
    metrics = ["Align", "Aniso", "Rank", "Entropy"]
    for metric in metrics:
        base_key = f"{metric}_base"
        ft_key = f"{metric}_ft"
        combined = [row[base_key] for row in rows] + [row[ft_key] for row in rows]
        normalized = minmax_normalize(combined)
        n = len(rows)
        for i, row in enumerate(rows):
            row[f"{base_key}_n"] = float(normalized[i])
            row[f"{ft_key}_n"] = float(normalized[n + i])
            row[f"{metric}_delta_n_ft_minus_base"] = float(normalized[n + i] - normalized[i])
    return rows


def init_stats(layer_indices):
    return {
        layer_id: {
            "alignment": [],
            "ood_vectors": [],
            "entropy": [],
        }
        for layer_id in layer_indices
    }


@torch.no_grad()
def compute_layer_metrics(model, dataloader, include_embedding_layer=True, desc="Computing RQ3 metrics"):
    model.eval()
    encoder = model.get_encoder()
    stats = None
    layer_indices = None

    for batch in tqdm(dataloader, desc=desc):
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
            use_cache=False,
            return_dict=True,
        )

        id_enc_states = id_encoder_outputs.hidden_states
        ood_enc_states = outputs.encoder_hidden_states
        dec_states = outputs.decoder_hidden_states
        n_hidden = min(len(id_enc_states), len(ood_enc_states), len(dec_states))

        if stats is None:
            start = 0 if include_embedding_layer else 1
            layer_indices = list(range(start, n_hidden))
            stats = init_stats(layer_indices)
            print(f">>> {desc}: hidden_states={n_hidden}, analyzed_layers={len(layer_indices)}, include_embedding={include_embedding_layer}")

        for layer_id in layer_indices:
            id_vec = masked_mean(id_enc_states[layer_id], id_attention_mask).detach().cpu()
            ood_vec = masked_mean(ood_enc_states[layer_id], ood_attention_mask).detach().cpu()
            alignment = F.cosine_similarity(
                F.normalize(id_vec.float(), p=2, dim=1),
                F.normalize(ood_vec.float(), p=2, dim=1),
                dim=1,
            )
            entropy = token_entropy_from_decoder_layer(model, dec_states[layer_id], labels)
            stats[layer_id]["alignment"].extend(alignment.numpy().tolist())
            stats[layer_id]["ood_vectors"].append(ood_vec.float())
            stats[layer_id]["entropy"].extend(entropy.numpy().tolist())

    if not stats:
        raise ValueError("No layer metrics were computed.")

    metrics = {}
    for layer_id in layer_indices:
        vectors = torch.cat(stats[layer_id]["ood_vectors"], dim=0)
        metrics[layer_id] = {
            "alignment": mean_or_zero(stats[layer_id]["alignment"]),
            "anisotropy": get_anisotropy(vectors),
            "effective_rank": get_effective_rank(vectors),
            "entropy": mean_or_zero(stats[layer_id]["entropy"]),
            "num_samples": int(vectors.size(0)),
        }
    return metrics


# ============================================================
# 5. Output
# ============================================================

def build_rows(model_type, shift_name, base_metrics, ft_metrics):
    rows = []
    common_layers = sorted(set(base_metrics.keys()) & set(ft_metrics.keys()))
    for layer_id in common_layers:
        base = base_metrics[layer_id]
        ft = ft_metrics[layer_id]
        rows.append({
            "Model": model_type,
            "Task": "CodeTrans",
            "Shift": shift_name,
            "Layer": layer_label(layer_id),
            "Align_base": float(base["alignment"]),
            "Align_ft": float(ft["alignment"]),
            "Aniso_base": float(base["anisotropy"]),
            "Aniso_ft": float(ft["anisotropy"]),
            "Rank_base": float(base["effective_rank"]),
            "Rank_ft": float(ft["effective_rank"]),
            "Entropy_base": float(base["entropy"]),
            "Entropy_ft": float(ft["entropy"]),
            "Samples": int(ft["num_samples"]),
        })
    return add_normalized_columns(rows)


def build_excel_raw_rows(rows):
    keys = [
        "Layer",
        "Align_base",
        "Align_ft",
        "Aniso_base",
        "Aniso_ft",
        "Rank_base",
        "Rank_ft",
        "Entropy_base",
        "Entropy_ft",
    ]
    return [{key: row[key] for key in keys} for row in rows]


def build_excel_normalized_rows(rows):
    keys = [
        "Layer",
        "Align_base_n",
        "Align_ft_n",
        "Aniso_base_n",
        "Aniso_ft_n",
        "Rank_base_n",
        "Rank_ft_n",
        "Entropy_base_n",
        "Entropy_ft_n",
    ]
    return [{key: row[key] for key in keys} for row in rows]


def print_excel_tables(model_type, shift_name, rows):
    print("\n" + "=" * 120)
    print(f"EXCEL_COPY_RAW\tCodeTrans\t{shift_name}\t{model_type}")
    print("=" * 120)
    raw_rows = build_excel_raw_rows(rows)
    print("\t".join(raw_rows[0].keys()))
    for row in raw_rows:
        print("\t".join(str(row[key]) if key == "Layer" else f"{float(row[key]):.4f}" for key in row))
    print("=" * 120)
    print(f"EXCEL_COPY_NORMALIZED\tCodeTrans\t{shift_name}\t{model_type}")
    print("=" * 120)
    norm_rows = build_excel_normalized_rows(rows)
    print("\t".join(norm_rows[0].keys()))
    for row in norm_rows:
        print("\t".join(str(row[key]) if key == "Layer" else f"{float(row[key]):.4f}" for key in row))
    print("=" * 120)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path, table_rows):
    if not table_rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)


# ============================================================
# 6. Run
# ============================================================

def run_one(model_type, model_path, direction_cfg, batch_size=BATCH_SIZE, max_samples=MAX_ANALYSIS_SAMPLES, include_embedding_layer=True):
    model_type = normalize_model_type(model_type)
    model_path = resolve_model_path(model_type, model_path)
    output_paths = get_output_paths(model_type, direction_cfg)
    rq3_paths = get_rq3_paths(model_type, direction_cfg)
    checkpoint = output_paths["best_model"]
    start_time = time.time()

    print("=" * 120)
    print("CodeXGLUE-CodeTrans RQ3: Layer-wise Representation Diagnosis")
    print("=" * 120)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("ID_DIRECTION:", direction_cfg["shift_name"])
    print("RQ1_BEST_MODEL:", checkpoint)
    print("ID Test :", direction_cfg["id_test_src_file"], "->", direction_cfg["id_test_tgt_file"])
    print("OOD Test:", direction_cfg["ood_test_src_file"], "->", direction_cfg["ood_test_tgt_file"])
    print("SAVE_DIR:", rq3_paths["save_dir"])
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id_data = read_parallel_files(direction_cfg["id_test_src_file"], direction_cfg["id_test_tgt_file"], max_samples=max_samples)
    ood_data = read_parallel_files(direction_cfg["ood_test_src_file"], direction_cfg["ood_test_tgt_file"], max_samples=max_samples)
    paired = align_id_ood_pairs(id_data, ood_data, max_samples=max_samples)
    dataset = PairedCodeTransDataset(paired, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    base_model = load_seq2seq_model(model_type, model_path, checkpoint_path=None)
    configure_special_tokens(tokenizer, base_model)
    base_metrics = compute_layer_metrics(
        base_model,
        dataloader,
        include_embedding_layer=include_embedding_layer,
        desc=f"{model_type} base",
    )
    del base_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    ft_model = load_seq2seq_model(model_type, model_path, checkpoint_path=checkpoint)
    configure_special_tokens(tokenizer, ft_model)
    ft_metrics = compute_layer_metrics(
        ft_model,
        dataloader,
        include_embedding_layer=include_embedding_layer,
        desc=f"{model_type} fine-tuned",
    )
    del ft_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    rows = build_rows(model_type, direction_cfg["shift_name"], base_metrics, ft_metrics)
    write_csv(rq3_paths["csv"], rows)
    write_tsv(rq3_paths["raw_tsv"], build_excel_raw_rows(rows))
    write_tsv(rq3_paths["normalized_tsv"], build_excel_normalized_rows(rows))
    payload = {
        "task": "Code-to-Code Translation",
        "dataset": "CodeXGLUE-CodeTrans",
        "model_type": model_type,
        "model_path": model_path,
        "rq1_best_model": checkpoint,
        "shift_name": direction_cfg["shift_name"],
        "id_direction": direction_cfg["id_label"],
        "ood_direction": direction_cfg["ood_label"],
        "metrics": {
            "alignment": "one-to-one cosine similarity between paired ID source encoder representation and OOD source encoder representation",
            "anisotropy": "average pairwise cosine similarity among all OOD source encoder representations",
            "effective_rank": "centered SVD effective rank of all OOD source encoder representations",
            "token_entropy": "per-OOD-sample decoder token predictive entropy under shifted teacher forcing, averaged over valid target tokens and then over OOD samples",
        },
        "normalization": "min-max per model over base and fine-tuned layer values; raw TSV and normalized TSV are both exported",
        "include_embedding_layer": bool(include_embedding_layer),
        "num_samples": len(dataset),
        "rows": rows,
        "files": rq3_paths,
        "time": {
            "total_seconds": float(time.time() - start_time),
            "total_readable": format_time(time.time() - start_time),
        },
    }
    with open(rq3_paths["json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print_excel_tables(model_type, direction_cfg["shift_name"], rows)
    print("CSV saved:", rq3_paths["csv"])
    print("JSON saved:", rq3_paths["json"])
    print("Raw paste TSV saved:", rq3_paths["raw_tsv"])
    print("Normalized paste TSV saved:", rq3_paths["normalized_tsv"])
    print("Total time:", format_time(time.time() - start_time))
    return payload


def main():
    args = build_arg_parser().parse_args()
    MODEL_PATHS["codeT5"] = args.codeT5_path
    MODEL_PATHS["PLBART"] = args.PLBART_path
    direction_cfg = get_direction_config(args.id_direction)
    os.makedirs(direction_cfg["rq3_save_dir"], exist_ok=True)

    selected = normalize_model_type(args.model_type)
    if selected == "all":
        experiments = [
            ("codeT5", args.codeT5_path),
            ("PLBART", args.PLBART_path),
        ]
    else:
        experiments = [(selected, resolve_model_path(selected))]

    outputs = []
    for model_type, model_path in experiments:
        try:
            outputs.append(run_one(
                model_type,
                model_path,
                direction_cfg,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                include_embedding_layer=args.include_embedding,
            ))
        except Exception as exc:
            if selected != "all":
                raise
            outputs.append({"status": "failed", "model_type": model_type, "error": repr(exc)})
            print(f"[Error] RQ3 failed for {model_type}: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if selected == "all":
        summary_path = os.path.join(direction_cfg["rq3_save_dir"], f"{direction_cfg['rq3_prefix']}_all_models.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "id_direction": direction_cfg["id_label"],
                "ood_direction": direction_cfg["ood_label"],
                "shift_name": direction_cfg["shift_name"],
                "results": outputs,
            }, f, ensure_ascii=False, indent=2)
        print("All-model RQ3 summary saved:", summary_path)


if __name__ == "__main__":
    main()
