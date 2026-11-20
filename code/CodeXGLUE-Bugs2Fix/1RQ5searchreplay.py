# -*- coding: utf-8 -*-
"""
CodeRefinement / Bugs2Fix RQ5 Stage 1.

Goal:
  Keep the RQ4 Replay continual-learning configuration fixed, and search only
  the OOD 80/20 split strategy using task-adapted RQ3 indicators.

RQ4 baseline:
  original OOD pool -> random 80/20 split -> fixed Replay CL -> ID/OOD test.

RQ5:
  original OOD pool -> RQ3-guided 80/20 split -> same fixed Replay CL
  -> select the split with the best OOD Exact Match.
"""

import csv
import itertools
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


# ============================================================
# 1. Config
# ============================================================

MODEL_TYPE = "PLBART"  # "codeT5", "PLBART"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codeT5", "PLBART"]

DATA_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement/data"
SUBSET_NAME = "small"
SUBSET_DIR = os.path.join(DATA_ROOT, SUBSET_NAME)
RQ1_OUTPUT_ROOT = os.path.join(DATA_ROOT, "rq1_outputs")
SAVE_ROOT = os.path.join(DATA_ROOT, "rq5_rq3_guided_replay_stage1_results")

ID_TRAIN_FILE = os.path.join(SUBSET_DIR, "id_train.jsonl")
ID_TEST_FILE = os.path.join(SUBSET_DIR, "id_test.jsonl")
OOD_POOL_FILE_CANDIDATES = [
    os.path.join(SUBSET_DIR, "ood_test_cst.jsonl"),
    os.path.join(SUBSET_DIR, "ood_test_cst"),
]
OOD_POOL_FALLBACK_FILES = [
    os.path.join(SUBSET_DIR, "ood_train_cst.jsonl"),
    os.path.join(SUBSET_DIR, "ood_train_cst"),
    os.path.join(SUBSET_DIR, "ood_test_cst_new.jsonl"),
    os.path.join(SUBSET_DIR, "ood_test_cst_new"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_BEAMS = 5
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_ANALYSIS_SAMPLES = None

# Fixed RQ4 Replay configuration. Do not search these.
EPOCHS = 8
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
REPLAY_LAMBDA = 0.5
REPLAY_BATCH_SIZE = 4
REPLAY_BUFFER_SIZE = 1000
REPLAY_BUFFER_SEED = SEED

# RQ3-guided split search.
FIXED_TEST_RATIO = 0.20
# Global geometry follows the original code-classification reference:
# all-pair OOD cosine similarity and one SVD over the complete OOD matrix.
LOCAL_GEOMETRY_CHUNK_SIZE = 128
ENTROPY_CHUNK_SIZE = 16
MIN_ID_OOD_PAIR_COVERAGE = 0.95
RQ3_METRIC_NAMES = ["alignment", "entropy", "anisotropy", "rank"]

SEARCH_WEIGHT_BOUNDS = {
    "alignment": (0.0, 1.0),
    "entropy": (0.0, 1.0),
    "anisotropy": (0.0, 1.0),
    "rank": (0.0, 1.0),
}
SEARCH_RATIO_BOUNDS = {
    "hard": (0.02, 0.20),
    "middle": (0.04, 0.30),
    "easy": (0.06, 0.40),
}
SEARCH_BOUNDARY_BOUNDS = {
    "hard_end": (0.20, 0.35),
    "middle_end": (0.65, 0.85),
}
DEFAULT_BOUNDARIES = {"hard_end": 0.25, "middle_end": 0.75}
SPLIT_WITH_TRAIN_PRIORITY = True
TRAIN_PRIORITY_HARD_FRACTION = 0.60
EASY_TEST_START_FRACTION = 0.35

SMART_STAGE1_TRIALS = 8
SMART_STAGE1_EPOCHS = 1
SMART_STAGE2_TRIALS = 4
SMART_STAGE2_EPOCHS = 1
SMART_STAGE2_FROM_STAGE1 = 2
SMART_STAGE2_RANDOM = 1
SMART_FINAL_TRIALS = 1
SMART_FINAL_EPOCHS = EPOCHS
SMART_TOP_CONFIGS_K = 5

RQ5_METRIC_CACHE_VERSION = 3
RQ5_SEARCH_SPACE_VERSION = 2

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
    for name in MODEL_PATHS:
        if lower == name.lower():
            return name
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {list(MODEL_PATHS.keys())}")


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))[:160]


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def write_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_first_existing(candidates, role):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find {role}. Tried:\n" + "\n".join(f"  - {p}" for p in candidates))


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


def normalize_code(text):
    if text is None:
        return ""
    return " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").strip().split())


def exact_match_score(preds, refs):
    return 0.0 if not preds else 100.0 * sum(normalize_code(p) == normalize_code(r) for p, r in zip(preds, refs)) / len(preds)


def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu_score(preds, refs, max_n=4, smooth=1.0):
    clipped_counts = [0 for _ in range(max_n)]
    total_counts = [0 for _ in range(max_n)]
    pred_len = 0
    ref_len = 0
    for pred, ref in zip(preds, refs):
        pred_tokens = normalize_code(pred).split()
        ref_tokens = normalize_code(ref).split()
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            clipped_counts[n - 1] += sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in pred_ngrams.items())
            total_counts[n - 1] += max(sum(pred_ngrams.values()), 0)
    if pred_len == 0:
        return 0.0
    precisions = [(clipped_counts[i] + smooth) / (total_counts[i] + smooth) for i in range(max_n)]
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n) * 100.0)


def codebleu_score(preds, refs, lang="java"):
    try:
        from codebleu import calc_codebleu
    except Exception as exc:
        print(f"[Warning] Cannot import codebleu; CodeBLEU=-1.0. {repr(exc)}")
        return -1.0
    norm_preds = [normalize_code(x) for x in preds]
    norm_refs = [normalize_code(x) for x in refs]
    try:
        result = calc_codebleu(references=[[r] for r in norm_refs], predictions=norm_preds, lang=lang)
    except TypeError:
        result = calc_codebleu([[r] for r in norm_refs], norm_preds, lang)
    except Exception as exc:
        print(f"[Warning] CodeBLEU failed; CodeBLEU=-1.0. {repr(exc)}")
        return -1.0
    value = result.get("codebleu") if isinstance(result, dict) else result
    if value is None:
        return -1.0
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def compute_repair_metrics(preds, refs):
    return {
        "exact_match": float(exact_match_score(preds, refs)),
        "bleu": float(corpus_bleu_score(preds, refs)),
        "codebleu": float(codebleu_score(preds, refs, lang="java")),
        "num_samples": len(preds),
    }


def id_ood_drop(id_eval, ood_eval):
    return {
        "exact_match_drop": id_eval["exact_match"] - ood_eval["exact_match"],
        "bleu_drop": id_eval["bleu"] - ood_eval["bleu"],
        "codebleu_drop": id_eval["codebleu"] - ood_eval["codebleu"],
    }


# ============================================================
# 3. Data
# ============================================================

class Bugs2FixDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_samples=None, shuffle_before_limit=False, name="dataset"):
        self.tokenizer = tokenizer
        self.examples = []
        data = list(raw_data)
        if shuffle_before_limit:
            rng = random.Random(SEED)
            rng.shuffle(data)
        if max_samples is not None and len(data) > max_samples:
            data = data[:max_samples]
        skipped = 0
        for line_idx, js in enumerate(data):
            source = get_source(js)
            target = get_target(js)
            if not source or not target:
                skipped += 1
                continue
            self.examples.append({
                "idx": get_sample_key(js, line_idx),
                "source": source,
                "target": target,
                "shift_type": js.get("shift_type", "id"),
                "raw": js,
            })
        print(f">>> Loaded {len(self.examples)} Bugs2Fix samples for {name}; skipped={skipped}")
        print(f">>> shift_type distribution: {dict(Counter(x['shift_type'] for x in self.examples))}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        source_inputs = self.tokenizer(
            item["source"], max_length=MAX_SOURCE_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_inputs = self.tokenizer(
            item["target"], max_length=MAX_TARGET_LEN, padding="max_length", truncation=True, return_tensors="pt"
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


class SourceOnlyDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_samples=None, name="source_dataset"):
        self.tokenizer = tokenizer
        self.examples = []
        data = list(raw_data)
        if max_samples is not None and len(data) > max_samples:
            data = data[:max_samples]
        skipped = 0
        for line_idx, js in enumerate(data):
            source = get_source(js)
            target = get_target(js)
            if not source or not target:
                skipped += 1
                continue
            self.examples.append({
                "idx": get_sample_key(js, line_idx),
                "source": source,
                "target": target,
                "raw": js,
            })
        print(f">>> Loaded {len(self.examples)} source samples for {name}; skipped={skipped}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        source_inputs = self.tokenizer(
            item["source"], max_length=MAX_SOURCE_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_inputs = self.tokenizer(
            item["target"], max_length=MAX_TARGET_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = target_inputs["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": source_inputs["input_ids"].squeeze(0),
            "attention_mask": source_inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "idx": item["idx"],
            "raw": item["raw"],
        }


def build_ood_pool():
    for path in OOD_POOL_FILE_CANDIDATES:
        if os.path.exists(path):
            data = read_jsonl(path)
            print(f">>> OOD pool loaded from original file: {path}, size={len(data)}")
            return data, [path]
    existing = [p for p in OOD_POOL_FALLBACK_FILES if os.path.exists(p)]
    if not existing:
        tried = OOD_POOL_FILE_CANDIDATES + OOD_POOL_FALLBACK_FILES
        raise FileNotFoundError("Cannot build OOD pool. Tried:\n" + "\n".join(f"  - {p}" for p in tried))
    merged = []
    for path in existing:
        part = read_jsonl(path)
        print(f">>> OOD pool fallback part: {path}, size={len(part)}")
        merged.extend(part)
    print(f">>> Reconstructed OOD pool size: {len(merged)}")
    return merged, existing


def align_id_ood_raw(id_raw, ood_raw):
    id_map = {}
    for i, item in enumerate(id_raw):
        key = get_sample_key(item, i)
        if key not in id_map:
            id_map[key] = item

    aligned_id = []
    aligned_ood = []
    aligned_keys = []
    unmatched = 0
    for i, item in enumerate(ood_raw):
        key = get_sample_key(item, i)
        if key not in id_map:
            unmatched += 1
            continue
        aligned_id.append(id_map[key])
        aligned_ood.append(item)
        aligned_keys.append(key)

    coverage = len(aligned_ood) / max(1, len(ood_raw))
    print("=" * 100)
    print("ID/OOD pairing report")
    print("ID raw size:", len(id_raw), "OOD raw size:", len(ood_raw))
    print("Aligned pairs:", len(aligned_ood), "Coverage:", f"{coverage:.2%}", "Unmatched:", unmatched)

    if coverage < MIN_ID_OOD_PAIR_COVERAGE and len(id_raw) == len(ood_raw):
        print("[Warning] Low key coverage; falling back to positional alignment.")
        aligned_id = list(id_raw)
        aligned_ood = list(ood_raw)
        aligned_keys = [get_sample_key(x, i) for i, x in enumerate(ood_raw)]
        coverage = 1.0

    if coverage < MIN_ID_OOD_PAIR_COVERAGE:
        raise RuntimeError(f"ID/OOD pair coverage {coverage:.2%} is lower than {MIN_ID_OOD_PAIR_COVERAGE:.2%}.")
    print("=" * 100)
    return aligned_id, aligned_ood, aligned_keys


def make_loader(dataset, batch_size, shuffle=False, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        generator=generator,
        collate_fn=collate_bugs2fix_batch,
    )


def collate_bugs2fix_batch(batch):
    tensor_keys = ("input_ids", "attention_mask", "labels")
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in batch[0]:
        if key not in out:
            out[key] = [item[key] for item in batch]
    return out


# ============================================================
# 4. Model / generation
# ============================================================

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


def build_generation_model(model_type, model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    return model


def load_rq1_model(model_type, model_path, rq1_best_model, tokenizer):
    if not os.path.exists(rq1_best_model):
        raise FileNotFoundError(f"Cannot find RQ1 best model: {rq1_best_model}")
    model = build_generation_model(model_type, model_path)
    configure_special_tokens(tokenizer, model)
    model.load_state_dict(safe_torch_load(rq1_best_model, map_location=DEVICE))
    print(">>> Loaded RQ1 checkpoint:", rq1_best_model)
    return model


def build_shifted_decoder_inputs(model, labels):
    pad_token_id = model.config.pad_token_id
    clean_labels = labels.clone()
    clean_labels[clean_labels == -100] = pad_token_id
    if hasattr(model, "prepare_decoder_input_ids_from_labels"):
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=clean_labels)
    else:
        start_id = model.config.decoder_start_token_id or pad_token_id
        decoder_input_ids = clean_labels.new_full(clean_labels.shape, pad_token_id)
        decoder_input_ids[:, 0] = start_id
        decoder_input_ids[:, 1:] = clean_labels[:, :-1]
    decoder_attention_mask = decoder_input_ids.ne(pad_token_id).long()
    return decoder_input_ids, decoder_attention_mask


@torch.no_grad()
def extract_final_source_vectors(model, dataloader, desc):
    model.eval()
    vectors = []
    idxs = []
    raws = []
    encoder = model.get_encoder()
    use_amp = DEVICE.type == "cuda"
    for batch in tqdm(dataloader, desc=desc, leave=False):
        with autocast(enabled=use_amp):
            outputs = encoder(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].to(DEVICE).unsqueeze(-1).to(hidden.dtype)
            vec = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        vectors.append(vec.detach().cpu().float())
        idxs.extend([str(x) for x in batch["idx"]])
        raws.extend(list(batch["raw"]))
    if not vectors:
        raise ValueError(f"{desc}: no vectors extracted.")
    return {"vecs": torch.cat(vectors, dim=0), "idx": idxs, "raw": raws}


@torch.no_grad()
def ood_token_entropy(model, dataloader, chunk_desc):
    model.eval()
    entropies = []
    for batch in tqdm(dataloader, desc=chunk_desc, leave=False):
        labels = batch["labels"].to(DEVICE)
        decoder_input_ids, decoder_attention_mask = build_shifted_decoder_inputs(model, labels)
        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        valid_mask = labels.ne(-100)
        denom = valid_mask.sum(dim=1).clamp(min=1)
        entropy = (token_entropy * valid_mask).sum(dim=1) / denom
        entropies.extend(entropy.detach().cpu().numpy().tolist())
    return np.asarray(entropies, dtype=np.float64)


def seq2seq_loss(model, batch, use_amp=False):
    with autocast(enabled=use_amp):
        return model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
        ).loss


@torch.no_grad()
def evaluate_generation(model, tokenizer, dataloader, desc="Evaluating", save_path=None, num_beams=None):
    model.eval()
    preds, refs, records = [], [], []
    use_amp = DEVICE.type == "cuda"
    for batch in tqdm(dataloader, desc=desc, leave=False):
        with autocast(enabled=use_amp):
            generated_ids = model.generate(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                max_length=MAX_TARGET_LEN,
                num_beams=NUM_BEAMS if num_beams is None else int(num_beams),
                early_stopping=True,
            )
        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        batch_refs = list(batch["target"])
        preds.extend(batch_preds)
        refs.extend(batch_refs)
        if save_path is not None:
            for i, pred in enumerate(batch_preds):
                records.append({
                    "idx": batch["idx"][i],
                    "shift_type": batch["shift_type"][i],
                    "source": batch["source"][i],
                    "target": batch_refs[i],
                    "prediction": pred,
                    "exact_match": normalize_code(pred) == normalize_code(batch_refs[i]),
                })
    result = compute_repair_metrics(preds, refs)
    if save_path is not None:
        write_jsonl(save_path, records)
    return result


# ============================================================
# 5. RQ3 sample metrics for split
# ============================================================

def effective_rank(x):
    if x.size(0) <= 1:
        return 1.0
    y = x.float() - x.float().mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(y)
    except Exception:
        s = torch.svd(y)[1]
    s = torch.clamp(s, min=1e-12)
    p = s / (s.sum() + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    return float(torch.exp(entropy).item())


def global_geometry_scores(ood_vecs, chunk_size=128):
    """Compute exact global OOD geometry and sample-level contributions."""
    x = F.normalize(ood_vecs.float(), p=2, dim=1)
    n = x.size(0)
    if n <= 1:
        summary = {"anisotropy": 0.0, "effective_rank": 1.0}
        return np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64), summary

    anisotropy_parts = []
    for start in tqdm(range(0, n, chunk_size), desc="Computing global OOD anisotropy", leave=False):
        end = min(start + chunk_size, n)
        sim = torch.matmul(x[start:end], x.T)
        rows = torch.arange(end - start)
        cols = torch.arange(start, end)
        sim[rows, cols] = 0.0
        anisotropy_parts.append((sim.sum(dim=1) / (n - 1)).cpu())
    anisotropy = torch.cat(anisotropy_parts).numpy().astype(np.float64)

    centered = ood_vecs.float() - ood_vecs.float().mean(dim=0, keepdim=True)
    try:
        _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        _, singular_values, right_vectors = torch.svd(centered)
        right_vectors = right_vectors.T
    singular_values = torch.clamp(singular_values, min=1e-12)
    singular_probs = singular_values / singular_values.sum().clamp(min=1e-12)
    global_rank = torch.exp(
        -torch.sum(singular_probs * torch.log(singular_probs + 1e-12))
    )

    # A global effective rank is one dataset-level scalar.  For filtering,
    # assign each sample its participation rank in the same global SVD axes.
    coordinates = torch.abs(torch.matmul(centered, right_vectors.T))
    coordinate_probs = coordinates / coordinates.sum(dim=1, keepdim=True).clamp(min=1e-12)
    sample_rank = torch.exp(
        -torch.sum(coordinate_probs * torch.log(coordinate_probs + 1e-12), dim=1)
    ).cpu().numpy().astype(np.float64)
    summary = {
        "anisotropy": float(np.mean(anisotropy)),
        "effective_rank": float(global_rank.item()),
    }
    return anisotropy, sample_rank, summary


def percentile_normalize(values, higher_is_harder=True):
    arr = np.asarray(values, dtype=np.float64)
    out = np.full_like(arr, 0.5, dtype=np.float64)
    finite = np.isfinite(arr)
    idx = np.where(finite)[0]
    if len(idx) <= 1:
        return out
    vals = arr[idx]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.zeros(len(vals), dtype=np.float64)
    ranks[order] = np.arange(len(vals), dtype=np.float64) / max(1, len(vals) - 1)
    out[idx] = ranks
    if not higher_is_harder:
        out[idx] = 1.0 - out[idx]
    return out


def normalize_positive_weights(weights):
    vals = {k: max(0.0, float(weights.get(k, 0.0))) for k in RQ3_METRIC_NAMES}
    total = sum(vals.values())
    if total <= 1e-12:
        return {k: 1.0 / len(RQ3_METRIC_NAMES) for k in RQ3_METRIC_NAMES}
    return {k: v / total for k, v in vals.items()}


def compute_rq3_sample_metrics(model_type, model_path, rq1_best_model, tokenizer, id_raw, ood_raw, cache_file):
    if os.path.exists(cache_file):
        cached = json.load(open(cache_file, "r", encoding="utf-8"))
        if (
            isinstance(cached, list)
            and cached
            and cached[0].get("metric_cache_version") == RQ5_METRIC_CACHE_VERSION
            and all(k in cached[0].get("metrics", {}) for k in RQ3_METRIC_NAMES)
            and isinstance(cached[0].get("global_metrics"), dict)
        ):
            print(">>> Loading cached RQ3 split metrics:", cache_file)
            return cached

    aligned_id, aligned_ood, keys = align_id_ood_raw(id_raw, ood_raw)
    id_ds = SourceOnlyDataset(aligned_id, tokenizer, max_samples=MAX_ANALYSIS_SAMPLES, name="paired_id_for_rq5")
    ood_ds = SourceOnlyDataset(aligned_ood, tokenizer, max_samples=MAX_ANALYSIS_SAMPLES, name="paired_ood_for_rq5")
    if len(id_ds) != len(ood_ds):
        raise RuntimeError("ID/OOD valid paired datasets have different sizes after tokenization.")

    model = load_rq1_model(model_type, model_path, rq1_best_model, tokenizer)
    id_vecs = extract_final_source_vectors(model, make_loader(id_ds, BATCH_SIZE, shuffle=False), "Extract ID source vectors")["vecs"]
    ood_pack = extract_final_source_vectors(model, make_loader(ood_ds, BATCH_SIZE, shuffle=False), "Extract OOD source vectors")
    ood_vecs = ood_pack["vecs"]
    entropy = ood_token_entropy(model, make_loader(ood_ds, BATCH_SIZE, shuffle=False), "Compute OOD generation entropy")

    alignment = torch.sum(F.normalize(id_vecs, p=2, dim=1) * F.normalize(ood_vecs, p=2, dim=1), dim=1).numpy()
    anisotropy, rank, geometry_summary = global_geometry_scores(
        ood_vecs,
        chunk_size=LOCAL_GEOMETRY_CHUNK_SIZE,
    )
    global_metrics = {
        "alignment": float(np.mean(alignment)),
        "entropy": float(np.mean(entropy)),
        "anisotropy": float(geometry_summary["anisotropy"]),
        "effective_rank": float(geometry_summary["effective_rank"]),
    }

    n_pack = len(ood_pack["raw"])
    records = []
    for i, raw in enumerate(ood_pack["raw"]):
        if i < len(ood_pack["idx"]):
            sid = str(ood_pack["idx"][i])
        elif i < len(keys):
            sid = str(keys[i])
        else:
            sid = f"fallback_{i}"
        records.append({
            "pool_index": i,
            "idx": sid,
            "metric_cache_version": RQ5_METRIC_CACHE_VERSION,
            "global_metrics": global_metrics if i == 0 else None,
            "raw": raw,
            "metrics": {
                "alignment": float(alignment[i]),
                "entropy": float(entropy[i]),
                "anisotropy": float(anisotropy[i]),
                "rank": float(rank[i]),
            },
        })
    save_json(cache_file, records)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return records


def score_metric_records(metric_records, weights):
    actual_weights = normalize_positive_weights(weights)
    ali = np.asarray([r["metrics"]["alignment"] for r in metric_records], dtype=np.float64)
    ent = np.asarray([r["metrics"]["entropy"] for r in metric_records], dtype=np.float64)
    ani = np.asarray([r["metrics"]["anisotropy"] for r in metric_records], dtype=np.float64)
    rnk = np.asarray([r["metrics"]["rank"] for r in metric_records], dtype=np.float64)
    hard_metrics = {
        "alignment": percentile_normalize(ali, higher_is_harder=False),
        "entropy": percentile_normalize(ent, higher_is_harder=False),
        "anisotropy": percentile_normalize(ani, higher_is_harder=True),
        "rank": percentile_normalize(rnk, higher_is_harder=False),
    }
    scores = np.zeros(len(metric_records), dtype=np.float64)
    for name in RQ3_METRIC_NAMES:
        scores += actual_weights[name] * hard_metrics[name]
    scored = []
    for i, rec in enumerate(metric_records):
        item = dict(rec)
        item["normalized_difficulty_metrics"] = {name: float(hard_metrics[name][i]) for name in RQ3_METRIC_NAMES}
        item["difficulty_score"] = float(scores[i])
        scored.append(item)
    return scored, actual_weights


# ============================================================
# 6. RQ3-guided split
# ============================================================

def sanitize_ratios(ratios):
    hard = min(max(float(ratios.get("hard", 0.10)), SEARCH_RATIO_BOUNDS["hard"][0]), SEARCH_RATIO_BOUNDS["hard"][1])
    middle = min(max(float(ratios.get("middle", 0.20)), SEARCH_RATIO_BOUNDS["middle"][0]), SEARCH_RATIO_BOUNDS["middle"][1])
    easy = min(max(float(ratios.get("easy", 0.30)), SEARCH_RATIO_BOUNDS["easy"][0]), SEARCH_RATIO_BOUNDS["easy"][1])
    middle = max(middle, hard)
    easy = max(easy, middle)
    return {"hard": hard, "middle": middle, "easy": easy}


def sanitize_boundaries(boundaries):
    hard_end = min(max(float(boundaries.get("hard_end", DEFAULT_BOUNDARIES["hard_end"])), SEARCH_BOUNDARY_BOUNDS["hard_end"][0]), SEARCH_BOUNDARY_BOUNDS["hard_end"][1])
    middle_end = min(max(float(boundaries.get("middle_end", DEFAULT_BOUNDARIES["middle_end"])), SEARCH_BOUNDARY_BOUNDS["middle_end"][0]), SEARCH_BOUNDARY_BOUNDS["middle_end"][1])
    middle_end = max(middle_end, hard_end + 0.20)
    middle_end = min(middle_end, SEARCH_BOUNDARY_BOUNDS["middle_end"][1])
    return {"hard_end": hard_end, "middle_end": middle_end}


def pick_evenly_spaced(items, n_select):
    if n_select <= 0:
        return []
    if n_select >= len(items):
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=n_select)
    indices = sorted(set(int(round(pos)) for pos in positions))
    cursor = 0
    while len(indices) < n_select and cursor < len(items):
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return [items[i] for i in sorted(indices[:n_select])]


def split_group_train_priority(group_name, items, n_test):
    ranked = sorted(items, key=lambda x: x["difficulty_score"], reverse=True)
    if n_test <= 0:
        return ranked, []
    if n_test >= len(ranked):
        return [], ranked
    if group_name == "hard":
        reserve = int(round(len(ranked) * TRAIN_PRIORITY_HARD_FRACTION))
        candidates = ranked[reserve:] or ranked[-n_test:]
        test = pick_evenly_spaced(candidates, n_test)
    elif group_name == "easy":
        candidates = ranked[int(round(len(ranked) * EASY_TEST_START_FRACTION)):] or ranked
        test = pick_evenly_spaced(candidates, n_test)
    else:
        test = pick_evenly_spaced(ranked, n_test)
    test_ids = {id(x) for x in test}
    train = [x for x in ranked if id(x) not in test_ids]
    return train, test


def build_geometry_split(metric_records, search_cfg):
    scored, actual_weights = score_metric_records(metric_records, search_cfg["weights"])
    ratios = sanitize_ratios(search_cfg["ratios"])
    boundaries = sanitize_boundaries(search_cfg.get("boundaries", DEFAULT_BOUNDARIES))
    sorted_items = sorted(scored, key=lambda x: x["difficulty_score"], reverse=True)
    n = len(sorted_items)
    if n < 4:
        raise ValueError("OOD pool is too small for RQ5 split.")
    h_end = max(1, min(int(round(n * boundaries["hard_end"])), n - 2))
    m_end = max(h_end + 1, min(int(round(n * boundaries["middle_end"])), n - 1))
    groups = {"hard": sorted_items[:h_end], "middle": sorted_items[h_end:m_end], "easy": sorted_items[m_end:]}
    target_test = max(1, min(int(round(n * FIXED_TEST_RATIO)), n - 1))
    capacity = {name: max(0, len(items) - 1) for name, items in groups.items()}
    weights = {name: len(items) * ratios[name] if capacity[name] > 0 else 0.0 for name, items in groups.items()}
    if sum(weights.values()) <= 1e-12:
        weights = {name: float(len(items)) if capacity[name] > 0 else 0.0 for name, items in groups.items()}
    target_test = min(target_test, sum(capacity.values()))
    test_counts = {name: 0 for name in groups}
    raw_alloc = {name: target_test * weights[name] / max(sum(weights.values()), 1e-12) for name in groups}
    fractions = []
    for name in groups:
        base = min(int(np.floor(raw_alloc[name])), capacity[name])
        test_counts[name] = base
        fractions.append((raw_alloc[name] - base, name))
    current = sum(test_counts.values())
    for _, name in sorted(fractions, reverse=True):
        if current >= target_test:
            break
        if test_counts[name] < capacity[name]:
            test_counts[name] += 1
            current += 1

    train_records, test_records, group_report = [], [], {}
    for name, items in groups.items():
        train_part, test_part = split_group_train_priority(name, items, int(test_counts[name]))
        train_records.extend(train_part)
        test_records.extend(test_part)
        group_report[name] = {
            "available": len(items),
            "train": len(train_part),
            "test": len(test_part),
            "allocation_preference": ratios[name],
            "difficulty_mean": float(np.mean([x["difficulty_score"] for x in items])) if items else 0.0,
        }
    rng = np.random.RandomState(int(search_cfg.get("seed", SEED)))
    rng.shuffle(train_records)
    rng.shuffle(test_records)
    train_raw = [x["raw"] for x in train_records]
    test_raw = [x["raw"] for x in test_records]
    report = {
        "search_config": search_cfg,
        "actual_weights": actual_weights,
        "ratios": ratios,
        "boundaries": boundaries,
        "fixed_test_ratio": FIXED_TEST_RATIO,
        "num_total": n,
        "num_train": len(train_raw),
        "num_test": len(test_raw),
        "groups": group_report,
        "train_difficulty_mean": float(np.mean([x["difficulty_score"] for x in train_records])) if train_records else 0.0,
        "test_difficulty_mean": float(np.mean([x["difficulty_score"] for x in test_records])) if test_records else 0.0,
        "global_rq3_metrics": next(
            (
                x["global_metrics"]
                for x in metric_records
                if isinstance(x.get("global_metrics"), dict)
            ),
            None,
        ),
        "sample_rank_definition": (
            "spectral participation rank in the axes from the single global OOD SVD"
        ),
        "note": "RQ3-guided 80/20 split for code refinement; only the split changes, Replay CL configuration is inherited from RQ4.",
    }
    return train_raw, test_raw, report


# ============================================================
# 7. Search configs
# ============================================================

def make_search_config(name, weights, ratios, boundaries=None, seed=SEED):
    return {"name": name, "weights": dict(weights), "ratios": dict(ratios), "boundaries": dict(boundaries or DEFAULT_BOUNDARIES), "seed": int(seed)}


def initial_geometry_configs(seed=SEED):
    weight_sets = [
        ("balanced", {"alignment": 0.25, "entropy": 0.25, "anisotropy": 0.25, "rank": 0.25}),
        ("alignment_focus", {"alignment": 0.55, "entropy": 0.15, "anisotropy": 0.15, "rank": 0.15}),
        ("entropy_focus", {"alignment": 0.15, "entropy": 0.55, "anisotropy": 0.15, "rank": 0.15}),
        ("anisotropy_focus", {"alignment": 0.15, "entropy": 0.15, "anisotropy": 0.55, "rank": 0.15}),
        ("rank_focus", {"alignment": 0.15, "entropy": 0.15, "anisotropy": 0.15, "rank": 0.55}),
        ("align_entropy", {"alignment": 0.40, "entropy": 0.40, "anisotropy": 0.10, "rank": 0.10}),
    ]
    ratio_sets = [
        ("representative_20", {"hard": 0.20, "middle": 0.20, "easy": 0.20}),
        ("train_hard_light", {"hard": 0.12, "middle": 0.18, "easy": 0.24}),
        ("train_hard_medium", {"hard": 0.08, "middle": 0.18, "easy": 0.30}),
        ("train_hard_strong", {"hard": 0.05, "middle": 0.16, "easy": 0.32}),
        ("middle_easy_test", {"hard": 0.10, "middle": 0.24, "easy": 0.30}),
        ("balanced_ood_test", {"hard": 0.15, "middle": 0.22, "easy": 0.25}),
    ]
    boundary_sets = [
        ("standard", {"hard_end": 0.25, "middle_end": 0.75}),
        ("more_hard_train", {"hard_end": 0.35, "middle_end": 0.80}),
        ("middle_wide", {"hard_end": 0.20, "middle_end": 0.85}),
        ("hard_middle_balanced", {"hard_end": 0.30, "middle_end": 0.70}),
    ]
    configs, idx = [], 0
    for w_name, weights in weight_sets:
        for r_name, ratios in ratio_sets:
            b_name, boundaries = boundary_sets[idx % len(boundary_sets)]
            configs.append(make_search_config(f"{w_name}_{r_name}_{b_name}", weights, ratios, boundaries, seed + idx))
            idx += 1
    return configs


def random_geometry_config(rng, name, seed=SEED):
    weights = {k: float(rng.uniform(*SEARCH_WEIGHT_BOUNDS[k])) for k in RQ3_METRIC_NAMES}
    ratios = sanitize_ratios({k: float(rng.uniform(*SEARCH_RATIO_BOUNDS[k])) for k in ("hard", "middle", "easy")})
    boundaries = sanitize_boundaries({
        "hard_end": float(rng.uniform(*SEARCH_BOUNDARY_BOUNDS["hard_end"])),
        "middle_end": float(rng.uniform(*SEARCH_BOUNDARY_BOUNDS["middle_end"])),
    })
    return make_search_config(name, weights, ratios, boundaries, seed)


def config_feature_vector(cfg):
    weights = normalize_positive_weights(cfg["weights"])
    ratios = sanitize_ratios(cfg["ratios"])
    boundaries = sanitize_boundaries(cfg.get("boundaries", DEFAULT_BOUNDARIES))
    return [
        float(weights["alignment"]),
        float(weights["entropy"]),
        float(weights["anisotropy"]),
        float(weights["rank"]),
        float(boundaries["hard_end"]),
        float(boundaries["middle_end"]),
        float(ratios["hard"]),
        float(ratios["middle"]),
        float(ratios["easy"]),
    ]


def config_signature(cfg):
    return tuple(round(x, 8) for x in config_feature_vector(cfg))


def fit_surrogate_and_propose(results, number_needed, seed=SEED, pool_size=800):
    valid = [r for r in results if r.get("status") == "ok"]
    rng = np.random.RandomState(seed)
    if len(valid) < 8:
        return [random_geometry_config(rng, f"random_{i:03d}", seed + i) for i in range(number_needed)]
    x = np.asarray([config_feature_vector(r["config"]) for r in valid], dtype=np.float64)
    y = np.asarray([r["select_score"] for r in valid], dtype=np.float64)
    model = ExtraTreesRegressor(n_estimators=256, random_state=seed, n_jobs=-1)
    model.fit(x, y)
    candidates = [random_geometry_config(rng, f"surrogate_pool_{i:04d}", seed + 1000 + i) for i in range(pool_size)]
    features = np.asarray([config_feature_vector(c) for c in candidates], dtype=np.float64)
    preds = np.stack([tree.predict(features) for tree in model.estimators_], axis=0)
    acquisition = preds.mean(axis=0) + 0.15 * preds.std(axis=0)
    selected, seen = [], set()
    for idx in np.argsort(-acquisition):
        cfg = candidates[int(idx)]
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        selected.append(cfg)
        if len(selected) >= number_needed:
            break
    return selected


def unique_top_configs(rows, top_k):
    if int(top_k) <= 0:
        return []
    valid = [r for r in rows if r.get("status") == "ok"]
    valid.sort(key=lambda r: float(r["select_score"]), reverse=True)
    configs, seen = [], set()
    for row in valid:
        sig = config_signature(row["config"])
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(row["config"])
        if len(configs) >= top_k:
            break
    return configs


# ============================================================
# 8. Replay training trials
# ============================================================

def build_replay_buffer_dataset(id_train_ds, buffer_size=None, seed=SEED):
    if buffer_size is None or buffer_size <= 0 or buffer_size >= len(id_train_ds):
        indices = list(range(len(id_train_ds)))
        return id_train_ds, indices
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(id_train_ds), size=int(buffer_size), replace=False).tolist())
    return Subset(id_train_ds, indices), indices


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def train_replay_epoch(model, ood_loader, replay_iter, optimizer, scheduler, scaler, desc):
    model.train()
    total_sum, ood_sum, replay_sum, steps = 0.0, 0.0, 0.0, 0
    use_amp = DEVICE.type == "cuda"
    for batch in tqdm(ood_loader, desc=desc, leave=False):
        optimizer.zero_grad()
        ood_loss = seq2seq_loss(model, batch, use_amp=use_amp)
        replay_loss = seq2seq_loss(model, next(replay_iter), use_amp=use_amp)
        loss = ood_loss + REPLAY_LAMBDA * replay_loss
        if torch.isnan(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_sum += float(loss.detach().item())
        ood_sum += float(ood_loss.detach().item())
        replay_sum += float(replay_loss.detach().item())
        steps += 1
    denom = max(steps, 1)
    return {"total_loss": total_sum / denom, "ood_loss": ood_sum / denom, "replay_loss": replay_sum / denom, "steps": steps}


def train_replay_trial(model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds, train_raw, test_raw, trial_name, trial_dir, epochs, keep_model=False, eval_num_beams=1):
    if keep_model:
        os.makedirs(trial_dir, exist_ok=True)

    model = load_rq1_model(model_type, model_path, rq1_best_model, tokenizer)
    ood_train_ds = Bugs2FixDataset(train_raw, tokenizer, max_samples=MAX_TRAIN_SAMPLES, shuffle_before_limit=False, name=f"{trial_name}_ood_train")
    ood_test_ds = Bugs2FixDataset(test_raw, tokenizer, max_samples=MAX_EVAL_SAMPLES, shuffle_before_limit=False, name=f"{trial_name}_ood_test")
    id_eval_ds = id_test_ds
    if MAX_EVAL_SAMPLES is not None and len(id_eval_ds) > MAX_EVAL_SAMPLES:
        id_eval_ds = Subset(id_eval_ds, range(MAX_EVAL_SAMPLES))

    ood_train_loader = make_loader(ood_train_ds, BATCH_SIZE, shuffle=True, seed=SEED)
    id_loader = make_loader(id_eval_ds, EVAL_BATCH_SIZE, shuffle=False, seed=SEED)
    ood_loader = make_loader(ood_test_ds, EVAL_BATCH_SIZE, shuffle=False, seed=SEED)
    replay_loader = make_loader(replay_ds, REPLAY_BATCH_SIZE, shuffle=True, seed=REPLAY_BUFFER_SEED)
    replay_iter = cycle_loader(replay_loader)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(ood_train_loader) * int(epochs), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    best_score, best_epoch = -1.0, 0
    best_id_metrics, best_ood_metrics = None, None
    best_model_path = os.path.join(trial_dir, f"best_rq5_replay_{model_type}_{safe_name(trial_name)}.bin")
    epoch_metrics = []
    start_time = time.time()
    for epoch in range(1, int(epochs) + 1):
        loss = train_replay_epoch(model, ood_train_loader, replay_iter, optimizer, scheduler, scaler, f"[{trial_name}] epoch {epoch}/{epochs}")
        id_eval = evaluate_generation(model, tokenizer, id_loader, desc=f"[{trial_name}] epoch {epoch} ID", num_beams=eval_num_beams)
        ood_eval = evaluate_generation(model, tokenizer, ood_loader, desc=f"[{trial_name}] epoch {epoch} OOD", num_beams=eval_num_beams)
        score = float(ood_eval["exact_match"])
        is_best = score > best_score
        if is_best:
            best_score, best_epoch = score, epoch
            best_id_metrics = dict(id_eval)
            best_ood_metrics = dict(ood_eval)
            if keep_model:
                torch.save(model.state_dict(), best_model_path)
        epoch_metrics.append({"epoch": epoch, "loss": loss, "id_test": id_eval, "ood_test": ood_eval, "select_score": score, "is_best": is_best})
        print(f"{trial_name} | epoch={epoch} | ID_EM={id_eval['exact_match']:.4f} | OOD_EM={ood_eval['exact_match']:.4f} | loss={loss['total_loss']:.4f}")

    final_id = best_id_metrics
    final_ood = best_ood_metrics
    result = {
        "trial_name": trial_name,
        "best_epoch": int(best_epoch),
        "select_score": float(final_ood["exact_match"]),
        "id_test": final_id,
        "ood_test": final_ood,
        "drop_id_minus_ood": id_ood_drop(final_id, final_ood),
        "epoch_metrics": epoch_metrics,
        "files": {"best_model": best_model_path if keep_model else None, "ood_train_file": None, "ood_test_file": None, "id_predictions": None, "ood_predictions": None},
        "time_seconds": time.time() - start_time,
        "time_readable": format_time(time.time() - start_time),
    }
    if not keep_model:
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        result["files"]["best_model"] = None
        if os.path.isdir(trial_dir) and not os.listdir(trial_dir):
            os.rmdir(trial_dir)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_trial(model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds, metric_records, search_cfg, phase, index, epochs, save_dir, keep_model=False, eval_num_beams=1):
    trial_name = f"{phase}_{index:03d}_{search_cfg['name']}"
    train_raw, test_raw, split_report = build_geometry_split(metric_records, search_cfg)
    result = train_replay_trial(
        model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds, train_raw, test_raw,
        trial_name, os.path.join(save_dir, phase, safe_name(trial_name)), epochs, keep_model, eval_num_beams,
    )
    return {
        "status": "ok",
        "phase": phase,
        "trial": trial_name,
        "config": search_cfg,
        "features": config_feature_vector(search_cfg),
        "split_report": split_report,
        "select_score": result["select_score"],
        "id_test": result["id_test"],
        "ood_test": result["ood_test"],
        "result": result,
    }


def run_phase(model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds, metric_records, configs, phase, epochs, save_dir, keep_models=False, eval_num_beams=1):
    rows, seen_test_hashes = [], set()
    for i, cfg in enumerate(configs, start=1):
        print("\n" + "=" * 120)
        print(f"RQ5 {phase}: trial {i}/{len(configs)} | model={model_type} | epochs={epochs}")
        print("weights:", cfg["weights"], "ratios:", cfg["ratios"], "boundaries:", cfg.get("boundaries"))
        print("=" * 120)
        try:
            _, _, preview = build_geometry_split(metric_records, cfg)
            split_key = json.dumps(preview["groups"], sort_keys=True) + json.dumps(cfg, sort_keys=True)
            if split_key in seen_test_hashes:
                rows.append({"status": "duplicate", "phase": phase, "trial_index": i, "config": cfg, "split_report": preview})
                continue
            seen_test_hashes.add(split_key)
            row = run_trial(model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds, metric_records, cfg, phase, i, epochs, save_dir, keep_models, eval_num_beams)
            rows.append(row)
        except Exception as exc:
            print(f"[Error] {phase} trial {i} failed: {repr(exc)}")
            rows.append({"status": "failed", "phase": phase, "trial_index": i, "config": cfg, "error": repr(exc)})
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def propose_stage2(stage1_rows):
    valid = [r for r in stage1_rows if r.get("status") == "ok"]
    valid.sort(key=lambda r: float(r["select_score"]), reverse=True)
    configs = [r["config"] for r in valid[:SMART_STAGE2_FROM_STAGE1]]
    rng = np.random.RandomState(SEED + 2025)
    for i in range(SMART_STAGE2_RANDOM):
        configs.append(random_geometry_config(rng, f"stage2_random_{i}", SEED + 300 + i))
    configs.extend(fit_surrogate_and_propose(stage1_rows, max(0, SMART_STAGE2_TRIALS - len(configs)), seed=SEED + 900))
    unique, seen = [], set()
    for cfg in configs:
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(cfg)
        if len(unique) >= SMART_STAGE2_TRIALS:
            break
    return unique


# ============================================================
# 9. Output
# ============================================================

def cleanup_stale_trial_checkpoints(root_dir):
    if not os.path.isdir(root_dir):
        return 0
    removed = 0
    for dirpath, _, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            if not (filename.startswith("best_rq5_replay_") and filename.endswith(".bin")):
                continue
            path = os.path.join(dirpath, filename)
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        if dirpath != root_dir and os.path.isdir(dirpath) and not os.listdir(dirpath):
            os.rmdir(dirpath)
    if removed:
        print(f">>> Removed stale intermediate RQ5 checkpoints: {removed}")
    return removed

def keep_only_final_model(final_rows, best, model_type):
    if best is None:
        return None

    source_path = best.get("result", {}).get("files", {}).get("best_model")
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError("Final full-budget checkpoint was not created.")

    source_trial_dir = os.path.dirname(source_path)
    final_path = os.path.join(
        SAVE_ROOT,
        f"rq5_final_best_{SUBSET_NAME}_cst_{model_type}.bin",
    )
    if os.path.abspath(source_path) != os.path.abspath(final_path):
        if os.path.exists(final_path):
            os.remove(final_path)
        os.replace(source_path, final_path)

    best["result"]["files"]["best_model"] = final_path

    if os.path.isdir(source_trial_dir) and not os.listdir(source_trial_dir):
        os.rmdir(source_trial_dir)

    for row in final_rows:
        path = row.get("result", {}).get("files", {}).get("best_model")
        if path and os.path.abspath(path) != os.path.abspath(final_path):
            if os.path.exists(path):
                os.remove(path)
            row["result"]["files"]["best_model"] = None

        trial_dir = os.path.dirname(path) if path else None
        if trial_dir and os.path.isdir(trial_dir) and not os.listdir(trial_dir):
            os.rmdir(trial_dir)

    return final_path


def metric_cache_file(model_type):
    return os.path.join(SAVE_ROOT, f"rq5_rq3_sample_metrics_{SUBSET_NAME}_cst_{model_type}.json")


def results_file(model_type):
    return os.path.join(SAVE_ROOT, f"rq5_stage1_split_search_results_{SUBSET_NAME}_cst_{model_type}.json")


def top_configs_file(model_type):
    return os.path.join(SAVE_ROOT, f"rq5_stage1_top_configs_{SUBSET_NAME}_cst_{model_type}.json")


def final_best_tsv_file(model_type):
    return os.path.join(SAVE_ROOT, f"rq5_stage1_final_best_{SUBSET_NAME}_cst_{model_type}_excel.tsv")


def build_final_rows(model_type, best):
    if best is None:
        return []
    rows = []
    for metric_name, key, drop_key in [
        ("Exact Match", "exact_match", "exact_match_drop"),
        ("BLEU", "bleu", "bleu_drop"),
        ("CodeBLEU", "codebleu", "codebleu_drop"),
    ]:
        rows.append({
            "Model": model_type,
            "Subset": SUBSET_NAME,
            "Shift": "cst",
            "Trial": best["trial"],
            "BestEpoch": best["result"]["best_epoch"],
            "SelectScore": best["select_score"],
            "Metric": metric_name,
            "ID_Test": best["id_test"][key],
            "OOD_Test": best["ood_test"][key],
            "Drop_ID_minus_OOD": best["result"]["drop_id_minus_ood"][drop_key],
        })
    return rows


def write_final_tsv(path, rows):
    header = ["Model", "Subset", "Shift", "Trial", "BestEpoch", "SelectScore", "Metric", "ID_Test", "OOD_Test", "Drop_ID_minus_OOD"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def print_final_best(model_type, best, tsv_path):
    print("\n" + "=" * 120)
    print(f"RQ5 STAGE-1 FINAL BEST\tBugs2Fix-{SUBSET_NAME}\tcst\t{model_type}")
    print("=" * 120)
    if best is None:
        print("No valid best result.")
        return
    print("Trial:", best["trial"])
    print("Best epoch:", best["result"]["best_epoch"])
    print("Select score:", f"{best['select_score']:.6f}")
    print("Weights:", best["config"]["weights"])
    print("Ratios:", best["config"]["ratios"])
    print("Boundaries:", best["config"].get("boundaries"))
    print("-" * 120)
    print(f"{'Metric':<14} | {'ID Test':<10} | {'OOD Test':<10} | {'Drop ID-OOD':<12}")
    for row in build_final_rows(model_type, best):
        print(f"{row['Metric']:<14} | {row['ID_Test']:<10.4f} | {row['OOD_Test']:<10.4f} | {row['Drop_ID_minus_OOD']:<12.4f}")
    print("-" * 120)
    print("EXCEL_COPY_STAGE1_FINAL_BEST")
    print("\t".join(["Model", "Subset", "Shift", "Trial", "BestEpoch", "SelectScore", "Metric", "ID_Test", "OOD_Test", "Drop_ID_minus_OOD"]))
    for row in build_final_rows(model_type, best):
        print("\t".join([
            str(row["Model"]), str(row["Subset"]), str(row["Shift"]), str(row["Trial"]),
            str(row["BestEpoch"]), f"{row['SelectScore']:.6f}", str(row["Metric"]),
            f"{row['ID_Test']:.4f}", f"{row['OOD_Test']:.4f}", f"{row['Drop_ID_minus_OOD']:.4f}",
        ]))
    print("Final best TSV:", tsv_path)
    print("=" * 120)


# ============================================================
# 10. Run
# ============================================================

def run_one_model(model_type):
    set_seed(SEED)
    model_type = normalize_model_type(model_type)
    model_path = MODEL_PATHS[model_type]
    rq1_best_model = os.path.join(RQ1_OUTPUT_ROOT, model_type, SUBSET_NAME, f"best_bugs2fix_{model_type}.bin")
    model_save_dir = os.path.join(SAVE_ROOT, model_type, SUBSET_NAME)
    os.makedirs(model_save_dir, exist_ok=True)
    cleanup_stale_trial_checkpoints(model_save_dir)

    print("=" * 120)
    print("CodeRefinement RQ5 Stage 1: RQ3-guided Replay split search")
    print("=" * 120)
    print("MODEL_TYPE:", model_type)
    print("RQ1_BEST_MODEL:", rq1_best_model)
    print("Fixed Replay:", {"lambda": REPLAY_LAMBDA, "batch_size": REPLAY_BATCH_SIZE, "buffer_size": REPLAY_BUFFER_SIZE, "lr": LR, "epochs": EPOCHS})
    print("Search dimensions: w_alignment, w_entropy, w_anisotropy, w_rank, hard_end, middle_end, test_ratio_hard/middle/easy")
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id_train_raw = read_jsonl(ID_TRAIN_FILE)
    id_test_raw = read_jsonl(ID_TEST_FILE)
    ood_pool_raw, ood_sources = build_ood_pool()

    id_train_ds = Bugs2FixDataset(id_train_raw, tokenizer, max_samples=MAX_TRAIN_SAMPLES, name="id_train")
    id_test_ds = Bugs2FixDataset(id_test_raw, tokenizer, max_samples=MAX_EVAL_SAMPLES, name="id_test")
    replay_ds, replay_indices = build_replay_buffer_dataset(id_train_ds, REPLAY_BUFFER_SIZE, REPLAY_BUFFER_SEED)

    metric_records = compute_rq3_sample_metrics(
        model_type, model_path, rq1_best_model, tokenizer, id_test_raw, ood_pool_raw, metric_cache_file(model_type)
    )

    stage1_configs = initial_geometry_configs(SEED)[:SMART_STAGE1_TRIALS]
    stage1_rows = run_phase(model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds, metric_records, stage1_configs, "stage1_probe", SMART_STAGE1_EPOCHS, model_save_dir)
    stage2_rows = run_phase(model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds, metric_records, propose_stage2(stage1_rows), "stage2_surrogate", SMART_STAGE2_EPOCHS, model_save_dir)
    all_rows = stage1_rows + stage2_rows
    final_configs = unique_top_configs(all_rows, SMART_FINAL_TRIALS)
    final_rows = run_phase(
        model_type, model_path, rq1_best_model, tokenizer, id_test_ds, replay_ds,
        metric_records, final_configs, "stage3_full", SMART_FINAL_EPOCHS, model_save_dir,
        keep_models=True, eval_num_beams=NUM_BEAMS
    )

    final_ok = [r for r in final_rows if r.get("status") == "ok"]
    fallback_ok = [r for r in all_rows if r.get("status") == "ok"]
    candidates = final_ok or fallback_ok
    candidates.sort(key=lambda r: float(r["select_score"]), reverse=True)
    best = candidates[0] if candidates else None
    final_best_model = keep_only_final_model(
        final_rows,
        best if final_ok else None,
        model_type,
    )

    best_train_file = best_test_file = None
    if best is not None:
        train_raw, test_raw, best_split_report = build_geometry_split(metric_records, best["config"])
    else:
        best_split_report = None

    final_tsv = final_best_tsv_file(model_type)
    write_final_tsv(final_tsv, build_final_rows(model_type, best))
    output = {
        "task": "Code Repair / Code Refinement",
        "method": "RQ5 Stage-1 RQ3-guided OOD split search with fixed RQ4 Replay",
        "dataset": "CodeXGLUE Bugs2Fix",
        "subset": SUBSET_NAME,
        "shift_type": "cst",
        "model_type": model_type,
        "rq1_best_model": rq1_best_model,
        "ood_pool_sources": ood_sources,
        "fixed_replay_settings": {
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "replay_lambda": REPLAY_LAMBDA,
            "replay_batch_size": REPLAY_BATCH_SIZE,
            "replay_buffer_size": REPLAY_BUFFER_SIZE,
            "replay_buffer_indices": replay_indices,
        },
        "rq3_metrics": {
            "alignment": "one-to-one cosine(ID source representation, OOD source representation); lower is harder",
            "entropy": "OOD decoder predictive token entropy under shifted teacher forcing; higher is harder",
            "anisotropy": "local OOD source representation concentration; higher is harder",
            "rank": "local OOD source effective rank; lower is harder",
        },
        "search_space_version": RQ5_SEARCH_SPACE_VERSION,
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "final_rows": final_rows,
        "best_result": best,
        "best_split_report": best_split_report,
        "files": {
            "metrics_cache": metric_cache_file(model_type),
            "best_ood_train": best_train_file,
            "best_ood_test": best_test_file,
            "final_best_model": final_best_model,
            "final_best_tsv": final_tsv,
        },
    }
    save_json(results_file(model_type), output)
    top_source = sorted((final_ok + fallback_ok), key=lambda r: float(r["select_score"]), reverse=True)
    save_json(top_configs_file(model_type), {
        "model_type": model_type,
        "subset": SUBSET_NAME,
        "shift_type": "cst",
        "search_space_version": RQ5_SEARCH_SPACE_VERSION,
        "source_results_file": results_file(model_type),
        "top_k": SMART_TOP_CONFIGS_K,
        "top_configs": [
            {
                "rank": i + 1,
                "config": row["config"],
                "select_score": row["select_score"],
                "id_test": row["id_test"],
                "ood_test": row["ood_test"],
                "split_report": row["split_report"],
            }
            for i, row in enumerate(top_source[:SMART_TOP_CONFIGS_K])
        ],
    })
    print_final_best(model_type, best, final_tsv)
    print("Results saved:", results_file(model_type))
    print("Top configs saved:", top_configs_file(model_type))
    return output


def main():
    models = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    outputs = []
    for model_type in models:
        try:
            outputs.append(run_one_model(model_type))
        except Exception as exc:
            if not RUN_ALL_MODELS:
                raise
            outputs.append({"status": "failed", "model_type": model_type, "error": repr(exc)})
            print(f"[Error] {model_type} failed: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if RUN_ALL_MODELS:
        save_json(os.path.join(SAVE_ROOT, f"rq5_stage1_all_models_{SUBSET_NAME}_cst.json"), {"results": outputs})


if __name__ == "__main__":
    main()
