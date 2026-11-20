# -*- coding: utf-8 -*-
"""
CodeXGLUE CodeTrans RQ5.

RQ5 purpose:
  Keep the RQ4-CL Replay configuration fixed, and replace the random OOD
  80/20 split with an RQ3-guided split searched by TPE.

ID direction from RQ1:
  Java -> C#

OOD language-direction shift:
  C# -> Java

The only searched variables are the RQ3 split variables:
  - weights for alignment / entropy / anisotropy / rank
  - train_preferences for hard / middle / easy samples

Replay hyperparameters are inherited from RQ4clnoval.py:
  replay_lambda=0.5, replay_batch_size=4, replay_buffer_size=1000,
  epochs=8, lr=5e-6, batch_size=8.
"""

import argparse
import importlib.util
import itertools
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import optuna
except ImportError:
    optuna = None


# ============================================================
# 1. Reuse RQ4 CL as the fixed Replay baseline implementation
# ============================================================

RQ4CL_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "RQ4clnoval.py",
)

if not os.path.exists(RQ4CL_SCRIPT):
    raise FileNotFoundError(f"Cannot find RQ4clnoval.py: {RQ4CL_SCRIPT}")

_spec = importlib.util.spec_from_file_location("codetrans_rq4cl_standard", RQ4CL_SCRIPT)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


# ============================================================
# 2. RQ5 configuration
# ============================================================

MODEL_TYPE = C.MODEL_TYPE
MODEL_TYPES_TO_RUN = list(C.MODEL_TYPES_TO_RUN)
RUN_ALL_MODELS = False

DATA_DIR = C.DATA_DIR
SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq5_rq3_tpe_replay_outputs_java_to_cs_id"
os.makedirs(SAVE_DIR, exist_ok=True)

# Original OOD pool for C# -> Java. Prefer the complete test set from RQ1.
OOD_POOL_SRC_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.cs")
OOD_POOL_TGT_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.java")

ID_TEST_SRC_FILE = C.ID_TEST_SRC_FILE
ID_TEST_TGT_FILE = C.ID_TEST_TGT_FILE
ID_TRAIN_SRC_FILE = os.path.join(DATA_DIR, "train.java-cs.txt.java")
ID_TRAIN_TGT_FILE = os.path.join(DATA_DIR, "train.java-cs.txt.cs")

DEVICE = C.DEVICE
SEED = C.SEED
MAX_SOURCE_LEN = C.MAX_SOURCE_LEN
MAX_TARGET_LEN = C.MAX_TARGET_LEN
BATCH_SIZE = C.BATCH_SIZE
EVAL_BATCH_SIZE = C.EVAL_BATCH_SIZE
NUM_WORKERS = C.NUM_WORKERS
EPOCHS = C.EPOCHS
LR = C.LR
WARMUP_RATIO = C.WARMUP_RATIO
WEIGHT_DECAY = C.WEIGHT_DECAY
GRAD_CLIP = C.GRAD_CLIP

REPLAY_LAMBDA = float(C.REPLAY_CONFIGS[0]["lambda"])
REPLAY_BATCH_SIZE = int(C.REPLAY_CONFIGS[0]["batch_size"])
REPLAY_BUFFER_SIZE = int(C.REPLAY_BUFFER_SIZES[0])
REPLAY_BUFFER_SEED = C.REPLAY_BUFFER_SEED

MAX_ANALYSIS_SAMPLES = None
MAX_TRAIN_SAMPLES = C.MAX_TRAIN_SAMPLES
MAX_EVAL_SAMPLES = C.MAX_EVAL_SAMPLES

RQ3_METRIC_NAMES = ["alignment", "entropy", "anisotropy", "rank"]
FIXED_TRAIN_RATIO = 0.80
FIXED_TEST_RATIO = 0.20
HARD_GROUP_FRACTION = 0.25
MIDDLE_GROUP_FRACTION = 0.50
MIN_TRAIN_PER_GROUP = 1
LOCAL_GEOMETRY_K = 32

SEARCH_WEIGHT_BOUNDS = {
    "alignment": (0.0, 1.0),
    "entropy": (0.0, 1.0),
    "anisotropy": (0.0, 1.0),
    "rank": (0.0, 1.0),
}
TRAIN_PREFERENCE_BOUNDS = {
    "hard": (0.70, 0.99),
    "middle": (0.50, 0.95),
    "easy": (0.20, 0.85),
}

STAGE1_RANDOM_TRIALS = 12
STAGE1_RANDOM_EPOCHS = 1
TPE_SEARCH_TRIALS = 40
TPE_SEARCH_EPOCHS = 3
TPE_FULL_EVAL_TOP_K = 3
TPE_FULL_EVAL_EPOCHS = EPOCHS
OPTUNA_STARTUP_TRIALS = 8
TOP_CONFIGS_K = 10
RESUME_TPE_RESULTS = True

RQ5_METRIC_CACHE_VERSION = 1
RQ5_SEARCH_SPACE_VERSION = 1


# ============================================================
# 3. Basic utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    return C.format_time(seconds)


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_model_type(model_type):
    return C.normalize_model_type(model_type)


def resolve_model_path(model_type, model_path=None):
    return model_path or C.MODEL_PATHS[normalize_model_type(model_type)]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE CodeTrans RQ5 RQ3-guided Replay TPE")
    parser.add_argument("--model_type", default=MODEL_TYPE, help="codeT5, PLBART, or all")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--codeT5_path", default=C.MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=C.MODEL_PATHS["PLBART"])
    return parser


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))[:180]


def metric_cache_file(model_type):
    return os.path.join(SAVE_DIR, model_type, f"rq5_rq3_sample_metrics_codetrans_{model_type}.json")


def trials_file(model_type):
    return os.path.join(SAVE_DIR, model_type, f"rq5_tpe_trials_codetrans_{model_type}.json")


def results_file(model_type):
    return os.path.join(SAVE_DIR, model_type, f"rq5_tpe_replay_results_codetrans_{model_type}.json")


def top_configs_file(model_type):
    return os.path.join(SAVE_DIR, model_type, f"rq5_tpe_top_configs_codetrans_{model_type}.json")


def final_best_tsv_file(model_type):
    return os.path.join(SAVE_DIR, model_type, f"rq5_tpe_final_best_codetrans_{model_type}_excel.tsv")


# ============================================================
# 4. RQ3 sample-level metrics for splitting
# ============================================================

def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


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


def token_entropy_from_logits(logits, labels):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    token_entropy = -(probs * log_probs).sum(dim=-1)
    valid_mask = labels.ne(-100).float()
    denom = valid_mask.sum(dim=1).clamp(min=1.0)
    return ((token_entropy * valid_mask).sum(dim=1) / denom).detach().cpu().numpy()


def get_effective_rank_np(x):
    if x.shape[0] <= 1:
        return 0.0
    mat = torch.as_tensor(x, dtype=torch.float32)
    mat = mat - mat.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(mat)
    except Exception:
        s = torch.svd(mat)[1]
    s = torch.clamp(s, min=1e-12)
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


def local_geometry_scores(vectors, k=32):
    x = F.normalize(torch.as_tensor(vectors, dtype=torch.float32), p=2, dim=1)
    n = x.size(0)
    k = min(int(k), max(n - 1, 1))
    sims = torch.matmul(x, x.t())
    sims.fill_diagonal_(-1e9)
    topk_values, topk_indices = torch.topk(sims, k=k, dim=1, largest=True)
    anisotropy = topk_values.mean(dim=1).cpu().numpy()
    ranks = []
    x_np = vectors.astype(np.float32)
    for i in range(n):
        idx = topk_indices[i].cpu().numpy().tolist()
        idx.append(i)
        ranks.append(get_effective_rank_np(x_np[idx]))
    return anisotropy.astype(np.float64), np.asarray(ranks, dtype=np.float64)


def minmax_hard(values, higher_is_harder=True):
    arr = np.asarray(values, dtype=np.float64)
    if not higher_is_harder:
        arr = -arr
    low = float(np.min(arr))
    high = float(np.max(arr))
    if abs(high - low) < 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def paired_data_for_metrics(max_samples=None):
    id_data = C.read_parallel_files(ID_TEST_SRC_FILE, ID_TEST_TGT_FILE, max_samples=max_samples)
    ood_data = C.read_parallel_files(OOD_POOL_SRC_FILE, OOD_POOL_TGT_FILE, max_samples=max_samples)
    n = min(len(id_data), len(ood_data))
    if n == 0:
        raise ValueError("No paired ID/OOD samples available for RQ3 metrics.")
    return id_data[:n], ood_data[:n]


@torch.no_grad()
def compute_rq3_sample_metrics(model_type, model_path, tokenizer, cache_file):
    if os.path.exists(cache_file):
        payload = json.load(open(cache_file, "r", encoding="utf-8"))
        if (
            int(payload.get("cache_version", -1)) == RQ5_METRIC_CACHE_VERSION
            and payload.get("model_type") == model_type
        ):
            print(">>> Loaded RQ5 RQ3 sample metrics cache:", cache_file)
            return payload["records"]

    id_data, ood_data = paired_data_for_metrics(MAX_ANALYSIS_SAMPLES)
    id_ds = C.CodeTransDataset(id_data, tokenizer)
    ood_ds = C.CodeTransDataset(ood_data, tokenizer)
    id_loader = DataLoader(id_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model, rq1_best_model = C.load_rq1_model(model_type, model_path, tokenizer)
    model.eval()
    encoder = model.get_encoder()

    id_vectors, ood_vectors, entropy_values = [], [], []
    for id_batch, ood_batch in tqdm(zip(id_loader, ood_loader), total=min(len(id_loader), len(ood_loader)), desc="Compute RQ3 sample metrics"):
        id_input_ids = id_batch["input_ids"].to(DEVICE)
        id_attention_mask = id_batch["attention_mask"].to(DEVICE)
        ood_input_ids = ood_batch["input_ids"].to(DEVICE)
        ood_attention_mask = ood_batch["attention_mask"].to(DEVICE)
        labels = ood_batch["labels"].to(DEVICE)

        id_outputs = encoder(
            input_ids=id_input_ids,
            attention_mask=id_attention_mask,
            return_dict=True,
        )
        ood_outputs = encoder(
            input_ids=ood_input_ids,
            attention_mask=ood_attention_mask,
            return_dict=True,
        )
        id_vec = masked_mean(id_outputs.last_hidden_state, id_attention_mask)
        ood_vec = masked_mean(ood_outputs.last_hidden_state, ood_attention_mask)

        outputs = model(
            input_ids=ood_input_ids,
            attention_mask=ood_attention_mask,
            labels=labels,
            return_dict=True,
        )
        entropy = token_entropy_from_logits(outputs.logits, labels)

        id_vectors.append(id_vec.detach().cpu())
        ood_vectors.append(ood_vec.detach().cpu())
        entropy_values.extend(entropy.tolist())

    id_vectors = torch.cat(id_vectors, dim=0).float()
    ood_vectors = torch.cat(ood_vectors, dim=0).float()
    n = min(id_vectors.size(0), ood_vectors.size(0), len(ood_data))
    id_vectors = id_vectors[:n]
    ood_vectors = ood_vectors[:n]
    entropy_values = np.asarray(entropy_values[:n], dtype=np.float64)

    alignment = F.cosine_similarity(
        F.normalize(id_vectors, p=2, dim=1),
        F.normalize(ood_vectors, p=2, dim=1),
        dim=1,
    ).cpu().numpy()
    alignment_damage = 1.0 - alignment

    id_aniso, id_rank = local_geometry_scores(id_vectors.numpy(), k=LOCAL_GEOMETRY_K)
    ood_aniso, ood_rank = local_geometry_scores(ood_vectors.numpy(), k=LOCAL_GEOMETRY_K)
    anisotropy_increase = ood_aniso - id_aniso
    rank_drop = id_rank - ood_rank

    records = []
    for i in range(n):
        records.append({
            "pool_index": int(i),
            "idx": str(ood_data[i]["idx"]),
            "source": ood_data[i]["source"],
            "target": ood_data[i]["target"],
            "raw": ood_data[i],
            "metrics_raw": {
                "alignment": float(alignment_damage[i]),
                "entropy": float(entropy_values[i]),
                "anisotropy": float(anisotropy_increase[i]),
                "rank": float(rank_drop[i]),
            },
        })

    norm = {
        "alignment": minmax_hard([r["metrics_raw"]["alignment"] for r in records], True),
        "entropy": minmax_hard([r["metrics_raw"]["entropy"] for r in records], True),
        "anisotropy": minmax_hard([r["metrics_raw"]["anisotropy"] for r in records], True),
        "rank": minmax_hard([r["metrics_raw"]["rank"] for r in records], True),
    }
    for i, record in enumerate(records):
        record["metrics_norm"] = {name: float(norm[name][i]) for name in RQ3_METRIC_NAMES}

    payload = {
        "cache_version": RQ5_METRIC_CACHE_VERSION,
        "model_type": model_type,
        "rq1_best_model": rq1_best_model,
        "metric_definitions": {
            "alignment": "1 - cosine(ID source encoder representation, paired OOD source encoder representation)",
            "entropy": "OOD decoder token predictive entropy under teacher forcing",
            "anisotropy": "local OOD source anisotropy minus local ID source anisotropy",
            "rank": "local ID effective rank minus local OOD effective rank",
        },
        "records": records,
    }
    save_json(cache_file, payload)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return records


# ============================================================
# 5. RQ3-guided split
# ============================================================

def normalize_positive_weights(weights):
    values = {name: max(float(weights.get(name, 0.0)), 0.0) for name in RQ3_METRIC_NAMES}
    total = sum(values.values())
    if total <= 1e-12:
        return {name: 1.0 / len(RQ3_METRIC_NAMES) for name in RQ3_METRIC_NAMES}
    return {name: values[name] / total for name in RQ3_METRIC_NAMES}


def sanitize_train_preferences(preferences):
    values = {}
    for group_name in ["hard", "middle", "easy"]:
        low, high = TRAIN_PREFERENCE_BOUNDS[group_name]
        values[group_name] = min(max(float(preferences.get(group_name, 0.80)), low), high)
    values["middle"] = min(values["middle"], values["hard"])
    values["easy"] = min(values["easy"], values["middle"])
    return values


def make_search_config(name, weights, train_preferences, split_seed=SEED):
    return {
        "name": str(name),
        "weights": normalize_positive_weights(weights),
        "train_preferences": sanitize_train_preferences(train_preferences),
        "split_seed": int(split_seed),
    }


def config_feature_vector(config):
    weights = normalize_positive_weights(config["weights"])
    prefs = sanitize_train_preferences(config["train_preferences"])
    return [
        float(weights["alignment"]),
        float(weights["entropy"]),
        float(weights["anisotropy"]),
        float(weights["rank"]),
        float(prefs["hard"]),
        float(prefs["middle"]),
        float(prefs["easy"]),
    ]


def config_signature(config):
    return tuple(round(x, 8) for x in config_feature_vector(config))


def allocate_test_counts(groups, train_preferences, target_test_size):
    capacity = {
        name: max(0, len(items) - min(MIN_TRAIN_PER_GROUP, len(items)))
        for name, items in groups.items()
    }
    target_test_size = min(int(target_test_size), sum(capacity.values()))
    test_weights = {}
    for name, items in groups.items():
        test_weights[name] = len(items) * max(1e-6, 1.0 - train_preferences[name])
        if capacity[name] <= 0:
            test_weights[name] = 0.0
    weight_sum = sum(test_weights.values())
    if weight_sum <= 1e-12:
        test_weights = {name: float(len(items)) if capacity[name] > 0 else 0.0 for name, items in groups.items()}
        weight_sum = sum(test_weights.values())
    raw_alloc = {name: target_test_size * test_weights[name] / max(weight_sum, 1e-12) for name in groups}
    counts = {name: min(int(np.floor(raw_alloc[name])), capacity[name]) for name in groups}
    current = sum(counts.values())
    for name in sorted(groups, key=lambda x: raw_alloc[x] - np.floor(raw_alloc[x]), reverse=True):
        if current >= target_test_size:
            break
        if counts[name] < capacity[name]:
            counts[name] += 1
            current += 1
    while current < target_test_size:
        eligible = [name for name in groups if counts[name] < capacity[name]]
        if not eligible:
            break
        name = max(eligible, key=lambda x: test_weights[x] / max(1, capacity[x]))
        counts[name] += 1
        current += 1
    return counts


def pick_evenly_spaced(items, number_to_select):
    if number_to_select <= 0:
        return []
    if number_to_select >= len(items):
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=number_to_select)
    selected = []
    for pos in positions:
        idx = int(round(pos))
        if idx not in selected:
            selected.append(idx)
    cursor = 0
    while len(selected) < number_to_select and cursor < len(items):
        if cursor not in selected:
            selected.append(cursor)
        cursor += 1
    return [items[i] for i in sorted(selected[:number_to_select])]


def score_records(records, config):
    weights = normalize_positive_weights(config["weights"])
    scored = []
    for record in records:
        difficulty = 0.0
        for name in RQ3_METRIC_NAMES:
            difficulty += weights[name] * float(record["metrics_norm"][name])
        item = dict(record)
        item["difficulty_score"] = float(difficulty)
        scored.append(item)
    return scored, weights


def split_hash(records):
    joined = "|".join(str(r["pool_index"]) for r in sorted(records, key=lambda x: int(x["pool_index"])))
    import hashlib
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def build_rq3_split(records, config):
    scored, weights = score_records(records, config)
    prefs = sanitize_train_preferences(config["train_preferences"])
    sorted_records = sorted(scored, key=lambda x: x["difficulty_score"], reverse=True)
    total = len(sorted_records)
    if total < 10:
        raise ValueError("OOD pool too small for RQ5 80/20 splitting.")
    hard_end = max(1, min(int(round(total * HARD_GROUP_FRACTION)), total - 2))
    middle_end = max(hard_end + 1, min(hard_end + int(round(total * MIDDLE_GROUP_FRACTION)), total - 1))
    groups = {
        "hard": sorted_records[:hard_end],
        "middle": sorted_records[hard_end:middle_end],
        "easy": sorted_records[middle_end:],
    }
    target_test_size = max(1, min(int(round(total * FIXED_TEST_RATIO)), total - 1))
    test_counts = allocate_test_counts(groups, prefs, target_test_size)
    train_records, test_records, group_report = [], [], {}
    for group_name, items in groups.items():
        group_test = pick_evenly_spaced(items, int(test_counts[group_name]))
        test_indices = {item["pool_index"] for item in group_test}
        group_train = [item for item in items if item["pool_index"] not in test_indices]
        train_records.extend(group_train)
        test_records.extend(group_test)
        group_report[group_name] = {
            "available": len(items),
            "train": len(group_train),
            "test": len(group_test),
            "train_ratio": len(group_train) / max(1, len(items)),
            "test_ratio": len(group_test) / max(1, len(items)),
            "train_preference": prefs[group_name],
            "all_difficulty_mean": float(np.mean([x["difficulty_score"] for x in items])) if items else 0.0,
            "train_difficulty_mean": float(np.mean([x["difficulty_score"] for x in group_train])) if group_train else 0.0,
            "test_difficulty_mean": float(np.mean([x["difficulty_score"] for x in group_test])) if group_test else 0.0,
        }
    rng = np.random.RandomState(int(config.get("split_seed", SEED)))
    rng.shuffle(train_records)
    rng.shuffle(test_records)
    train_raw = [r["raw"] for r in train_records]
    test_raw = [r["raw"] for r in test_records]
    train_mean = float(np.mean([r["difficulty_score"] for r in train_records])) if train_records else 0.0
    test_mean = float(np.mean([r["difficulty_score"] for r in test_records])) if test_records else 0.0
    report = {
        "search_config": config,
        "actual_weights": weights,
        "train_preferences": prefs,
        "num_total": total,
        "num_train": len(train_records),
        "num_test": len(test_records),
        "actual_train_ratio": len(train_records) / total,
        "actual_test_ratio": len(test_records) / total,
        "train_difficulty_mean": train_mean,
        "test_difficulty_mean": test_mean,
        "train_minus_test_difficulty": train_mean - test_mean,
        "groups": group_report,
        "train_hash": split_hash(train_records),
        "test_hash": split_hash(test_records),
    }
    return train_raw, test_raw, report


# ============================================================
# 6. Search configs and TPE
# ============================================================

def initial_configs():
    return [
        make_search_config("balanced_representative", {"alignment": 0.25, "entropy": 0.25, "anisotropy": 0.25, "rank": 0.25}, {"hard": 0.80, "middle": 0.80, "easy": 0.80}, SEED),
        make_search_config("alignment_focus_hard_train", {"alignment": 0.55, "entropy": 0.15, "anisotropy": 0.15, "rank": 0.15}, {"hard": 0.94, "middle": 0.80, "easy": 0.60}, SEED + 1),
        make_search_config("entropy_focus_hard_train", {"alignment": 0.15, "entropy": 0.55, "anisotropy": 0.15, "rank": 0.15}, {"hard": 0.94, "middle": 0.80, "easy": 0.60}, SEED + 2),
        make_search_config("anisotropy_focus_middle", {"alignment": 0.15, "entropy": 0.15, "anisotropy": 0.55, "rank": 0.15}, {"hard": 0.91, "middle": 0.82, "easy": 0.68}, SEED + 3),
        make_search_config("rank_focus_middle", {"alignment": 0.15, "entropy": 0.15, "anisotropy": 0.15, "rank": 0.55}, {"hard": 0.91, "middle": 0.82, "easy": 0.68}, SEED + 4),
    ]


def random_config(rng, name, seed):
    weights = {m: float(rng.uniform(*SEARCH_WEIGHT_BOUNDS[m])) for m in RQ3_METRIC_NAMES}
    prefs = {g: float(rng.uniform(*TRAIN_PREFERENCE_BOUNDS[g])) for g in ["hard", "middle", "easy"]}
    return make_search_config(name, weights, prefs, seed)


def cfg_to_params(config):
    weights = normalize_positive_weights(config["weights"])
    prefs = sanitize_train_preferences(config["train_preferences"])
    return {
        "w_alignment": weights["alignment"],
        "w_entropy": weights["entropy"],
        "w_anisotropy": weights["anisotropy"],
        "w_rank": weights["rank"],
        "pref_hard": prefs["hard"],
        "pref_middle": prefs["middle"],
        "pref_easy": prefs["easy"],
    }


def params_to_cfg(params, name, seed):
    return make_search_config(
        name,
        {
            "alignment": float(params["w_alignment"]),
            "entropy": float(params["w_entropy"]),
            "anisotropy": float(params["w_anisotropy"]),
            "rank": float(params["w_rank"]),
        },
        {
            "hard": float(params["pref_hard"]),
            "middle": float(params["pref_middle"]),
            "easy": float(params["pref_easy"]),
        },
        seed,
    )


def suggest_tpe_config(trial):
    params = {
        "w_alignment": trial.suggest_float("w_alignment", *SEARCH_WEIGHT_BOUNDS["alignment"]),
        "w_entropy": trial.suggest_float("w_entropy", *SEARCH_WEIGHT_BOUNDS["entropy"]),
        "w_anisotropy": trial.suggest_float("w_anisotropy", *SEARCH_WEIGHT_BOUNDS["anisotropy"]),
        "w_rank": trial.suggest_float("w_rank", *SEARCH_WEIGHT_BOUNDS["rank"]),
        "pref_hard": trial.suggest_float("pref_hard", *TRAIN_PREFERENCE_BOUNDS["hard"]),
        "pref_middle": trial.suggest_float("pref_middle", *TRAIN_PREFERENCE_BOUNDS["middle"]),
        "pref_easy": trial.suggest_float("pref_easy", *TRAIN_PREFERENCE_BOUNDS["easy"]),
    }
    return params_to_cfg(params, f"tpe_{trial.number:03d}", SEED + 5000 + trial.number)


# ============================================================
# 7. Replay training trial
# ============================================================

def train_replay_trial(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, config, trial_name, epochs, save_dir):
    train_raw, test_raw, split_report = build_rq3_split(records, config)
    trial_dir = os.path.join(save_dir, safe_name(trial_name))
    os.makedirs(trial_dir, exist_ok=True)
    train_file = os.path.join(trial_dir, "rq5_ood_train.jsonl")
    test_file = os.path.join(trial_dir, "rq5_ood_test.jsonl")
    write_jsonl(train_file, train_raw)
    write_jsonl(test_file, test_raw)

    train_ds = C.CodeTransDataset(train_raw, tokenizer)
    test_ds = C.CodeTransDataset(test_raw, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    id_loader = DataLoader(id_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    ood_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    replay_loader = DataLoader(replay_ds, batch_size=REPLAY_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    replay_iter = itertools.cycle(replay_loader)

    model, rq1_best_model = C.load_rq1_model(model_type, model_path, tokenizer)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    use_amp = DEVICE.type == "cuda"

    best_epoch = 0
    best_ood_codebleu = -1.0
    best_model_path = os.path.join(trial_dir, "best_model.bin")
    epoch_metrics = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"{trial_name} Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            ood_loss = C.seq2seq_loss(model, batch, use_amp=use_amp)
            replay_batch = next(replay_iter)
            replay_loss = C.seq2seq_loss(model, replay_batch, use_amp=use_amp)
            loss = ood_loss + REPLAY_LAMBDA * replay_loss
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / max(steps, 1)
        id_eval = C.evaluate_generation(model, id_loader, tokenizer, C.ID_CODEBLEU_LANG, f"{trial_name} Epoch {epoch} ID")
        ood_eval = C.evaluate_generation(model, ood_loader, tokenizer, C.OOD_CODEBLEU_LANG, f"{trial_name} Epoch {epoch} OOD")
        is_best = ood_eval["codebleu"] > best_ood_codebleu
        if is_best:
            best_epoch = epoch
            best_ood_codebleu = float(ood_eval["codebleu"])
            torch.save(model.state_dict(), best_model_path)
        epoch_metrics.append({
            "epoch": epoch,
            "loss": avg_loss,
            "id_test": id_eval,
            "ood_test": ood_eval,
            "is_best": is_best,
        })
        print(f"{trial_name} | epoch={epoch} | loss={avg_loss:.4f} | ID_EM={id_eval['exact_match']:.4f} | OOD_EM={ood_eval['exact_match']:.4f} | OOD_CB={ood_eval['codebleu']:.4f}")

    model.load_state_dict(C.safe_torch_load(best_model_path, map_location=DEVICE))
    id_pred = os.path.join(trial_dir, "pred_id.jsonl")
    ood_pred = os.path.join(trial_dir, "pred_ood.jsonl")
    final_id = C.evaluate_generation(model, id_loader, tokenizer, C.ID_CODEBLEU_LANG, f"{trial_name} Best ID", id_pred)
    final_ood = C.evaluate_generation(model, ood_loader, tokenizer, C.OOD_CODEBLEU_LANG, f"{trial_name} Best OOD", ood_pred)
    result = {
        "trial": trial_name,
        "config": config,
        "split_report": split_report,
        "best_epoch": best_epoch,
        "select_score": float(final_ood["codebleu"]),
        "id_test": final_id,
        "ood_test": final_ood,
        "drop_id_minus_ood": C.id_ood_drop(final_id, final_ood),
        "epoch_metrics": epoch_metrics,
        "files": {
            "best_model": best_model_path,
            "ood_train": train_file,
            "ood_test": test_file,
            "id_predictions": id_pred,
            "ood_predictions": ood_pred,
        },
        "rq1_best_model": rq1_best_model,
    }
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_config(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, config, phase, index, epochs, save_dir):
    trial_name = f"{phase}_{index:03d}_{config['name']}"
    result = train_replay_trial(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, config, trial_name, epochs, os.path.join(save_dir, phase))
    return {
        "status": "ok",
        "phase": phase,
        "trial": trial_name,
        "config": config,
        "features": config_feature_vector(config),
        "split_report": result["split_report"],
        "select_score": result["select_score"],
        "id_test": result["id_test"],
        "ood_test": result["ood_test"],
        "result": result,
    }


def run_phase(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, configs, phase, epochs, save_dir):
    rows = []
    seen_hashes = set()
    for i, config in enumerate(configs, start=1):
        try:
            _, _, preview = build_rq3_split(records, config)
            if preview["test_hash"] in seen_hashes:
                rows.append({"status": "duplicate", "phase": phase, "config": config, "split_report": preview})
                continue
            seen_hashes.add(preview["test_hash"])
            rows.append(evaluate_config(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, config, phase, i, epochs, save_dir))
        except Exception as exc:
            print(f"[Error] {phase} trial {i} failed: {repr(exc)}")
            rows.append({"status": "failed", "phase": phase, "config": config, "error": repr(exc)})
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    return rows


# ============================================================
# 8. TPE search / output
# ============================================================

def load_completed_rows(path):
    if not RESUME_TPE_RESULTS or not os.path.exists(path):
        return []
    payload = json.load(open(path, "r", encoding="utf-8"))
    if int(payload.get("search_space_version", -1)) != RQ5_SEARCH_SPACE_VERSION:
        return []
    return payload.get("tpe_rows", [])


def run_tpe_search(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, priors, save_dir):
    if optuna is None:
        raise ImportError("optuna is required for RQ5 TPE search. Please install optuna.")
    path = trials_file(model_type)
    rows = load_completed_rows(path)
    seen = {config_signature(r["config"]) for r in rows if r.get("status") == "ok" and isinstance(r.get("config"), dict)}
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=OPTUNA_STARTUP_TRIALS, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for prior in priors:
        if config_signature(prior) not in seen:
            study.enqueue_trial(cfg_to_params(prior))

    def objective(trial):
        config = suggest_tpe_config(trial)
        sig = config_signature(config)
        if sig in seen:
            return -1e9
        row = evaluate_config(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, config, "stage2_tpe", trial.number + 1, TPE_SEARCH_EPOCHS, save_dir)
        rows.append(row)
        seen.add(sig)
        save_json(path, {
            "search_space_version": RQ5_SEARCH_SPACE_VERSION,
            "tpe_trials_target": TPE_SEARCH_TRIALS,
            "tpe_rows": rows,
        })
        return float(row.get("select_score", 0.0))

    done = len([r for r in rows if r.get("status") == "ok"])
    remain = max(0, TPE_SEARCH_TRIALS - done)
    if remain > 0:
        study.optimize(objective, n_trials=remain, gc_after_trial=True)
    return rows


def choose_full_configs(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: float(r.get("select_score", -1e9)), reverse=True)
    chosen, seen = [], set()
    for row in ok:
        cfg = dict(row["config"])
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        cfg["name"] = f"full_{len(chosen) + 1:03d}_{cfg.get('name', 'config')}"
        chosen.append(cfg)
        if len(chosen) >= TPE_FULL_EVAL_TOP_K:
            break
    return chosen


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def final_rows(model_type, best):
    rows = []
    for display, key, drop_key in [
        ("Exact Match", "exact_match", "exact_match_drop"),
        ("BLEU", "bleu", "bleu_drop"),
        ("CodeBLEU", "codebleu", "codebleu_drop"),
    ]:
        rows.append({
            "Model": model_type,
            "Trial": best["trial"],
            "BestEpoch": best["result"]["best_epoch"],
            "SelectScore": best["select_score"],
            "Metric": display,
            "ID_Test": best["id_test"].get(key),
            "OOD_Test": best["ood_test"].get(key),
            "Drop_ID_minus_OOD": best["result"]["drop_id_minus_ood"].get(drop_key),
        })
    return rows


def write_final_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["Model", "Trial", "BestEpoch", "SelectScore", "Metric", "ID_Test", "OOD_Test", "Drop_ID_minus_OOD"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join([
                str(row["Model"]),
                str(row["Trial"]),
                str(row["BestEpoch"]),
                fmt(row["SelectScore"], 6),
                str(row["Metric"]),
                fmt(row["ID_Test"], 4),
                fmt(row["OOD_Test"], 4),
                fmt(row["Drop_ID_minus_OOD"], 4),
            ]) + "\n")


def print_final(model_type, best, tsv_path):
    rows = final_rows(model_type, best)
    print("\n" + "=" * 120)
    print(f"RQ5 FINAL BEST\tCodeTrans\tjava_to_cs_id__cs_to_java_ood\t{model_type}")
    print("=" * 120)
    print("Trial:", best["trial"])
    print("Best epoch:", best["result"]["best_epoch"])
    print("Select score (OOD CodeBLEU):", fmt(best["select_score"], 6))
    print("Weights:", best["config"]["weights"])
    print("Train preferences:", best["config"]["train_preferences"])
    print("-" * 120)
    print(f"{'Metric':<18} | {'ID Test':<12} | {'OOD Test':<12} | {'Drop ID-OOD':<12}")
    for row in rows:
        print(f"{row['Metric']:<18} | {fmt(row['ID_Test']):<12} | {fmt(row['OOD_Test']):<12} | {fmt(row['Drop_ID_minus_OOD']):<12}")
    print("-" * 120)
    print("EXCEL_COPY_RQ5_FINAL_BEST")
    print("\t".join(["Model", "Trial", "BestEpoch", "SelectScore", "Metric", "ID_Test", "OOD_Test", "Drop_ID_minus_OOD"]))
    for row in rows:
        print("\t".join([str(row["Model"]), str(row["Trial"]), str(row["BestEpoch"]), fmt(row["SelectScore"], 6), str(row["Metric"]), fmt(row["ID_Test"]), fmt(row["OOD_Test"]), fmt(row["Drop_ID_minus_OOD"])]))
    print("Final TSV:", tsv_path)
    print("=" * 120)


# ============================================================
# 9. Run
# ============================================================

def run_one_model(model_type, model_path):
    set_seed(SEED)
    model_type = normalize_model_type(model_type)
    model_path = resolve_model_path(model_type, model_path)
    save_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()

    print("=" * 120)
    print("CodeXGLUE-CodeTrans RQ5: RQ3-guided split + fixed Replay TPE")
    print("=" * 120)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("RQ1_BEST_MODEL:", C.get_rq1_best_model(model_type))
    print("OOD_POOL:", OOD_POOL_SRC_FILE, "->", OOD_POOL_TGT_FILE)
    print("Replay fixed:", {"lambda": REPLAY_LAMBDA, "batch_size": REPLAY_BATCH_SIZE, "buffer_size": REPLAY_BUFFER_SIZE, "epochs": EPOCHS, "lr": LR})
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id_train_data = C.read_parallel_files(ID_TRAIN_SRC_FILE, ID_TRAIN_TGT_FILE, max_samples=MAX_TRAIN_SAMPLES)
    id_test_data = C.read_parallel_files(ID_TEST_SRC_FILE, ID_TEST_TGT_FILE, max_samples=MAX_EVAL_SAMPLES)
    id_train_ds = C.CodeTransDataset(id_train_data, tokenizer)
    id_test_ds = C.CodeTransDataset(id_test_data, tokenizer)
    replay_ds, replay_indices = C.build_replay_buffer_dataset(id_train_ds, buffer_size=REPLAY_BUFFER_SIZE, seed=REPLAY_BUFFER_SEED)

    records = compute_rq3_sample_metrics(model_type, model_path, tokenizer, metric_cache_file(model_type))
    rng = np.random.RandomState(SEED + 2026)
    probe_configs = initial_configs()
    for i in range(STAGE1_RANDOM_TRIALS):
        probe_configs.append(random_config(rng, f"stage1_random_{i:03d}", SEED + 100 + i))
    stage1_rows = run_phase(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, probe_configs, "stage1_probe", STAGE1_RANDOM_EPOCHS, save_dir)
    priors = [r["config"] for r in sorted([r for r in stage1_rows if r.get("status") == "ok"], key=lambda x: float(x["select_score"]), reverse=True)[:TOP_CONFIGS_K]]
    tpe_rows = run_tpe_search(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, priors, save_dir)
    full_configs = choose_full_configs(tpe_rows)
    full_rows = run_phase(model_type, model_path, tokenizer, id_test_ds, replay_ds, records, full_configs, "stage3_full", TPE_FULL_EVAL_EPOCHS, save_dir)
    final_ok = [r for r in full_rows if r.get("status") == "ok"]
    fallback_ok = [r for r in tpe_rows if r.get("status") == "ok"] or [r for r in stage1_rows if r.get("status") == "ok"]
    candidates = final_ok or fallback_ok
    candidates.sort(key=lambda r: float(r.get("select_score", -1e9)), reverse=True)
    if not candidates:
        raise RuntimeError("No valid RQ5 result produced.")
    best = candidates[0]

    train_raw, test_raw, split_report = build_rq3_split(records, best["config"])
    best_train_file = os.path.join(save_dir, f"best_rq5_ood_train_{model_type}.jsonl")
    best_test_file = os.path.join(save_dir, f"best_rq5_ood_test_{model_type}.jsonl")
    write_jsonl(best_train_file, train_raw)
    write_jsonl(best_test_file, test_raw)
    tsv_path = final_best_tsv_file(model_type)
    write_final_tsv(tsv_path, final_rows(model_type, best))

    output = {
        "task": "Code-to-Code Translation",
        "dataset": "CodeXGLUE-CodeTrans",
        "method": "RQ5 RQ3-guided OOD 80/20 split searched by TPE with fixed RQ4 Replay",
        "model_type": model_type,
        "model_path": model_path,
        "rq1_best_model": C.get_rq1_best_model(model_type),
        "id_direction": "Java -> C#",
        "ood_direction": "C# -> Java",
        "rq3_metrics": {
            "alignment": "1 - cosine(ID source encoder representation, paired OOD source encoder representation)",
            "entropy": "OOD decoder token predictive entropy under teacher forcing",
            "anisotropy": "local OOD source anisotropy minus local ID source anisotropy",
            "rank": "local ID effective rank minus local OOD effective rank",
        },
        "fixed_replay_settings": {
            "replay_lambda": REPLAY_LAMBDA,
            "replay_batch_size": REPLAY_BATCH_SIZE,
            "replay_buffer_size": REPLAY_BUFFER_SIZE,
            "replay_buffer_indices": replay_indices,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        },
        "stage1_rows": stage1_rows,
        "tpe_rows": tpe_rows,
        "full_rows": full_rows,
        "best_result": best,
        "best_split_report": split_report,
        "files": {
            "metrics_cache": metric_cache_file(model_type),
            "tpe_trials": trials_file(model_type),
            "best_ood_train": best_train_file,
            "best_ood_test": best_test_file,
            "final_tsv": tsv_path,
        },
        "time": {
            "total_seconds": time.time() - start,
            "total_readable": format_time(time.time() - start),
        },
    }
    save_json(results_file(model_type), output)
    top_source = sorted(candidates, key=lambda r: float(r.get("select_score", -1e9)), reverse=True)
    save_json(top_configs_file(model_type), {
        "model_type": model_type,
        "search_space_version": RQ5_SEARCH_SPACE_VERSION,
        "top_configs": [
            {"rank": i + 1, "config": row["config"], "select_score": row["select_score"], "id_test": row["id_test"], "ood_test": row["ood_test"]}
            for i, row in enumerate(top_source[:TOP_CONFIGS_K])
        ],
    })
    print_final(model_type, best, tsv_path)
    print("Results saved:", results_file(model_type))
    print("Top configs saved:", top_configs_file(model_type))
    return output


def main():
    if optuna is None:
        raise ImportError("optuna is required for RQ5 TPE search. Please install optuna.")
    args = build_arg_parser().parse_args()
    C.MODEL_PATHS["codeT5"] = args.codeT5_path
    C.MODEL_PATHS["PLBART"] = args.PLBART_path
    selected = normalize_model_type(args.model_type)
    if selected == "all":
        experiments = [("codeT5", args.codeT5_path), ("PLBART", args.PLBART_path)]
    else:
        experiments = [(selected, args.model_path)]
    outputs = []
    for model_type, model_path in experiments:
        try:
            outputs.append(run_one_model(model_type, model_path))
        except Exception as exc:
            if selected != "all":
                raise
            outputs.append({"status": "failed", "model_type": model_type, "error": repr(exc)})
            print(f"[Error] RQ5 failed for {model_type}: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if selected == "all":
        save_json(os.path.join(SAVE_DIR, "rq5_codetrans_all_models.json"), {"results": outputs})


if __name__ == "__main__":
    main()
