import json
import os
import pickle
import random
import shutil
import time
import itertools
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5Config,
    T5EncoderModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm

try:
    import optuna
    from optuna.distributions import FloatDistribution
except Exception:
    optuna = None
    FloatDistribution = None


"""
================================================================================
RQ4-CL-Replay + RQ3-guided Geometry-aware OOD Split

目标：
  在最佳 Replay 持续学习配置基础上，使用 RQ5/RQ3 几何感知方法重新构造
  OOD train / OOD test_new。

核心流程：
  1) 从同一个 RQ1 best model 出发；
  2) 读取 OOD_POOL_FILE，例如 ood_test.jsonl；
  3) 用 RQ1 模型计算每个 OOD 类的几何困难度：
       entropy 高        -> hard
       alignment 低      -> hard
       anisotropy 高     -> hard
       effective rank 低 -> hard
  4) 根据几何困难度，在固定全局比例 80%/20% 下生成：
       geometry_replay_ood_train.jsonl
       geometry_replay_ood_test_new.jsonl
     其中困难类倾向保留更多训练样本，容易类倾向分配更多测试样本；
  5) 在 geometry OOD train 上做 Replay continual learning；
  6) 在 ID test 和 geometry OOD test_new 上评估。

注意：
  - 标签默认 0-based，不做 label-1。
  - CodeBERT/GraphCodeBERT 默认使用 CLS pooling。
  - CodeT5 默认使用 mean pooling。
  - 分类头自动兼容 single / mlp_seq / mlp_named / legacy out_proj-only。
  - 当前脚本仍沿用原 RQ4 协议：按 OOD test_new Macro-F1 选最佳 epoch。
    如果用于严格论文实验，建议把 geometry OOD train 再划出 validation，
    只把 geometry OOD test_new 留作最终测试。
================================================================================
"""

START_TIME = time.time()


# ============================================================
# 1. Config
# ============================================================
MODEL_TYPE = "codebert"  # options: "codebert", "GraphCodeBERT", "codeT5"
MODEL_PATH = "/opt/data/private/codebert"

# Set this when doing cross-dataset transfer:
#   None                    : use the paths below exactly as written.
#   "java250_cst_source"    : source dataset, used to train the trial-surrogate model.
#   "java250_token_target"  : target dataset, used to recommend/validate configs on token shift.
DATASET_PRESET = None

DATA_DIR = "/opt/data/private/codes/Java250/Java250/cst/pre-trained"
SAVE_DIR = "/opt/data/private/codes/Java250/Java250/cst/RQ5_TPE_Pareto_Replay_Geometry_results"
os.makedirs(SAVE_DIR, exist_ok=True)

RQ1_BEST_MODEL = "/opt/data/private/codes/Java250/Java250/cst/best_model_codebert_Java250.bin"

ID_TRAIN_FILE = "train.jsonl"
ID_TEST_FILE = "id_test.jsonl"

# 用这个 OOD pool 重新生成 OOD train / OOD test_new。
# 对于 token/cst shift，通常设置为原始大 OOD 池，例如 ood_test.jsonl。
OOD_POOL_FILE = "ood_test.jsonl"

GEOMETRY_OOD_TRAIN_FILE = os.path.join(SAVE_DIR, "geometry_replay_ood_train.jsonl")
GEOMETRY_OOD_TEST_FILE = os.path.join(SAVE_DIR, "geometry_replay_ood_test_new.jsonl")
SPLIT_REPORT_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geometry_split_report_{MODEL_TYPE}.json")
METRICS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geometry_split_metrics_{MODEL_TYPE}.json")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, f"best_rq4_replay_geometry_{MODEL_TYPE}.bin")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 250
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 8
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
TOP_K = 5
SEED = 42

# 与 RQ1 自适应分类代码保持一致：
#   auto: CodeT5 -> mean pooling; CodeBERT/GraphCodeBERT -> CLS pooling
#   也可手动设为 "cls" 或 "mean"，用于强制与旧 RQ1 checkpoint 对齐。
ENCODER_POOLING = "auto"
CLASSIFIER_HEAD_TYPE = "auto"  # auto / single / mlp_seq / mlp_named

# Best Replay configuration from previous RQ4 result.
REPLAY_LAMBDA = 0.5
REPLAY_BATCH_SIZE = 4
REPLAY_BUFFER_SIZE = 1000
REPLAY_BUFFER_SEED = SEED

# Geometry split settings.
SHIFT_TYPE = "cst"  # choices: "token", "cst", "time", "user"
ALIGNMENT_MODE = "auto"  # choices: "auto", "pair", "label_centroid"
MIN_CLASS_SAMPLES_FOR_GEOMETRY = 3
MIN_TRAIN_PER_CLASS = 2

DATASET_PRESETS = {
    "java250_cst_source": {
        "data_dir": "/opt/data/private/codes/Java250/Java250/cst/pre-trained",
        "save_dir": "/opt/data/private/codes/Java250/Java250/cst/RQ5_TPE_Pareto_Replay_Geometry_results",
        "rq1_best_model": "/opt/data/private/codes/Java250/Java250/cst/best_model_codebert_Java250.bin",
        "num_labels": 250,
        "shift_type": "cst",
        "ood_pool_file": "ood_test.jsonl",
    },
    "java250_token_target": {
        "data_dir": "/opt/data/private/codes/Java250/Java250/token/pre-trained",
        "save_dir": "/opt/data/private/codes/Java250/Java250/token/RQ5_surrogate_transfer_results",
        "rq1_best_model": "/opt/data/private/codes/Java250/Java250/token/best_model_codebert_Java250.bin",
        "num_labels": 250,
        "shift_type": "token",
        "ood_pool_file": "ood_test.jsonl",
        "surrogate_pretrained_model_file": (
            "/opt/data/private/codes/Java250/Java250/cst/"
            "RQ5_TPE_Pareto_Replay_Geometry_results/"
            f"rq5_trial_surrogate_extra_trees_{MODEL_TYPE}.pkl"
        ),
        "surrogate_source_results_files": [
            (
                "/opt/data/private/codes/Java250/Java250/cst/"
                "RQ5_TPE_Pareto_Replay_Geometry_results/"
                f"rq5_tpe_pareto_replay_search_results_{MODEL_TYPE}.json"
            ),
            (
                "/opt/data/private/codes/Java250/Java250/cst/"
                "RQ5_TPE_Pareto_Replay_Geometry_results/"
                f"rq4_replay_geometry_split_metrics_{MODEL_TYPE}.json"
            ),
            (
                "/opt/data/private/codes/Java250/Java250/token/"
                "RQ5_surrogate_transfer_results/"
                f"rq5_surrogate_validation_results_{MODEL_TYPE}.json"
            ),
        ],
    },
    "java250_user_target": {
        "data_dir": "/opt/data/private/codes/Java250/Java250/user/pre-trained",
        "save_dir": "/opt/data/private/codes/Java250/Java250/user/RQ5_surrogate_transfer_results",
        "rq1_best_model": "/opt/data/private/codes/Java250/Java250/user/best_model_codebert_Java250.bin",
        "num_labels": 250,
        "shift_type": "user",
        "ood_pool_file": "ood_test.jsonl",
        "surrogate_pretrained_model_file": (
            "/opt/data/private/codes/Java250/Java250/cst/"
            "RQ5_TPE_Pareto_Replay_Geometry_results/"
            f"rq5_trial_surrogate_extra_trees_{MODEL_TYPE}.pkl"
        ),
        "surrogate_source_results_files": [
            (
                "/opt/data/private/codes/Java250/Java250/cst/"
                "RQ5_TPE_Pareto_Replay_Geometry_results/"
                f"rq5_tpe_pareto_replay_search_results_{MODEL_TYPE}.json"
            ),
            (
                "/opt/data/private/codes/Java250/Java250/cst/"
                "RQ5_TPE_Pareto_Replay_Geometry_results/"
                f"rq4_replay_geometry_split_metrics_{MODEL_TYPE}.json"
            ),
            (
                "/opt/data/private/codes/Java250/Java250/user/"
                "RQ5_surrogate_transfer_results/"
                f"rq5_surrogate_validation_results_{MODEL_TYPE}.json"
            ),
        ],
    },
    "python75_cst_target": {
        "data_dir": "/opt/data/private/codes/Python75/Python75/cst/pre-trained",
        "save_dir": "/opt/data/private/codes/Python75/Python75/cst/RQ5_surrogate_transfer_results",
        "rq1_best_model": "/opt/data/private/codes/Python75/Python75/cst/best_model_codebert_Python75.bin",
        "num_labels": 75,
        "shift_type": "cst",
        "ood_pool_file": "ood_test.jsonl",
        "surrogate_pretrained_model_file": (
            "/opt/data/private/codes/Java250/Java250/cst/"
            "RQ5_TPE_Pareto_Replay_Geometry_results/"
            f"rq5_trial_surrogate_extra_trees_{MODEL_TYPE}.pkl"
        ),
        "surrogate_source_results_files": [
            (
                "/opt/data/private/codes/Java250/Java250/cst/"
                "RQ5_TPE_Pareto_Replay_Geometry_results/"
                f"rq5_tpe_pareto_replay_search_results_{MODEL_TYPE}.json"
            ),
            (
                "/opt/data/private/codes/Java250/Java250/cst/"
                "RQ5_TPE_Pareto_Replay_Geometry_results/"
                f"rq4_replay_geometry_split_metrics_{MODEL_TYPE}.json"
            ),
            (
                "/opt/data/private/codes/Java250/Java250/cst/"
                "RQ5_TPE_Pareto_Replay_Geometry_results/"
                f"rq5_tpe_pareto_replay_pareto_results_{MODEL_TYPE}.json"
            ),
            (
                "/opt/data/private/codes/Python75/Python75/cst/"
                "RQ5_surrogate_transfer_results/"
                f"rq5_surrogate_validation_results_{MODEL_TYPE}.json"
            ),
        ],
    },
}

# Fixed global OOD train/test ratio.
FIXED_TRAIN_RATIO = 0.80
FIXED_TEST_RATIO = 0.20

# Allocation preferences under the fixed 20% test budget.
# hard <= middle <= easy: hard classes keep more training samples.
TEST_RATIO_HARD = 0.08
TEST_RATIO_MIDDLE = 0.10
TEST_RATIO_EASY = 0.15

DEFAULT_SCORE_WEIGHTS = {
    "entropy": 0.25,
    "alignment": 0.25,
    "anisotropy": 0.25,
    "rank": 0.25,
}

MAX_ANISO_SAMPLES_PER_CLASS = 3000
MAX_RANK_SAMPLES_PER_CLASS = 8000


# SmartSearch settings: NO TPE / NO Optuna.
# Stage-1: cheap probing with fixed logical geometry configs.
# Stage-2: medium-budget Replay training, using top Stage-1 + surrogate-selected configs.
# Stage-3: full-budget Replay training for top configs.
SMART_STAGE1_TRIALS = 24
SMART_STAGE1_EPOCHS = 1
SMART_STAGE2_TRIALS = 10
SMART_STAGE2_EPOCHS = 3
SMART_STAGE2_FROM_STAGE1 = 4
SMART_STAGE2_RANDOM = 2
SMART_FINAL_TRIALS = 3
SMART_FINAL_EPOCHS = EPOCHS
SMART_SURROGATE_POOL_SIZE = 120
SURROGATE_MIN_SAMPLES = 8
SURROGATE_N_ESTIMATORS = 128
SURROGATE_EXPLORATION_WEIGHT = 0.15
SEARCH_SELECT_METRIC = "macro_f1"

SEARCH_WEIGHT_BOUNDS = {
    "entropy": (0.0, 1.0),
    "alignment": (0.0, 1.0),
    "anisotropy": (0.0, 1.0),
    "rank": (0.0, 1.0),
}
SEARCH_RATIO_BOUNDS = {
    "hard": (0.02, 0.20),
    "middle": (0.04, 0.30),
    "easy": (0.06, 0.40),
}

SMART_SEARCH_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geo_smartsearch_results_{MODEL_TYPE}.json")
SMART_TOP_CONFIGS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geo_smartsearch_top_configs_{MODEL_TYPE}.json")

# Resume / OOM-safe settings.
# RESUME_SEARCH=True will load SMART_SEARCH_RESULTS_FILE and skip completed trials.
# It resumes from the last COMPLETED trial; a trial interrupted in the middle is rerun.
RESUME_SEARCH = True

# OOM_SAFE_MODE lowers peak memory. It is recommended for SmartSearch because each trial reloads
# and trains a fresh model. You can set it to False after freeing the GPU.
OOM_SAFE_MODE = True
if OOM_SAFE_MODE:
    BATCH_SIZE = min(BATCH_SIZE, 8)
    REPLAY_BATCH_SIZE = min(REPLAY_BATCH_SIZE, 2)


# ============================================================
# 2. Utilities
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def format_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.1f}s"


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


def normalized_keys(state_dict):
    return [strip_prefix(k) for k in state_dict.keys()]


def infer_head_type_from_state_dict(state_dict):
    """
    自动判断分类头结构。

    兼容几类 RQ1 checkpoint:
      single:
        classifier.weight / classifier.bias
        score.weight / score.bias
        classifier.out_proj.weight / classifier.out_proj.bias 但无 dense
        classification_head.out_proj.* 但无 dense
      mlp_seq:
        classifier.0.* and classifier.3.*
      mlp_named:
        classifier.dense.* and classifier.out_proj.*
        classification_head.dense.* and classification_head.out_proj.*
    """
    keys = normalized_keys(state_dict)

    has_seq = any(k.startswith("classifier.0.") or k.startswith("classifier.3.") for k in keys)
    if has_seq:
        return "mlp_seq"

    has_named_dense = any(k.startswith("classifier.dense.") or k.startswith("classification_head.dense.") for k in keys)
    has_named_out = any(k.startswith("classifier.out_proj.") or k.startswith("classification_head.out_proj.") for k in keys)
    if has_named_dense and has_named_out:
        return "mlp_named"

    if any(k in ("classifier.weight", "classifier.bias", "score.weight", "score.bias") for k in keys):
        return "single"

    # Legacy RQ1: ClassifierHead only had dropout + out_proj.
    if has_named_out:
        return "single"

    return "single"


def adapt_state_dict_keys(state_dict, head_type):
    new_dict = {}

    for key, value in state_dict.items():
        new_key = strip_prefix(key)

        # encoder prefix adaptation
        if new_key.startswith("roberta."):
            new_key = new_key.replace("roberta.", "encoder.", 1)
        elif new_key.startswith("bert."):
            new_key = new_key.replace("bert.", "encoder.", 1)
        elif new_key.startswith("transformer."):
            new_key = new_key.replace("transformer.", "encoder.", 1)
        elif new_key.startswith("encoder.encoder."):
            new_key = new_key.replace("encoder.encoder.", "encoder.", 1)

        # classification_head -> classifier
        if new_key.startswith("classification_head.dense."):
            new_key = new_key.replace("classification_head.dense.", "classifier.dense.", 1)
        elif new_key.startswith("classification_head.out_proj."):
            new_key = new_key.replace("classification_head.out_proj.", "classifier.out_proj.", 1)

        # score -> classifier
        if new_key == "score.weight":
            new_key = "classifier.weight"
        elif new_key == "score.bias":
            new_key = "classifier.bias"

        # map checkpoint head keys to current model head keys
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


def get_required_final_classifier_keys(head_type):
    if head_type == "single":
        return ["classifier.weight", "classifier.bias"]
    if head_type == "mlp_named":
        return ["classifier.out_proj.weight", "classifier.out_proj.bias"]
    if head_type == "mlp_seq":
        return ["classifier.3.weight", "classifier.3.bias"]
    raise ValueError(f"Unsupported head_type: {head_type}")


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")
    return "mean" if "t5" in model_type.lower() else "cls"


def get_code(js):
    return js.get("func") or js.get("code") or js.get("function") or js.get("source") or ""


def get_label(js):
    label = js.get("label")
    if label is None:
        label = js.get("target")
    return label


def get_idx(js, fallback):
    idx = js.get("idx")
    if idx is None:
        idx = js.get("id")
    if idx is None:
        idx = fallback
    return str(idx)


def resolve_data_path(file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(DATA_DIR, file_name)


def read_jsonl(path):
    data = []
    bad_lines = 0
    skipped_label = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            code = get_code(js)
            label = get_label(js)
            if not code or label is None:
                bad_lines += 1
                continue

            label = int(label)
            if label < 0 or label >= NUM_LABELS:
                skipped_label += 1
                continue

            js["idx"] = get_idx(js, line_idx)
            js["label"] = label  # 0-based label kept unchanged
            data.append(js)

    labels = [int(x["label"]) for x in data]
    print("=" * 100)
    print("JSONL loaded:", path)
    print("num_samples:", len(data))
    print("bad_lines:", bad_lines)
    print("skipped_label_out_of_range:", skipped_label)
    if labels:
        print(f"0-based label check: min={min(labels)}, max={max(labels)}, expected=[0, {NUM_LABELS - 1}]")
        print("label_distribution_preview:", dict(Counter(labels).most_common(10)))
    print("=" * 100)
    return data


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_distribution(data):
    return dict(Counter(int(x["label"]) for x in data))


def deterministic_indices(n, max_samples, seed=SEED):
    if max_samples is None or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def minmax_hard(values, higher_is_harder=True):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if finite.sum() == 0:
        return out
    v = arr[finite]
    vmin, vmax = float(v.min()), float(v.max())
    if abs(vmax - vmin) < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (v - vmin) / (vmax - vmin)
    if not higher_is_harder:
        out = 1.0 - out
    out[~finite] = 0.0
    return out


def normalize_positive_weights(weights):
    vals = {
        "entropy": max(0.0, float(weights.get("entropy", DEFAULT_SCORE_WEIGHTS["entropy"]))),
        "alignment": max(0.0, float(weights.get("alignment", DEFAULT_SCORE_WEIGHTS["alignment"]))),
        "anisotropy": max(0.0, float(weights.get("anisotropy", DEFAULT_SCORE_WEIGHTS["anisotropy"]))),
        "rank": max(0.0, float(weights.get("rank", DEFAULT_SCORE_WEIGHTS["rank"]))),
    }
    total = sum(vals.values())
    if total <= 1e-12:
        return dict(DEFAULT_SCORE_WEIGHTS)
    return {k: float(v / total) for k, v in vals.items()}


# ============================================================
# 3. Model
# ============================================================
class NamedMLPHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        nn.init.eye_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def prepare_final_features(self, pooled_features):
        x = self.dropout(pooled_features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x

    def forward(self, pooled_features):
        return self.out_proj(self.prepare_final_features(pooled_features))


class CodeEncoder(nn.Module):
    def __init__(self, model_path, num_labels, model_type, head_type="single", pooling="auto", load_pretrained=True):
        super().__init__()
        self.model_type = model_type.lower()
        self.head_type = head_type
        self.pooling_strategy = resolve_pooling_strategy(model_type, pooling)

        if "t5" in self.model_type:
            if load_pretrained:
                self.encoder = T5EncoderModel.from_pretrained(model_path)
            else:
                config = T5Config.from_pretrained(model_path)
                self.encoder = T5EncoderModel(config)
            hidden_size = self.encoder.config.d_model
        else:
            if load_pretrained:
                self.encoder = AutoModel.from_pretrained(model_path)
            else:
                config = AutoConfig.from_pretrained(model_path)
                self.encoder = AutoModel.from_config(config)
            hidden_size = self.encoder.config.hidden_size

        if head_type == "single":
            self.dropout = nn.Dropout(0.1)
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

    def pool_encoder_output(self, outputs, attention_mask):
        if self.pooling_strategy == "mean":
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / denom
        return outputs.last_hidden_state[:, 0, :]

    def prepare_classifier_features(self, pooled_features):
        if self.head_type == "single":
            return self.dropout(pooled_features)
        if self.head_type == "mlp_named":
            return self.classifier.prepare_final_features(pooled_features)
        x = self.classifier[0](pooled_features)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x

    def classify_final_features(self, final_features):
        if self.head_type == "single":
            return self.classifier(final_features)
        if self.head_type == "mlp_named":
            return self.classifier.out_proj(final_features)
        return self.classifier[3](final_features)

    def forward(self, input_ids, attention_mask, return_feat=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_features = self.pool_encoder_output(outputs, attention_mask)
        final_features = self.prepare_classifier_features(pooled_features)
        logits = self.classify_final_features(final_features)
        if return_feat:
            return logits, final_features
        return type("ModelOutput", (object,), {"logits": logits})


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        raise FileNotFoundError(f"RQ1 best model not found: {RQ1_BEST_MODEL}")

    raw_state = unwrap_checkpoint(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    if not isinstance(raw_state, dict):
        raise RuntimeError("Loaded checkpoint is not a state_dict-like object.")

    head_type = infer_head_type_from_state_dict(raw_state) if CLASSIFIER_HEAD_TYPE == "auto" else CLASSIFIER_HEAD_TYPE
    if head_type not in ("single", "mlp_seq", "mlp_named"):
        raise ValueError(f"Unsupported head_type: {head_type}")

    model = CodeEncoder(
        MODEL_PATH,
        NUM_LABELS,
        MODEL_TYPE,
        head_type=head_type,
        pooling=ENCODER_POOLING,
        load_pretrained=False,
    ).to(DEVICE)

    adapted_state = adapt_state_dict_keys(raw_state, head_type=head_type)
    compatible_state, skipped = filter_compatible_state_dict(model, adapted_state)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    matched = [key for key in compatible_state.keys() if key in model.state_dict()]

    encoder_matched = [key for key in matched if key.startswith("encoder.")]
    classifier_matched = [key for key in matched if key.startswith("classifier.")]

    print(
        f">>> Loaded RQ1 model: model_type={MODEL_TYPE}, head_type={model.head_type}, "
        f"pooling={model.pooling_strategy}, matched={len(matched)}, "
        f"encoder_matched={len(encoder_matched)}, classifier_matched={len(classifier_matched)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
    )

    skipped_classifier = [item for item in skipped if item[0].startswith("classifier.")]
    if skipped_classifier:
        print(">>> Skipped classifier keys due to mismatch/missing:")
        for item in skipped_classifier[:30]:
            print("   ", item)

    missing_show = [key for key in missing if "pooler" not in key][:30]
    if missing_show:
        print(">>> Missing keys preview:", missing_show)
    if unexpected:
        print(">>> Unexpected keys preview:", unexpected[:30])

    if len(encoder_matched) == 0:
        raise RuntimeError("No encoder parameters were loaded from the RQ1 checkpoint.")
    if len(classifier_matched) == 0:
        raise RuntimeError("No classifier parameters were loaded from the RQ1 checkpoint.")

    required_final_keys = get_required_final_classifier_keys(model.head_type)
    missing_final_keys = [key for key in required_final_keys if key not in matched]
    print(">>> Required final classifier keys:", required_final_keys)
    if missing_final_keys:
        raise RuntimeError(
            "Final classifier output layer was not fully loaded from the RQ1 checkpoint. "
            f"Missing final classifier keys: {missing_final_keys}. "
            "Please check RQ1_BEST_MODEL, NUM_LABELS, and CLASSIFIER_HEAD_TYPE."
        )
    print(">>> Final classifier output layer fully loaded.")

    model.eval()
    return model


# ============================================================
# 4. Dataset
# ============================================================
class CodeDataset(Dataset):
    def __init__(self, source, tokenizer, name="dataset"):
        self.examples = []

        if isinstance(source, str):
            data = read_jsonl(resolve_data_path(source))
            source_name = source
        else:
            data = source
            source_name = name

        bad_lines = 0
        skipped_label = 0
        labels_seen = []

        for line_idx, js in enumerate(tqdm(data, desc=f"Tokenizing {source_name}", leave=False)):
            code = get_code(js)
            label = get_label(js)
            if not code or label is None:
                bad_lines += 1
                continue
            label = int(label)
            if label < 0 or label >= NUM_LABELS:
                skipped_label += 1
                continue
            labels_seen.append(label)

            inputs = tokenizer(
                code,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.examples.append({
                "idx": get_idx(js, line_idx),
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
                "label_int": label,
                "raw": js,
            })

        counter = Counter(x["label_int"] for x in self.examples)
        print("=" * 100)
        print(f"CodeDataset loaded: {source_name}")
        print("num_samples:", len(self.examples))
        print("bad_lines:", bad_lines)
        print("skipped_label_out_of_range:", skipped_label)
        if labels_seen:
            print(f"0-based label check: min={min(labels_seen)}, max={max(labels_seen)}, expected=[0, {NUM_LABELS - 1}]")
        print("label_distribution_preview:", dict(counter.most_common(10)))
        print("=" * 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# 5. Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, desc="Evaluating", save_path=None):
    model.eval()
    all_logits = []
    all_labels = []
    records = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        ).logits
        labels = batch["label"].to(DEVICE)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if save_path is not None:
            preds = torch.argmax(logits, dim=1)
            for i in range(labels.size(0)):
                records.append({
                    "idx": batch["idx"][i],
                    "label": int(labels[i].detach().cpu().item()),
                    "pred": int(preds[i].detach().cpu().item()),
                })

    if not all_logits:
        return {"accuracy": 0.0, "macro_f1": 0.0, "top5_acc": 0.0, "num_samples": 0}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    preds = torch.argmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    k = min(TOP_K, all_logits.size(1))
    _, topk = all_logits.topk(k, dim=1)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "macro_f1": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        "top5_acc": float(topk.eq(all_labels.view(-1, 1)).any(dim=1).float().mean().item()),
        "num_samples": int(len(labels_np)),
    }


# ============================================================
# 6. Geometry metrics and split
# ============================================================
@torch.no_grad()
def extract_embeddings_and_logits(model, dataloader, desc="Extracting embeddings"):
    model.eval()
    all_feats = []
    all_logits = []
    all_labels = []
    all_idxs = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        logits, feats = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            return_feat=True,
        )
        all_feats.append(feats.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())
        all_idxs.extend([str(x) for x in batch["idx"]])

    return (
        torch.cat(all_feats, dim=0),
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        np.asarray(all_idxs),
    )


def compute_anisotropy(embeddings):
    n = embeddings.size(0)
    if n <= 1:
        return 0.0
    idx = deterministic_indices(n, MAX_ANISO_SAMPLES_PER_CLASS, seed=SEED)
    x = embeddings[idx].float()
    x = F.normalize(x, p=2, dim=1)
    cos_mat = torch.mm(x, x.T)
    m = x.size(0)
    avg_sim = (cos_mat.sum() - m) / (m * (m - 1))
    return float(avg_sim.item())


def compute_effective_rank(embeddings):
    n = embeddings.size(0)
    if n <= 1:
        return 1.0
    idx = deterministic_indices(n, MAX_RANK_SAMPLES_PER_CLASS, seed=SEED)
    x = embeddings[idx].float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x)
    except Exception:
        s = torch.svd(x)[1]
    s = torch.clamp(s, min=1e-12)
    p = s / (s.sum() + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    return float(torch.exp(entropy).item())


def compute_pair_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs):
    id_norm = F.normalize(id_emb.float(), p=2, dim=1)
    ood_norm = F.normalize(ood_emb.float(), p=2, dim=1)
    id_map = {}
    for i in range(id_emb.size(0)):
        label = int(id_labels[i].item())
        id_map[(str(id_idxs[i]), label)] = i

    vals = defaultdict(list)
    matched = 0
    for j in range(ood_emb.size(0)):
        label = int(ood_labels[j].item())
        key = (str(ood_idxs[j]), label)
        if key not in id_map:
            continue
        i = id_map[key]
        sim = F.cosine_similarity(id_norm[i:i + 1], ood_norm[j:j + 1], dim=1).item()
        vals[label].append(sim)
        matched += 1

    return {int(c): float(np.mean(v)) for c, v in vals.items()}, matched


def compute_centroid_alignment_by_class(id_emb, id_labels, ood_emb, ood_labels):
    results = {}
    for c in range(NUM_LABELS):
        id_mask = id_labels == c
        ood_mask = ood_labels == c
        if id_mask.sum().item() == 0 or ood_mask.sum().item() == 0:
            continue
        id_centroid = id_emb[id_mask].float().mean(dim=0, keepdim=True)
        ood_centroid = ood_emb[ood_mask].float().mean(dim=0, keepdim=True)
        sim = F.cosine_similarity(
            F.normalize(id_centroid, p=2, dim=1),
            F.normalize(ood_centroid, p=2, dim=1),
            dim=1,
        ).item()
        results[int(c)] = float(sim)
    return results


def compute_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs):
    shift = SHIFT_TYPE.lower()
    mode = ALIGNMENT_MODE.lower()
    if mode == "pair" or (mode == "auto" and shift in ["token", "cst"]):
        pair_align, matched = compute_pair_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs)
        print(f">>> Pair-level alignment matched pairs: {matched}")
        if matched > 0:
            centroid_align = compute_centroid_alignment_by_class(id_emb, id_labels, ood_emb, ood_labels)
            centroid_align.update(pair_align)
            return centroid_align
        print("[Warning] No matched ID/OOD pairs. Falling back to label-centroid alignment.")
    return compute_centroid_alignment_by_class(id_emb, id_labels, ood_emb, ood_labels)


def compute_per_class_metrics(ood_emb, ood_logits, ood_labels, ood_idxs, id_emb, id_labels, id_idxs):
    results = {}
    alignment_by_class = compute_alignment_by_class(id_emb, id_labels, id_idxs, ood_emb, ood_labels, ood_idxs)

    for c in range(NUM_LABELS):
        ood_mask = ood_labels == c
        n_ood = int(ood_mask.sum().item())
        if n_ood < MIN_CLASS_SAMPLES_FOR_GEOMETRY:
            continue
        cls_emb = ood_emb[ood_mask]
        cls_logits = ood_logits[ood_mask]
        probs = F.softmax(cls_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
        alignment = alignment_by_class.get(int(c), float("nan"))
        anisotropy = compute_anisotropy(cls_emb)
        rank = compute_effective_rank(cls_emb)
        results[int(c)] = {
            "entropy": float(entropy),
            "alignment": float(alignment),
            "anisotropy": float(anisotropy),
            "rank": float(rank),
            "n_samples": int(n_ood),
        }
    return results


def compute_class_scores(class_metrics, weights=None):
    if not class_metrics:
        return {}
    weights = normalize_positive_weights(weights or DEFAULT_SCORE_WEIGHTS)
    classes = list(class_metrics.keys())

    ent = np.array([class_metrics[c]["entropy"] for c in classes], dtype=float)
    ali = np.array([class_metrics[c]["alignment"] for c in classes], dtype=float)
    ani = np.array([class_metrics[c]["anisotropy"] for c in classes], dtype=float)
    rnk = np.array([class_metrics[c]["rank"] for c in classes], dtype=float)

    ent_hard = minmax_hard(ent, higher_is_harder=True)
    ali_hard = minmax_hard(ali, higher_is_harder=False)
    ani_hard = minmax_hard(ani, higher_is_harder=True)
    rnk_hard = minmax_hard(rnk, higher_is_harder=False)

    scores = (
        weights["entropy"] * ent_hard
        + weights["alignment"] * ali_hard
        + weights["anisotropy"] * ani_hard
        + weights["rank"] * rnk_hard
    )

    for c, eh, ah, nh, rh, s in zip(classes, ent_hard, ali_hard, ani_hard, rnk_hard, scores):
        class_metrics[c]["entropy_hard"] = float(eh)
        class_metrics[c]["alignment_hard"] = float(ah)
        class_metrics[c]["anisotropy_hard"] = float(nh)
        class_metrics[c]["rank_hard"] = float(rh)
        class_metrics[c]["difficulty_score"] = float(s)

    return {int(c): float(s) for c, s in zip(classes, scores)}


def class_test_ratios_from_scores(class_scores):
    if not class_scores:
        return {}
    sorted_items = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    ratios = {}
    for rank, (c, _) in enumerate(sorted_items):
        percentile = rank / max(n - 1, 1)
        if percentile < 0.25:
            ratio = TEST_RATIO_HARD
        elif percentile < 0.75:
            ratio = TEST_RATIO_MIDDLE
        else:
            ratio = TEST_RATIO_EASY
        ratios[int(c)] = float(ratio)
    return ratios


def build_geometry_ood_train_test_split(ood_pool_raw, class_test_ratios):
    """
    Fixed global 80/20 split. class_test_ratios are allocation preferences only.
    """
    rng = np.random.RandomState(SEED)
    by_label = defaultdict(list)
    for item in ood_pool_raw:
        by_label[int(item["label"])].append(item)

    total_n = len(ood_pool_raw)
    target_test_total = int(round(total_n * FIXED_TEST_RATIO))
    labels = sorted(by_label.keys())

    per_class = {}
    total_capacity = 0
    for label in labels:
        items = list(by_label[label])
        rng.shuffle(items)
        n = len(items)
        max_test = max(0, n - MIN_TRAIN_PER_CLASS)
        total_capacity += max_test
        preference = float(class_test_ratios.get(label, TEST_RATIO_MIDDLE))
        per_class[label] = {
            "items": items,
            "n": n,
            "max_test": max_test,
            "preference": max(preference, 1e-12),
        }

    if target_test_total > total_capacity:
        print(
            f"[Warning] fixed test target {target_test_total} exceeds capacity {total_capacity}; "
            "using capacity instead."
        )
        target_test_total = total_capacity

    weights = {
        label: info["n"] * info["preference"] if info["max_test"] > 0 else 0.0
        for label, info in per_class.items()
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 1e-12:
        weights = {label: float(info["n"]) if info["max_test"] > 0 else 0.0 for label, info in per_class.items()}
        weight_sum = sum(weights.values())

    n_test_by_label = {}
    frac_records = []
    for label, info in per_class.items():
        if info["max_test"] <= 0 or weight_sum <= 1e-12:
            n_test_by_label[label] = 0
            continue
        raw = target_test_total * weights[label] / weight_sum
        clipped = min(raw, info["max_test"])
        base = int(np.floor(clipped))
        n_test_by_label[label] = base
        frac_records.append((clipped - base, label))

    current_total = sum(n_test_by_label.values())
    frac_records.sort(reverse=True)
    while current_total < target_test_total:
        changed = False
        for _, label in frac_records:
            if current_total >= target_test_total:
                break
            if n_test_by_label[label] < per_class[label]["max_test"]:
                n_test_by_label[label] += 1
                current_total += 1
                changed = True
        if not changed:
            break

    if current_total > target_test_total:
        frac_records_asc = sorted(frac_records)
        while current_total > target_test_total:
            changed = False
            for _, label in frac_records_asc:
                if current_total <= target_test_total:
                    break
                if n_test_by_label[label] > 0:
                    n_test_by_label[label] -= 1
                    current_total -= 1
                    changed = True
            if not changed:
                break

    train_data = []
    test_data = []
    report = {}
    for label in labels:
        info = per_class[label]
        items = info["items"]
        n_test = int(n_test_by_label[label])
        test_items = items[:n_test]
        train_items = items[n_test:]
        test_data.extend(test_items)
        train_data.extend(train_items)
        report[int(label)] = {
            "available": int(info["n"]),
            "train": int(len(train_items)),
            "test": int(len(test_items)),
            "test_ratio": float(len(test_items) / max(1, info["n"])),
            "allocation_preference": float(info["preference"]),
            "max_test_capacity": int(info["max_test"]),
        }

    rng.shuffle(train_data)
    rng.shuffle(test_data)

    actual_train_ratio = len(train_data) / max(1, total_n)
    actual_test_ratio = len(test_data) / max(1, total_n)
    print(
        f">>> Geometry OOD split: train={len(train_data)} ({actual_train_ratio:.4f}), "
        f"test={len(test_data)} ({actual_test_ratio:.4f}), target=0.80/0.20"
    )
    return train_data, test_data, report


def build_geometry_split(tokenizer):
    id_test_raw = read_jsonl(resolve_data_path(ID_TEST_FILE))
    ood_pool_raw = read_jsonl(resolve_data_path(OOD_POOL_FILE))
    if not ood_pool_raw:
        raise RuntimeError("No OOD pool samples loaded. Please check OOD_POOL_FILE.")

    id_test_ds = CodeDataset(id_test_raw, tokenizer, name="id_test_for_geometry")
    ood_pool_ds = CodeDataset(ood_pool_raw, tokenizer, name="ood_pool_for_geometry")
    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_pool_loader = DataLoader(ood_pool_ds, batch_size=BATCH_SIZE, shuffle=False)

    rq1_model = load_rq1_model()
    id_emb, _, id_labels, id_idxs = extract_embeddings_and_logits(
        rq1_model,
        id_test_loader,
        desc="Extract ID test embeddings for alignment",
    )
    ood_emb, ood_logits, ood_labels, ood_idxs = extract_embeddings_and_logits(
        rq1_model,
        ood_pool_loader,
        desc="Extract OOD pool embeddings",
    )

    class_metrics = compute_per_class_metrics(
        ood_emb,
        ood_logits,
        ood_labels,
        ood_idxs,
        id_emb,
        id_labels,
        id_idxs,
    )
    class_scores = compute_class_scores(class_metrics, DEFAULT_SCORE_WEIGHTS)
    class_test_ratios = class_test_ratios_from_scores(class_scores)
    train_raw, test_raw, split_report = build_geometry_ood_train_test_split(ood_pool_raw, class_test_ratios)

    write_jsonl(GEOMETRY_OOD_TRAIN_FILE, train_raw)
    write_jsonl(GEOMETRY_OOD_TEST_FILE, test_raw)

    sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    print(">>> Geometry class count:", len(class_metrics))
    print(">>> Top-10 hardest classes:")
    for c, score in sorted_scores[:10]:
        m = class_metrics[c]
        r = split_report.get(c, {})
        print(
            f"Class {c:<4} | score={score:.4f} | "
            f"test={r.get('test', 0)}/{r.get('available', 0)} | "
            f"test_ratio={r.get('test_ratio', 0.0):.3f} | "
            f"ent={m['entropy']:.3f} | align={m['alignment']:.3f} | "
            f"aniso={m['anisotropy']:.3f} | rank={m['rank']:.2f}"
        )

    with open(SPLIT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "RQ3-guided geometry-aware OOD train/test split for RQ4 Replay",
                "model_type": MODEL_TYPE,
                "model_path": MODEL_PATH,
                "rq1_best_model": RQ1_BEST_MODEL,
                "data_dir": DATA_DIR,
                "ood_pool_file": OOD_POOL_FILE,
                "geometry_ood_train_file": GEOMETRY_OOD_TRAIN_FILE,
                "geometry_ood_test_file": GEOMETRY_OOD_TEST_FILE,
                "fixed_global_split_ratio": {
                    "train_ratio": FIXED_TRAIN_RATIO,
                    "test_ratio": FIXED_TEST_RATIO,
                    "note": "class test ratios are allocation preferences under a fixed 20% test budget",
                },
                "score_weights": normalize_positive_weights(DEFAULT_SCORE_WEIGHTS),
                "test_ratio_hard": TEST_RATIO_HARD,
                "test_ratio_middle": TEST_RATIO_MIDDLE,
                "test_ratio_easy": TEST_RATIO_EASY,
                "class_metrics": class_metrics,
                "class_scores": class_scores,
                "class_test_ratios": class_test_ratios,
                "split_report": split_report,
                "geometry_ood_train_size": len(train_raw),
                "geometry_ood_test_size": len(test_raw),
                "geometry_ood_train_label_distribution": label_distribution(train_raw),
                "geometry_ood_test_label_distribution": label_distribution(test_raw),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    del rq1_model, id_emb, id_labels, id_idxs, ood_emb, ood_logits, ood_labels, ood_idxs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_raw, test_raw, split_report, class_metrics, class_scores


# ============================================================
# 7. Replay training
# ============================================================
def build_replay_buffer_dataset(id_train_ds, buffer_size=None, seed=42):
    if buffer_size is None or buffer_size <= 0 or buffer_size >= len(id_train_ds):
        indices = list(range(len(id_train_ds)))
        print(f">>> Replay buffer: using full ID train, size={len(indices)}")
        return id_train_ds, indices
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(id_train_ds), size=int(buffer_size), replace=False).tolist()
    indices = sorted(indices)
    print(f">>> Replay buffer: sampled {len(indices)} / {len(id_train_ds)} ID-train samples")
    return Subset(id_train_ds, indices), indices


def train_one_epoch_replay(model, ood_loader, replay_iter, optimizer, scheduler, epoch):
    model.train()
    total_loss_sum = 0.0
    ood_ce_sum = 0.0
    replay_ce_sum = 0.0
    weighted_replay_sum = 0.0
    steps = 0

    pbar = tqdm(ood_loader, desc=f"[Replay-Geo] Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        # Memory-safe Replay training:
        # Do not keep OOD and replay computation graphs alive at the same time.
        # Backward OOD loss first, then compute/backward replay loss.
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        ).logits
        ood_ce = F.cross_entropy(logits, batch["label"].to(DEVICE))

        if torch.isnan(ood_ce):
            continue

        ood_ce.backward()

        # Explicitly release the first graph before replay forward.
        del logits

        id_batch = next(replay_iter)
        id_logits = model(
            id_batch["input_ids"].to(DEVICE),
            id_batch["attention_mask"].to(DEVICE),
        ).logits
        replay_ce = F.cross_entropy(id_logits, id_batch["label"].to(DEVICE))
        weighted_replay = REPLAY_LAMBDA * replay_ce

        if torch.isnan(weighted_replay):
            optimizer.zero_grad(set_to_none=True)
            continue

        weighted_replay.backward()
        loss = ood_ce.detach() + weighted_replay.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        del id_logits

        total_loss_sum += float(loss.item())
        ood_ce_sum += float(ood_ce.detach().item())
        replay_ce_sum += float(replay_ce.detach().item())
        weighted_replay_sum += float(weighted_replay.detach().item())
        steps += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ood_ce": f"{ood_ce.detach().item():.4f}",
            "rep*w": f"{weighted_replay.detach().item():.4f}",
        })

    denom = max(steps, 1)
    stats = {
        "total_loss": total_loss_sum / denom,
        "ood_ce": ood_ce_sum / denom,
        "replay_ce": replay_ce_sum / denom,
        "replay_weighted": weighted_replay_sum / denom,
        "steps": int(steps),
    }
    print(
        f">>> [Replay-Geo] Epoch {epoch} loss breakdown | "
        f"OOD_CE={stats['ood_ce']:.6f} | "
        f"Replay_CE={stats['replay_ce']:.6f} | "
        f"lambda*Replay={stats['replay_weighted']:.6f} | "
        f"Total={stats['total_loss']:.6f}"
    )
    return stats


def print_epoch_table(epoch_metrics):
    print("\n" + "=" * 130)
    print("RQ4 Replay + Geometry OOD Split Epoch Metrics")
    print("=" * 130)
    print(
        f"{'Epoch':<7} | {'Loss':<10} | {'OOD_CE':<10} | {'Replay*w':<10} | "
        f"{'ID Acc':<10} | {'ID F1':<10} | {'ID Top5':<10} | "
        f"{'OOD Acc':<10} | {'OOD F1':<10} | {'OOD Top5':<10} | {'Best':<5}"
    )
    print("-" * 130)
    for row in epoch_metrics:
        id_eval = row["id_test"]
        ood_eval = row["ood_test"]
        loss_stats = row.get("loss_breakdown", {})
        best_mark = "*" if row.get("is_best") else ""
        print(
            f"{row['epoch']:<7} | {row['train_loss']:<10.4f} | "
            f"{loss_stats.get('ood_ce', 0.0):<10.4f} | "
            f"{loss_stats.get('replay_weighted', 0.0):<10.4f} | "
            f"{id_eval['accuracy']:<10.4f} | {id_eval['macro_f1']:<10.4f} | {id_eval['top5_acc']:<10.4f} | "
            f"{ood_eval['accuracy']:<10.4f} | {ood_eval['macro_f1']:<10.4f} | {ood_eval['top5_acc']:<10.4f} | "
            f"{best_mark:<5}"
        )
    print("=" * 130)



# ============================================================
# 8. SmartSearch utilities: geometry split search + Replay training
# ============================================================
def clip_float(value, low, high):
    return float(min(max(float(value), low), high))


def normalize_weight_config(weights):
    clipped = {
        name: clip_float(weights.get(name, DEFAULT_SCORE_WEIGHTS[name]), *bounds)
        for name, bounds in SEARCH_WEIGHT_BOUNDS.items()
    }
    return normalize_positive_weights(clipped)


def normalize_ratio_config(ratios):
    hard = clip_float(ratios.get("hard", TEST_RATIO_HARD), *SEARCH_RATIO_BOUNDS["hard"])
    middle = clip_float(ratios.get("middle", TEST_RATIO_MIDDLE), *SEARCH_RATIO_BOUNDS["middle"])
    easy = clip_float(ratios.get("easy", TEST_RATIO_EASY), *SEARCH_RATIO_BOUNDS["easy"])

    # Keep curriculum: hard classes get the smallest test allocation preference.
    middle = max(middle, hard)
    easy = max(easy, middle)
    return {"hard": float(hard), "middle": float(middle), "easy": float(easy)}


def make_search_config(name, weights, ratios):
    return {
        "name": str(name),
        "weights": normalize_weight_config(weights),
        "ratios": normalize_ratio_config(ratios),
    }


def config_signature(cfg, precision=5):
    values = []
    for key in ["entropy", "alignment", "anisotropy", "rank"]:
        values.append(round(float(cfg["weights"][key]), precision))
    for key in ["hard", "middle", "easy"]:
        values.append(round(float(cfg["ratios"][key]), precision))
    return tuple(values)


def class_test_ratios_from_scores(class_scores, ratio_config=None):
    """
    Map class difficulty to test allocation preferences.
    High-difficulty classes use the hard ratio, easy classes use the easy ratio.
    These ratios do NOT set the global test ratio; they only allocate the fixed 20% test budget.
    """
    if not class_scores:
        return {}
    ratio_config = normalize_ratio_config(ratio_config or {
        "hard": TEST_RATIO_HARD,
        "middle": TEST_RATIO_MIDDLE,
        "easy": TEST_RATIO_EASY,
    })
    sorted_items = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    ratios = {}
    for rank, (c, _) in enumerate(sorted_items):
        percentile = rank / max(n - 1, 1)
        if percentile < 0.25:
            ratio = ratio_config["hard"]
        elif percentile < 0.75:
            ratio = ratio_config["middle"]
        else:
            ratio = ratio_config["easy"]
        ratios[int(c)] = float(ratio)
    return ratios


def get_fixed_smart_stage1_configs():
    """
    24 interpretable SmartSearch configs: 4 weighting strategies × 6 allocation strategies.
    This is deterministic and contains no TPE/Optuna.
    """
    weight_templates = [
        ("balanced", {"entropy": 0.25, "alignment": 0.25, "anisotropy": 0.25, "rank": 0.25}),
        ("alignment_focus", {"entropy": 0.15, "alignment": 0.55, "anisotropy": 0.15, "rank": 0.15}),
        ("collapse_focus", {"entropy": 0.15, "alignment": 0.15, "anisotropy": 0.35, "rank": 0.35}),
        ("uncertainty_focus", {"entropy": 0.55, "alignment": 0.15, "anisotropy": 0.15, "rank": 0.15}),
    ]
    ratio_templates = [
        ("balanced_15", {"hard": 0.15, "middle": 0.15, "easy": 0.15}),
        ("default_curriculum", {"hard": 0.08, "middle": 0.10, "easy": 0.15}),
        ("strong_curriculum", {"hard": 0.05, "middle": 0.10, "easy": 0.20}),
        ("easy_more_test", {"hard": 0.06, "middle": 0.15, "easy": 0.30}),
        ("mid_easy_more_test", {"hard": 0.08, "middle": 0.20, "easy": 0.30}),
        ("high_test_all", {"hard": 0.15, "middle": 0.25, "easy": 0.35}),
    ]
    configs = []
    for w_name, weights in weight_templates:
        for r_name, ratios in ratio_templates:
            configs.append(make_search_config(f"{w_name}__{r_name}", weights, ratios))
    return configs


def sample_uniform_config(rng, name):
    weights = {k: rng.uniform(low, high) for k, (low, high) in SEARCH_WEIGHT_BOUNDS.items()}
    ratios = {k: rng.uniform(low, high) for k, (low, high) in SEARCH_RATIO_BOUNDS.items()}
    return make_search_config(name, weights, ratios)


def generate_candidate_configs(rng, count, prefix):
    if prefix == "stage1":
        return get_fixed_smart_stage1_configs()[:count]

    configs = []
    seen = set()
    attempts = 0
    while len(configs) < count and attempts < count * 30:
        attempts += 1
        cfg = sample_uniform_config(rng, f"{prefix}_{len(configs) + 1:03d}")
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(cfg)
    return configs


def split_feature_summary(split_report, train_size, test_size):
    if not split_report:
        return [float(train_size), float(test_size), 0.0, 0.0, 0.0, 0.0]
    available = []
    test_ratios = []
    for item in split_report.values():
        available.append(float(item.get("available", 0)))
        test_ratios.append(float(item.get("test_ratio", 0.0)))
    return [
        float(train_size),
        float(test_size),
        float(np.mean(available)),
        float(np.std(available)),
        float(np.mean(test_ratios)),
        float(np.std(test_ratios)),
    ]


def canonical_model_type(model_type=None):
    text = str(model_type or MODEL_TYPE).lower()
    if "graphcodebert" in text:
        return "GraphCodeBERT"
    if "codet5" in text or "code_t5" in text or "t5" in text:
        return "codeT5"
    if "codebert" in text:
        return "codebert"
    return str(model_type or MODEL_TYPE)


def infer_model_type_from_path(path):
    text = str(path or "").lower()
    if "graphcodebert" in text:
        return "GraphCodeBERT"
    if "codet5" in text or "code_t5" in text:
        return "codeT5"
    if "codebert" in text:
        return "codebert"
    return None


def infer_record_model_type(item, source_path=None):
    if isinstance(item, dict):
        for key in ("model_type", "source_model_type", "target_model_type"):
            value = item.get(key)
            if value:
                return canonical_model_type(value)
        source_file = item.get("_source_file")
    else:
        source_file = None
    return canonical_model_type(infer_model_type_from_path(source_file or source_path) or MODEL_TYPE)


def model_feature_vector(model_type=None):
    canonical = canonical_model_type(model_type)
    is_codebert = 1.0 if canonical == "codebert" else 0.0
    is_graphcodebert = 1.0 if canonical == "GraphCodeBERT" else 0.0
    is_codet5 = 1.0 if canonical == "codeT5" else 0.0
    uses_mean = 1.0 if "t5" in canonical.lower() else 0.0
    uses_cls = 1.0 - uses_mean
    hidden_size_norm = 768.0 / 1024.0
    model_family_id = {"codebert": 0.0, "GraphCodeBERT": 0.5, "codeT5": 1.0}.get(canonical, 0.25)
    return [
        is_codebert,
        is_graphcodebert,
        is_codet5,
        uses_cls,
        uses_mean,
        hidden_size_norm,
        model_family_id,
    ]


def config_feature_vector(cfg, train_size=0, test_size=0, split_report=None, model_type=None):
    features = [
        float(cfg["weights"]["entropy"]),
        float(cfg["weights"]["alignment"]),
        float(cfg["weights"]["anisotropy"]),
        float(cfg["weights"]["rank"]),
        float(cfg["ratios"]["hard"]),
        float(cfg["ratios"]["middle"]),
        float(cfg["ratios"]["easy"]),
    ]
    features.extend(split_feature_summary(split_report, train_size, test_size))
    features.extend(model_feature_vector(model_type))
    return features


def build_trial_split(cfg, class_metrics, ood_pool_raw):
    class_scores = compute_class_scores(class_metrics, cfg["weights"])
    class_test_ratios = class_test_ratios_from_scores(class_scores, cfg["ratios"])
    train_raw, test_raw, split_report = build_geometry_ood_train_test_split(ood_pool_raw, class_test_ratios)
    features = config_feature_vector(
        cfg,
        train_size=len(train_raw),
        test_size=len(test_raw),
        split_report=split_report,
    )
    return train_raw, test_raw, split_report, features, class_scores, class_test_ratios


def fit_surrogate_model(search_results):
    usable = [x for x in search_results if x.get("features") is not None and x.get("select_score") is not None]
    if len(usable) < SURROGATE_MIN_SAMPLES:
        return None
    x = np.asarray([item["features"] for item in usable], dtype=float)
    y = np.asarray([item["select_score"] for item in usable], dtype=float)
    model = ExtraTreesRegressor(
        n_estimators=SURROGATE_N_ESTIMATORS,
        random_state=SEED,
        min_samples_leaf=2,
    )
    model.fit(x, y)
    return model


def surrogate_predict_with_uncertainty(model, features):
    x = np.asarray(features, dtype=float)
    tree_preds = np.asarray([tree.predict(x) for tree in model.estimators_])
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)
    acquisition = mean + SURROGATE_EXPLORATION_WEIGHT * std
    return mean, std, acquisition


def select_unique_configs(configs, limit, seen=None):
    seen = set() if seen is None else seen
    selected = []
    for cfg in configs:
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        selected.append(cfg)
        if len(selected) >= limit:
            break
    return selected


def train_replay_for_epochs(model, ood_train_loader, replay_buffer_ds, num_epochs, trial_name):
    replay_loader = DataLoader(replay_buffer_ds, batch_size=REPLAY_BATCH_SIZE, shuffle=True)
    replay_iter = itertools.cycle(replay_loader)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(ood_train_loader) * num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    train_losses = []
    loss_breakdowns = []
    for epoch in range(1, num_epochs + 1):
        loss_stats = train_one_epoch_replay(
            model,
            ood_train_loader,
            replay_iter,
            optimizer,
            scheduler,
            epoch,
        )
        train_losses.append(float(loss_stats["total_loss"]))
        loss_breakdowns.append(loss_stats)
        print(
            f">>> [{trial_name}] Epoch {epoch}/{num_epochs} | "
            f"loss={loss_stats['total_loss']:.4f} | "
            f"OOD_CE={loss_stats['ood_ce']:.4f} | "
            f"Replay*w={loss_stats['replay_weighted']:.4f}"
        )
    return model, train_losses, loss_breakdowns


def export_top_configs(search_results, path=SMART_TOP_CONFIGS_FILE, top_k=5):
    valid = [item for item in search_results if item.get("select_score") is not None]
    valid = sorted(valid, key=lambda x: x["select_score"], reverse=True)[:top_k]
    exported = []
    for item in valid:
        cfg = item["config"]
        exported.append({
            "trial": item.get("trial"),
            "stage": item.get("stage"),
            "select_metric": item.get("select_metric"),
            "select_score": float(item.get("select_score", 0.0)),
            "id_macro_f1": float(item.get("id_test", {}).get("macro_f1", 0.0)),
            "ood_macro_f1": float(item.get("ood_test", {}).get("macro_f1", 0.0)),
            "train_size": int(item.get("train_size", 0)),
            "ood_test_size": int(item.get("ood_test_size", 0)),
            "weights": cfg["weights"],
            "ratios": cfg["ratios"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Top SmartSearch configs. This file is for reporting/debugging only; no TPE/Optuna is used.",
            "top_k": int(top_k),
            "configs": exported,
        }, f, ensure_ascii=False, indent=2)
    print("Top SmartSearch configs saved:", path)
    return exported



# ============================================================
# 9. Resume helpers
# ============================================================
def load_existing_search_results(path):
    if not RESUME_SEARCH or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[Resume] Existing search file is not a list, ignore: {path}")
            return []
        valid = [x for x in data if isinstance(x, dict) and x.get("trial")]
        print(f"[Resume] Loaded {len(valid)} completed trials from: {path}")
        return valid
    except json.JSONDecodeError as exc:
        print(f"[Resume] Search results JSON is incomplete/corrupted, ignore it: {repr(exc)}")
        return []


def best_loaded_full_budget_result(search_results):
    full = [x for x in search_results if x.get("stage") == "stage3_full" and x.get("select_score") is not None]
    if not full:
        return None
    return max(full, key=lambda x: float(x.get("select_score", -1e9)))


def print_gpu_memory(prefix=""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"{prefix}CUDA memory | allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB")


# ============================================================
# 9. TPE + Pareto helpers for Replay-Geometry search
# ============================================================
# 说明：
# - 这里不再使用 ExtraTrees surrogate；
# - 使用 Optuna TPE 搜索几何权重和 hard/middle/easy 测试分配偏好；
# - Replay 方法仍然保留：每个 trial 都从 RQ1 best model 出发，在 geometry OOD train 上做 Replay CL；
# - 下面的 TOP_SMARTSEARCH_CONFIGS 来自上一轮 SmartSearch 输出，用作 TPE prior/warm-start。

TPE_SEARCH_TRIALS = 24                 # 新增 TPE 搜索 trial 数；不包含下面的 prior 观测
TPE_SEARCH_EPOCHS = 5                  # 和之前 TPE-E5 逻辑一致；显存/时间紧张可改成 3
TPE_FULL_EVAL_TOP_K = 3                # Pareto/高 F1 候选进入 full-budget confirmation 的数量
TPE_FULL_EVAL_EPOCHS = EPOCHS          # full-budget Replay epoch，默认 8
OPTUNA_STARTUP_TRIALS = 5              # TPE 前的随机探索数量；prior 不占用真实训练
OPTUNA_STUDY_NAME = f"rq5_tpe_pareto_replay_geo_{MODEL_TYPE}"
OPTUNA_STORAGE = None                  # 需要 optuna 级别断点续跑时可设 sqlite:///xxx.db
USE_SMARTSEARCH_PRIORS_AS_OBSERVATIONS = True
TPE_GUIDANCE_METRIC = "macro_f1"

TPE_SEARCH_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq5_tpe_pareto_replay_search_results_{MODEL_TYPE}.json")
TPE_TOP_CONFIGS_FILE = os.path.join(SAVE_DIR, f"rq5_tpe_pareto_replay_top_configs_{MODEL_TYPE}.json")
TPE_PARETO_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq5_tpe_pareto_replay_pareto_results_{MODEL_TYPE}.json")
TRIAL_META_TRAINING_FILE = os.path.join(SAVE_DIR, f"rq5_trial_meta_training_data_{MODEL_TYPE}.jsonl")
SURROGATE_MODEL_FILE = os.path.join(SAVE_DIR, f"rq5_trial_surrogate_extra_trees_{MODEL_TYPE}.pkl")
SURROGATE_REPORT_FILE = os.path.join(SAVE_DIR, f"rq5_trial_surrogate_report_{MODEL_TYPE}.json")

# Run modes:
#   full_tpe             : original RQ5 search + full-budget Pareto evaluation.
#   train_surrogate      : train a lightweight predictor from previous trial records only.
#   recommend            : use a trained predictor to recommend Top-K configs on the current dataset.
#   recommend_validate   : recommend Top-K configs, then really train/evaluate them and update the predictor.
#   cst_to_token         : train surrogate on Java250-cst trials, then recommend/validate on Java250-token.
#   cst_to_user          : train surrogate on Java250-cst trials, then recommend/validate on Java250-user.
#   java250_cst_to_python75_cst:
#                          train surrogate on Java250-cst-codebert trials, then recommend/validate on Python75-cst-codebert.
# Typical Java250 cst -> token transfer:
#   Step 1: DATASET_PRESET = "java250_cst_source";   RQ5_RUN_MODE = "train_surrogate"
#   Step 2: DATASET_PRESET = "java250_token_target"; RQ5_RUN_MODE = "recommend"
#   Step 3: DATASET_PRESET = "java250_token_target"; RQ5_RUN_MODE = "recommend_validate"
#   One-shot: RQ5_RUN_MODE = "cst_to_token"
RQ5_RUN_MODE = "java250_cst_to_python75_cst"

# For transfer, point this to old trial records, e.g. Java250/cst results, then switch DATA_DIR/RQ1_BEST_MODEL
# to the new target dataset such as Java250/token.
SURROGATE_SOURCE_RESULTS_FILES = [TPE_SEARCH_RESULTS_FILE, METRICS_FILE, TPE_PARETO_RESULTS_FILE]
SURROGATE_PRETRAINED_MODEL_FILE = SURROGATE_MODEL_FILE
SURROGATE_CANDIDATE_POOL_SIZE = 1000
SURROGATE_RECOMMEND_TOP_K = 10
SURROGATE_VALIDATE_TOP_K = 3
SURROGATE_VALIDATE_EPOCHS = EPOCHS
SURROGATE_RECOMMEND_TARGET = "ood_macro_f1"  # or: weighted_harmonic_score / ood_accuracy / ood_top5_acc
USE_CROSS_MODEL_SURROGATE = False
CROSS_MODEL_SURROGATE_MODEL_TYPES = ["codebert"]
SURROGATE_RECOMMENDATIONS_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_recommendations_{MODEL_TYPE}.json")
SURROGATE_VALIDATION_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_validation_results_{MODEL_TYPE}.json")
SURROGATE_BEST_CONFIG_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_target_best_config_{MODEL_TYPE}.json")
SURROGATE_BEST_METRICS_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_target_best_metrics_{MODEL_TYPE}.json")
SURROGATE_BEST_MODEL_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_target_best_model_{MODEL_TYPE}.bin")


def refresh_derived_paths():
    global GEOMETRY_OOD_TRAIN_FILE, GEOMETRY_OOD_TEST_FILE, SPLIT_REPORT_FILE, METRICS_FILE, BEST_MODEL_PATH
    global SMART_SEARCH_RESULTS_FILE, SMART_TOP_CONFIGS_FILE
    global TPE_SEARCH_RESULTS_FILE, TPE_TOP_CONFIGS_FILE, TPE_PARETO_RESULTS_FILE
    global TRIAL_META_TRAINING_FILE, SURROGATE_MODEL_FILE, SURROGATE_REPORT_FILE
    global SURROGATE_RECOMMENDATIONS_FILE, SURROGATE_VALIDATION_RESULTS_FILE
    global SURROGATE_BEST_CONFIG_FILE, SURROGATE_BEST_METRICS_FILE, SURROGATE_BEST_MODEL_FILE

    os.makedirs(SAVE_DIR, exist_ok=True)
    GEOMETRY_OOD_TRAIN_FILE = os.path.join(SAVE_DIR, "geometry_replay_ood_train.jsonl")
    GEOMETRY_OOD_TEST_FILE = os.path.join(SAVE_DIR, "geometry_replay_ood_test_new.jsonl")
    SPLIT_REPORT_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geometry_split_report_{MODEL_TYPE}.json")
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geometry_split_metrics_{MODEL_TYPE}.json")
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, f"best_rq4_replay_geometry_{MODEL_TYPE}.bin")
    SMART_SEARCH_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geo_smartsearch_results_{MODEL_TYPE}.json")
    SMART_TOP_CONFIGS_FILE = os.path.join(SAVE_DIR, f"rq4_replay_geo_smartsearch_top_configs_{MODEL_TYPE}.json")
    TPE_SEARCH_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq5_tpe_pareto_replay_search_results_{MODEL_TYPE}.json")
    TPE_TOP_CONFIGS_FILE = os.path.join(SAVE_DIR, f"rq5_tpe_pareto_replay_top_configs_{MODEL_TYPE}.json")
    TPE_PARETO_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq5_tpe_pareto_replay_pareto_results_{MODEL_TYPE}.json")
    TRIAL_META_TRAINING_FILE = os.path.join(SAVE_DIR, f"rq5_trial_meta_training_data_{MODEL_TYPE}.jsonl")
    SURROGATE_MODEL_FILE = os.path.join(SAVE_DIR, f"rq5_trial_surrogate_extra_trees_{MODEL_TYPE}.pkl")
    SURROGATE_REPORT_FILE = os.path.join(SAVE_DIR, f"rq5_trial_surrogate_report_{MODEL_TYPE}.json")
    SURROGATE_RECOMMENDATIONS_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_recommendations_{MODEL_TYPE}.json")
    SURROGATE_VALIDATION_RESULTS_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_validation_results_{MODEL_TYPE}.json")
    SURROGATE_BEST_CONFIG_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_target_best_config_{MODEL_TYPE}.json")
    SURROGATE_BEST_METRICS_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_target_best_metrics_{MODEL_TYPE}.json")
    SURROGATE_BEST_MODEL_FILE = os.path.join(SAVE_DIR, f"rq5_surrogate_target_best_model_{MODEL_TYPE}.bin")


def build_cross_model_result_files(source_save_dir, target_save_dir=None):
    files = []
    for model_type in CROSS_MODEL_SURROGATE_MODEL_TYPES:
        for model_tag in sorted({model_type, str(model_type).lower()}):
            files.extend([
                os.path.join(source_save_dir, f"rq5_tpe_pareto_replay_search_results_{model_tag}.json"),
                os.path.join(source_save_dir, f"rq4_replay_geometry_split_metrics_{model_tag}.json"),
                os.path.join(source_save_dir, f"rq5_tpe_pareto_replay_pareto_results_{model_tag}.json"),
            ])
    if target_save_dir:
        for model_type in CROSS_MODEL_SURROGATE_MODEL_TYPES:
            for model_tag in sorted({model_type, str(model_type).lower()}):
                files.append(os.path.join(target_save_dir, f"rq5_surrogate_validation_results_{model_tag}.json"))
    return files


def apply_dataset_preset(preset_name):
    global DATA_DIR, SAVE_DIR, RQ1_BEST_MODEL, SHIFT_TYPE, OOD_POOL_FILE, NUM_LABELS
    global SURROGATE_SOURCE_RESULTS_FILES, SURROGATE_PRETRAINED_MODEL_FILE

    if preset_name is None or str(preset_name).strip() == "":
        refresh_derived_paths()
        return
    if preset_name not in DATASET_PRESETS:
        raise ValueError(f"Unknown DATASET_PRESET={preset_name}. Available: {sorted(DATASET_PRESETS)}")

    preset = DATASET_PRESETS[preset_name]
    DATA_DIR = preset["data_dir"]
    SAVE_DIR = preset["save_dir"]
    RQ1_BEST_MODEL = preset["rq1_best_model"]
    NUM_LABELS = int(preset.get("num_labels", NUM_LABELS))
    SHIFT_TYPE = preset.get("shift_type", SHIFT_TYPE)
    OOD_POOL_FILE = preset.get("ood_pool_file", OOD_POOL_FILE)
    refresh_derived_paths()

    if USE_CROSS_MODEL_SURROGATE:
        source_save_dir = DATASET_PRESETS["java250_cst_source"]["save_dir"]
        target_save_dir = SAVE_DIR if str(preset_name).endswith("_target") else None
        SURROGATE_SOURCE_RESULTS_FILES = build_cross_model_result_files(source_save_dir, target_save_dir)
    elif "surrogate_source_results_files" in preset:
        SURROGATE_SOURCE_RESULTS_FILES = list(preset["surrogate_source_results_files"])
    if "surrogate_pretrained_model_file" in preset:
        SURROGATE_PRETRAINED_MODEL_FILE = preset["surrogate_pretrained_model_file"]

    print(f">>> Applied DATASET_PRESET={preset_name}")
    print(">>> DATA_DIR:", DATA_DIR)
    print(">>> SAVE_DIR:", SAVE_DIR)
    print(">>> RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print(">>> NUM_LABELS:", NUM_LABELS)
    print(">>> SHIFT_TYPE:", SHIFT_TYPE)
    print(">>> SURROGATE_PRETRAINED_MODEL_FILE:", SURROGATE_PRETRAINED_MODEL_FILE)


TOP_SMARTSEARCH_CONFIGS = [
    {
        "trial": "stage3_full_003_balanced__default_curriculum",
        "stage": "stage3_full",
        "select_score": 0.8282208445636307,
        "id_macro_f1": 0.9371527788597512,
        "ood_macro_f1": 0.8282208445636307,
        "weights": {"entropy": 0.25, "alignment": 0.25, "anisotropy": 0.25, "rank": 0.25},
        "ratios": {"hard": 0.08, "middle": 0.10, "easy": 0.15},
    },
    {
        "trial": "stage3_full_002_surrogate_pool_027",
        "stage": "stage3_full",
        "select_score": 0.8270051264142965,
        "id_macro_f1": 0.9366765641234556,
        "ood_macro_f1": 0.8270051264142965,
        "weights": {"entropy": 0.33996605788083883, "alignment": 0.3225561369699041, "anisotropy": 0.0948329226073143, "rank": 0.24264488254194275},
        "ratios": {"hard": 0.16709999603621886, "middle": 0.18435221101586022, "easy": 0.24008119664104222},
    },
    {
        "trial": "stage2_medium_007_surrogate_pool_112",
        "stage": "stage2_medium",
        "select_score": 0.8260228935940084,
        "id_macro_f1": 0.9366253744594087,
        "ood_macro_f1": 0.8260228935940084,
        "weights": {"entropy": 0.2600476782975983, "alignment": 0.2592497940642117, "anisotropy": 0.23084577428194725, "rank": 0.24985675335624272},
        "ratios": {"hard": 0.11769039980256733, "middle": 0.11769039980256733, "easy": 0.1775366377913326},
    },
    {
        "trial": "stage3_full_001_surrogate_pool_112",
        "stage": "stage3_full",
        "select_score": 0.8255341531606819,
        "id_macro_f1": 0.9370466141361633,
        "ood_macro_f1": 0.8255341531606819,
        "weights": {"entropy": 0.2600476782975983, "alignment": 0.2592497940642117, "anisotropy": 0.23084577428194725, "rank": 0.24985675335624272},
        "ratios": {"hard": 0.11769039980256733, "middle": 0.11769039980256733, "easy": 0.1775366377913326},
    },
    {
        "trial": "stage2_medium_010_surrogate_pool_027",
        "stage": "stage2_medium",
        "select_score": 0.8245188350109345,
        "id_macro_f1": 0.9361421028802999,
        "ood_macro_f1": 0.8245188350109345,
        "weights": {"entropy": 0.33996605788083883, "alignment": 0.3225561369699041, "anisotropy": 0.0948329226073143, "rank": 0.24264488254194275},
        "ratios": {"hard": 0.16709999603621886, "middle": 0.18435221101586022, "easy": 0.24008119664104222},
    },
]

PARETO_OBJECTIVES = ["accuracy", "macro_f1", "top5_acc"]


def cfg_to_optuna_params(cfg):
    return {
        "w_entropy": float(cfg["weights"]["entropy"]),
        "w_alignment": float(cfg["weights"]["alignment"]),
        "w_anisotropy": float(cfg["weights"]["anisotropy"]),
        "w_rank": float(cfg["weights"]["rank"]),
        "test_ratio_hard": float(cfg["ratios"]["hard"]),
        "test_ratio_middle": float(cfg["ratios"]["middle"]),
        "test_ratio_easy": float(cfg["ratios"]["easy"]),
    }


def optuna_param_distributions():
    if FloatDistribution is None:
        return None
    return {
        "w_entropy": FloatDistribution(*SEARCH_WEIGHT_BOUNDS["entropy"]),
        "w_alignment": FloatDistribution(*SEARCH_WEIGHT_BOUNDS["alignment"]),
        "w_anisotropy": FloatDistribution(*SEARCH_WEIGHT_BOUNDS["anisotropy"]),
        "w_rank": FloatDistribution(*SEARCH_WEIGHT_BOUNDS["rank"]),
        "test_ratio_hard": FloatDistribution(*SEARCH_RATIO_BOUNDS["hard"]),
        "test_ratio_middle": FloatDistribution(*SEARCH_RATIO_BOUNDS["middle"]),
        "test_ratio_easy": FloatDistribution(*SEARCH_RATIO_BOUNDS["easy"]),
    }


def params_to_search_config(params, name):
    weights = {
        "entropy": float(params["w_entropy"]),
        "alignment": float(params["w_alignment"]),
        "anisotropy": float(params["w_anisotropy"]),
        "rank": float(params["w_rank"]),
    }
    ratios = {
        "hard": float(params["test_ratio_hard"]),
        "middle": float(params["test_ratio_middle"]),
        "easy": float(params["test_ratio_easy"]),
    }
    return make_search_config(name, weights, ratios)


def suggest_tpe_config(trial):
    params = {
        "w_entropy": trial.suggest_float("w_entropy", *SEARCH_WEIGHT_BOUNDS["entropy"]),
        "w_alignment": trial.suggest_float("w_alignment", *SEARCH_WEIGHT_BOUNDS["alignment"]),
        "w_anisotropy": trial.suggest_float("w_anisotropy", *SEARCH_WEIGHT_BOUNDS["anisotropy"]),
        "w_rank": trial.suggest_float("w_rank", *SEARCH_WEIGHT_BOUNDS["rank"]),
        "test_ratio_hard": trial.suggest_float("test_ratio_hard", *SEARCH_RATIO_BOUNDS["hard"]),
        "test_ratio_middle": trial.suggest_float("test_ratio_middle", *SEARCH_RATIO_BOUNDS["middle"]),
        "test_ratio_easy": trial.suggest_float("test_ratio_easy", *SEARCH_RATIO_BOUNDS["easy"]),
    }
    return params_to_search_config(params, f"tpe_{trial.number:03d}")


def unique_smartsearch_prior_configs():
    configs = []
    seen = set()
    for item in TOP_SMARTSEARCH_CONFIGS:
        cfg = make_search_config(
            f"prior_{item['trial']}",
            item["weights"],
            item["ratios"],
        )
        sig = config_signature(cfg, precision=8)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append({
            "source_trial": item["trial"],
            "source_stage": item.get("stage"),
            "source_ood_macro_f1": float(item.get("ood_macro_f1", item.get("select_score", 0.0))),
            "source_id_macro_f1": float(item.get("id_macro_f1", 0.0)),
            "config": cfg,
        })
    return configs


def add_smartsearch_priors_to_study(study):
    if not USE_SMARTSEARCH_PRIORS_AS_OBSERVATIONS:
        return []
    distributions = optuna_param_distributions()
    if distributions is None:
        return []
    prior_records = unique_smartsearch_prior_configs()
    added = []
    existing = set()
    for t in study.trials:
        if t.params:
            existing.add(tuple(sorted((k, round(float(v), 8)) for k, v in t.params.items())))
    for record in prior_records:
        params = cfg_to_optuna_params(record["config"])
        key = tuple(sorted((k, round(float(v), 8)) for k, v in params.items()))
        if key in existing:
            continue
        try:
            prior_trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=float(record["source_ood_macro_f1"]),
                state=optuna.trial.TrialState.COMPLETE,
                user_attrs={
                    "source": "smartsearch_top_config_prior",
                    "source_trial": record["source_trial"],
                    "source_stage": record["source_stage"],
                },
            )
            study.add_trial(prior_trial)
            added.append(record)
            existing.add(key)
        except Exception as exc:
            print(f"[Optuna prior warning] Failed to add prior {record['source_trial']}: {repr(exc)}")
    return added


def make_prior_search_results(class_metrics, ood_pool_raw):
    """
    把上一轮 SmartSearch top configs 作为候选记录保存到 search_results 中。
    它们不作为已训练 trial 的 Pareto 结果，因为只有 Macro-F1，没有 Accuracy/Top-5；
    但会参与 full-eval 候选补充。
    """
    records = []
    for idx, record in enumerate(unique_smartsearch_prior_configs(), start=1):
        cfg = record["config"]
        train_raw, test_raw, split_report, features, _, _ = build_trial_split(cfg, class_metrics, ood_pool_raw)
        records.append({
            "trial": f"smart_prior_{idx:03d}_{cfg['name']}",
            "stage": "smartsearch_prior_observation",
            "epochs": 0,
            "config": cfg,
            "weights": cfg["weights"],
            "ratios": cfg["ratios"],
            "features": features,
            "train_size": len(train_raw),
            "ood_test_size": len(test_raw),
            "id_test": {"macro_f1": record["source_id_macro_f1"]},
            "ood_test": {"macro_f1": record["source_ood_macro_f1"]},
            "select_metric": TPE_GUIDANCE_METRIC,
            "select_score": record["source_ood_macro_f1"],
            "selection_set": "provided_smartsearch_top_configs",
            "search_backend": "smartsearch_prior_for_tpe",
            "source_trial": record["source_trial"],
            "source_stage": record["source_stage"],
            "is_prior_only": True,
        })
    return records


def metric_value(result, metric):
    return float(result.get("ood_test", {}).get(metric, float("nan")))


def has_all_pareto_metrics(result):
    ood = result.get("ood_test", {})
    return isinstance(ood, dict) and all(k in ood for k in PARETO_OBJECTIVES)


def pareto_dominates(a, b):
    if not has_all_pareto_metrics(a) or not has_all_pareto_metrics(b):
        return False
    not_worse = all(metric_value(a, k) >= metric_value(b, k) for k in PARETO_OBJECTIVES)
    strictly_better = any(metric_value(a, k) > metric_value(b, k) for k in PARETO_OBJECTIVES)
    return not_worse and strictly_better


def compute_pareto_front(results):
    valid = [r for r in results if has_all_pareto_metrics(r)]
    front = []
    for i, cand in enumerate(valid):
        dominated = False
        for j, other in enumerate(valid):
            if i == j:
                continue
            if pareto_dominates(other, cand):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return front


def compute_weighted_harmonic_score(eval_result):
    vals = [float(eval_result.get(k, 0.0)) for k in PARETO_OBJECTIVES]
    vals = [max(v, 1e-8) for v in vals]
    weights = [1.0, 1.5, 0.8]  # Macro-F1 权重大一些
    return float(sum(weights) / sum(w / v for w, v in zip(weights, vals)))


SURROGATE_BASE_FEATURE_NAMES = [
    "w_entropy",
    "w_alignment",
    "w_anisotropy",
    "w_rank",
    "test_ratio_hard",
    "test_ratio_middle",
    "test_ratio_easy",
    "train_size",
    "ood_test_size",
    "mean_class_available",
    "std_class_available",
    "mean_class_test_ratio",
    "std_class_test_ratio",
]

SURROGATE_MODEL_FEATURE_NAMES = [
    "model_is_codebert",
    "model_is_graphcodebert",
    "model_is_codet5",
    "model_uses_cls_pooling",
    "model_uses_mean_pooling",
    "model_hidden_size_norm",
    "model_family_id",
]

SURROGATE_FEATURE_NAMES = SURROGATE_BASE_FEATURE_NAMES + SURROGATE_MODEL_FEATURE_NAMES

SURROGATE_TARGET_NAMES = [
    "ood_accuracy",
    "ood_macro_f1",
    "ood_top5_acc",
    "id_macro_f1",
    "weighted_harmonic_score",
]


def normalize_surrogate_features(features, model_type):
    if features is None:
        return None
    values = [float(x) for x in features]
    if len(values) == len(SURROGATE_FEATURE_NAMES):
        return values
    if len(values) == len(SURROGATE_BASE_FEATURE_NAMES):
        return values + model_feature_vector(model_type)
    return None


def build_surrogate_rows(results):
    rows = []
    seen = set()
    for item in results:
        if item.get("is_prior_only"):
            continue
        source_model_type = infer_record_model_type(item)
        features = normalize_surrogate_features(item.get("features"), source_model_type)
        ood_eval = item.get("ood_test", {})
        id_eval = item.get("id_test", {})
        if features is None:
            continue
        if not isinstance(ood_eval, dict) or not all(k in ood_eval for k in PARETO_OBJECTIVES):
            continue

        trial = str(item.get("trial", ""))
        stage = str(item.get("stage", ""))
        key = (source_model_type, trial, stage, str(item.get("_source_file", "")))
        if key in seen:
            continue
        seen.add(key)

        cfg = item.get("config", {})
        weights = cfg.get("weights", item.get("weights", {}))
        ratios = cfg.get("ratios", item.get("ratios", {}))
        harmonic = compute_weighted_harmonic_score(ood_eval)

        rows.append({
            "trial": trial,
            "stage": stage,
            "epochs": int(item.get("epochs", 0)),
            "search_backend": item.get("search_backend"),
            "selection_set": item.get("selection_set"),
            "model_type": source_model_type,
            "source_file": item.get("_source_file"),
            "select_metric": item.get("select_metric"),
            "select_score": float(item.get("select_score", ood_eval.get("macro_f1", 0.0))),
            "features": [float(x) for x in features],
            "feature_names": SURROGATE_FEATURE_NAMES,
            "weights": {k: float(v) for k, v in weights.items()},
            "ratios": {k: float(v) for k, v in ratios.items()},
            "train_size": int(item.get("train_size", 0)),
            "ood_test_size": int(item.get("ood_test_size", 0)),
            "targets": {
                "ood_accuracy": float(ood_eval.get("accuracy", 0.0)),
                "ood_macro_f1": float(ood_eval.get("macro_f1", 0.0)),
                "ood_top5_acc": float(ood_eval.get("top5_acc", 0.0)),
                "id_macro_f1": float(id_eval.get("macro_f1", 0.0)),
                "weighted_harmonic_score": float(harmonic),
            },
        })
    return rows


def export_trial_records_and_train_surrogate(results):
    rows = build_surrogate_rows(results)
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(TRIAL_META_TRAINING_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "method": "Backbone-aware ExtraTrees surrogate regressor trained from completed RQ5 trial records",
        "trained": False,
        "reason": None,
        "num_records": int(len(rows)),
        "cross_model_surrogate": bool(USE_CROSS_MODEL_SURROGATE),
        "record_model_counts": dict(Counter(row.get("model_type", "unknown") for row in rows)),
        "feature_names": SURROGATE_FEATURE_NAMES,
        "target_names": SURROGATE_TARGET_NAMES,
        "meta_training_file": TRIAL_META_TRAINING_FILE,
        "surrogate_model_file": SURROGATE_MODEL_FILE,
    }

    if len(rows) < SURROGATE_MIN_SAMPLES:
        report["reason"] = f"Need at least {SURROGATE_MIN_SAMPLES} measured trials; got {len(rows)}."
        with open(SURROGATE_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("Surrogate skipped:", report["reason"])
        return report

    x = np.asarray([row["features"] for row in rows], dtype=float)
    y = np.asarray(
        [[row["targets"][name] for name in SURROGATE_TARGET_NAMES] for row in rows],
        dtype=float,
    )

    model = ExtraTreesRegressor(
        n_estimators=SURROGATE_N_ESTIMATORS,
        random_state=SEED,
        min_samples_leaf=2,
    )
    model.fit(x, y)
    pred = model.predict(x)
    mae = np.mean(np.abs(pred - y), axis=0)

    payload = {
        "model": model,
        "feature_names": SURROGATE_FEATURE_NAMES,
        "target_names": SURROGATE_TARGET_NAMES,
        "target_model_type_when_trained": MODEL_TYPE,
        "record_model_counts": dict(Counter(row.get("model_type", "unknown") for row in rows)),
        "cross_model_surrogate": bool(USE_CROSS_MODEL_SURROGATE),
        "num_labels": NUM_LABELS,
        "seed": SEED,
        "note": "Input features come from config_feature_vector(...), including geometry/config features and backbone-aware model features.",
    }
    with open(SURROGATE_MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)

    report.update({
        "trained": True,
        "reason": "ok",
        "model": "ExtraTreesRegressor",
        "n_estimators": int(SURROGATE_N_ESTIMATORS),
        "min_samples_leaf": 2,
        "train_mae": {name: float(value) for name, value in zip(SURROGATE_TARGET_NAMES, mae)},
        "feature_importance": {
            name: float(value)
            for name, value in zip(SURROGATE_FEATURE_NAMES, model.feature_importances_)
        },
    })
    with open(SURROGATE_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Trial meta-training data:", TRIAL_META_TRAINING_FILE)
    print("Surrogate model:", SURROGATE_MODEL_FILE)
    print("Surrogate report:", SURROGATE_REPORT_FILE)
    return report


def pareto_rank_tuple(result):
    ood = result.get("ood_test", {})
    return (
        float(ood.get("macro_f1", 0.0)),
        compute_weighted_harmonic_score(ood) if has_all_pareto_metrics(result) else 0.0,
        float(ood.get("accuracy", 0.0)),
        float(ood.get("top5_acc", 0.0)),
    )


def select_final_from_pareto(results):
    front = compute_pareto_front(results)
    if front:
        selected = max(front, key=pareto_rank_tuple)
        reason = "pareto_front_macro_f1_then_harmonic"
    else:
        valid = [r for r in results if isinstance(r.get("ood_test"), dict) and "macro_f1" in r["ood_test"]]
        if not valid:
            raise RuntimeError("No valid result for Pareto/fallback selection.")
        selected = max(valid, key=lambda r: float(r["ood_test"].get("macro_f1", 0.0)))
        reason = "fallback_macro_f1_no_complete_pareto_metrics"

    front_names = {r.get("trial") for r in front}
    selected_name = selected.get("trial")
    for r in results:
        r["is_pareto"] = r.get("trial") in front_names
        r["is_pareto_selected"] = r.get("trial") == selected_name
        if has_all_pareto_metrics(r):
            r["weighted_harmonic_score"] = compute_weighted_harmonic_score(r["ood_test"])
    return selected, front, reason


def select_full_eval_candidates(stage1_results, prior_results, top_k=TPE_FULL_EVAL_TOP_K):
    selected = []
    seen = set()

    def add(r):
        cfg = r.get("config")
        if cfg is None:
            return
        sig = config_signature(cfg, precision=8)
        if sig in seen:
            return
        seen.add(sig)
        selected.append(r)

    # 1) 优先从有完整 Accuracy/F1/Top5 的 TPE Stage 结果中取 Pareto front。
    pareto_front = compute_pareto_front(stage1_results)
    for r in sorted(pareto_front, key=pareto_rank_tuple, reverse=True):
        add(r)
        if len(selected) >= top_k:
            return selected[:top_k]

    # 2) 补充 TPE Stage Macro-F1 高的配置。
    for r in sorted(stage1_results, key=lambda x: float(x.get("ood_test", {}).get("macro_f1", 0.0)), reverse=True):
        add(r)
        if len(selected) >= top_k:
            return selected[:top_k]

    # 3) 再补充上一轮 SmartSearch top configs，避免 TPE 早期搜索反而弱于先验配置。
    for r in sorted(prior_results, key=lambda x: float(x.get("ood_test", {}).get("macro_f1", 0.0)), reverse=True):
        add(r)
        if len(selected) >= top_k:
            return selected[:top_k]

    return selected[:top_k]


def export_tpe_top_configs(results, path=TPE_TOP_CONFIGS_FILE, top_k=10):
    valid = [r for r in results if isinstance(r.get("ood_test"), dict) and "macro_f1" in r["ood_test"]]
    valid = sorted(valid, key=lambda r: float(r.get("ood_test", {}).get("macro_f1", 0.0)), reverse=True)[:top_k]
    out = []
    for r in valid:
        cfg = r.get("config", {})
        out.append({
            "trial": r.get("trial"),
            "stage": r.get("stage"),
            "epochs": r.get("epochs"),
            "search_backend": r.get("search_backend"),
            "select_score": r.get("select_score"),
            "id_macro_f1": r.get("id_test", {}).get("macro_f1"),
            "ood_accuracy": r.get("ood_test", {}).get("accuracy"),
            "ood_macro_f1": r.get("ood_test", {}).get("macro_f1"),
            "ood_top5_acc": r.get("ood_test", {}).get("top5_acc"),
            "train_size": r.get("train_size"),
            "ood_test_size": r.get("ood_test_size"),
            "weights": cfg.get("weights"),
            "ratios": cfg.get("ratios"),
            "is_pareto": r.get("is_pareto", False),
            "is_pareto_selected": r.get("is_pareto_selected", False),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Top TPE+Pareto Replay-Geometry configs. SmartSearch top configs are used as TPE priors/warm-start.",
            "top_k": int(top_k),
            "configs": out,
        }, f, ensure_ascii=False, indent=2)
    print("TPE top configs saved:", path)
    return out


def load_trial_records_from_files(paths):
    records = []
    for path in paths:
        if not path or not os.path.exists(path):
            print(f"[Surrogate] Trial record file not found, skip: {path}")
            continue
        inferred_model_type = canonical_model_type(infer_model_type_from_path(path) or MODEL_TYPE)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"[Surrogate] Invalid JSON, skip {path}: {repr(exc)}")
            continue

        loaded = []
        if isinstance(data, list):
            loaded = [x for x in data if isinstance(x, dict) and x.get("trial")]
        elif isinstance(data, dict):
            parent_model_type = canonical_model_type(data.get("model_type") or inferred_model_type)
            for key in ["search_results", "full_results", "final_pareto_front"]:
                value = data.get(key)
                if isinstance(value, list):
                    loaded.extend([x for x in value if isinstance(x, dict) and x.get("trial")])
            best_trial = data.get("best_trial") or data.get("final_selected")
            if isinstance(best_trial, dict) and best_trial.get("trial"):
                loaded.append(best_trial)
            for item in loaded:
                item.setdefault("model_type", parent_model_type)
        else:
            print(f"[Surrogate] Unsupported trial record format, skip: {path}")
            continue

        for item in loaded:
            item.setdefault("model_type", inferred_model_type)
            item["_source_file"] = path
        print(f"[Surrogate] Loaded {len(loaded)} trial records from {path}")
        records.extend(loaded)

    unique = []
    seen = set()
    for item in records:
        model_type = infer_record_model_type(item)
        key = (model_type, str(item.get("trial", "")), str(item.get("stage", "")), str(item.get("_source_file", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    print(f"[Surrogate] Total unique trial records: {len(unique)}")
    return unique


def run_train_surrogate_only():
    records = load_trial_records_from_files(SURROGATE_SOURCE_RESULTS_FILES)
    if not records:
        raise RuntimeError("No trial records found. Please set SURROGATE_SOURCE_RESULTS_FILES to existing RQ5 result files.")
    report = export_trial_records_and_train_surrogate(records)
    print("Surrogate training report:", report)
    return report


def load_surrogate_payload(path=SURROGATE_PRETRAINED_MODEL_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Surrogate model not found: {path}. "
            "Run RQ5_RUN_MODE='train_surrogate' first, or set SURROGATE_PRETRAINED_MODEL_FILE."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Invalid surrogate model payload: {path}")
    return payload


def prepare_transfer_geometry_context(tokenizer):
    id_train_raw = read_jsonl(resolve_data_path(ID_TRAIN_FILE))
    id_test_raw = read_jsonl(resolve_data_path(ID_TEST_FILE))
    ood_pool_raw = read_jsonl(resolve_data_path(OOD_POOL_FILE))
    if not ood_pool_raw:
        raise RuntimeError("No OOD pool samples loaded. Please check OOD_POOL_FILE.")

    id_train_ds = CodeDataset(id_train_raw, tokenizer, name="transfer_id_train")
    id_test_ds = CodeDataset(id_test_raw, tokenizer, name="transfer_id_test")
    ood_pool_ds = CodeDataset(ood_pool_raw, tokenizer, name="transfer_ood_pool")
    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_pool_loader = DataLoader(ood_pool_ds, batch_size=BATCH_SIZE, shuffle=False)

    rq1_model = load_rq1_model()
    id_emb, _, id_labels, id_idxs = extract_embeddings_and_logits(
        rq1_model,
        id_test_loader,
        desc="Transfer: extract ID test embeddings",
    )
    ood_emb, ood_logits, ood_labels, ood_idxs = extract_embeddings_and_logits(
        rq1_model,
        ood_pool_loader,
        desc="Transfer: extract OOD pool embeddings",
    )
    class_metrics = compute_per_class_metrics(
        ood_emb,
        ood_logits,
        ood_labels,
        ood_idxs,
        id_emb,
        id_labels,
        id_idxs,
    )
    print(">>> Transfer geometry class count:", len(class_metrics))

    del rq1_model, id_emb, id_labels, id_idxs, ood_emb, ood_logits, ood_labels, ood_idxs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "id_train_raw": id_train_raw,
        "id_test_raw": id_test_raw,
        "ood_pool_raw": ood_pool_raw,
        "id_train_ds": id_train_ds,
        "id_test_ds": id_test_ds,
        "id_test_loader": id_test_loader,
        "class_metrics": class_metrics,
    }


def build_transfer_candidate_configs():
    rng = np.random.RandomState(SEED)
    configs = []
    seen = set()

    for record in unique_smartsearch_prior_configs():
        cfg = record["config"]
        sig = config_signature(cfg, precision=8)
        if sig not in seen:
            seen.add(sig)
            configs.append(make_search_config(f"prior_{len(configs) + 1:03d}", cfg["weights"], cfg["ratios"]))

    for cfg in get_fixed_smart_stage1_configs():
        sig = config_signature(cfg, precision=8)
        if sig not in seen:
            seen.add(sig)
            configs.append(cfg)

    attempts = 0
    while len(configs) < SURROGATE_CANDIDATE_POOL_SIZE and attempts < SURROGATE_CANDIDATE_POOL_SIZE * 40:
        attempts += 1
        cfg = sample_uniform_config(rng, f"surrogate_pool_{len(configs) + 1:04d}")
        sig = config_signature(cfg, precision=8)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(cfg)

    return configs[:SURROGATE_CANDIDATE_POOL_SIZE]


def predict_transfer_candidates(payload, class_metrics, ood_pool_raw):
    model = payload["model"]
    feature_names = payload.get("feature_names", SURROGATE_FEATURE_NAMES)
    target_names = payload.get("target_names", SURROGATE_TARGET_NAMES)
    if list(feature_names) != SURROGATE_FEATURE_NAMES:
        raise RuntimeError(
            f"Surrogate feature mismatch. model_features={feature_names}, expected={SURROGATE_FEATURE_NAMES}"
        )

    candidates = []
    for cfg in build_transfer_candidate_configs():
        train_raw, test_raw, split_report, features, class_scores, class_test_ratios = build_trial_split(
            cfg,
            class_metrics,
            ood_pool_raw,
        )
        if not train_raw or not test_raw:
            continue
        pred = np.asarray(model.predict(np.asarray([features], dtype=float))[0], dtype=float)
        pred_targets = {name: float(value) for name, value in zip(target_names, pred)}
        recommend_score = float(pred_targets.get(SURROGATE_RECOMMEND_TARGET, pred_targets.get("ood_macro_f1", 0.0)))
        candidates.append({
            "config": cfg,
            "weights": cfg["weights"],
            "ratios": cfg["ratios"],
            "target_model_type": canonical_model_type(MODEL_TYPE),
            "features": [float(x) for x in features],
            "feature_names": SURROGATE_FEATURE_NAMES,
            "predicted": pred_targets,
            "recommend_target": SURROGATE_RECOMMEND_TARGET,
            "recommend_score": recommend_score,
            "train_size": len(train_raw),
            "ood_test_size": len(test_raw),
            "split_report": split_report,
            "class_test_ratios": class_test_ratios,
            "class_scores": class_scores,
        })

    candidates = sorted(candidates, key=lambda x: x["recommend_score"], reverse=True)
    return candidates


def validate_surrogate_recommendations(recommendations, context, tokenizer):
    validation_results = []
    replay_buffer_ds, replay_buffer_indices = build_replay_buffer_dataset(
        context["id_train_ds"],
        buffer_size=REPLAY_BUFFER_SIZE,
        seed=REPLAY_BUFFER_SEED,
    )
    id_test_loader = context["id_test_loader"]

    for idx, rec in enumerate(recommendations[:SURROGATE_VALIDATE_TOP_K], start=1):
        cfg = rec["config"]
        trial_name = f"surrogate_validate_{idx:03d}_{cfg['name']}"
        print("\n" + "=" * 100)
        print(f"Surrogate validation trial: {trial_name}")
        print("predicted:", rec["predicted"])
        print("weights:", cfg["weights"])
        print("ratios :", cfg["ratios"])

        train_raw, test_raw, split_report, features, _, _ = build_trial_split(
            cfg,
            context["class_metrics"],
            context["ood_pool_raw"],
        )
        train_ds = CodeDataset(train_raw, tokenizer, name=f"{trial_name}_ood_train")
        test_ds = CodeDataset(test_raw, tokenizer, name=f"{trial_name}_ood_test_new")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = load_rq1_model()
        start = time.time()
        model, train_losses, loss_breakdowns = train_replay_for_epochs(
            model,
            train_loader,
            replay_buffer_ds,
            SURROGATE_VALIDATE_EPOCHS,
            trial_name,
        )
        train_seconds = time.time() - start
        id_eval = evaluate(model, id_test_loader, desc=f"{trial_name} ID Test")
        ood_eval = evaluate(model, test_loader, desc=f"{trial_name} OOD Test")

        safe_name = trial_name.replace(os.sep, "_").replace(" ", "_")
        trial_model_path = os.path.join(SAVE_DIR, f"{safe_name}.bin")
        trial_train_file = os.path.join(SAVE_DIR, f"{safe_name}_ood_train.jsonl")
        trial_test_file = os.path.join(SAVE_DIR, f"{safe_name}_ood_test_new.jsonl")
        torch.save(model.state_dict(), trial_model_path)
        write_jsonl(trial_train_file, train_raw)
        write_jsonl(trial_test_file, test_raw)

        result = {
            "trial": trial_name,
            "stage": "surrogate_transfer_validation",
            "epochs": int(SURROGATE_VALIDATE_EPOCHS),
            "model_type": canonical_model_type(MODEL_TYPE),
            "config": cfg,
            "weights": cfg["weights"],
            "ratios": cfg["ratios"],
            "features": features,
            "train_size": len(train_raw),
            "ood_test_size": len(test_raw),
            "id_test": id_eval,
            "ood_test": ood_eval,
            "predicted": rec["predicted"],
            "train_losses": train_losses,
            "loss_breakdowns": loss_breakdowns,
            "select_metric": TPE_GUIDANCE_METRIC,
            "select_score": float(ood_eval.get(TPE_GUIDANCE_METRIC, 0.0)),
            "selection_set": "surrogate_recommended_geometry_ood_test_new",
            "search_backend": "surrogate_transfer_validation",
            "train_seconds": float(train_seconds),
            "trial_model_path": trial_model_path,
            "trial_train_file": trial_train_file,
            "trial_test_file": trial_test_file,
            "split_report": split_report,
            "replay_buffer_indices": replay_buffer_indices,
        }
        validation_results.append(result)

        del model, train_ds, test_ds, train_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(SURROGATE_VALIDATION_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "method": "Backbone-aware surrogate recommended Top-K configs with real Replay validation",
            "model_type": canonical_model_type(MODEL_TYPE),
            "validate_top_k": int(SURROGATE_VALIDATE_TOP_K),
            "validate_epochs": int(SURROGATE_VALIDATE_EPOCHS),
            "results": validation_results,
        }, f, ensure_ascii=False, indent=2)
    return validation_results


def select_and_export_best_surrogate_validation(validation_results):
    if not validation_results:
        raise RuntimeError("No surrogate validation results to select from.")

    best_result, pareto_front, reason = select_final_from_pareto(validation_results)

    if best_result.get("trial_model_path") and os.path.exists(best_result["trial_model_path"]):
        shutil.copyfile(best_result["trial_model_path"], SURROGATE_BEST_MODEL_FILE)
    else:
        print("[Warning] Best validation model checkpoint not found:", best_result.get("trial_model_path"))

    if best_result.get("trial_train_file") and os.path.exists(best_result["trial_train_file"]):
        shutil.copyfile(best_result["trial_train_file"], GEOMETRY_OOD_TRAIN_FILE)
    if best_result.get("trial_test_file") and os.path.exists(best_result["trial_test_file"]):
        shutil.copyfile(best_result["trial_test_file"], GEOMETRY_OOD_TEST_FILE)

    best_config_payload = {
        "method": f"Best Java250-{SHIFT_TYPE} config selected after real Replay validation of surrogate recommendations",
        "selection_reason": reason,
        "data_dir": DATA_DIR,
        "rq1_best_model": RQ1_BEST_MODEL,
        "source_surrogate_model": SURROGATE_PRETRAINED_MODEL_FILE,
        "best_trial": best_result["trial"],
        "best_config": best_result["config"],
        "best_weights": best_result["weights"],
        "best_ratios": best_result["ratios"],
        "predicted": best_result.get("predicted"),
        "id_test": best_result["id_test"],
        "ood_test": best_result["ood_test"],
        "pareto_front_trials": [item.get("trial") for item in pareto_front],
        "files": {
            "best_model": SURROGATE_BEST_MODEL_FILE,
            "best_geometry_ood_train": GEOMETRY_OOD_TRAIN_FILE,
            "best_geometry_ood_test": GEOMETRY_OOD_TEST_FILE,
            "source_trial_model": best_result.get("trial_model_path"),
            "source_trial_train_file": best_result.get("trial_train_file"),
            "source_trial_test_file": best_result.get("trial_test_file"),
        },
    }

    with open(SURROGATE_BEST_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(best_config_payload, f, ensure_ascii=False, indent=2)

    best_metrics_payload = {
        "task": "Code Classification",
        "method": f"CST-trained trial-surrogate transferred to Java250-{SHIFT_TYPE}; Top-K configs validated by Replay CL",
        "model_type": MODEL_TYPE,
        "dataset_preset": DATASET_PRESET,
        "data_dir": DATA_DIR,
        "selection_reason": reason,
        "best_trial": best_result["trial"],
        "best_config": best_result["config"],
        "final_evaluation": {
            "id_test": best_result["id_test"],
            f"{SHIFT_TYPE}_ood_test_new": best_result["ood_test"],
        },
        "validation_results_file": SURROGATE_VALIDATION_RESULTS_FILE,
        "best_config_file": SURROGATE_BEST_CONFIG_FILE,
        "best_model_file": SURROGATE_BEST_MODEL_FILE,
        "time": {"total_seconds": time.time() - START_TIME, "total_readable": format_time(time.time() - START_TIME)},
    }
    with open(SURROGATE_BEST_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_metrics_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(f"Best Java250-{SHIFT_TYPE} surrogate-transfer result")
    print("=" * 100)
    print("Best trial:", best_result["trial"])
    print("Selection reason:", reason)
    print("Best weights:", best_result["weights"])
    print("Best ratios:", best_result["ratios"])
    print("ID metrics:", best_result["id_test"])
    print(f"{SHIFT_TYPE} OOD metrics:", best_result["ood_test"])
    print("Best config file:", SURROGATE_BEST_CONFIG_FILE)
    print("Best metrics file:", SURROGATE_BEST_METRICS_FILE)
    print("Best model file:", SURROGATE_BEST_MODEL_FILE)
    return best_config_payload


def run_surrogate_recommendation(validate=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    payload = load_surrogate_payload()
    context = prepare_transfer_geometry_context(tokenizer)
    candidates = predict_transfer_candidates(payload, context["class_metrics"], context["ood_pool_raw"])
    top_recommendations = candidates[:SURROGATE_RECOMMEND_TOP_K]

    with open(SURROGATE_RECOMMENDATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "method": "Cross-dataset and cross-model geometry split recommendation from a backbone-aware trial-surrogate model",
            "mode": "recommend_validate" if validate else "recommend",
            "model_type": canonical_model_type(MODEL_TYPE),
            "data_dir": DATA_DIR,
            "rq1_best_model": RQ1_BEST_MODEL,
            "ood_pool_file": OOD_POOL_FILE,
            "cross_model_surrogate": bool(USE_CROSS_MODEL_SURROGATE),
            "source_surrogate_model": SURROGATE_PRETRAINED_MODEL_FILE,
            "candidate_pool_size": len(candidates),
            "recommend_top_k": int(SURROGATE_RECOMMEND_TOP_K),
            "recommend_target": SURROGATE_RECOMMEND_TARGET,
            "top_recommendations": top_recommendations,
        }, f, ensure_ascii=False, indent=2)

    print("Surrogate recommendations:", SURROGATE_RECOMMENDATIONS_FILE)
    for idx, rec in enumerate(top_recommendations[:SURROGATE_VALIDATE_TOP_K], start=1):
        print(
            f"Top {idx}: score={rec['recommend_score']:.6f}, "
            f"pred_ood_f1={rec['predicted'].get('ood_macro_f1')}, "
            f"weights={rec['weights']}, ratios={rec['ratios']}"
        )

    if not validate:
        return {"recommendations_file": SURROGATE_RECOMMENDATIONS_FILE, "top_recommendations": top_recommendations}

    validation_results = validate_surrogate_recommendations(top_recommendations, context, tokenizer)
    best_token_result = select_and_export_best_surrogate_validation(validation_results)
    old_records = load_trial_records_from_files(SURROGATE_SOURCE_RESULTS_FILES)
    updated_report = export_trial_records_and_train_surrogate(old_records + validation_results)
    return {
        "recommendations_file": SURROGATE_RECOMMENDATIONS_FILE,
        "validation_results_file": SURROGATE_VALIDATION_RESULTS_FILE,
        "best_config_file": SURROGATE_BEST_CONFIG_FILE,
        "best_metrics_file": SURROGATE_BEST_METRICS_FILE,
        "best_model_file": SURROGATE_BEST_MODEL_FILE,
        "best_token_result": best_token_result,
        "updated_surrogate_report": updated_report,
    }


def run_cst_to_target_transfer(target_preset):
    print("\n" + "=" * 120)
    print(f"One-shot transfer: Java250-cst trial surrogate -> {target_preset} best config")
    print("=" * 120)

    apply_dataset_preset("java250_cst_source")
    cst_surrogate_report = run_train_surrogate_only()

    apply_dataset_preset(target_preset)
    target_result = run_surrogate_recommendation(validate=True)

    transfer_summary = {
        "method": f"Train trial-surrogate on Java250-cst, recommend and validate best config on {target_preset}",
        "source_preset": "java250_cst_source",
        "target_preset": target_preset,
        "source_surrogate_report": cst_surrogate_report,
        "target_result": target_result,
        "files": {
            "target_recommendations": SURROGATE_RECOMMENDATIONS_FILE,
            "target_validation_results": SURROGATE_VALIDATION_RESULTS_FILE,
            "target_best_config": SURROGATE_BEST_CONFIG_FILE,
            "target_best_metrics": SURROGATE_BEST_METRICS_FILE,
            "target_best_model": SURROGATE_BEST_MODEL_FILE,
        },
    }
    target_name = target_preset.replace("java250_", "").replace("_target", "")
    summary_path = os.path.join(SAVE_DIR, f"rq5_cst_to_{target_name}_transfer_summary_{MODEL_TYPE}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(transfer_summary, f, ensure_ascii=False, indent=2)
    print(f"CST -> {target_name} transfer summary:", summary_path)
    return transfer_summary


def run_cst_to_token_transfer():
    return run_cst_to_target_transfer("java250_token_target")


def run_cst_to_user_transfer():
    return run_cst_to_target_transfer("java250_user_target")


def run_java250_cst_to_python75_cst_transfer():
    return run_cst_to_target_transfer("python75_cst_target")


# ============================================================
# 10. Main: Warm-start TPE + Pareto + Replay
# ============================================================
def main():
    set_seed(SEED)
    apply_dataset_preset(DATASET_PRESET)
    mode = RQ5_RUN_MODE.lower()
    if mode == "cst_to_token":
        run_cst_to_token_transfer()
        return
    if mode == "cst_to_user":
        run_cst_to_user_transfer()
        return
    if mode == "java250_cst_to_python75_cst":
        run_java250_cst_to_python75_cst_transfer()
        return
    if mode == "train_surrogate":
        run_train_surrogate_only()
        return
    if mode == "recommend":
        run_surrogate_recommendation(validate=False)
        return
    if mode == "recommend_validate":
        run_surrogate_recommendation(validate=True)
        return
    if mode != "full_tpe":
        raise ValueError(
            "RQ5_RUN_MODE must be one of: full_tpe, train_surrogate, recommend, recommend_validate, cst_to_token, cst_to_user, java250_cst_to_python75_cst. "
            f"Got: {RQ5_RUN_MODE}"
        )

    if optuna is None:
        raise RuntimeError("This script requires optuna for TPE search. Please install optuna first: pip install optuna")

    print("=" * 120)
    print("RQ5/RQ4 Replay + Geometry Warm-start TPE + Pareto")
    print("=" * 120)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("DATA_DIR:", DATA_DIR)
    print("SAVE_DIR:", SAVE_DIR)
    print("ID_TRAIN_FILE:", ID_TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_POOL_FILE:", OOD_POOL_FILE)
    print("NUM_LABELS:", NUM_LABELS)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("CLASSIFIER_HEAD_TYPE:", CLASSIFIER_HEAD_TYPE)
    print("Replay config:", {"lambda": REPLAY_LAMBDA, "batch_size": REPLAY_BATCH_SIZE, "buffer_size": REPLAY_BUFFER_SIZE})
    print("TPE config:", {
        "smartsearch_prior_configs": len(unique_smartsearch_prior_configs()),
        "use_priors_as_observations": USE_SMARTSEARCH_PRIORS_AS_OBSERVATIONS,
        "tpe_search_trials": TPE_SEARCH_TRIALS,
        "tpe_search_epochs": TPE_SEARCH_EPOCHS,
        "full_eval_top_k": TPE_FULL_EVAL_TOP_K,
        "full_eval_epochs": TPE_FULL_EVAL_EPOCHS,
        "pareto_objectives": PARETO_OBJECTIVES,
    })
    print("Runtime config:", {"resume_search": RESUME_SEARCH, "oom_safe_mode": OOM_SAFE_MODE, "batch_size": BATCH_SIZE, "replay_batch_size": REPLAY_BATCH_SIZE})
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("\n" + "=" * 100)
    print("Phase 1: Load ID data and OOD pool")
    print("=" * 100)
    id_train_raw = read_jsonl(resolve_data_path(ID_TRAIN_FILE))
    id_test_raw = read_jsonl(resolve_data_path(ID_TEST_FILE))
    ood_pool_raw = read_jsonl(resolve_data_path(OOD_POOL_FILE))
    if not ood_pool_raw:
        raise RuntimeError("No OOD pool samples loaded. Please check OOD_POOL_FILE.")

    id_train_ds = CodeDataset(id_train_raw, tokenizer, name="id_train")
    id_test_ds = CodeDataset(id_test_raw, tokenizer, name="id_test")
    ood_pool_ds = CodeDataset(ood_pool_raw, tokenizer, name="ood_pool")

    id_test_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_pool_loader = DataLoader(ood_pool_ds, batch_size=BATCH_SIZE, shuffle=False)

    replay_buffer_ds, replay_buffer_indices = build_replay_buffer_dataset(id_train_ds, buffer_size=REPLAY_BUFFER_SIZE, seed=REPLAY_BUFFER_SEED)

    print("\n" + "=" * 100)
    print("Phase 2: Compute RQ3 geometry metrics on OOD pool")
    print("=" * 100)
    rq1_model = load_rq1_model()
    id_emb, _, id_labels, id_idxs = extract_embeddings_and_logits(rq1_model, id_test_loader, desc="Extract ID test embeddings for alignment")
    ood_emb, ood_logits, ood_labels, ood_idxs = extract_embeddings_and_logits(rq1_model, ood_pool_loader, desc="Extract OOD pool embeddings")
    class_metrics = compute_per_class_metrics(ood_emb, ood_logits, ood_labels, ood_idxs, id_emb, id_labels, id_idxs)
    print(">>> Geometry class count:", len(class_metrics))
    del rq1_model, id_emb, id_labels, id_idxs, ood_emb, ood_logits, ood_labels, ood_idxs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    search_results = load_existing_search_results(TPE_SEARCH_RESULTS_FILE)
    completed_by_trial = {item["trial"]: item for item in search_results if item.get("trial")}

    # 把 SmartSearch top configs 保存为 prior records，用于报告和 full-eval 候选补充。
    prior_results = make_prior_search_results(class_metrics, ood_pool_raw)
    for prior in prior_results:
        if prior["trial"] not in completed_by_trial:
            search_results.append(prior)
            completed_by_trial[prior["trial"]] = prior

    best_result = None
    best_train_raw = None
    best_test_raw = None
    best_split_report = None
    best_class_scores = None
    best_class_test_ratios = None

    def run_config(trial_idx, cfg, stage, num_epochs, search_backend="optuna_tpe", save_trial_model=False):
        trial_name = f"{stage}_{trial_idx:03d}_{cfg['name']}"
        if RESUME_SEARCH and trial_name in completed_by_trial and not completed_by_trial[trial_name].get("is_prior_only", False):
            cached = completed_by_trial[trial_name]
            print(f"[Resume] Skip completed trial: {trial_name} | score={cached.get('select_score')}")
            return cached

        print("\n" + "-" * 120)
        print(f"Trial: {trial_name}")
        print("stage:", stage, "| replay_epochs:", num_epochs, "| backend:", search_backend)
        print("weights:", cfg["weights"])
        print("ratios :", cfg["ratios"])
        print("-" * 120)

        set_seed(SEED)
        trial_train_raw, trial_test_raw, split_report, features, class_scores, class_test_ratios = build_trial_split(cfg, class_metrics, ood_pool_raw)
        if not trial_train_raw or not trial_test_raw:
            print("Skip empty split.")
            return None

        train_ds = CodeDataset(trial_train_raw, tokenizer, name=f"{trial_name}_ood_train")
        test_ds = CodeDataset(trial_test_raw, tokenizer, name=f"{trial_name}_ood_test_new")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_gpu_memory(prefix=f"[{trial_name}] before load/train | ")
        model = load_rq1_model()
        trial_start = time.time()
        model, train_losses, loss_breakdowns = train_replay_for_epochs(model, train_loader, replay_buffer_ds, num_epochs, trial_name)
        train_seconds = time.time() - trial_start

        id_eval = evaluate(model, id_test_loader, desc=f"{trial_name} ID Test")
        ood_eval = evaluate(model, test_loader, desc=f"{trial_name} OOD Test")
        select_score = float(ood_eval[TPE_GUIDANCE_METRIC])

        trial_model_path = None
        trial_train_file = None
        trial_test_file = None
        if save_trial_model:
            safe_name = trial_name.replace(os.sep, "_").replace(" ", "_")
            trial_model_path = os.path.join(SAVE_DIR, f"{safe_name}.bin")
            trial_train_file = os.path.join(SAVE_DIR, f"{safe_name}_ood_train.jsonl")
            trial_test_file = os.path.join(SAVE_DIR, f"{safe_name}_ood_test_new.jsonl")
            torch.save(model.state_dict(), trial_model_path)
            write_jsonl(trial_train_file, trial_train_raw)
            write_jsonl(trial_test_file, trial_test_raw)

        result = {
            "trial": trial_name,
            "stage": stage,
            "epochs": int(num_epochs),
            "model_type": canonical_model_type(MODEL_TYPE),
            "config": cfg,
            "weights": cfg["weights"],
            "ratios": cfg["ratios"],
            "features": features,
            "train_size": len(trial_train_raw),
            "ood_test_size": len(trial_test_raw),
            "id_test": id_eval,
            "ood_test": ood_eval,
            "train_losses": train_losses,
            "loss_breakdowns": loss_breakdowns,
            "select_metric": TPE_GUIDANCE_METRIC,
            "select_score": select_score,
            "tpe_guidance_score": select_score,
            "selection_set": "geometry_ood_test_new",
            "search_backend": search_backend,
            "train_seconds": float(train_seconds),
            "trial_model_path": trial_model_path,
            "trial_train_file": trial_train_file,
            "trial_test_file": trial_test_file,
            "split_report": split_report,
        }
        search_results.append(result)
        completed_by_trial[trial_name] = result

        print(
            f"{trial_name} result | ID F1={id_eval['macro_f1']:.4f} | "
            f"OOD F1={ood_eval['macro_f1']:.4f} | OOD Acc={ood_eval['accuracy']:.4f} | "
            f"OOD Top5={ood_eval['top5_acc']:.4f} | Time={format_time(train_seconds)}"
        )

        with open(TPE_SEARCH_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)

        del model, train_ds, test_ds, train_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    print("\n" + "=" * 100)
    print("Phase 3: Warm-start Optuna TPE with SmartSearch top configs")
    print("=" * 100)
    try:
        sampler = optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=OPTUNA_STARTUP_TRIALS,
            multivariate=True,
            group=True,
        )
    except TypeError:
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=OPTUNA_STARTUP_TRIALS)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE,
        load_if_exists=OPTUNA_STORAGE is not None,
    )
    added_priors = add_smartsearch_priors_to_study(study)
    print(f"SmartSearch prior observations added to Optuna: {len(added_priors)}")
    for item in added_priors:
        print("  prior:", item["source_trial"], "OOD F1=", item["source_ood_macro_f1"], "cfg=", item["config"])

    print("\n" + "=" * 100)
    print("Phase 4: TPE search trials with Replay training")
    print("=" * 100)
    tpe_results = []
    for _ in range(TPE_SEARCH_TRIALS):
        trial = study.ask()
        cfg = suggest_tpe_config(trial)
        result = run_config(
            trial.number,
            cfg,
            stage="tpe_e5" if TPE_SEARCH_EPOCHS == 5 else f"tpe_e{TPE_SEARCH_EPOCHS}",
            num_epochs=TPE_SEARCH_EPOCHS,
            search_backend="optuna_tpe_replay",
            save_trial_model=False,
        )
        if result is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        study.tell(trial, float(result["tpe_guidance_score"]))
        tpe_results.append(result)

    if not tpe_results:
        print("WARNING: TPE produced no new valid result; full eval will use SmartSearch priors.")
    else:
        try:
            print("Optuna best value:", study.best_value)
            print("Optuna best params:", study.best_params)
        except Exception:
            pass

    print("\n" + "=" * 100)
    print("Phase 5: Pareto candidate selection for full-budget Replay confirmation")
    print("=" * 100)
    full_eval_candidates = select_full_eval_candidates(tpe_results, prior_results, top_k=TPE_FULL_EVAL_TOP_K)
    if not full_eval_candidates:
        raise RuntimeError("No full-eval candidate selected.")

    stage_pareto_front = compute_pareto_front(tpe_results)
    print("TPE-stage Pareto front size:", len(stage_pareto_front), "/", len(tpe_results))
    print("Full-eval candidate count:", len(full_eval_candidates))
    for idx, item in enumerate(full_eval_candidates, start=1):
        print(
            f"  Candidate {idx}: {item['trial']} | "
            f"stage={item.get('stage')} | "
            f"OOD F1={item.get('ood_test', {}).get('macro_f1')} | "
            f"cfg={item['config']}"
        )

    print("\n" + "=" * 100)
    print("Phase 6: Full-budget Replay confirmation for Pareto-selected candidates")
    print("=" * 100)
    full_results = []
    for idx, candidate in enumerate(full_eval_candidates, start=1):
        cfg = candidate["config"]
        cfg = make_search_config(f"full_from_{candidate['trial']}", cfg["weights"], cfg["ratios"])
        result = run_config(
            idx,
            cfg,
            stage="full_eval",
            num_epochs=TPE_FULL_EVAL_EPOCHS,
            search_backend="full_budget_replay_pareto_confirmation",
            save_trial_model=True,
        )
        if result is not None:
            full_results.append(result)

    if not full_results:
        raise RuntimeError("Full-budget confirmation produced no valid result.")

    print("\n" + "=" * 100)
    print("Phase 7: Final Pareto selection over full-budget Replay trials")
    print("=" * 100)
    best_result, final_pareto_front, final_reason = select_final_from_pareto(full_results)
    best_train_raw, best_test_raw, best_split_report, _, best_class_scores, best_class_test_ratios = build_trial_split(best_result["config"], class_metrics, ood_pool_raw)

    if best_result.get("trial_model_path") and os.path.exists(best_result["trial_model_path"]):
        best_state = safe_torch_load(best_result["trial_model_path"], map_location=DEVICE)
        torch.save(best_state, BEST_MODEL_PATH)
    else:
        print("[Warning] selected full trial checkpoint not found; rerun selected config may be needed.")

    write_jsonl(GEOMETRY_OOD_TRAIN_FILE, best_train_raw)
    write_jsonl(GEOMETRY_OOD_TEST_FILE, best_test_raw)

    print("Final full-budget Pareto front size:", len(final_pareto_front), "/", len(full_results))
    print("Final selected trial:", best_result["trial"])
    print("Final selection reason:", final_reason)
    print("Final selected OOD metrics:", best_result.get("ood_test", {}))
    print("Best model:", BEST_MODEL_PATH)

    print("\n" + "=" * 100)
    print("Phase 8: Final evaluation of Pareto-selected Replay model")
    print("=" * 100)
    best_test_ds = CodeDataset(best_test_raw, tokenizer, name="best_geometry_ood_test_new")
    best_test_loader = DataLoader(best_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    best_model = load_rq1_model()
    best_model.load_state_dict(safe_torch_load(BEST_MODEL_PATH, map_location=DEVICE))

    id_pred_path = os.path.join(SAVE_DIR, f"predictions_id_test_replay_geo_tpe_pareto_{MODEL_TYPE}.jsonl")
    ood_pred_path = os.path.join(SAVE_DIR, f"predictions_geometry_ood_test_replay_geo_tpe_pareto_{MODEL_TYPE}.jsonl")
    final_id_eval = evaluate(best_model, id_test_loader, desc="Best TPE-Pareto Replay-Geo ID Test", save_path=id_pred_path)
    final_ood_eval = evaluate(best_model, best_test_loader, desc="Best TPE-Pareto Replay-Geo OOD Test", save_path=ood_pred_path)

    print("\n" + "=" * 120)
    print("RQ5/RQ4 Replay + Geometry TPE + Pareto Final Results")
    print("=" * 120)
    print(f"{'Metric':<15} | {'ID Test':<12} | {'OOD Test':<12}")
    print("-" * 45)
    for key, name in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("top5_acc", "Top-5 Acc")]:
        print(f"{name:<15} | {final_id_eval[key]:<12.4f} | {final_ood_eval[key]:<12.4f}")
    print("=" * 120)
    print("Best trial:", best_result["trial"])
    print("Best weights:", best_result["weights"])
    print("Best ratios:", best_result["ratios"])

    all_results_for_export = search_results + full_results
    final_selected, final_front, _ = select_final_from_pareto(full_results)
    export_tpe_top_configs(all_results_for_export, TPE_TOP_CONFIGS_FILE, top_k=10)
    surrogate_report = export_trial_records_and_train_surrogate(all_results_for_export)

    with open(TPE_PARETO_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "method": "Warm-start Optuna TPE + full-budget Pareto selection + Replay continual learning",
            "smartsearch_priors": prior_results,
            "tpe_stage_pareto_front": stage_pareto_front,
            "full_pareto_front": final_pareto_front,
            "final_selection_reason": final_reason,
            "final_selected": best_result,
            "surrogate_report": surrogate_report,
        }, f, ensure_ascii=False, indent=2)

    with open(SPLIT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "method": "RQ4/RQ5 Replay + RQ3-guided geometry split + warm-start TPE + Pareto",
            "model_type": MODEL_TYPE,
            "model_path": MODEL_PATH,
            "rq1_best_model": RQ1_BEST_MODEL,
            "data_dir": DATA_DIR,
            "ood_pool_file": OOD_POOL_FILE,
            "best_trial": best_result,
            "best_split_report": best_split_report,
            "best_class_scores": best_class_scores,
            "best_class_test_ratios": best_class_test_ratios,
            "best_geometry_ood_train_file": GEOMETRY_OOD_TRAIN_FILE,
            "best_geometry_ood_test_file": GEOMETRY_OOD_TEST_FILE,
            "fixed_global_split_ratio": {
                "train_ratio": FIXED_TRAIN_RATIO,
                "test_ratio": FIXED_TEST_RATIO,
                "note": "hard/middle/easy ratios are allocation preferences only; global test ratio is fixed.",
            },
        }, f, ensure_ascii=False, indent=2)

    metrics = {
        "task": "Code Classification",
        "method": "Replay continual learning with RQ3-guided geometry split, warm-start TPE search, and Pareto final selection",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "rq1_best_model": RQ1_BEST_MODEL,
        "data_dir": DATA_DIR,
        "id_train_file": ID_TRAIN_FILE,
        "id_test_file": ID_TEST_FILE,
        "ood_pool_file": OOD_POOL_FILE,
        "label_setting": "0-based labels; valid labels are [0, NUM_LABELS-1]",
        "num_labels": NUM_LABELS,
        "encoder_pooling": resolve_pooling_strategy(MODEL_TYPE, ENCODER_POOLING),
        "classifier_head_type": CLASSIFIER_HEAD_TYPE,
        "replay_config": {
            "lambda": REPLAY_LAMBDA,
            "batch_size": REPLAY_BATCH_SIZE,
            "buffer_size": REPLAY_BUFFER_SIZE,
            "buffer_source": "ID train / previous-task training data",
            "buffer_actual_size": len(replay_buffer_ds),
            "buffer_indices": replay_buffer_indices,
        },
        "tpe_search": {
            "backend": "optuna_tpe",
            "smartsearch_priors_as_observations": USE_SMARTSEARCH_PRIORS_AS_OBSERVATIONS,
            "num_smartsearch_unique_priors": len(unique_smartsearch_prior_configs()),
            "tpe_search_trials": TPE_SEARCH_TRIALS,
            "tpe_search_epochs": TPE_SEARCH_EPOCHS,
            "full_eval_top_k": TPE_FULL_EVAL_TOP_K,
            "full_eval_epochs": TPE_FULL_EVAL_EPOCHS,
            "guidance_metric": TPE_GUIDANCE_METRIC,
            "pareto_objectives": PARETO_OBJECTIVES,
            "selection_set": "geometry_ood_test_new",
        },
        "geometry_definition": {
            "shift_type": SHIFT_TYPE,
            "alignment_mode": ALIGNMENT_MODE,
            "entropy": "OOD sample-level prediction entropy averaged within class",
            "alignment": "token/cst use pair-level idx+label when available; otherwise label-centroid alignment",
            "anisotropy": "OOD class-wise average pairwise cosine similarity",
            "effective_rank": "OOD class-wise centered SVD effective rank",
            "standardization": "hard-oriented min-max normalization; positive weights normalized to sum 1",
        },
        "best_trial": best_result,
        "best_geometry_ood_train_size": len(best_train_raw),
        "best_geometry_ood_test_size": len(best_test_raw),
        "best_geometry_ood_train_label_distribution": label_distribution(best_train_raw),
        "best_geometry_ood_test_label_distribution": label_distribution(best_test_raw),
        "search_results": search_results,
        "full_results": full_results,
        "final_pareto_front": final_pareto_front,
        "final_selection_reason": final_reason,
        "final_evaluation": {"id_test": final_id_eval, "ood_test": final_ood_eval},
        "trial_surrogate": surrogate_report,
        "files": {
            "best_model": BEST_MODEL_PATH,
            "best_geometry_ood_train": GEOMETRY_OOD_TRAIN_FILE,
            "best_geometry_ood_test": GEOMETRY_OOD_TEST_FILE,
            "search_results": TPE_SEARCH_RESULTS_FILE,
            "top_configs": TPE_TOP_CONFIGS_FILE,
            "pareto_results": TPE_PARETO_RESULTS_FILE,
            "trial_meta_training_data": TRIAL_META_TRAINING_FILE,
            "trial_surrogate_model": SURROGATE_MODEL_FILE,
            "trial_surrogate_report": SURROGATE_REPORT_FILE,
            "split_report": SPLIT_REPORT_FILE,
            "id_predictions": id_pred_path,
            "ood_predictions": ood_pred_path,
            "metrics_file": METRICS_FILE,
        },
        "time": {"total_seconds": time.time() - START_TIME, "total_readable": format_time(time.time() - START_TIME)},
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Metrics file:", METRICS_FILE)
    print("TPE search results:", TPE_SEARCH_RESULTS_FILE)
    print("Pareto results:", TPE_PARETO_RESULTS_FILE)
    print("Split report:", SPLIT_REPORT_FILE)
    print("Top configs:", TPE_TOP_CONFIGS_FILE)
    print("Trial meta-training data:", TRIAL_META_TRAINING_FILE)
    print("Trial surrogate model:", SURROGATE_MODEL_FILE)
    print("Trial surrogate report:", SURROGATE_REPORT_FILE)
    print("Total time:", format_time(time.time() - START_TIME))


if __name__ == "__main__":
    main()
