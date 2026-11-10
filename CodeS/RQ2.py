import os
import json
import csv
import time
import sys
import io
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# ============================================================
# 0. Environment
# ============================================================

if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

START_TIME = time.time()


# ============================================================
# 1. Config: CodeS / Code Classification
# ============================================================

TASK_NAME = "Code Classification"
DATASET_NAME = "CodeS"

# 只需要保证 MODEL_TYPE、MODEL_PATH、DATA_DIR、MODEL_WEIGHTS 是对应的。
# 分类头会从 checkpoint 的 key 自动判断；encoder pooling 会按模型类型自动选择。
SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

# Set RUN_ALL_MODELS=True to evaluate all five models sequentially.
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_TYPE = "RoBERTa"

DATA_DIR = "/opt/data/private/codes/Java250/Java250/user/pre-trained"
SAVE_DIR = "/opt/data/private/codes/Java250/Java250/user/RQ2_results"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_WEIGHTS_DIR = os.path.dirname(DATA_DIR)
MODEL_WEIGHTS_DATASET_NAME = "Java250"

ID_TRAIN_FILE = "train.jsonl"
ID_TEST_FILE = "id_test.jsonl"
OOD_TEST_FILES = {
    "ood": "ood_test.jsonl",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change this according to the dataset: Python75=75, Java250-S=250, Python800-S=800, etc.
NUM_LABELS = 250

MAX_LEN = 256
BATCH_SIZE = 16
SEED = 42
LABEL_OFFSET = 0

# auto 会根据 checkpoint 自动判断分类头结构。
# 支持: single / mlp_seq / mlp_named
CLASSIFIER_HEAD_TYPE = "auto"
ENCODER_POOLING = "auto"

# auto: CodeT5 使用 mean pooling；CodeBERT / GraphCodeBERT 使用 first-token/CLS pooling。
# 也可手动设为 "cls" 或 "mean"，用于和 RQ1 训练代码强制对齐。
ENCODER_POOLING = "auto"

# OOD detector settings. We keep 9 detectors:
# MSP, Entropy, MaxLogit, Energy, Mahalanobis, Mahalanobis++, KNN, ViM, GradNorm.
# Logit-based detectors use logits; feature-based detectors use final task/classifier features.
KNN_K = 5
VIM_MAX_PCA_COMPONENTS = 512
VIM_PRINCIPAL_DIM = 128
VIM_ALPHA_EPS = 1e-12

# GradNorm is sample-wise and slow. Set False for a quick run.
ENABLE_GRADNORM = True

# GradNorm 方向不再硬编码在函数里。
# "neg": score = -||grad||_1，保持 larger score = more OOD-like 的旧设定；
# "pos": score =  ||grad||_1，用于你确认 raw norm 才是 OOD score 的情况。
# 注意：不要根据 OOD test 结果临时切换方向，否则会引入测试集调参。
GRADNORM_SCORE_MODE = "neg"  # optional: "neg", "pos"

ALL_MODELS_SUMMARY_JSON = os.path.join(SAVE_DIR, "rq2_ood_summary_all_models.json")
ALL_MODELS_SUMMARY_CSV = os.path.join(SAVE_DIR, "rq2_ood_summary_all_models.csv")
ALL_MODELS_PASTE_TABLE_TSV = os.path.join(SAVE_DIR, "rq2_ood_paste_table_all_models.tsv")


def canonical_model_type(model_type):
    for name in SUPPORTED_MODEL_TYPES:
        if str(model_type).lower() == name.lower():
            return name
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {SUPPORTED_MODEL_TYPES}")


def resolve_model_path(model_type):
    return MODEL_PATHS[canonical_model_type(model_type)]


def build_model_weights_path(model_type):
    model_type = canonical_model_type(model_type)
    return os.path.join(MODEL_WEIGHTS_DIR, f"best_model_{model_type}_{MODEL_WEIGHTS_DATASET_NAME}.bin")


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, MODEL_WEIGHTS
    global RESULT_SUMMARY_JSON, RESULT_SUMMARY_CSV, RESULT_PASTE_TABLE_TSV

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = resolve_model_path(MODEL_TYPE)
    MODEL_WEIGHTS = build_model_weights_path(MODEL_TYPE)
    RESULT_SUMMARY_JSON = os.path.join(SAVE_DIR, f"rq2_ood_summary_{MODEL_TYPE}.json")
    RESULT_SUMMARY_CSV = os.path.join(SAVE_DIR, f"rq2_ood_summary_{MODEL_TYPE}.csv")
    RESULT_PASTE_TABLE_TSV = os.path.join(SAVE_DIR, f"rq2_ood_paste_table_{MODEL_TYPE}.tsv")


configure_model_runtime(MODEL_TYPE)

PASTE_TABLE_METRICS = [
    ("AUROC", "auroc"),
    ("AUPR", "aupr"),
    ("TNR@95", "tnr95"),
]


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
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds"


def safe_torch_load(path, map_location="cpu"):
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
    return [strip_prefix(key) for key in state_dict.keys()]


def infer_head_type_from_state_dict(state_dict):
    """
    根据 checkpoint 中分类头参数的 key 自动判断分类头结构。

    single:
        classifier.weight / classifier.bias
        score.weight / score.bias
    mlp_seq:
        classifier.0.* 和 classifier.3.*
    mlp_named:
        classifier.dense.* 和 classifier.out_proj.*
        classification_head.dense.* 和 classification_head.out_proj.*
    """
    keys = normalized_keys(state_dict)

    # RoBERTa-style classification_head 统一看作 named MLP head。
    if any(key.startswith("classification_head.dense.") or key.startswith("classification_head.out_proj.") for key in keys):
        return "mlp_named"

    if any(key.startswith("classifier.0.") or key.startswith("classifier.3.") for key in keys):
        return "mlp_seq"
    if any(key.startswith("classifier.dense.") or key.startswith("classifier.out_proj.") for key in keys):
        return "mlp_named"
    if any(key in ("classifier.weight", "classifier.bias", "score.weight", "score.bias") for key in keys):
        return "single"

    # 兜底按 CodeS RQ1 的常用结构处理：encoder + 单层 classifier。
    # 如果 checkpoint 里没有任何可匹配的分类头参数，后续加载检查会直接报错。
    return "single"


def adapt_state_dict_keys(state_dict, head_type):
    new_dict = {}

    for key, value in state_dict.items():
        new_key = strip_prefix(key)

        # encoder 前缀适配
        if new_key.startswith("roberta."):
            new_key = new_key.replace("roberta.", "encoder.", 1)
        elif new_key.startswith("bert."):
            new_key = new_key.replace("bert.", "encoder.", 1)
        elif new_key.startswith("transformer."):
            new_key = new_key.replace("transformer.", "encoder.", 1)
        elif new_key.startswith("encoder.encoder."):
            # 防止某些封装重复 encoder 前缀
            new_key = new_key.replace("encoder.encoder.", "encoder.", 1)

        # 常见分类头命名适配
        if new_key.startswith("classification_head.dense."):
            new_key = new_key.replace("classification_head.dense.", "classifier.dense.", 1)
        elif new_key.startswith("classification_head.out_proj."):
            new_key = new_key.replace("classification_head.out_proj.", "classifier.out_proj.", 1)

        if new_key == "score.weight":
            new_key = "classifier.weight"
        elif new_key == "score.bias":
            new_key = "classifier.bias"

        # 根据自动识别出的 head_type，把 checkpoint key 映射到当前模型结构。
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
    """
    只加载 key 存在且 shape 一致的参数。
    这样可以把真正的 head mismatch 打印出来，而不是直接在 load_state_dict 处崩溃。
    """
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


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    model_type_lower = model_type.lower()

    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")

    # 与 RQ1 保持一致：
    # CodeT5 没有专门 CLS token，因此使用 non-padding mean pooling；
    # CodeBERT / GraphCodeBERT 使用 CLS/first-token pooling。
    return "mean" if "t5" in model_type_lower else "cls"


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = pooling.lower()
    model_type_lower = model_type.lower()

    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING: {pooling}")

    # Match CodeS RQ1: CodeT5/PLBART use mean pooling; CodeBERT/GraphCodeBERT use CLS.
    return "mean" if ("t5" in model_type_lower or "plbart" in model_type_lower) else "cls"


# ============================================================
# 3. Model aligned with CodeS RQ1
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

    def prepare_final_features(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x

    def forward(self, features):
        return self.out_proj(self.prepare_final_features(features))


class CodeModelForOOD(nn.Module):
    def __init__(self, model_path, model_type, num_labels, head_type="mlp_seq", pooling="auto"):
        super().__init__()
        self.model_type = model_type.lower()
        self.head_type = head_type
        self.pooling_strategy = resolve_pooling_strategy(model_type, pooling)

        if "t5" in self.model_type:
            self.encoder = T5EncoderModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            base_model = AutoModel.from_pretrained(model_path)
            if not hasattr(base_model, "get_encoder"):
                raise ValueError(
                    "PLBART model should expose get_encoder(); "
                    f"please check MODEL_PATH={model_path}"
                )
            self.encoder = base_model.get_encoder()
            hidden_size = getattr(base_model.config, "d_model", None)
            if hidden_size is None:
                hidden_size = getattr(base_model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot infer PLBART hidden size from model config.")
        else:
            self.encoder = AutoModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.hidden_size

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

    def pool_encoder_output(self, outputs, attention_mask):
        if self.pooling_strategy == "mean":
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / denom
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, return_feat=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # raw_features: encoder-side representation.
        #   - CodeBERT / GraphCodeBERT: CLS/first-token pooling
        #   - CodeT5: mean pooling by default
        # final_features: task/classifier representation immediately before the final classifier.
        raw_features = self.pool_encoder_output(outputs, attention_mask)
        final_features = self.prepare_classifier_features(raw_features)
        logits = self.classify_final_features(final_features)

        if return_feat:
            return logits, final_features
        return logits

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

    def get_final_classifier_weight(self):
        if self.head_type == "single":
            return self.classifier.weight
        if self.head_type == "mlp_named":
            return self.classifier.out_proj.weight
        return self.classifier[3].weight

    def get_final_classifier_bias(self):
        if self.head_type == "single":
            return self.classifier.bias
        if self.head_type == "mlp_named":
            return self.classifier.out_proj.bias
        return self.classifier[3].bias


# ============================================================
# 4. Dataset
# ============================================================

class CodeDataset(Dataset):
    def __init__(self, file_name, tokenizer, require_label=False, name=None):
        self.examples = []
        self.path = resolve_data_path(file_name)
        self.name = name or os.path.basename(self.path)

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Cannot find data file: {self.path}")

        bad_lines = 0
        skipped_label = 0
        raw_labels = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Loading {self.name}", leave=False)):
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

                if not code:
                    bad_lines += 1
                    continue

                if label is None:
                    if require_label:
                        bad_lines += 1
                        continue
                    fixed_label = -1
                    raw_label = -1
                else:
                    raw_label = int(label)
                    fixed_label = raw_label - LABEL_OFFSET
                    if fixed_label < 0 or fixed_label >= NUM_LABELS:
                        if require_label:
                            skipped_label += 1
                            continue
                        fixed_label = -1
                    else:
                        raw_labels.append(raw_label)

                inputs = tokenizer(
                    code,
                    max_length=MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                self.examples.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "label": torch.tensor(fixed_label, dtype=torch.long),
                    "label_int": fixed_label,
                    "raw_label": raw_label,
                    "idx": get_idx(js, line_idx),
                    "shift_type": js.get("shift_type", ""),
                })

        label_counter = Counter(x["label_int"] for x in self.examples if x["label_int"] >= 0)

        print("=" * 100)
        print(f"CodeDataset loaded: {self.name}")
        print("path:", self.path)
        print("num_samples:", len(self.examples))
        print("bad_lines:", bad_lines)
        print("skipped_label:", skipped_label)
        print("label_distribution:", dict(label_counter))
        print("=" * 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# 5. OOD metrics
# ============================================================

def get_metrics(id_scores, ood_scores):
    """
    Larger score means more OOD-like.

    AUROC and AUPR are computed with OOD as the positive class.
    TNR@95 is computed with ID as the positive class, so it means:
    when 95% of ID samples are accepted, what fraction of OOD samples
    can be correctly rejected.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_scores, dtype=np.float64).reshape(-1)

    id_scores = np.nan_to_num(id_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    ood_scores = np.nan_to_num(ood_scores, nan=0.0, posinf=1e6, neginf=-1e6)

    y_true_ood = np.concatenate([
        np.zeros(len(id_scores), dtype=np.int64),
        np.ones(len(ood_scores), dtype=np.int64),
    ])
    y_scores_ood = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true_ood, y_scores_ood)
    aupr = average_precision_score(y_true_ood, y_scores_ood)

    y_true_id = np.concatenate([
        np.ones(len(id_scores), dtype=np.int64),
        np.zeros(len(ood_scores), dtype=np.int64),
    ])
    y_scores_id = np.concatenate([
        -id_scores,
        -ood_scores,
    ])

    fpr, tpr, _ = roc_curve(y_true_id, y_scores_id, pos_label=1)
    valid = np.where(tpr >= 0.95)[0]
    fpr95 = 1.0 if len(valid) == 0 else fpr[valid[0]]
    tnr95 = 1.0 - fpr95

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr95": float(fpr95),
        "tnr95": float(tnr95),
        "id_score_mean": float(np.mean(id_scores)),
        "ood_score_mean": float(np.mean(ood_scores)),
        "id_score_std": float(np.std(id_scores)),
        "ood_score_std": float(np.std(ood_scores)),
        "num_id": int(len(id_scores)),
        "num_ood": int(len(ood_scores)),
    }


def format_metric_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return ""
    return f"{float(value):.4f}"


def build_paste_table_rows(results, detectors, model_name=MODEL_TYPE):
    rows = [[model_name] + detectors]

    for metric_name, metric_key in PASTE_TABLE_METRICS:
        row = [metric_name]
        for detector in detectors:
            metric_value = results.get(detector, {}).get(metric_key)
            row.append(format_metric_value(metric_value))
        rows.append(row)

    return rows


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def print_paste_table(rows, title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    for row in rows:
        print("\t".join(row))
    print("=" * 120)


def build_summary_paste_table_rows(all_results):
    rows = []

    for shift_name, detector_results in all_results.items():
        detectors = list(detector_results.keys())
        model_label = f"{MODEL_TYPE}-{shift_name}" if len(all_results) > 1 else MODEL_TYPE
        if rows:
            rows.append([])
        rows.extend(build_paste_table_rows(detector_results, detectors, model_name=model_label))

    return rows


def flatten_model_results(summary_payloads):
    rows = []
    for payload in summary_payloads:
        model_type = payload.get("model_type", "")
        for shift_name, detector_results in payload.get("results", {}).items():
            for detector, metrics in detector_results.items():
                row = {
                    "model_type": model_type,
                    "shift_name": shift_name,
                    "detector": detector,
                }
                row.update(metrics)
                rows.append(row)
    return rows


def write_all_models_csv(path, rows):
    fieldnames = [
        "model_type",
        "shift_name",
        "detector",
        "auroc",
        "aupr",
        "tnr95",
        "fpr95",
        "id_score_mean",
        "ood_score_mean",
        "id_score_std",
        "ood_score_std",
        "num_id",
        "num_ood",
        "score_mode",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_all_models_paste_rows(summary_payloads):
    rows = [["Model", "Shift", "Detector", "AUROC", "AUPR", "TNR@95"]]
    for payload in summary_payloads:
        model_type = payload.get("model_type", "")
        for shift_name, detector_results in payload.get("results", {}).items():
            for detector in detector_results:
                metrics = detector_results[detector]
                rows.append([
                    model_type,
                    shift_name,
                    detector,
                    format_metric_value(metrics.get("auroc")),
                    format_metric_value(metrics.get("aupr")),
                    format_metric_value(metrics.get("tnr95")),
                ])
    return rows


# ============================================================
# 6. OOD detection engine
# ============================================================

class OODMaster:
    def __init__(self, model, id_train_loader):
        self.model = model
        self.model.eval()

        id_feats = []
        id_logits = []
        id_labels = []

        print(">>> Extracting ID train task features for detector calibration...")

        with torch.no_grad():
            for batch in tqdm(id_train_loader, desc="Extract ID train task features", leave=False):
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)

                logits, feats = self.model(ids, mask, return_feat=True)

                id_feats.append(feats.detach().cpu().numpy())
                id_logits.append(logits.detach().cpu().numpy())
                id_labels.append(batch["label"].detach().cpu().numpy())

        if not id_feats:
            raise RuntimeError("ID train loader produced no samples. Cannot calibrate OOD detectors.")

        self.id_feats = np.concatenate(id_feats, axis=0)
        self.id_logits = np.concatenate(id_logits, axis=0)
        self.id_labels = np.concatenate(id_labels, axis=0).astype(np.int64)

        self.num_samples, self.feat_dim = self.id_feats.shape

        print("ID train feature shape:", self.id_feats.shape)
        print("ID train label distribution:", dict(Counter(self.id_labels.tolist())))

        self._prepare_mahalanobis()
        self._prepare_mahalanobis_pp()
        self._prepare_vim()
        self._prepare_knn()

        print(">>> OODMaster initialized.")

    def _prepare_mahalanobis(self):
        print(">>> Preparing Mahalanobis statistics...")

        self.global_mean = np.mean(self.id_feats, axis=0)
        class_means = []
        valid_class_mask = []

        for cls_id in range(NUM_LABELS):
            mask = self.id_labels == cls_id
            if np.sum(mask) == 0:
                class_means.append(self.global_mean)
                valid_class_mask.append(False)
            else:
                class_means.append(np.mean(self.id_feats[mask], axis=0))
                valid_class_mask.append(True)

        self.class_means = np.array(class_means)
        self.valid_class_mask = np.array(valid_class_mask)

        centered_feats = self.id_feats - self.class_means[self.id_labels]
        self.precision = self._fit_shrinkage_precision(centered_feats, name="Mahalanobis")

    def _prepare_mahalanobis_pp(self):
        print(">>> Preparing Mahalanobis++ normalized statistics...")

        norm_feats = self.id_feats / (np.linalg.norm(self.id_feats, axis=1, keepdims=True) + 1e-12)
        self.global_mean_norm = np.mean(norm_feats, axis=0)

        class_means_norm = []
        valid_class_mask_norm = []
        for cls_id in range(NUM_LABELS):
            mask = self.id_labels == cls_id
            if np.sum(mask) == 0:
                class_means_norm.append(self.global_mean_norm)
                valid_class_mask_norm.append(False)
            else:
                class_means_norm.append(np.mean(norm_feats[mask], axis=0))
                valid_class_mask_norm.append(True)

        self.class_means_norm = np.array(class_means_norm)
        self.valid_class_mask_norm = np.array(valid_class_mask_norm)
        centered_norm = norm_feats - self.class_means_norm[self.id_labels]

        self.precision_norm = self._fit_shrinkage_precision(centered_norm, name="Mahalanobis++")

    def _fit_shrinkage_precision(self, centered_features, name):
        try:
            cov = LedoitWolf().fit(centered_features)
            print(f">>> {name} covariance: LedoitWolf shrinkage={cov.shrinkage_:.6f}")
            return cov.precision_
        except Exception as exc:
            print(f"[Warning] {name} LedoitWolf failed; using ridge pseudo-inverse. error={repr(exc)}")
            cov_matrix = np.cov(centered_features, rowvar=False)
            cov_matrix = np.atleast_2d(cov_matrix)
            ridge = max(1e-6, float(np.trace(cov_matrix)) / max(1, cov_matrix.shape[0]) * 1e-6)
            cov_matrix += np.eye(cov_matrix.shape[0]) * ridge
            return np.linalg.pinv(cov_matrix)

    def _prepare_vim(self):
        print(">>> Preparing ViM space...")

        weight = self.model.get_final_classifier_weight().detach().cpu().numpy()
        bias_tensor = self.model.get_final_classifier_bias()
        if bias_tensor is None:
            self.vim_u = np.mean(self.id_feats, axis=0)
            print("[Warning] ViM classifier bias is None; using ID feature mean as origin.")
        else:
            bias = bias_tensor.detach().cpu().numpy()
            try:
                self.vim_u = -np.matmul(np.linalg.pinv(weight), bias)
            except Exception as exc:
                print(f"[Warning] ViM origin failed; using ID feature mean. error={repr(exc)}")
                self.vim_u = np.mean(self.id_feats, axis=0)

        n_components = min(
            VIM_MAX_PCA_COMPONENTS,
            self.id_feats.shape[0] - 1,
            self.id_feats.shape[1],
        )

        if n_components <= 1:
            self.NS = None
            print("[Warning] Too few ID train samples for a valid ViM PCA space.")
        else:
            pca = PCA(n_components=n_components)
            pca.fit(self.id_feats - self.vim_u)

            principal_dim = min(VIM_PRINCIPAL_DIM, n_components - 1)
            self.NS = pca.components_[principal_dim:]
            if self.NS.shape[0] == 0:
                self.NS = pca.components_[-1:]

            print(
                f"ViM PCA n_components={n_components}, "
                f"principal_dim={principal_dim}, "
                f"NS shape={self.NS.shape}"
            )

        if self.NS is None:
            self.vim_alpha = 1.0
        else:
            id_residual = np.linalg.norm(np.dot(self.id_feats - self.vim_u, self.NS.T), axis=1)
            id_energy = self._logsumexp_np(self.id_logits, axis=1)
            self.vim_alpha = float(
                np.mean(np.abs(id_energy)) / (np.mean(id_residual) + VIM_ALPHA_EPS)
            )
        print(f">>> ViM alpha: {self.vim_alpha:.6f}")

    def _logsumexp_np(self, values, axis=1):
        max_values = np.max(values, axis=axis, keepdims=True)
        return (np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True)) + max_values).squeeze(axis)

    def _prepare_knn(self):
        print(">>> Building KNN index...")
        k = min(KNN_K, max(1, self.id_feats.shape[0]))
        self.knn_k = k
        self.knn_index = NearestNeighbors(n_neighbors=k, metric="cosine").fit(self.id_feats)

    def msp(self, logits):
        return -torch.max(F.softmax(logits, dim=1), dim=1)[0].detach().cpu().numpy()

    def entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * torch.log(probs + 1e-12)).sum(dim=1).detach().cpu().numpy()

    def energy(self, logits):
        return -torch.logsumexp(logits, dim=1).detach().cpu().numpy()

    def maxlogit(self, logits):
        return -torch.max(logits, dim=1)[0].detach().cpu().numpy()

    def mahalanobis(self, feats):
        features = feats.detach().cpu().numpy()
        return self._mahalanobis_score(features, self.class_means, self.precision, self.valid_class_mask)

    def mahalanobis_pp(self, feats):
        features = feats.detach().cpu().numpy()
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        return self._mahalanobis_score(features, self.class_means_norm, self.precision_norm, self.valid_class_mask_norm)

    def _mahalanobis_score(self, features, class_means, precision, valid_class_mask):
        scores = []
        for item in features:
            diff = item - class_means
            distances = np.matmul(np.matmul(diff, precision), diff.T).diagonal()
            if np.any(valid_class_mask):
                distances = distances[valid_class_mask]
            scores.append(np.min(distances))
        return np.array(scores)

    def knn(self, feats):
        features = feats.detach().cpu().numpy()
        dists, _ = self.knn_index.kneighbors(features)
        return dists[:, -1]

    def vim(self, logits, feats):
        features = feats.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()

        if self.NS is None:
            return -self._logsumexp_np(logits_np, axis=1)

        residual = np.linalg.norm(np.dot(features - self.vim_u, self.NS.T), axis=1)
        virtual_logit = self.vim_alpha * residual
        energy = self._logsumexp_np(logits_np, axis=1)
        return virtual_logit - energy

    def gradnorm(self, ids, mask):
        self.model.eval()
        mode = GRADNORM_SCORE_MODE.lower()
        if mode not in ("neg", "pos"):
            raise ValueError(f"GRADNORM_SCORE_MODE must be 'neg' or 'pos', got: {GRADNORM_SCORE_MODE}")

        scores = []
        for idx in range(ids.size(0)):
            self.model.zero_grad(set_to_none=True)

            one_id = ids[idx:idx + 1]
            one_mask = mask[idx:idx + 1]

            logits = self.model(one_id, one_mask)
            log_probs = F.log_softmax(logits, dim=1)
            objective = -log_probs.mean(dim=1).sum()
            weights = self.model.get_final_classifier_weight()

            grad = torch.autograd.grad(
                objective,
                weights,
                retain_graph=False,
                create_graph=False,
            )[0]

            norm = torch.norm(grad, p=1).detach().cpu().item()
            score = -norm if mode == "neg" else norm
            scores.append(score)

        return np.array(scores)


# ============================================================
# 7. Evaluation
# ============================================================

def get_detector_list():
    detectors = [
        "MSP",
        "Entropy",
        "MaxLogit",
        "Energy",
        "Mahalanobis",
        "Mahalanobis++",
        "KNN",
        "ViM",
    ]
    if ENABLE_GRADNORM:
        detectors.append("GradNorm")
    if len(detectors) != 9:
        raise RuntimeError(
            f"RQ2 requires 9 OOD detectors, but got {len(detectors)}: {detectors}. "
            "Set ENABLE_GRADNORM=True for the full detector suite."
        )
    return detectors


def evaluate_ood_split(model, master, id_test_loader, ood_test_loader, shift_name):
    detectors = get_detector_list()
    scores = {name: {"id": [], "ood": []} for name in detectors}

    for loader, dtype in [(id_test_loader, "id"), (ood_test_loader, "ood")]:
        print(f"\n>>> Computing {dtype.upper()} scores: {shift_name}")

        for batch in tqdm(loader, desc=f"{shift_name}-{dtype}"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)

            with torch.no_grad():
                logits, feats = model(ids, mask, return_feat=True)

                scores["MSP"][dtype].append(master.msp(logits))
                scores["Entropy"][dtype].append(master.entropy(logits))
                scores["MaxLogit"][dtype].append(master.maxlogit(logits))
                scores["Energy"][dtype].append(master.energy(logits))
                scores["Mahalanobis"][dtype].append(master.mahalanobis(feats))
                scores["Mahalanobis++"][dtype].append(master.mahalanobis_pp(feats))
                scores["KNN"][dtype].append(master.knn(feats))
                scores["ViM"][dtype].append(master.vim(logits, feats))

            if ENABLE_GRADNORM:
                grad_scores = master.gradnorm(ids, mask)
                scores["GradNorm"][dtype].append(grad_scores)

    print("\n" + "=" * 105)
    print(f"RQ2 OOD Detection Results: {shift_name}")
    print("=" * 105)
    print(
        f"{'OOD Detector':<17} | "
        f"{'AUROC':<10} | "
        f"{'AUPR':<10} | "
        f"{'TNR@95':<10} | "
        f"{'FPR@95':<10} | "
        f"{'ID Mean':<10} | "
        f"{'OOD Mean':<10}"
    )
    print("-" * 105)

    results = {}

    for name in detectors:
        id_arr = np.concatenate(scores[name]["id"]).flatten()
        ood_arr = np.concatenate(scores[name]["ood"]).flatten()
        metrics = get_metrics(id_arr, ood_arr)
        if name == "GradNorm":
            metrics["score_mode"] = GRADNORM_SCORE_MODE
        results[name] = metrics

        print(
            f"{name:<17} | "
            f"{metrics['auroc']:<10.4f} | "
            f"{metrics['aupr']:<10.4f} | "
            f"{metrics['tnr95']:<10.4f} | "
            f"{metrics['fpr95']:<10.4f} | "
            f"{metrics['id_score_mean']:<10.4f} | "
            f"{metrics['ood_score_mean']:<10.4f}"
        )

        if name == "GradNorm" and metrics["auroc"] < 0.5:
            print(
                f"[Warning] GradNorm AUROC={metrics['auroc']:.4f} < 0.5. "
                f"The score direction may be reversed under GRADNORM_SCORE_MODE='{GRADNORM_SCORE_MODE}'. "
                "For a pre-registered rerun, consider switching GRADNORM_SCORE_MODE between 'neg' and 'pos'."
            )

    print("=" * 105)

    split_json = os.path.join(SAVE_DIR, f"rq2_ood_{shift_name}_{MODEL_TYPE}.json")
    split_csv = os.path.join(SAVE_DIR, f"rq2_ood_{shift_name}_{MODEL_TYPE}.csv")
    split_paste_tsv = os.path.join(SAVE_DIR, f"rq2_ood_{shift_name}_{MODEL_TYPE}_paste_table.tsv")

    metadata = {
        "task": TASK_NAME,
        "dataset": DATASET_NAME,
        "shift_name": shift_name,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "model_weights": MODEL_WEIGHTS,
        "data_dir": DATA_DIR,
        "id_train_file": ID_TRAIN_FILE,
        "id_test_file": ID_TEST_FILE,
        "ood_test_file": OOD_TEST_FILES[shift_name],
        "num_labels": NUM_LABELS,
        "detectors": detectors,
        "settings": {
            "knn_k": master.knn_k,
            "vim_principal_dim": VIM_PRINCIPAL_DIM,
            "encoder_pooling": model.pooling_strategy,
            "classifier_head_type": model.head_type,
            "gradnorm_score_mode": GRADNORM_SCORE_MODE if ENABLE_GRADNORM else None,
            "score_convention": "larger scores indicate more OOD-like samples",
            "feature_choice": "feature-based detectors use final task/classifier features immediately before the final classifier",
            "auroc_aupr_positive_class": "OOD",
            "tnr95_definition": "ID is treated as the positive class; TNR@95 is the OOD rejection rate when 95% of ID samples are accepted",
        },
        "metrics": results,
    }

    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(split_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "detector",
            "auroc",
            "aupr",
            "tnr95",
            "fpr95",
            "id_score_mean",
            "ood_score_mean",
            "id_score_std",
            "ood_score_std",
            "num_id",
            "num_ood",
            "score_mode",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for detector, metrics in results.items():
            row = {"detector": detector}
            row.update(metrics)
            writer.writerow(row)

    paste_rows = build_paste_table_rows(results, detectors, model_name=MODEL_TYPE)
    write_tsv(split_paste_tsv, paste_rows)
    print_paste_table(
        paste_rows,
        title=f"Paste-ready table: {shift_name}"
    )

    print("Current split JSON:", split_json)
    print("Current split CSV :", split_csv)
    print("Current split paste table:", split_paste_tsv)

    return results


def load_rq1_model_for_ood():
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Cannot find RQ1 best CodeS model: {MODEL_WEIGHTS}")

    print(">>> Loading RQ1 CodeS checkpoint...")
    checkpoint = safe_torch_load(MODEL_WEIGHTS, map_location="cpu")
    raw_state = unwrap_checkpoint(checkpoint)

    if not isinstance(raw_state, dict):
        raise RuntimeError("Loaded checkpoint is not a state_dict-like object.")

    head_type = infer_head_type_from_state_dict(raw_state) if CLASSIFIER_HEAD_TYPE == "auto" else CLASSIFIER_HEAD_TYPE
    if head_type not in ("single", "mlp_seq", "mlp_named"):
        raise ValueError(f"Unsupported CLASSIFIER_HEAD_TYPE/head_type: {head_type}")

    print(f">>> Auto/selected classifier head_type: {head_type}")

    model = CodeModelForOOD(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        num_labels=NUM_LABELS,
        head_type=head_type,
        pooling=ENCODER_POOLING,
    ).to(DEVICE)

    print(f">>> Encoder pooling strategy: {model.pooling_strategy}")

    adapted_state = adapt_state_dict_keys(raw_state, head_type=head_type)
    compatible_state, skipped = filter_compatible_state_dict(model, adapted_state)

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    matched = [key for key in compatible_state.keys() if key in model.state_dict()]

    print(
        f">>> Model weights loaded: "
        f"head_type={head_type}, pooling={model.pooling_strategy}, "
        f"matched={len(matched)}, missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
    )

    skipped_classifier = [item for item in skipped if item[0].startswith("classifier.")]
    if skipped_classifier:
        print(">>> Skipped classifier keys due to mismatch/missing:")
        for item in skipped_classifier[:30]:
            print("   ", item)

    missing_show = [key for key in missing if "pooler" not in key][:30]
    if missing_show:
        print(">>> Missing keys first 30:", missing_show)
    if unexpected:
        print(">>> Unexpected keys first 30:", unexpected[:30])

    classifier_matched = [key for key in matched if key.startswith("classifier.")]
    encoder_matched = [key for key in matched if key.startswith("encoder.")]
    print(f">>> Encoder matched keys: {len(encoder_matched)}")
    print(f">>> Classifier matched keys: {len(classifier_matched)}")
    print(">>> Classifier matched key preview:", classifier_matched[:20])

    if len(encoder_matched) == 0:
        raise RuntimeError(
            "No encoder parameters were loaded from the checkpoint. "
            "Please check MODEL_TYPE, MODEL_PATH, and MODEL_WEIGHTS."
        )

    if len(classifier_matched) == 0:
        raise RuntimeError(
            "No classifier parameters were loaded from the checkpoint. "
            "Please check MODEL_WEIGHTS, NUM_LABELS, and whether the RQ1 classifier head type is supported."
        )

    final_weight = model.get_final_classifier_weight()
    if final_weight.shape[0] != NUM_LABELS:
        raise RuntimeError(
            f"Final classifier output dim mismatch: got {final_weight.shape[0]}, expected NUM_LABELS={NUM_LABELS}."
        )

    model.eval()
    param_norm = sum(param.detach().float().norm().item() for param in model.parameters())
    print(f">>> Parameter total norm: {param_norm:.6f}")

    return model


# ============================================================
# 8. Main
# ============================================================

def main(model_type=None):
    if model_type is not None:
        configure_model_runtime(model_type)
    else:
        configure_model_runtime(MODEL_TYPE)
    set_seed(SEED)

    print("=" * 120)
    print("CodeS RQ2: Code Classification OOD Detection with 9 Detectors")
    print("=" * 120)
    print("TASK_NAME:", TASK_NAME)
    print("DATASET_NAME:", DATASET_NAME)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("SUPPORTED_MODEL_TYPES:", SUPPORTED_MODEL_TYPES)
    print("MODEL_WEIGHTS:", MODEL_WEIGHTS)
    print("DATA_DIR:", DATA_DIR)
    print("ID_TRAIN_FILE:", ID_TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", OOD_TEST_FILES)
    print("NUM_LABELS:", NUM_LABELS)
    print("CLASSIFIER_HEAD_TYPE:", CLASSIFIER_HEAD_TYPE)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("ENABLE_GRADNORM:", ENABLE_GRADNORM)
    print("GRADNORM_SCORE_MODE:", GRADNORM_SCORE_MODE)
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_rq1_model_for_ood()

    train_loader = DataLoader(
        CodeDataset(ID_TRAIN_FILE, tokenizer, require_label=True, name="id_train"),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    id_test_loader = DataLoader(
        CodeDataset(ID_TEST_FILE, tokenizer, require_label=False, name="id_test"),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    master = OODMaster(model, train_loader)

    all_results = {}

    for shift_name, ood_file in OOD_TEST_FILES.items():
        ood_test_loader = DataLoader(
            CodeDataset(ood_file, tokenizer, require_label=False, name=f"ood_{shift_name}"),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        result = evaluate_ood_split(
            model=model,
            master=master,
            id_test_loader=id_test_loader,
            ood_test_loader=ood_test_loader,
            shift_name=shift_name,
        )
        all_results[shift_name] = result

    summary_payload = {
        "task": TASK_NAME,
        "dataset": DATASET_NAME,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "supported_model_types": SUPPORTED_MODEL_TYPES,
        "model_weights": MODEL_WEIGHTS,
        "data_dir": DATA_DIR,
        "id_train_file": ID_TRAIN_FILE,
        "id_test_file": ID_TEST_FILE,
        "ood_test_files": OOD_TEST_FILES,
        "num_labels": NUM_LABELS,
        "classifier_head_type": model.head_type,
        "encoder_pooling": model.pooling_strategy,
        "enable_gradnorm": ENABLE_GRADNORM,
        "gradnorm_score_mode": GRADNORM_SCORE_MODE if ENABLE_GRADNORM else None,
        "detector_settings": {
            "knn_k": KNN_K,
            "vim_principal_dim": VIM_PRINCIPAL_DIM,
            "score_convention": "larger scores indicate more OOD-like samples",
            "feature_choice": "feature-based detectors use final task/classifier features immediately before the final classifier",
            "auroc_aupr_positive_class": "OOD",
            "tnr95_definition": "ID is treated as the positive class; TNR@95 is the OOD rejection rate when 95% of ID samples are accepted",
        },
        "results": all_results,
        "elapsed_seconds": time.time() - START_TIME,
        "elapsed_readable": format_time(time.time() - START_TIME),
    }

    with open(RESULT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            summary_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(RESULT_SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "shift_name",
            "detector",
            "auroc",
            "aupr",
            "tnr95",
            "fpr95",
            "id_score_mean",
            "ood_score_mean",
            "id_score_std",
            "ood_score_std",
            "num_id",
            "num_ood",
            "score_mode",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for shift_name, detector_results in all_results.items():
            for detector, metrics in detector_results.items():
                row = {"shift_name": shift_name, "detector": detector}
                row.update(metrics)
                writer.writerow(row)

    summary_paste_rows = build_summary_paste_table_rows(all_results)
    write_tsv(RESULT_PASTE_TABLE_TSV, summary_paste_rows)

    print("\n" + "=" * 120)
    print("RQ2 CodeS OOD detection finished")
    print("=" * 120)
    print("Summary JSON:", RESULT_SUMMARY_JSON)
    print("Summary CSV :", RESULT_SUMMARY_CSV)
    print("Paste table :", RESULT_PASTE_TABLE_TSV)
    print("Elapsed:", format_time(time.time() - START_TIME))
    print("=" * 120)
    return summary_payload


if __name__ == "__main__":
    if RUN_ALL_MODELS:
        all_model_results = []
        for run_model_type in MODEL_TYPES_TO_RUN:
            print("\n\n" + "#" * 120)
            print(f"Running CodeS RQ2 OOD detection for model: {run_model_type}")
            print("#" * 120)
            all_model_results.append(main(run_model_type))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        with open(ALL_MODELS_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": TASK_NAME,
                    "dataset": DATASET_NAME,
                    "models": MODEL_TYPES_TO_RUN,
                    "results": all_model_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        all_model_rows = flatten_model_results(all_model_results)
        write_all_models_csv(ALL_MODELS_SUMMARY_CSV, all_model_rows)
        all_model_paste_rows = build_all_models_paste_rows(all_model_results)
        write_tsv(ALL_MODELS_PASTE_TABLE_TSV, all_model_paste_rows)
        print_paste_table(all_model_paste_rows, "EXCEL_COPY_ALL_MODELS_RQ2")
        print("All-model summary JSON:", ALL_MODELS_SUMMARY_JSON)
        print("All-model summary CSV :", ALL_MODELS_SUMMARY_CSV)
        print("All-model paste table  :", ALL_MODELS_PASTE_TABLE_TSV)
    else:
        main(MODEL_TYPE)
