import os
import json
import csv
import time
import sys
import io
import random
import gc
from collections import Counter
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm


# ============================================================
# 0. Environment
# ============================================================

if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

START_TIME = time.time()


# ============================================================
# 1. Core config: clone detection RQ2
# ============================================================

SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = list(SUPPORTED_MODEL_TYPES)
MODEL_TYPE = "codebert"
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}
MODEL_PATH = MODEL_PATHS[MODEL_TYPE]

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset"
RQ1_SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/rq1_outputs"
SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/rq2_outputs"

MODEL_WEIGHTS = os.path.join(RQ1_SAVE_DIR, f"best_clone_{MODEL_TYPE}.bin")

ID_TRAIN_FILE = "id_train.jsonl"
ID_TEST_FILE = "id_test.jsonl"
OOD_TEST_FILES = {
    "token": "ood_test_token.jsonl",
    "cst": "ood_test_cst.jsonl",
}

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 2
MAX_LEN = 512
BATCH_SIZE = 32
SEED = 42
NUM_DATALOADER_WORKERS = 4
USE_AMP_INFERENCE = True
DYNAMIC_PADDING = True

# Keep these aligned with RQ1. Set either value to None for full data.
MAX_ID_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

# Full-data, bounded-memory settings.
STREAM_CHUNK_SIZE = 2048
KNN_REFERENCE_CHUNK_SIZE = 4096
KNN_QUERY_CHUNK_SIZE = 256
COVARIANCE_RIDGE = 1e-4
TEMP_DIR = os.path.join(SAVE_DIR, "rq2_streaming_tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

# OOD detector settings.
# Keep the detector set aligned with codes/RQ2.py:
# MSP, Entropy, MaxLogit, Energy, Mahalanobis, Mahalanobis++, KNN, ViM, GradNorm.
KNN_K = 5
KNN_KEEP_NORMALIZED_ON_GPU = True
KNN_GPU_MAX_MB = 4096
KNN_FULL_DISTANCE_MAX_MB = 1536

VIM_MAX_PCA_COMPONENTS = 512
VIM_PRINCIPAL_DIM = 128
VIM_ALPHA_EPS = 1e-12

# GradNorm keeps the original final-layer definition but uses its exact closed form.
ENABLE_GRADNORM = True
GRADNORM_SCORE_MODE = "neg"  # "neg": -norm, usually larger means more OOD-like; "pos": norm
GRADNORM_CACHE_FLUSH_INTERVAL = 100

RESULT_SUMMARY_JSON = os.path.join(SAVE_DIR, f"rq2_clone_ood_summary_{MODEL_TYPE}.json")
RESULT_SUMMARY_CSV = os.path.join(SAVE_DIR, f"rq2_clone_ood_summary_{MODEL_TYPE}.csv")
RESULT_SUMMARY_EXCEL_TSV = os.path.join(SAVE_DIR, f"rq2_clone_ood_summary_excel_{MODEL_TYPE}.tsv")
ALL_MODELS_SUMMARY_JSON = os.path.join(SAVE_DIR, "rq2_clone_ood_summary_all_models.json")


# ============================================================
# 2. Basic utilities
# ============================================================

def canonical_model_type(model_type):
    aliases = {
        "codebert": "codebert",
        "graphcodebert": "GraphCodeBERT",
        "codet5": "codeT5",
        "code_t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
        "roberta": "RoBERTa",
        "roberta-base": "RoBERTa",
    }
    key = str(model_type).strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported MODEL_TYPE={model_type}. Supported: {SUPPORTED_MODEL_TYPES}"
        )
    return aliases[key]


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, MODEL_WEIGHTS
    global RESULT_SUMMARY_JSON, RESULT_SUMMARY_CSV, RESULT_SUMMARY_EXCEL_TSV

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
    MODEL_WEIGHTS = os.path.join(RQ1_SAVE_DIR, f"best_clone_{MODEL_TYPE}.bin")
    RESULT_SUMMARY_JSON = os.path.join(
        SAVE_DIR, f"rq2_clone_ood_summary_{MODEL_TYPE}.json"
    )
    RESULT_SUMMARY_CSV = os.path.join(
        SAVE_DIR, f"rq2_clone_ood_summary_{MODEL_TYPE}.csv"
    )
    RESULT_SUMMARY_EXCEL_TSV = os.path.join(
        SAVE_DIR, f"rq2_clone_ood_summary_excel_{MODEL_TYPE}.tsv"
    )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def inference_autocast():
    if USE_AMP_INFERENCE and DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


FORWARD_MICROBATCH_LIMIT = None


def fast_model_forward(model, ids, mask):
    """Use batch 32 when possible and automatically fall back after CUDA OOM."""
    global FORWARD_MICROBATCH_LIMIT

    if FORWARD_MICROBATCH_LIMIT is not None and ids.size(0) > FORWARD_MICROBATCH_LIMIT:
        logits_parts = []
        feat_parts = []
        for start in range(0, ids.size(0), FORWARD_MICROBATCH_LIMIT):
            end = min(start + FORWARD_MICROBATCH_LIMIT, ids.size(0))
            logits, feats = fast_model_forward(model, ids[start:end], mask[start:end])
            logits_parts.append(logits)
            feat_parts.append(feats)
        return torch.cat(logits_parts, dim=0), torch.cat(feat_parts, dim=0)

    try:
        with inference_autocast():
            return model(ids, mask, return_feat=True)
    except torch.cuda.OutOfMemoryError:
        if ids.size(0) <= 1:
            raise
        torch.cuda.empty_cache()
        FORWARD_MICROBATCH_LIMIT = max(1, ids.size(0) // 2)
        print(
            f"[Warning] CUDA OOM at batch={ids.size(0)}; "
            f"automatically using micro-batch={FORWARD_MICROBATCH_LIMIT}."
        )
        return fast_model_forward(model, ids, mask)


def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds"


def resolve_data_path(file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(DATA_DIR, file_name)


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
    if key.startswith("module."):
        key = key.replace("module.", "", 1)
    if key.startswith("model."):
        key = key.replace("model.", "", 1)
    return key


def adapt_state_dict_keys(state_dict):
    new_dict = {}

    for key, value in state_dict.items():
        new_key = strip_prefix(key)

        if new_key.startswith("roberta."):
            new_key = new_key.replace("roberta.", "encoder.", 1)
        elif new_key.startswith("bert."):
            new_key = new_key.replace("bert.", "encoder.", 1)
        elif new_key.startswith("transformer."):
            new_key = new_key.replace("transformer.", "encoder.", 1)

        # Some checkpoints save the head under common alternatives.
        if new_key == "score.weight":
            new_key = "classifier.weight"
        elif new_key == "score.bias":
            new_key = "classifier.bias"

        new_dict[new_key] = value

    return new_dict


def get_label(js):
    label = js.get("label")
    if label is None:
        label = js.get("target")
    return label


def get_pair_code(js):
    code1 = (
        js.get("code1")
        or js.get("func1")
        or js.get("function1")
        or js.get("source1")
        or js.get("code_a")
        or js.get("codeA")
        or ""
    )
    code2 = (
        js.get("code2")
        or js.get("func2")
        or js.get("function2")
        or js.get("source2")
        or js.get("code_b")
        or js.get("codeB")
        or ""
    )
    return code1, code2


def get_idx(js, key, fallback):
    idx = js.get(key)
    if idx is None:
        idx = js.get("id")
    if idx is None:
        idx = fallback
    return str(idx)


# ============================================================
# 3. Clone pair dataset
# ============================================================

class ClonePairDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_samples=None, shuffle_before_limit=False):
        self.path = resolve_data_path(file_name)
        self.tokenizer = tokenizer
        self.entries = []
        self._handle = None

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Cannot find data file: {self.path}")

        bad_lines = 0
        skipped_label = 0

        with open(self.path, "rb") as f:
            line_idx = 0
            progress = tqdm(desc=f"Indexing {os.path.basename(self.path)}", leave=False)
            while True:
                offset = f.tell()
                raw_line = f.readline()
                if not raw_line:
                    break
                progress.update(1)
                line = raw_line.strip()
                if not line:
                    line_idx += 1
                    continue

                try:
                    js = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    bad_lines += 1
                    line_idx += 1
                    continue

                code1, code2 = get_pair_code(js)
                label = get_label(js)

                if not code1 or not code2 or label is None:
                    bad_lines += 1
                    line_idx += 1
                    continue

                label = int(label)
                if label < 0 or label >= NUM_LABELS:
                    skipped_label += 1
                    line_idx += 1
                    continue

                self.entries.append((offset, line_idx, label))
                line_idx += 1
            progress.close()

        if shuffle_before_limit:
            random.shuffle(self.entries)

        if max_samples is not None and len(self.entries) > max_samples:
            self.entries = self.entries[:max_samples]

        label_counter = Counter(item[2] for item in self.entries)

        print("=" * 100)
        print(f"ClonePairDataset loaded: {os.path.basename(self.path)}")
        print("path:", self.path)
        print("num_samples:", len(self.entries))
        print("bad_lines:", bad_lines)
        print("skipped_label:", skipped_label)
        print("label_distribution:", dict(label_counter))
        print("=" * 100)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        offset, line_idx, _ = self.entries[index]
        if self._handle is None or self._handle.closed:
            self._handle = open(self.path, "rb")
        self._handle.seek(offset)
        js = json.loads(self._handle.readline().decode("utf-8"))
        code1, code2 = get_pair_code(js)
        label = int(get_label(js))

        inputs = self.tokenizer(
            code1,
            code2,
            max_length=MAX_LEN,
            padding=False if DYNAMIC_PADDING else "max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "idx1": js.get("idx1", get_idx(js, "id1", line_idx)),
            "idx2": js.get("idx2", get_idx(js, "id2", line_idx)),
            "shift_type": js.get("shift_type", ""),
        }

    def close(self):
        if self._handle is not None and not self._handle.closed:
            self._handle.close()

    def __del__(self):
        self.close()


class CloneBatchCollator:
    """Pads only to the longest sequence in a batch, rounded to 8 tokens."""

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def __call__(self, batch):
        if not DYNAMIC_PADDING:
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "label": torch.stack([item["label"] for item in batch]),
                "idx1": [item["idx1"] for item in batch],
                "idx2": [item["idx2"] for item in batch],
                "shift_type": [item["shift_type"] for item in batch],
            }

        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        rounded_length = min(MAX_LEN, ((input_ids.size(1) + 7) // 8) * 8)
        if rounded_length > input_ids.size(1):
            pad_width = rounded_length - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, pad_width), value=self.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_width), value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.stack([item["label"] for item in batch]),
            "idx1": [item["idx1"] for item in batch],
            "idx2": [item["idx2"] for item in batch],
            "shift_type": [item["shift_type"] for item in batch],
        }


# ============================================================
# 4. Clone detection model aligned with RQ1
# ============================================================

class CloneClassifierForOOD(nn.Module):
    def __init__(self, model_path, model_type, num_labels):
        super().__init__()
        self.model_type = model_type.lower()

        if "t5" in self.model_type:
            self.encoder = T5EncoderModel.from_pretrained(model_path)
            hidden_size = self.encoder.config.d_model
        elif "plbart" in self.model_type:
            base_model = AutoModel.from_pretrained(model_path)
            if not hasattr(base_model, "get_encoder"):
                raise ValueError("PLBART requires a seq2seq model with get_encoder().")
            self.encoder = base_model.get_encoder()
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

    def _pool(self, hidden_states, attention_mask):
        if "t5" in self.model_type or "plbart" in self.model_type:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return hidden_states[:, 0, :]

    def forward(self, input_ids, attention_mask, return_feat=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        features = self._pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(features))

        if return_feat:
            return logits, features
        return logits

    def classify_features(self, features):
        return self.classifier(self.dropout(features))

    def get_final_classifier_weight(self):
        return self.classifier.weight

    def get_final_classifier_bias(self):
        return self.classifier.bias


# ============================================================
# 5. OOD metrics
# ============================================================

def get_metrics(id_scores, ood_scores):
    """Larger scores mean more OOD-like."""
    id_scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_scores, dtype=np.float64).reshape(-1)

    id_scores = np.nan_to_num(id_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    ood_scores = np.nan_to_num(ood_scores, nan=0.0, posinf=1e6, neginf=-1e6)

    y_true = np.concatenate([
        np.zeros(len(id_scores), dtype=np.int64),
        np.ones(len(ood_scores), dtype=np.int64),
    ])
    y_scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
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


# ============================================================
# 6. OOD detection engine
# ============================================================

class OODMaster:
    def __init__(self, model, id_train_loader):
        self.model = model
        self.model.eval()
        self.temp_paths = []
        self.id_score_cache = {}
        cursor = 0
        total_examples = len(id_train_loader.dataset)

        if total_examples == 0:
            raise ValueError("ID calibration dataset is empty.")

        print(">>> Extracting ID train features to disk memmap...")

        with torch.inference_mode():
            for batch in tqdm(id_train_loader, desc="Extract ID train features", leave=False):
                ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

                logits, feats = fast_model_forward(self.model, ids, mask)

                feats_np = feats.detach().float().cpu().numpy()
                logits_np = logits.detach().float().cpu().numpy()
                labels_np = batch["label"].detach().cpu().numpy().astype(np.int64)

                if cursor == 0:
                    prefix = canonical_model_type(MODEL_TYPE).lower()
                    feat_path = os.path.join(TEMP_DIR, f"{prefix}_id_features.dat")
                    logit_path = os.path.join(TEMP_DIR, f"{prefix}_id_logits.dat")
                    label_path = os.path.join(TEMP_DIR, f"{prefix}_id_labels.dat")
                    self.temp_paths = [feat_path, logit_path, label_path]
                    self.id_feats = np.memmap(
                        feat_path, mode="w+", dtype=np.float32,
                        shape=(total_examples, feats_np.shape[1]),
                    )
                    self.id_logits = np.memmap(
                        logit_path, mode="w+", dtype=np.float32,
                        shape=(total_examples, logits_np.shape[1]),
                    )
                    self.id_labels = np.memmap(
                        label_path, mode="w+", dtype=np.int64,
                        shape=(total_examples,),
                    )

                end = cursor + feats_np.shape[0]
                self.id_feats[cursor:end] = feats_np
                self.id_logits[cursor:end] = logits_np
                self.id_labels[cursor:end] = labels_np
                cursor = end

        if cursor == 0:
            raise ValueError("No ID calibration samples were extracted.")

        self.id_feats.flush()
        self.id_logits.flush()
        self.id_labels.flush()
        self.num_samples = cursor
        self.feat_dim = self.id_feats.shape[1]

        print("ID train feature shape:", (self.num_samples, self.feat_dim))
        counts = np.bincount(np.asarray(self.id_labels[:cursor]), minlength=NUM_LABELS)
        print("ID train label distribution:", dict(enumerate(counts.tolist())))

        self._prepare_mahalanobis()
        self._prepare_mahalanobis_pp()
        self._prepare_vim()
        self._prepare_knn()
        gc.collect()

        print(">>> OODMaster ready.")

    def _feature_chunks(self):
        for start in range(0, self.num_samples, STREAM_CHUNK_SIZE):
            end = min(start + STREAM_CHUNK_SIZE, self.num_samples)
            yield (
                np.asarray(self.id_feats[start:end], dtype=np.float64),
                np.asarray(self.id_labels[start:end], dtype=np.int64),
            )

    def _class_statistics(self, normalize=False):
        sums = np.zeros((NUM_LABELS, self.feat_dim), dtype=np.float64)
        counts = np.zeros(NUM_LABELS, dtype=np.int64)
        for features, labels in self._feature_chunks():
            if normalize:
                features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            np.add.at(sums, labels, features)
            counts += np.bincount(labels, minlength=NUM_LABELS)
        global_mean = sums.sum(axis=0) / max(int(counts.sum()), 1)
        means = np.empty_like(sums)
        valid = counts > 0
        for label_id in range(NUM_LABELS):
            means[label_id] = sums[label_id] / counts[label_id] if valid[label_id] else global_mean
        return global_mean, means, valid

    def _streaming_precision(self, class_means, normalize, name):
        scatter = np.zeros((self.feat_dim, self.feat_dim), dtype=np.float64)
        count = 0
        for features, labels in tqdm(
            self._feature_chunks(), desc=f"{name} covariance", leave=False,
            total=(self.num_samples + STREAM_CHUNK_SIZE - 1) // STREAM_CHUNK_SIZE,
        ):
            if normalize:
                features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            centered = features - class_means[labels]
            scatter += centered.T @ centered
            count += centered.shape[0]
        covariance = scatter / max(count - 1, 1)
        scale = max(float(np.trace(covariance)) / max(self.feat_dim, 1), 1e-8)
        covariance.flat[:: self.feat_dim + 1] += COVARIANCE_RIDGE * scale
        print(f">>> {name} covariance uses all {count} ID samples.")
        return np.linalg.pinv(covariance, hermitian=True)

    def _prepare_mahalanobis(self):
        print(">>> Preparing Mahalanobis statistics...")
        self.global_mean, self.class_means, self.valid_class_mask = self._class_statistics(False)
        self.precision = self._streaming_precision(self.class_means, False, "Mahalanobis")

    def _prepare_mahalanobis_pp(self):
        print(">>> Preparing Mahalanobis++ statistics...")
        self.global_mean_pp, self.class_means_pp, self.valid_class_mask_pp = self._class_statistics(True)
        self.precision_pp = self._streaming_precision(self.class_means_pp, True, "Mahalanobis++")

    def _prepare_vim(self):
        print(">>> Preparing ViM space...")
        weight = self.model.get_final_classifier_weight().detach().cpu().numpy()
        bias = self.model.get_final_classifier_bias()
        if bias is not None:
            bias = bias.detach().cpu().numpy()

        try:
            if bias is None:
                raise ValueError("classifier bias is None")
            self.vim_u = -np.matmul(np.linalg.pinv(weight), bias)
        except Exception as exc:
            print(f"[Warning] ViM origin from classifier failed; using ID mean. error={repr(exc)}")
            self.vim_u = self.global_mean.copy()

        n_components = min(
            VIM_MAX_PCA_COMPONENTS,
            self.num_samples - 1,
            self.feat_dim,
        )

        if n_components <= 1:
            self.NS = None
            print("[Warning] Too few ID train samples for a valid ViM PCA space.")
        else:
            scatter = np.zeros((self.feat_dim, self.feat_dim), dtype=np.float64)
            for features, _ in tqdm(
                self._feature_chunks(), desc="ViM covariance", leave=False,
                total=(self.num_samples + STREAM_CHUNK_SIZE - 1) // STREAM_CHUNK_SIZE,
            ):
                centered = features - self.vim_u
                scatter += centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(scatter / max(self.num_samples - 1, 1))
            order = np.argsort(eigenvalues)[::-1][:n_components]
            components = eigenvectors[:, order].T
            principal_dim = min(VIM_PRINCIPAL_DIM, n_components - 1)
            self.NS = components[principal_dim:]
            if self.NS.shape[0] == 0:
                self.NS = components[-1:]

            print(
                f"ViM PCA n_components={n_components}, "
                f"principal_dim={principal_dim}, "
                f"NS shape={self.NS.shape}"
            )

        if self.NS is None:
            self.vim_alpha = 1.0
        else:
            residual_sum = energy_sum = 0.0
            count = 0
            for start in range(0, self.num_samples, STREAM_CHUNK_SIZE):
                end = min(start + STREAM_CHUNK_SIZE, self.num_samples)
                features = np.asarray(self.id_feats[start:end], dtype=np.float64)
                logits = np.asarray(self.id_logits[start:end], dtype=np.float64)
                residual_sum += np.linalg.norm((features - self.vim_u) @ self.NS.T, axis=1).sum()
                energy_sum += np.abs(self._logsumexp_np(logits)).sum()
                count += end - start
            self.vim_alpha = float(
                (energy_sum / count) / (residual_sum / count + VIM_ALPHA_EPS)
            )

        print(f">>> ViM alpha: {self.vim_alpha:.6f}")

    @staticmethod
    def _logsumexp_np(values, axis=1):
        max_values = np.max(values, axis=axis, keepdims=True)
        return np.squeeze(max_values, axis=axis) + np.log(
            np.sum(np.exp(values - max_values), axis=axis)
        )

    def _prepare_knn(self):
        self.knn_k = min(KNN_K, max(1, self.num_samples))
        prefix = canonical_model_type(MODEL_TYPE).lower()
        norm_path = os.path.join(TEMP_DIR, f"{prefix}_id_features_normalized.dat")
        self.id_feats_norm = np.memmap(
            norm_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.num_samples, self.feat_dim),
        )
        self.temp_paths.append(norm_path)
        for start in tqdm(
            range(0, self.num_samples, KNN_REFERENCE_CHUNK_SIZE),
            desc="Normalize KNN references",
            leave=False,
        ):
            end = min(start + KNN_REFERENCE_CHUNK_SIZE, self.num_samples)
            refs = np.asarray(self.id_feats[start:end], dtype=np.float32).copy()
            refs /= np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-12)
            self.id_feats_norm[start:end] = refs
        self.id_feats_norm.flush()
        self.knn_reference_tensor = None
        reference_mb = (
            self.num_samples * self.feat_dim * np.dtype(np.float32).itemsize
            / (1024 ** 2)
        )
        if (
            KNN_KEEP_NORMALIZED_ON_GPU
            and DEVICE.type == "cuda"
            and reference_mb <= KNN_GPU_MAX_MB
        ):
            try:
                self.knn_reference_tensor = torch.from_numpy(
                    np.asarray(self.id_feats_norm)
                ).to(DEVICE)
                print(f">>> KNN normalized references cached on GPU: {reference_mb:.1f} MB")
            except RuntimeError as exc:
                self.knn_reference_tensor = None
                torch.cuda.empty_cache()
                print(f"[Warning] KNN GPU cache unavailable; using CPU chunks: {exc}")
        print(
            f">>> KNN uses all {self.num_samples} ID references in "
            f"chunks of {KNN_REFERENCE_CHUNK_SIZE}."
        )

    def msp(self, logits):
        return -torch.max(F.softmax(logits, dim=1), dim=1)[0].detach().cpu().numpy()

    def entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        return (-(probs * torch.log(probs + 1e-12)).sum(dim=1)).detach().cpu().numpy()

    def energy(self, logits):
        return -torch.logsumexp(logits, dim=1).detach().cpu().numpy()

    def maxlogit(self, logits):
        return -torch.max(logits, dim=1)[0].detach().cpu().numpy()

    def mahalanobis(self, feats):
        features = feats.detach().cpu().numpy()
        return self._mahalanobis_score(
            features=features,
            class_means=self.class_means,
            precision=self.precision,
            valid_class_mask=self.valid_class_mask,
        )

    def mahalanobis_pp(self, feats):
        features = feats.detach().cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        return self._mahalanobis_score(
            features=features,
            class_means=self.class_means_pp,
            precision=self.precision_pp,
            valid_class_mask=self.valid_class_mask_pp,
        )

    @staticmethod
    def _mahalanobis_score(features, class_means, precision, valid_class_mask):
        diff = features[:, None, :] - class_means[None, :, :]
        distances = np.einsum(
            "bcd,de,bce->bc", diff, precision, diff, optimize=True
        )
        if np.any(valid_class_mask):
            distances = distances[:, valid_class_mask]
        return np.min(distances, axis=1)

    def vim(self, logits, feats):
        features = feats.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()

        if self.NS is None:
            return -self._logsumexp_np(logits_np)

        residual = np.linalg.norm(np.dot(features - self.vim_u, self.NS.T), axis=1)
        energy = self._logsumexp_np(logits_np)
        virtual_logit = self.vim_alpha * residual
        return virtual_logit - energy

    def knn(self, feats):
        if self.knn_reference_tensor is not None:
            queries = F.normalize(feats.detach().float(), p=2, dim=1)
            distance_mb = (
                queries.size(0) * self.num_samples * queries.element_size()
                / (1024 ** 2)
            )
            if distance_mb <= KNN_FULL_DISTANCE_MAX_MB:
                distances = 1.0 - queries @ self.knn_reference_tensor.T
                nearest = torch.topk(
                    distances,
                    k=self.knn_k,
                    dim=1,
                    largest=False,
                    sorted=False,
                ).values
                return torch.max(nearest, dim=1).values.detach().cpu().numpy()

            best = torch.full(
                (queries.size(0), self.knn_k),
                float("inf"),
                dtype=queries.dtype,
                device=queries.device,
            )
            for start in range(0, self.num_samples, KNN_REFERENCE_CHUNK_SIZE):
                end = min(start + KNN_REFERENCE_CHUNK_SIZE, self.num_samples)
                distances = 1.0 - queries @ self.knn_reference_tensor[start:end].T
                best = torch.topk(
                    torch.cat((best, distances), dim=1),
                    k=self.knn_k,
                    dim=1,
                    largest=False,
                    sorted=False,
                ).values
            return torch.max(best, dim=1).values.detach().cpu().numpy()

        queries = feats.detach().float().cpu().numpy()
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        best = np.full((queries.shape[0], self.knn_k), np.inf, dtype=np.float32)
        for start in range(0, self.num_samples, KNN_REFERENCE_CHUNK_SIZE):
            end = min(start + KNN_REFERENCE_CHUNK_SIZE, self.num_samples)
            refs = np.asarray(self.id_feats_norm[start:end], dtype=np.float32)
            distances = 1.0 - queries @ refs.T
            merged = np.concatenate((best, distances), axis=1)
            best = np.partition(merged, self.knn_k - 1, axis=1)[:, :self.knn_k]
        return np.max(best, axis=1)

    def close(self, remove_files=False):
        for name in (
            "id_feats", "id_logits", "id_labels", "id_feats_norm",
            "knn_reference_tensor",
        ):
            value = getattr(self, name, None)
            if value is not None and hasattr(value, "flush"):
                value.flush()
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if remove_files:
            for path in self.temp_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    def gradnorm(self, feats):
        """Exact closed form of the original final-layer GradNorm score.

        logits=W h+b and objective=logsumexp(logits), therefore
        d(objective)/dW=softmax(logits) outer h. Its L1 norm is exactly
        ||h||_1 because softmax is non-negative and sums to one.
        """
        norms = torch.norm(feats.detach().float(), p=1, dim=1).cpu().numpy()
        return norms if GRADNORM_SCORE_MODE == "pos" else -norms


# ============================================================
# 7. Evaluate one OOD split
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

    return detectors


def get_model_display_name():
    display_names = {
        "codebert": "CodeBERT",
        "graphcodebert": "GraphCodeBERT",
        "GraphCodeBERT": "GraphCodeBERT",
        "codet5": "CodeT5",
        "codeT5": "CodeT5",
        "plbart": "PLBART",
        "PLBART": "PLBART",
        "roberta": "RoBERTa",
        "RoBERTa": "RoBERTa",
    }
    return display_names.get(MODEL_TYPE, MODEL_TYPE)


def format_excel_tsv_block(results, detectors, title=None):
    """Build a tab-separated detector table that can be pasted directly into Excel."""
    metric_rows = [
        ("AUROC", "auroc"),
        ("AUPR", "aupr"),
        ("TNR@95", "tnr95"),
    ]
    first_cell = title or get_model_display_name()
    lines = ["\t".join([first_cell] + detectors)]
    for label, key in metric_rows:
        values = []
        for detector in detectors:
            metric_value = results.get(detector, {}).get(key, "")
            if metric_value == "":
                values.append("")
            else:
                values.append(f"{float(metric_value):.4f}".rstrip("0").rstrip("."))
        lines.append("\t".join([label] + values))
    return "\n".join(lines)


def write_excel_tsv(path, results, detectors, title=None):
    tsv_text = format_excel_tsv_block(results, detectors, title=title)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(tsv_text + "\n")
    return tsv_text


def write_summary_excel_tsv(path, all_results, detectors):
    blocks = []
    model_name = get_model_display_name()
    for shift_name, results in all_results.items():
        blocks.append(
            format_excel_tsv_block(
                results,
                detectors,
                title=f"{model_name}-{shift_name}",
            )
        )
    tsv_text = "\n\n".join(blocks)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(tsv_text + "\n")
    return tsv_text


def _safe_cache_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def open_gradnorm_cache(shift_name, dtype, total):
    domain = "id_shared" if dtype == "id" else f"{shift_name}_ood"
    precision_tag = "amp_v2" if USE_AMP_INFERENCE else "fp32_v2"
    stem = (
        f"{_safe_cache_name(MODEL_TYPE)}_gradnorm_"
        f"{_safe_cache_name(domain)}_{precision_tag}"
    )
    score_path = os.path.join(TEMP_DIR, stem + ".dat")
    progress_path = os.path.join(TEMP_DIR, stem + ".json")
    expected_size = total * np.dtype(np.float64).itemsize
    completed = 0

    if os.path.exists(score_path) and os.path.getsize(score_path) == expected_size:
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("total") == total and metadata.get("domain") == domain:
                completed = min(max(int(metadata.get("completed", 0)), 0), total)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            completed = 0

    mode = "r+" if completed > 0 else "w+"
    cache = np.memmap(score_path, mode=mode, dtype=np.float64, shape=(total,))
    return cache, completed, progress_path, domain


def save_gradnorm_progress(path, domain, total, completed):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            {"domain": domain, "total": total, "completed": completed},
            f,
        )
    os.replace(temp_path, path)


def cleanup_model_temp_files(model_type):
    prefix = _safe_cache_name(model_type)
    for name in os.listdir(TEMP_DIR):
        if name.startswith(prefix + "_"):
            path = os.path.join(TEMP_DIR, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


def evaluate_ood_split(model, master, id_test_loader, ood_test_loader, shift_name):
    detectors = get_detector_list()

    scores = {
        detector: {
            "id": [],
            "ood": [],
        }
        for detector in detectors
    }

    for loader, dtype in [(id_test_loader, "id"), (ood_test_loader, "ood")]:
        print(f"\n>>> Computing {dtype.upper()} scores for shift={shift_name}")

        if dtype == "id" and master.id_score_cache:
            print(">>> Reusing ID detector scores from the previous OOD shift.")
            for detector in detectors:
                scores[detector][dtype].append(master.id_score_cache[detector])
            continue

        sample_offset = 0
        grad_cache = None
        grad_completed = 0
        grad_progress_path = None
        grad_domain = None
        batches_since_grad_flush = 0
        total = len(loader.dataset)
        knn_domain = "id_shared" if dtype == "id" else f"{shift_name}_ood"
        knn_feature_path = os.path.join(
            TEMP_DIR,
            f"{_safe_cache_name(MODEL_TYPE)}_knn_queries_{_safe_cache_name(knn_domain)}.dat",
        )
        knn_features = np.memmap(
            knn_feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(total, master.feat_dim),
        )
        knn_cursor = 0
        if ENABLE_GRADNORM:
            grad_cache, grad_completed, grad_progress_path, grad_domain = open_gradnorm_cache(
                shift_name, dtype, total
            )
            print(f">>> GradNorm resume: {grad_completed}/{total} ({grad_domain})")

        for batch in tqdm(loader, desc=f"{shift_name}-{dtype}"):
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            with torch.inference_mode():
                logits, feats = fast_model_forward(model, ids, mask)

                scores["MSP"][dtype].append(master.msp(logits))
                scores["Entropy"][dtype].append(master.entropy(logits))
                scores["MaxLogit"][dtype].append(master.maxlogit(logits))
                scores["Energy"][dtype].append(master.energy(logits))
                scores["Mahalanobis"][dtype].append(master.mahalanobis(feats))
                scores["Mahalanobis++"][dtype].append(master.mahalanobis_pp(feats))
                scores["ViM"][dtype].append(master.vim(logits, feats))

                feats_np = feats.detach().float().cpu().numpy()
                knn_end = knn_cursor + feats_np.shape[0]
                knn_features[knn_cursor:knn_end] = feats_np
                knn_cursor = knn_end

            batch_size = int(ids.size(0))
            batch_end = sample_offset + batch_size
            if ENABLE_GRADNORM and batch_end > grad_completed:
                local_start = max(0, grad_completed - sample_offset)
                grad_scores = master.gradnorm(feats[local_start:])
                write_start = sample_offset + local_start
                grad_cache[write_start:batch_end] = grad_scores
                grad_completed = batch_end
                batches_since_grad_flush += 1
                if batches_since_grad_flush >= GRADNORM_CACHE_FLUSH_INTERVAL:
                    grad_cache.flush()
                    save_gradnorm_progress(
                        grad_progress_path, grad_domain, total, grad_completed
                    )
                    batches_since_grad_flush = 0
            sample_offset = batch_end

        knn_features.flush()
        print(
            f">>> Computing exact KNN in query chunks of {KNN_QUERY_CHUNK_SIZE} "
            f"for {knn_cursor} {dtype.upper()} samples..."
        )
        for start in tqdm(
            range(0, knn_cursor, KNN_QUERY_CHUNK_SIZE),
            desc=f"{shift_name}-{dtype}-KNN",
            leave=False,
        ):
            end = min(start + KNN_QUERY_CHUNK_SIZE, knn_cursor)
            query_np = np.asarray(knn_features[start:end], dtype=np.float32).copy()
            query_tensor = torch.from_numpy(query_np).to(DEVICE, non_blocking=True)
            scores["KNN"][dtype].append(master.knn(query_tensor))
            del query_tensor, query_np
        del knn_features
        gc.collect()
        try:
            os.remove(knn_feature_path)
        except FileNotFoundError:
            pass

        if ENABLE_GRADNORM:
            grad_cache.flush()
            save_gradnorm_progress(
                grad_progress_path, grad_domain, total, grad_completed
            )
            scores["GradNorm"][dtype].append(np.asarray(grad_cache).copy())
            del grad_cache

        if dtype == "id":
            master.id_score_cache = {
                detector: np.concatenate(scores[detector][dtype]).flatten()
                for detector in detectors
            }

    print("\n" + "=" * 95)
    print(f"RQ2 Clone OOD Detection Results: {shift_name}")
    print("=" * 95)
    print(
        f"{'OOD Detector':<15} | "
        f"{'AUROC':<10} | "
        f"{'AUPR':<10} | "
        f"{'TNR@95':<10} | "
        f"{'FPR@95':<10} | "
        f"{'ID Mean':<10} | "
        f"{'OOD Mean':<10}"
    )
    print("-" * 95)

    results = {}

    for detector in detectors:
        id_arr = np.concatenate(scores[detector]["id"]).flatten()
        ood_arr = np.concatenate(scores[detector]["ood"]).flatten()

        metrics = get_metrics(id_arr, ood_arr)
        if detector == "GradNorm":
            metrics["score_mode"] = GRADNORM_SCORE_MODE
            if metrics["auroc"] < 0.5:
                print(
                    "[Warning] GradNorm AUROC < 0.5. "
                    "If this is consistent, try GRADNORM_SCORE_MODE='pos'."
                )
        results[detector] = metrics

        print(
            f"{detector:<15} | "
            f"{metrics['auroc']:<10.4f} | "
            f"{metrics['aupr']:<10.4f} | "
            f"{metrics['tnr95']:<10.4f} | "
            f"{metrics['fpr95']:<10.4f} | "
            f"{metrics['id_score_mean']:<10.4f} | "
            f"{metrics['ood_score_mean']:<10.4f}"
        )

    print("=" * 95)

    split_json = os.path.join(SAVE_DIR, f"rq2_clone_ood_{shift_name}_{MODEL_TYPE}.json")
    split_csv = os.path.join(SAVE_DIR, f"rq2_clone_ood_{shift_name}_{MODEL_TYPE}.csv")
    split_excel_tsv = os.path.join(SAVE_DIR, f"rq2_clone_ood_{shift_name}_excel_{MODEL_TYPE}.tsv")

    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "Clone Detection",
                "dataset": "CodeXGLUE BigCloneBench",
                "shift_name": shift_name,
                "model_type": MODEL_TYPE,
                "model_path": MODEL_PATH,
                "model_weights": MODEL_WEIGHTS,
                "data_dir": DATA_DIR,
                "id_train_file": ID_TRAIN_FILE,
                "id_test_file": ID_TEST_FILE,
                "ood_test_file": OOD_TEST_FILES[shift_name],
                "num_labels": NUM_LABELS,
                "max_len": MAX_LEN,
                "detectors": detectors,
                "settings": {
                    "knn_k": master.knn_k,
                    "vim_principal_dim": VIM_PRINCIPAL_DIM,
                    "gradnorm_score_mode": GRADNORM_SCORE_MODE if ENABLE_GRADNORM else None,
                    "score_convention": "larger scores indicate more OOD-like samples",
                    "feature_choice": "feature-based detectors use clone-pair task features immediately before the final classifier",
                    "auroc_aupr_positive_class": "OOD",
                    "tnr95_definition": "ID is treated as positive class when computing TNR at 95% TPR",
                },
                "metrics": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

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

    print("Split JSON:", split_json)
    print("Split CSV :", split_csv)
    split_tsv_text = write_excel_tsv(
        split_excel_tsv,
        results,
        detectors,
        title=get_model_display_name(),
    )
    print("Split Excel TSV:", split_excel_tsv)
    print("\nExcel TSV block, copy directly into Excel:")
    print(split_tsv_text)

    return results


# ============================================================
# 8. Main
# ============================================================

def build_loader(file_name, tokenizer, batch_size, max_samples=None):
    dataset = ClonePairDataset(
        file_name=file_name,
        tokenizer=tokenizer,
        max_samples=max_samples,
        shuffle_before_limit=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CloneBatchCollator(tokenizer),
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_DATALOADER_WORKERS > 0,
        prefetch_factor=3,
    )


def run_one_model(requested_model_type):
    global FORWARD_MICROBATCH_LIMIT
    FORWARD_MICROBATCH_LIMIT = None
    configure_model_runtime(requested_model_type)
    set_seed(SEED)
    model_start = time.time()
    print("=" * 120)
    print("RQ2: Clone Detection OOD Detection with Multiple Detectors")
    print("=" * 120)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("MODEL_WEIGHTS:", MODEL_WEIGHTS)
    print("DATA_DIR:", DATA_DIR)
    print("ID_TRAIN_FILE:", ID_TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", OOD_TEST_FILES)
    print("NUM_LABELS:", NUM_LABELS)
    print("MAX_LEN:", MAX_LEN)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_DATALOADER_WORKERS:", NUM_DATALOADER_WORKERS)
    print("USE_AMP_INFERENCE:", USE_AMP_INFERENCE)
    print("DYNAMIC_PADDING:", DYNAMIC_PADDING)
    print("MAX_ID_TRAIN_SAMPLES:", MAX_ID_TRAIN_SAMPLES)
    print("MAX_EVAL_SAMPLES:", MAX_EVAL_SAMPLES)
    print("ENABLE_GRADNORM:", ENABLE_GRADNORM)
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Cannot find RQ1 best clone model: {MODEL_WEIGHTS}")

    print(">>> Loading RQ1 clone checkpoint...")
    checkpoint = safe_torch_load(MODEL_WEIGHTS, map_location="cpu")
    raw_state = unwrap_checkpoint(checkpoint)
    adapted_state = adapt_state_dict_keys(raw_state)

    model = CloneClassifierForOOD(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        num_labels=NUM_LABELS,
    ).to(DEVICE)

    missing, unexpected = model.load_state_dict(adapted_state, strict=False)
    matched = [key for key in adapted_state.keys() if key in model.state_dict()]
    encoder_matched = [key for key in matched if key.startswith("encoder.")]
    classifier_matched = [key for key in matched if key.startswith("classifier.")]

    if not encoder_matched:
        raise RuntimeError("No encoder parameters were loaded from the RQ1 checkpoint.")
    for key in ("classifier.weight", "classifier.bias"):
        if key not in classifier_matched:
            raise RuntimeError(f"Required RQ1 classifier parameter was not loaded: {key}")

    print(
        f">>> Model weights loaded: "
        f"matched={len(matched)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )

    missing_show = [key for key in missing if "pooler" not in key][:30]
    unexpected_show = unexpected[:30]

    if missing_show:
        print(">>> Missing keys first 30:", missing_show)
    if unexpected_show:
        print(">>> Unexpected keys first 30:", unexpected_show)

    model.eval()

    param_norm = 0.0
    for param in model.parameters():
        param_norm += param.detach().float().norm().item()
    print(f">>> Parameter total norm: {param_norm:.6f}")

    train_loader = build_loader(
        ID_TRAIN_FILE,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_samples=MAX_ID_TRAIN_SAMPLES,
    )

    id_test_loader = build_loader(
        ID_TEST_FILE,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_samples=MAX_EVAL_SAMPLES,
    )

    master = OODMaster(model, train_loader)

    all_results = {}

    for shift_name, ood_file in OOD_TEST_FILES.items():
        ood_test_loader = build_loader(
            ood_file,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_samples=MAX_EVAL_SAMPLES,
        )

        result = evaluate_ood_split(
            model=model,
            master=master,
            id_test_loader=id_test_loader,
            ood_test_loader=ood_test_loader,
            shift_name=shift_name,
        )
        all_results[shift_name] = result
        ood_test_loader.dataset.close()
        del ood_test_loader

    with open(RESULT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "Clone Detection",
                "dataset": "CodeXGLUE BigCloneBench",
                "model_type": MODEL_TYPE,
                "model_path": MODEL_PATH,
                "model_weights": MODEL_WEIGHTS,
                "data_dir": DATA_DIR,
                "id_train_file": ID_TRAIN_FILE,
                "id_test_file": ID_TEST_FILE,
                "ood_test_files": OOD_TEST_FILES,
                "num_labels": NUM_LABELS,
                "max_len": MAX_LEN,
                "max_id_train_samples": MAX_ID_TRAIN_SAMPLES,
                "max_eval_samples": MAX_EVAL_SAMPLES,
                "enable_gradnorm": ENABLE_GRADNORM,
                "detectors": get_detector_list(),
                "settings": {
                    "full_data": True,
                    "batch_size": BATCH_SIZE,
                    "num_dataloader_workers": NUM_DATALOADER_WORKERS,
                    "amp_inference": USE_AMP_INFERENCE,
                    "dynamic_padding": DYNAMIC_PADDING,
                    "knn_k": KNN_K,
                    "stream_chunk_size": STREAM_CHUNK_SIZE,
                    "knn_reference_chunk_size": KNN_REFERENCE_CHUNK_SIZE,
                    "knn_query_chunk_size": KNN_QUERY_CHUNK_SIZE,
                    "knn_gpu_cache": KNN_KEEP_NORMALIZED_ON_GPU,
                    "knn_gpu_max_mb": KNN_GPU_MAX_MB,
                    "knn_full_distance_max_mb": KNN_FULL_DISTANCE_MAX_MB,
                    "covariance_ridge": COVARIANCE_RIDGE,
                    "vim_max_pca_components": VIM_MAX_PCA_COMPONENTS,
                    "vim_principal_dim": VIM_PRINCIPAL_DIM,
                    "vim_alpha_eps": VIM_ALPHA_EPS,
                    "gradnorm_full_data": ENABLE_GRADNORM,
                    "gradnorm_disk_resume": ENABLE_GRADNORM,
                    "gradnorm_exact_closed_form": ENABLE_GRADNORM,
                    "gradnorm_cache_flush_interval": GRADNORM_CACHE_FLUSH_INTERVAL,
                    "gradnorm_score_mode": GRADNORM_SCORE_MODE if ENABLE_GRADNORM else None,
                    "score_convention": "larger scores indicate more OOD-like samples",
                    "feature_choice": "feature-based detectors use clone-pair task features immediately before the final classifier",
                    "detector_alignment": "same detector set as codes/RQ2.py",
                },
                "results": all_results,
                "elapsed_seconds": time.time() - model_start,
                "elapsed_readable": format_time(time.time() - model_start),
            },
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

    summary_tsv_text = write_summary_excel_tsv(
        RESULT_SUMMARY_EXCEL_TSV,
        all_results,
        get_detector_list(),
    )

    print("\n" + "=" * 120)
    print("RQ2 clone OOD detection completed")
    print("=" * 120)
    print("Summary JSON:", RESULT_SUMMARY_JSON)
    print("Summary CSV :", RESULT_SUMMARY_CSV)
    print("Summary Excel TSV:", RESULT_SUMMARY_EXCEL_TSV)
    print("\nSummary Excel TSV blocks, copy directly into Excel:")
    print(summary_tsv_text)
    print("Elapsed:", format_time(time.time() - model_start))
    print("=" * 120)

    master.close(remove_files=True)
    cleanup_model_temp_files(MODEL_TYPE)
    train_loader.dataset.close()
    id_test_loader.dataset.close()
    del master, model, train_loader, id_test_loader, tokenizer
    if "ood_test_loader" in locals():
        ood_test_loader.dataset.close()
        del ood_test_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return all_results


def main():
    requested_models = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    outputs = {}

    for requested_model in requested_models:
        model_name = canonical_model_type(requested_model)
        print("\n" + "#" * 120)
        print("Running RQ2 for model:", model_name)
        print("#" * 120)
        try:
            outputs[model_name] = {
                "status": "ok",
                "results": run_one_model(model_name),
            }
        except Exception as exc:
            outputs[model_name] = {
                "status": "failed",
                "error": repr(exc),
            }
            print(f"[Error] {model_name} failed: {repr(exc)}")
        finally:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    with open(ALL_MODELS_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "Clone Detection",
                "dataset": "CodeXGLUE BigCloneBench",
                "models": outputs,
                "detectors": get_detector_list(),
                "elapsed_seconds": time.time() - START_TIME,
                "elapsed_readable": format_time(time.time() - START_TIME),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("All-model summary:", ALL_MODELS_SUMMARY_JSON)


if __name__ == "__main__":
    main()
