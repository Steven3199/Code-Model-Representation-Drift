# -*- coding: utf-8 -*-

"""
CodeSearchNet RQ5 Stage 1:
RQ3-guided 80/20 OOD Split Search.

Research design
---------------
RQ4:
    Original OOD test
    -> random 80/20 split
    -> fixed Replay continual learning
    -> OOD MRR

RQ5:
    The same original OOD test
    -> RQ3-guided 80/20 split
    -> the same fixed Replay continual learning
    -> OOD MRR

Only the OOD split strategy is searched in this script.

The continual-learning settings are fixed to the RQ4 Replay settings:
    replay_lambda = 0.5
    replay_batch_size = 4
    replay_buffer_size = 1000
    batch_size = 32
    learning_rate = 5e-6
    epochs = 8

The four RQ3 signals are adapted to sample-level ID/OOD changes:

1. Alignment damage
   1 - cosine(ID code representation, OOD code representation)

2. Entropy increase
   retrieval_entropy(OOD) - retrieval_entropy(ID)

3. Anisotropy increase
   local_anisotropy(OOD) - local_anisotropy(ID)

4. Effective-rank drop
   local_effective_rank(ID) - local_effective_rank(OOD)

Higher values always indicate greater OOD difficulty.

Expected files
--------------
/opt/data/private/CodeSearchNet/{language}/
    id_train_full.jsonl
    id_test.jsonl
    ood_test_{shift}.jsonl

If the original OOD pool file is unavailable, the script can reconstruct it
from:
    ood_train_{shift}.jsonl
    ood_test_{shift}_new.jsonl
"""

import hashlib
import json
import os
import pickle
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5Config,
    T5EncoderModel,
    get_linear_schedule_with_warmup,
)


# ============================================================
# 1. User configuration
# ============================================================

LANGUAGE = "python"
SHIFT_TYPE = "token"

SUPPORTED_MODEL_TYPES = [
    "codebert",
    "GraphCodeBERT",
    "codeT5",
    "PLBART",
    "RoBERTa",
]

MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

RUN_ALL_MODELS = False

MODEL_TYPES_TO_RUN = [
    "codebert",
    "GraphCodeBERT",
    "codeT5",
    "PLBART",
    "RoBERTa",
]

MODEL_TYPE = "PLBART"


# ============================================================
# 2. Data paths
# ============================================================

DATA_ROOT = "/opt/data/private/CodeSearchNet"
DATA_DIR = os.path.join(DATA_ROOT, LANGUAGE)

RQ1_OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "rq1_outputs",
)

SAVE_DIR = os.path.join(
    DATA_DIR,
    "rq5_rq3_guided_split_results",
)

os.makedirs(SAVE_DIR, exist_ok=True)

ID_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "id_train_full.jsonl",
)

ID_TEST_FILE = os.path.join(
    DATA_DIR,
    "id_test.jsonl",
)

# Prefer the complete original OOD test generated in RQ1.
OOD_POOL_FILE_CANDIDATES = [
    os.path.join(
        DATA_DIR,
        f"ood_test_{SHIFT_TYPE}.jsonl",
    ),
]

# If the original OOD pool is unavailable, merge the existing
# RQ4 random 80/20 split to reconstruct the original pool.
OOD_POOL_FALLBACK_FILES = [
    os.path.join(
        DATA_DIR,
        f"ood_train_{SHIFT_TYPE}.jsonl",
    ),
    os.path.join(
        DATA_DIR,
        f"ood_test_{SHIFT_TYPE}_new.jsonl",
    ),
]


# ============================================================
# 3. RQ4 result used only for comparison
# ============================================================

# Fill in the Replay OOD MRR already obtained from RQ4.
#
# Example:
# RQ4_REPLAY_OOD_MRR = 0.5234
#
# Keep None when no comparison should be printed.
RQ4_REPLAY_OOD_MRR = None


# ============================================================
# 4. Runtime settings
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

MAX_LEN = 256
BATCH_SIZE = 32
NUM_WORKERS = 4

MAX_EVAL_SAMPLES = None

TEMPERATURE_SCALE = 20.0
ENTROPY_CHUNK_SIZE = 64
LOCAL_GEOMETRY_CHUNK_SIZE = 128
LOCAL_RANK_K = 32

MIN_ID_OOD_PAIR_COVERAGE = 0.95
CHECKPOINT_MIN_ENCODER_COVERAGE = 0.90


# ============================================================
# 5. Fixed RQ4 Replay settings
#
# These values are not searched in RQ5.
# ============================================================

LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EPOCHS = 8

REPLAY_LAMBDA = 0.5
REPLAY_BATCH_SIZE = 4
REPLAY_BUFFER_SIZE = 1000
REPLAY_BUFFER_SEED = SEED


# ============================================================
# 6. RQ5 split settings
# ============================================================

FIXED_TRAIN_RATIO = 0.80
FIXED_TEST_RATIO = 0.20

# Samples are divided according to the combined RQ3 difficulty.
#
# hard   = top 25%
# middle = middle 50%
# easy   = bottom 25%
HARD_GROUP_FRACTION = 0.25
MIDDLE_GROUP_FRACTION = 0.50
EASY_GROUP_FRACTION = 0.25

MIN_TRAIN_PER_GROUP = 1

# Formal RQ3 metrics only.
RQ3_METRIC_NAMES = [
    "alignment",
    "entropy",
    "anisotropy",
    "rank",
]

DEFAULT_SCORE_WEIGHTS = {
    "alignment": 0.25,
    "entropy": 0.25,
    "anisotropy": 0.25,
    "rank": 0.25,
}

SEARCH_WEIGHT_BOUNDS = {
    "alignment": (0.0, 1.0),
    "entropy": (0.0, 1.0),
    "anisotropy": (0.0, 1.0),
    "rank": (0.0, 1.0),
}

# These parameters represent the preference for retaining samples
# from each difficulty group in the 80% OOD training set.
#
# A larger value means that the group is more strongly preferred
# for OOD training and is less likely to enter the 20% test set.
TRAIN_PREFERENCE_BOUNDS = {
    "hard": (0.70, 0.99),
    "middle": (0.50, 0.95),
    "easy": (0.20, 0.85),
}


# ============================================================
# 7. Search stages inside this Stage-1 script
# ============================================================

SMART_STAGE1_TRIALS = 36
SMART_STAGE1_EPOCHS = 1

SMART_STAGE2_TRIALS = 16
SMART_STAGE2_EPOCHS = 3
SMART_STAGE2_FROM_STAGE1 = 4
SMART_STAGE2_RANDOM = 2

SMART_FINAL_TRIALS = 5
SMART_FINAL_EPOCHS = EPOCHS

SMART_TOP_CONFIGS_K = 10

RQ5_METRIC_CACHE_VERSION = 3
RQ5_SEARCH_SPACE_VERSION = 3


# ============================================================
# 8. General utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def format_time(seconds):
    hours, remain = divmod(seconds, 3600)
    minutes, seconds = divmod(remain, 60)

    return (
        f"{int(hours)}h "
        f"{int(minutes)}m "
        f"{seconds:.1f}s"
    )


def safe_name(text):
    output = []

    for char in str(text):
        if char.isalnum() or char in ("-", "_"):
            output.append(char)
        else:
            output.append("_")

    return "".join(output)[:160]


def safe_torch_load(path, map_location="cpu"):
    target = "cpu" if map_location is None else map_location
    try:
        return torch.load(
            path,
            map_location=target,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=target,
        )
    except pickle.UnpicklingError:
        # These are trusted checkpoints generated by the local RQ1/RQ5 code.
        return torch.load(
            path,
            map_location=target,
            weights_only=False,
        )


def unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if (
            "model_state_dict" in checkpoint
            and isinstance(checkpoint["model_state_dict"], dict)
        ):
            return checkpoint["model_state_dict"]

        if (
            "state_dict" in checkpoint
            and isinstance(checkpoint["state_dict"], dict)
        ):
            return checkpoint["state_dict"]

        if (
            "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
        ):
            return checkpoint["model"]

    return checkpoint


def read_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return data


def write_jsonl(path, data):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_json(path, value):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def canonical_model_type(model_type):
    aliases = {
        "codebert": "codebert",
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

    lower_key = key.lower()

    if lower_key in aliases:
        return aliases[lower_key]

    raise ValueError(
        f"Unsupported MODEL_TYPE={model_type}. "
        f"Choose from {list(MODEL_PATHS.keys())}"
    )


def get_nl_text(js):
    if js.get("docstring"):
        return js["docstring"]

    if js.get("doc"):
        return js["doc"]

    if js.get("nl"):
        return js["nl"]

    if js.get("query"):
        return js["query"]

    if "docstring_tokens" in js:
        return " ".join(js["docstring_tokens"])

    return ""


def get_code_text(js):
    code = (
        js.get("func")
        or js.get("code")
        or js.get("function")
        or ""
    )

    if not code and "code_tokens" in js:
        code = " ".join(js["code_tokens"])

    return code


def get_sample_key(js, fallback_index):
    for field in (
        "idx",
        "id",
        "url",
        "sample_id",
        "example_id",
    ):
        value = js.get(field)

        if value is not None and str(value) != "":
            return str(value)

    return f"position_{fallback_index}"


def deduplicate_raw_data(data):
    unique = []
    seen = set()

    for index, item in enumerate(data):
        key = get_sample_key(item, index)

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def build_ood_pool(primary_candidates, fallback_files):
    for path in primary_candidates:
        if os.path.exists(path):
            data = deduplicate_raw_data(
                read_jsonl(path)
            )

            print(
                f">>> OOD pool loaded from: {path}"
            )
            print(
                f">>> OOD pool size: {len(data)}"
            )

            return data, [path]

    existing_fallbacks = [
        path
        for path in fallback_files
        if os.path.exists(path)
    ]

    if not existing_fallbacks:
        tried = primary_candidates + fallback_files

        raise FileNotFoundError(
            "Cannot build the original OOD pool. Tried:\n"
            + "\n".join(
                f"  - {path}"
                for path in tried
            )
        )

    merged = []

    for path in existing_fallbacks:
        part = read_jsonl(path)

        print(
            f">>> OOD fallback part: "
            f"{path}, size={len(part)}"
        )

        merged.extend(part)

    merged = deduplicate_raw_data(merged)

    print(
        f">>> Reconstructed OOD pool size: "
        f"{len(merged)}"
    )

    return merged, existing_fallbacks


def align_id_ood_raw(
    id_raw,
    ood_raw,
    min_coverage=0.95,
):
    id_map = {}
    duplicate_id_keys = []

    for index, item in enumerate(id_raw):
        key = get_sample_key(item, index)

        if key in id_map:
            duplicate_id_keys.append(key)
            continue

        id_map[key] = item

    aligned_id = []
    aligned_ood = []
    aligned_keys = []
    unmatched_ood_keys = []

    for index, ood_item in enumerate(ood_raw):
        key = get_sample_key(ood_item, index)

        if key not in id_map:
            unmatched_ood_keys.append(key)
            continue

        id_item = id_map[key]

        if not get_nl_text(id_item):
            continue

        if not get_code_text(id_item):
            continue

        if not get_nl_text(ood_item):
            continue

        if not get_code_text(ood_item):
            continue

        aligned_id.append(id_item)
        aligned_ood.append(ood_item)
        aligned_keys.append(key)

    coverage = (
        len(aligned_ood)
        / max(1, len(ood_raw))
    )

    print("=" * 110)
    print("ID/OOD pairing report")
    print("=" * 110)
    print("ID raw size:", len(id_raw))
    print("OOD raw size:", len(ood_raw))
    print("Aligned pairs:", len(aligned_ood))
    print("Pair coverage:", f"{coverage:.2%}")
    print("Duplicate ID keys:", len(duplicate_id_keys))
    print("Unmatched OOD keys:", len(unmatched_ood_keys))

    if unmatched_ood_keys:
        print(
            "Unmatched key preview:",
            unmatched_ood_keys[:20],
        )

    print("=" * 110)

    if coverage < min_coverage:
        if len(id_raw) == len(ood_raw):
            print(
                "[Warning] Key-based coverage is too low. "
                "Falling back to positional ID/OOD alignment "
                "because both files have the same size."
            )

            positional_id = []
            positional_ood = []
            positional_keys = []

            for index, (id_item, ood_item) in enumerate(
                zip(id_raw, ood_raw)
            ):
                if not get_nl_text(id_item):
                    continue

                if not get_code_text(id_item):
                    continue

                if not get_nl_text(ood_item):
                    continue

                if not get_code_text(ood_item):
                    continue

                positional_id.append(id_item)
                positional_ood.append(ood_item)
                positional_keys.append(
                    get_sample_key(
                        ood_item,
                        index,
                    )
                )

            positional_coverage = (
                len(positional_ood)
                / max(1, len(ood_raw))
            )

            print(
                "Positional pair coverage:",
                f"{positional_coverage:.2%}",
            )

            if positional_coverage >= min_coverage:
                return (
                    positional_id,
                    positional_ood,
                    positional_keys,
                )

        raise RuntimeError(
            f"ID/OOD pair coverage is only {coverage:.2%}, "
            f"lower than the required "
            f"{min_coverage:.2%}. "
            f"Check whether ID and OOD files preserve idx."
        )

    return aligned_id, aligned_ood, aligned_keys


def split_hash(records):
    identifiers = sorted(
        str(record["pool_index"])
        for record in records
    )

    content = "\n".join(identifiers)

    return hashlib.md5(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# 9. Dataset
# ============================================================

class CodeSearchDataset(Dataset):
    def __init__(
        self,
        raw_data,
        tokenizer,
        max_len=256,
        name="dataset",
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        skipped = 0
        shift_counter = Counter()

        for line_index, js in enumerate(raw_data):
            nl_text = get_nl_text(js)
            code_text = get_code_text(js)

            if not nl_text or not code_text:
                skipped += 1
                continue

            # Remove a duplicated docstring from source code when present.
            if nl_text in code_text:
                code_text = code_text.replace(
                    nl_text,
                    "",
                )

                code_text = (
                    code_text
                    .replace('""""""', "")
                    .replace("''''''", "")
                )

            idx = get_sample_key(
                js,
                line_index,
            )

            shift_type = str(
                js.get(
                    "shift_type",
                    "id",
                )
            )

            self.examples.append({
                "nl": nl_text,
                "code": code_text,
                "idx": idx,
                "shift_type": shift_type,
                "raw": js,
            })

            shift_counter[shift_type] += 1

        print("=" * 100)
        print("CodeSearchDataset loaded")
        print("name:", name)
        print("num_samples:", len(self.examples))
        print("skipped:", skipped)
        print(
            "shift distribution:",
            dict(shift_counter),
        )
        print("=" * 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]

        nl_inputs = self.tokenizer(
            item["nl"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        code_inputs = self.tokenizer(
            item["code"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "nl_input_ids": (
                nl_inputs["input_ids"]
                .squeeze(0)
            ),
            "nl_attention_mask": (
                nl_inputs["attention_mask"]
                .squeeze(0)
            ),
            "pl_input_ids": (
                code_inputs["input_ids"]
                .squeeze(0)
            ),
            "pl_attention_mask": (
                code_inputs["attention_mask"]
                .squeeze(0)
            ),
            "idx": item["idx"],
            "shift_type": item["shift_type"],
        }


# ============================================================
# 10. Model
# ============================================================

class DualEncoderModel(nn.Module):
    def __init__(
        self,
        model_path,
        model_type,
        load_pretrained=True,
    ):
        super().__init__()

        self.model_type = model_type.lower()

        if "t5" in self.model_type:
            if load_pretrained:
                self.encoder = (
                    T5EncoderModel.from_pretrained(
                        model_path
                    )
                )
            else:
                config = T5Config.from_pretrained(
                    model_path
                )

                self.encoder = T5EncoderModel(
                    config
                )

            self.hidden_size = (
                self.encoder.config.d_model
            )

        elif "plbart" in self.model_type:
            if load_pretrained:
                base_model = AutoModel.from_pretrained(
                    model_path
                )
            else:
                config = AutoConfig.from_pretrained(
                    model_path
                )

                base_model = AutoModel.from_config(
                    config
                )

            if hasattr(base_model, "get_encoder"):
                self.encoder = (
                    base_model.get_encoder()
                )

            elif hasattr(base_model, "encoder"):
                self.encoder = base_model.encoder

            else:
                raise ValueError(
                    "PLBART model should expose "
                    "get_encoder() or encoder."
                )

            self.hidden_size = (
                getattr(
                    base_model.config,
                    "d_model",
                    None,
                )
                or getattr(
                    base_model.config,
                    "hidden_size",
                    None,
                )
            )

            if self.hidden_size is None:
                raise ValueError(
                    "Cannot infer PLBART hidden size."
                )

        else:
            if load_pretrained:
                self.encoder = AutoModel.from_pretrained(
                    model_path
                )
            else:
                config = AutoConfig.from_pretrained(
                    model_path
                )

                self.encoder = AutoModel.from_config(
                    config
                )

            self.hidden_size = (
                self.encoder.config.hidden_size
            )

    def get_vec(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if (
            "t5" in self.model_type
            or "plbart" in self.model_type
        ):
            mask = (
                attention_mask
                .unsqueeze(-1)
                .to(
                    outputs.last_hidden_state.dtype
                )
            )

            summed = torch.sum(
                outputs.last_hidden_state * mask,
                dim=1,
            )

            vector = summed / torch.clamp(
                mask.sum(dim=1),
                min=1e-9,
            )

        else:
            vector = (
                outputs
                .last_hidden_state[:, 0, :]
            )

        return vector

    def forward(
        self,
        nl_ids,
        nl_mask,
        pl_ids,
        pl_mask,
    ):
        nl_vector = self.get_vec(
            nl_ids,
            nl_mask,
        )

        code_vector = self.get_vec(
            pl_ids,
            pl_mask,
        )

        return (
            F.normalize(
                nl_vector,
                p=2,
                dim=1,
            ),
            F.normalize(
                code_vector,
                p=2,
                dim=1,
            ),
        )


# ============================================================
# 11. Checkpoint loading
# ============================================================

def strip_external_prefixes(key):
    changed = True

    while changed:
        changed = False

        for prefix in (
            "module.",
            "model.",
        ):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def build_checkpoint_key_candidates(raw_key):
    key = strip_external_prefixes(raw_key)

    candidates = [key]

    if key.startswith("roberta."):
        candidates.append(
            "encoder."
            + key[len("roberta."):]
        )

    if key.startswith("bert."):
        candidates.append(
            "encoder."
            + key[len("bert."):]
        )

    if key.startswith("transformer."):
        candidates.append(
            "encoder."
            + key[len("transformer."):]
        )

    if key.startswith("plbart.encoder."):
        candidates.append(
            "encoder."
            + key[len("plbart.encoder."):]
        )

    if key.startswith("plbart."):
        candidates.append(
            "encoder."
            + key[len("plbart."):]
        )

    if key.startswith("encoder.roberta."):
        candidates.append(
            "encoder."
            + key[len("encoder.roberta."):]
        )

    if key.startswith("encoder.bert."):
        candidates.append(
            "encoder."
            + key[len("encoder.bert."):]
        )

    # Only remove a truly duplicated external wrapper:
    #
    # encoder.encoder.encoder.layer.0...
    # ->
    # encoder.encoder.layer.0...
    #
    # Do not change:
    #
    # encoder.encoder.layer.0...
    if key.startswith(
        "encoder.encoder.encoder."
    ):
        candidates.append(
            "encoder.encoder."
            + key[
                len(
                    "encoder.encoder.encoder."
                ):
            ]
        )

    unique_candidates = []

    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def match_checkpoint_to_model(
    model,
    raw_state_dict,
):
    model_state = model.state_dict()

    compatible = {}
    skipped = []
    mappings = []

    for raw_key, value in raw_state_dict.items():
        candidates = build_checkpoint_key_candidates(
            raw_key
        )

        matched_key = None
        mismatch_info = []

        for candidate in candidates:
            if candidate not in model_state:
                continue

            checkpoint_shape = tuple(
                value.shape
            )

            model_shape = tuple(
                model_state[candidate].shape
            )

            if checkpoint_shape == model_shape:
                matched_key = candidate
                break

            mismatch_info.append({
                "candidate": candidate,
                "checkpoint_shape": checkpoint_shape,
                "model_shape": model_shape,
            })

        if matched_key is not None:
            compatible[matched_key] = value

            mappings.append({
                "raw_key": raw_key,
                "matched_key": matched_key,
                "shape": tuple(value.shape),
            })

        elif mismatch_info:
            skipped.append({
                "raw_key": raw_key,
                "reason": "shape_mismatch",
                "details": mismatch_info,
            })

        else:
            skipped.append({
                "raw_key": raw_key,
                "reason": "no_matching_key",
                "candidate_keys": candidates,
                "shape": (
                    tuple(value.shape)
                    if hasattr(value, "shape")
                    else None
                ),
            })

    return compatible, skipped, mappings


def load_rq1_model(
    model_type,
    model_path,
    rq1_best_model,
    device,
):
    if not os.path.exists(rq1_best_model):
        raise FileNotFoundError(
            f"RQ1 best model not found: "
            f"{rq1_best_model}"
        )

    checkpoint = safe_torch_load(
        rq1_best_model,
        map_location="cpu",
    )

    raw_state = unwrap_checkpoint(
        checkpoint
    )

    if not isinstance(raw_state, dict):
        raise RuntimeError(
            "RQ1 checkpoint is not a valid "
            "state_dict-like object."
        )

    model = DualEncoderModel(
        model_path,
        model_type,
        load_pretrained=False,
    ).to(device)

    (
        compatible_state,
        skipped,
        mappings,
    ) = match_checkpoint_to_model(
        model,
        raw_state,
    )

    missing, unexpected = model.load_state_dict(
        compatible_state,
        strict=False,
    )

    model_state = model.state_dict()

    encoder_model_keys = [
        key
        for key in model_state
        if (
            key.startswith("encoder.")
            and "pooler" not in key
        )
    ]

    encoder_matched_keys = [
        key
        for key in compatible_state
        if (
            key.startswith("encoder.")
            and "pooler" not in key
        )
    ]

    total_encoder_numel = sum(
        model_state[key].numel()
        for key in encoder_model_keys
    )

    matched_encoder_numel = sum(
        model_state[key].numel()
        for key in encoder_matched_keys
    )

    encoder_coverage = (
        matched_encoder_numel
        / max(1, total_encoder_numel)
    )

    print("=" * 110)
    print("RQ1 checkpoint loading")
    print("=" * 110)
    print("checkpoint:", rq1_best_model)
    print(
        "matched parameters:",
        len(compatible_state),
    )
    print(
        "matched encoder keys:",
        len(encoder_matched_keys),
    )
    print(
        "encoder parameter coverage:",
        f"{encoder_coverage:.2%}",
    )
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    print("skipped keys:", len(skipped))

    if skipped:
        print(
            "Skipped key preview:",
            skipped[:10],
        )

    missing_without_pooler = [
        key
        for key in missing
        if "pooler" not in key
    ]

    if missing_without_pooler:
        print(
            "Missing key preview:",
            missing_without_pooler[:20],
        )

    print("=" * 110)

    if encoder_coverage < (
        CHECKPOINT_MIN_ENCODER_COVERAGE
    ):
        raise RuntimeError(
            f"RQ1 encoder checkpoint coverage is "
            f"{encoder_coverage:.2%}, "
            f"lower than the required "
            f"{CHECKPOINT_MIN_ENCODER_COVERAGE:.2%}. "
            f"Check checkpoint key mappings."
        )

    return model


# ============================================================
# 12. DataLoader utilities
# ============================================================

def make_loader(
    dataset,
    batch_size,
    shuffle,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


# ============================================================
# 13. Vector extraction and evaluation
# ============================================================

@torch.no_grad()
def extract_vectors(
    model,
    dataloader,
    desc,
):
    model.eval()

    nl_vectors = []
    code_vectors = []
    indices = []
    shift_types = []

    use_amp = DEVICE.type == "cuda"

    for batch in tqdm(
        dataloader,
        desc=desc,
        leave=False,
    ):
        with autocast(enabled=use_amp):
            nl_vector, code_vector = model(
                batch["nl_input_ids"].to(DEVICE),
                batch["nl_attention_mask"].to(DEVICE),
                batch["pl_input_ids"].to(DEVICE),
                batch["pl_attention_mask"].to(DEVICE),
            )

        nl_vectors.append(
            nl_vector.detach().cpu().float()
        )

        code_vectors.append(
            code_vector.detach().cpu().float()
        )

        indices.extend(
            str(value)
            for value in batch["idx"]
        )

        shift_types.extend(
            str(value)
            for value in batch["shift_type"]
        )

    if not nl_vectors:
        raise ValueError(
            f"{desc}: empty dataloader."
        )

    return {
        "nl_vecs": torch.cat(
            nl_vectors,
            dim=0,
        ),
        "code_vecs": torch.cat(
            code_vectors,
            dim=0,
        ),
        "idx": indices,
        "shift_type": shift_types,
    }


@torch.no_grad()
def evaluate_search_metrics(
    model,
    dataloader,
    desc="Evaluating",
    save_path=None,
):
    vectors = extract_vectors(
        model,
        dataloader,
        desc=f"{desc} - extract",
    )

    nl_vectors = F.normalize(
        vectors["nl_vecs"],
        p=2,
        dim=1,
    )

    code_vectors = F.normalize(
        vectors["code_vecs"],
        p=2,
        dim=1,
    )

    scores = torch.matmul(
        nl_vectors,
        code_vectors.T,
    ).numpy()

    ranks = []
    records = []

    for index in tqdm(
        range(len(scores)),
        desc=f"{desc} - rank",
        leave=False,
    ):
        row = scores[index]

        sorted_indices = np.argsort(
            row,
            kind="mergesort",
        )[::-1]

        rank_position = np.where(
            sorted_indices == index
        )[0]

        if len(rank_position) == 0:
            raise RuntimeError(
                f"Cannot locate gold code for "
                f"query position {index}."
            )

        rank = int(
            rank_position[0]
        ) + 1

        ranks.append(rank)

        records.append({
            "idx": str(
                vectors["idx"][index]
            ),
            "shift_type": str(
                vectors["shift_type"][index]
            ),
            "rank": rank,
            "top1_index": int(
                sorted_indices[0]
            ),
            "top1_score": float(
                row[sorted_indices[0]]
            ),
            "gold_score": float(
                row[index]
            ),
        })

    ranks = np.asarray(
        ranks,
        dtype=np.int64,
    )

    result = {
        "mrr": float(
            np.mean(1.0 / ranks)
        ),
        "recall@1": float(
            np.mean(ranks <= 1)
        ),
        "recall@5": float(
            np.mean(ranks <= 5)
        ),
        "recall@10": float(
            np.mean(ranks <= 10)
        ),
        "num_samples": int(
            len(ranks)
        ),
    }

    if save_path is not None:
        write_jsonl(
            save_path,
            records,
        )

    return result


# ============================================================
# 14. Retrieval training
# ============================================================

def retrieval_contrastive_loss(
    model,
    batch,
):
    use_amp = DEVICE.type == "cuda"

    with autocast(enabled=use_amp):
        nl_vector, code_vector = model(
            batch["nl_input_ids"].to(DEVICE),
            batch["nl_attention_mask"].to(DEVICE),
            batch["pl_input_ids"].to(DEVICE),
            batch["pl_attention_mask"].to(DEVICE),
        )

        logits = torch.matmul(
            nl_vector,
            code_vector.T,
        ) * TEMPERATURE_SCALE

        labels = torch.arange(
            nl_vector.size(0),
            device=DEVICE,
        )

        nl_loss = F.cross_entropy(
            logits,
            labels,
        )

        code_loss = F.cross_entropy(
            logits.T,
            labels,
        )

        loss = (
            nl_loss + code_loss
        ) / 2.0

    return loss


def build_replay_buffer_dataset(
    id_train_dataset,
    buffer_size,
    seed,
):
    if (
        buffer_size is None
        or buffer_size <= 0
        or buffer_size >= len(id_train_dataset)
    ):
        indices = list(
            range(len(id_train_dataset))
        )

        return id_train_dataset, indices

    rng = np.random.RandomState(seed)

    indices = rng.choice(
        len(id_train_dataset),
        size=int(buffer_size),
        replace=False,
    ).tolist()

    indices = sorted(indices)

    return (
        Subset(
            id_train_dataset,
            indices,
        ),
        indices,
    )


def train_replay_epoch(
    model,
    ood_loader,
    replay_iterator,
    optimizer,
    scheduler,
    scaler,
    desc,
):
    model.train()

    total_loss_sum = 0.0
    ood_loss_sum = 0.0
    replay_loss_sum = 0.0
    steps = 0

    for ood_batch in tqdm(
        ood_loader,
        desc=desc,
        leave=False,
    ):
        optimizer.zero_grad()

        ood_loss = retrieval_contrastive_loss(
            model,
            ood_batch,
        )

        replay_batch = next(
            replay_iterator
        )

        replay_loss = retrieval_contrastive_loss(
            model,
            replay_batch,
        )

        total_loss = (
            ood_loss
            + REPLAY_LAMBDA * replay_loss
        )

        if not torch.isfinite(total_loss):
            continue

        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        scheduler.step()

        total_loss_sum += float(
            total_loss.detach().item()
        )

        ood_loss_sum += float(
            ood_loss.detach().item()
        )

        replay_loss_sum += float(
            replay_loss.detach().item()
        )

        steps += 1

    denominator = max(
        steps,
        1,
    )

    return {
        "total_loss": (
            total_loss_sum / denominator
        ),
        "ood_loss": (
            ood_loss_sum / denominator
        ),
        "replay_loss": (
            replay_loss_sum / denominator
        ),
        "steps": int(steps),
    }


# ============================================================
# 15. RQ3 sample-level metrics
# ============================================================

def ranking_entropy_scores(
    query_vectors,
    code_bank_vectors,
    temperature=20.0,
    chunk_size=64,
):
    queries = F.normalize(
        query_vectors.float(),
        p=2,
        dim=1,
    )

    codes = F.normalize(
        code_bank_vectors.float(),
        p=2,
        dim=1,
    )

    num_candidates = codes.size(0)

    normalizer = np.log(
        max(2, num_candidates)
    )

    all_entropies = []

    for start in tqdm(
        range(
            0,
            queries.size(0),
            chunk_size,
        ),
        desc="Computing retrieval entropy",
        leave=False,
    ):
        end = min(
            start + chunk_size,
            queries.size(0),
        )

        logits = torch.matmul(
            queries[start:end],
            codes.T,
        ) * float(temperature)

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities + 1e-12
            ),
            dim=1,
        )

        entropy = entropy / normalizer

        all_entropies.extend(
            entropy.cpu().numpy().tolist()
        )

    return np.asarray(
        all_entropies,
        dtype=np.float64,
    )


def effective_rank_of_neighbors(
    neighborhood_vectors,
):
    if neighborhood_vectors.size(0) <= 1:
        return 1.0

    centered = (
        neighborhood_vectors.float()
        - neighborhood_vectors
        .float()
        .mean(
            dim=0,
            keepdim=True,
        )
    )

    gram = torch.matmul(
        centered,
        centered.T,
    )

    try:
        eigenvalues = torch.linalg.eigvalsh(
            gram
        )
    except Exception:
        eigenvalues = torch.linalg.eigvals(
            gram
        ).real

    eigenvalues = torch.clamp(
        eigenvalues,
        min=0.0,
    )

    singular_values = torch.sqrt(
        eigenvalues + 1e-12
    )

    singular_values = singular_values[
        singular_values > 1e-10
    ]

    if singular_values.numel() == 0:
        return 1.0

    probability = (
        singular_values
        / singular_values.sum()
    )

    entropy = -torch.sum(
        probability
        * torch.log(
            probability + 1e-12
        )
    )

    return float(
        torch.exp(entropy).item()
    )


def local_geometry_scores(
    code_vectors,
    local_k=32,
    chunk_size=128,
):
    vectors = F.normalize(
        code_vectors.float(),
        p=2,
        dim=1,
    )

    num_samples = vectors.size(0)

    if num_samples <= 1:
        return (
            np.zeros(
                num_samples,
                dtype=np.float64,
            ),
            np.ones(
                num_samples,
                dtype=np.float64,
            ),
        )

    k = min(
        max(1, int(local_k)),
        num_samples - 1,
    )

    neighbor_indices = []
    neighbor_similarities = []

    for start in tqdm(
        range(
            0,
            num_samples,
            chunk_size,
        ),
        desc="Finding local code neighborhoods",
        leave=False,
    ):
        end = min(
            start + chunk_size,
            num_samples,
        )

        similarity = torch.matmul(
            vectors[start:end],
            vectors.T,
        )

        local_rows = torch.arange(
            end - start
        )

        global_columns = torch.arange(
            start,
            end,
        )

        similarity[
            local_rows,
            global_columns,
        ] = -1e30

        values, indices = torch.topk(
            similarity,
            k=k,
            dim=1,
            largest=True,
        )

        neighbor_indices.append(
            indices.cpu()
        )

        neighbor_similarities.append(
            values.cpu()
        )

    neighbor_indices = torch.cat(
        neighbor_indices,
        dim=0,
    )

    neighbor_similarities = torch.cat(
        neighbor_similarities,
        dim=0,
    )

    anisotropy = (
        neighbor_similarities
        .mean(dim=1)
        .numpy()
        .astype(np.float64)
    )

    effective_ranks = np.zeros(
        num_samples,
        dtype=np.float64,
    )

    for index in tqdm(
        range(num_samples),
        desc="Computing local effective rank",
        leave=False,
    ):
        local_indices = (
            neighbor_indices[index]
        )

        local_vectors = vectors[
            local_indices
        ]

        effective_ranks[index] = (
            effective_rank_of_neighbors(
                local_vectors
            )
        )

    return anisotropy, effective_ranks


def percentile_normalize(
    values,
):
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    output = np.full_like(
        array,
        0.5,
        dtype=np.float64,
    )

    finite_mask = np.isfinite(
        array
    )

    finite_indices = np.where(
        finite_mask
    )[0]

    if len(finite_indices) == 0:
        return output

    valid_values = array[
        finite_indices
    ]

    if len(valid_values) == 1:
        output[finite_indices] = 0.5
        return output

    order = np.argsort(
        valid_values,
        kind="mergesort",
    )

    sorted_values = valid_values[
        order
    ]

    ranks = np.zeros(
        len(valid_values),
        dtype=np.float64,
    )

    start = 0

    while start < len(sorted_values):
        end = start + 1

        while (
            end < len(sorted_values)
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = (
            start + end - 1
        ) / 2.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    ranks = ranks / max(
        1,
        len(valid_values) - 1,
    )

    output[
        finite_indices
    ] = ranks

    return output


def normalize_positive_weights(
    weights,
):
    normalized = {
        name: max(
            0.0,
            float(
                weights.get(
                    name,
                    0.0,
                )
            ),
        )
        for name in RQ3_METRIC_NAMES
    }

    total = sum(
        normalized.values()
    )

    if total <= 1e-12:
        return {
            name: 1.0 / len(
                RQ3_METRIC_NAMES
            )
            for name in RQ3_METRIC_NAMES
        }

    return {
        name: value / total
        for name, value in normalized.items()
    }


def compute_rq3_sample_metrics(
    model_type,
    model_path,
    rq1_best_model,
    tokenizer,
    id_test_raw,
    ood_pool_raw,
    cache_file,
):
    if os.path.exists(cache_file):
        print(
            ">>> Loading cached paired RQ3 metrics:",
            cache_file,
        )

        cached = json.load(
            open(
                cache_file,
                "r",
                encoding="utf-8",
            )
        )

        if (
            isinstance(cached, list)
            and cached
            and cached[0].get(
                "metric_cache_version"
            ) == RQ5_METRIC_CACHE_VERSION
            and "damage_metrics"
            in cached[0]
            and all(
                metric_name
                in cached[0]["damage_metrics"]
                for metric_name
                in RQ3_METRIC_NAMES
            )
        ):
            return cached

        print(
            ">>> Metric cache is outdated. "
            "Recomputing paired RQ3 metrics."
        )

    (
        aligned_id_raw,
        aligned_ood_raw,
        aligned_keys,
    ) = align_id_ood_raw(
        id_test_raw,
        ood_pool_raw,
        min_coverage=(
            MIN_ID_OOD_PAIR_COVERAGE
        ),
    )

    id_dataset = CodeSearchDataset(
        aligned_id_raw,
        tokenizer,
        max_len=MAX_LEN,
        name="paired_id_test_for_rq3",
    )

    ood_dataset = CodeSearchDataset(
        aligned_ood_raw,
        tokenizer,
        max_len=MAX_LEN,
        name="paired_ood_pool_for_rq3",
    )

    if len(id_dataset) != len(ood_dataset):
        raise RuntimeError(
            "The paired ID and OOD datasets "
            "have different valid sample counts."
        )

    id_loader = make_loader(
        id_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    ood_loader = make_loader(
        ood_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    model = load_rq1_model(
        model_type,
        model_path,
        rq1_best_model,
        DEVICE,
    )

    id_vectors = extract_vectors(
        model,
        id_loader,
        desc="Extract paired ID representations",
    )

    ood_vectors = extract_vectors(
        model,
        ood_loader,
        desc="Extract paired OOD representations",
    )

    id_query_vectors = (
        id_vectors["nl_vecs"]
    )

    id_code_vectors = (
        id_vectors["code_vecs"]
    )

    ood_query_vectors = (
        ood_vectors["nl_vecs"]
    )

    ood_code_vectors = (
        ood_vectors["code_vecs"]
    )

    # --------------------------------------------------------
    # 1. Alignment damage
    # --------------------------------------------------------

    alignment_similarity = torch.sum(
        F.normalize(
            id_code_vectors,
            p=2,
            dim=1,
        )
        * F.normalize(
            ood_code_vectors,
            p=2,
            dim=1,
        ),
        dim=1,
    ).numpy()

    alignment_damage = (
        1.0 - alignment_similarity
    )

    # --------------------------------------------------------
    # 2. Retrieval entropy increase
    # --------------------------------------------------------

    id_entropy = ranking_entropy_scores(
        id_query_vectors,
        id_code_vectors,
        temperature=TEMPERATURE_SCALE,
        chunk_size=ENTROPY_CHUNK_SIZE,
    )

    ood_entropy = ranking_entropy_scores(
        ood_query_vectors,
        ood_code_vectors,
        temperature=TEMPERATURE_SCALE,
        chunk_size=ENTROPY_CHUNK_SIZE,
    )

    entropy_increase = (
        ood_entropy - id_entropy
    )

    # --------------------------------------------------------
    # 3. Local anisotropy increase
    # --------------------------------------------------------

    (
        id_anisotropy,
        id_effective_rank,
    ) = local_geometry_scores(
        id_code_vectors,
        local_k=LOCAL_RANK_K,
        chunk_size=(
            LOCAL_GEOMETRY_CHUNK_SIZE
        ),
    )

    (
        ood_anisotropy,
        ood_effective_rank,
    ) = local_geometry_scores(
        ood_code_vectors,
        local_k=LOCAL_RANK_K,
        chunk_size=(
            LOCAL_GEOMETRY_CHUNK_SIZE
        ),
    )

    anisotropy_increase = (
        ood_anisotropy
        - id_anisotropy
    )

    # --------------------------------------------------------
    # 4. Effective-rank drop
    # --------------------------------------------------------

    effective_rank_drop = (
        id_effective_rank
        - ood_effective_rank
    )

    records = []

    for index in range(
        len(aligned_ood_raw)
    ):
        records.append({
            "pool_index": int(index),
            "idx": str(aligned_keys[index]),
            "shift_type": str(
                aligned_ood_raw[index].get(
                    "shift_type",
                    SHIFT_TYPE,
                )
            ),
            "metric_cache_version": (
                RQ5_METRIC_CACHE_VERSION
            ),
            "raw": aligned_ood_raw[index],
            "raw_metrics": {
                "alignment_similarity": float(
                    alignment_similarity[index]
                ),
                "alignment_damage": float(
                    alignment_damage[index]
                ),
                "id_entropy": float(
                    id_entropy[index]
                ),
                "ood_entropy": float(
                    ood_entropy[index]
                ),
                "entropy_increase": float(
                    entropy_increase[index]
                ),
                "id_anisotropy": float(
                    id_anisotropy[index]
                ),
                "ood_anisotropy": float(
                    ood_anisotropy[index]
                ),
                "anisotropy_increase": float(
                    anisotropy_increase[index]
                ),
                "id_effective_rank": float(
                    id_effective_rank[index]
                ),
                "ood_effective_rank": float(
                    ood_effective_rank[index]
                ),
                "effective_rank_drop": float(
                    effective_rank_drop[index]
                ),
            },
            # All four values use the same direction:
            # a larger value means a harder OOD sample.
            "damage_metrics": {
                "alignment": float(
                    alignment_damage[index]
                ),
                "entropy": float(
                    entropy_increase[index]
                ),
                "anisotropy": float(
                    anisotropy_increase[index]
                ),
                "rank": float(
                    effective_rank_drop[index]
                ),
            },
        })

    save_json(
        cache_file,
        records,
    )

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return records


def score_metric_records(
    metric_records,
    weights,
):
    actual_weights = normalize_positive_weights(
        weights
    )

    normalized_metrics = {}

    for metric_name in RQ3_METRIC_NAMES:
        raw_values = np.asarray(
            [
                record["damage_metrics"][
                    metric_name
                ]
                for record in metric_records
            ],
            dtype=np.float64,
        )

        normalized_metrics[
            metric_name
        ] = percentile_normalize(
            raw_values
        )

    difficulty_scores = np.zeros(
        len(metric_records),
        dtype=np.float64,
    )

    for metric_name in RQ3_METRIC_NAMES:
        difficulty_scores += (
            actual_weights[metric_name]
            * normalized_metrics[
                metric_name
            ]
        )

    scored_records = []

    for index, record in enumerate(
        metric_records
    ):
        scored = dict(record)

        scored["normalized_damage"] = {
            metric_name: float(
                normalized_metrics[
                    metric_name
                ][index]
            )
            for metric_name
            in RQ3_METRIC_NAMES
        }

        scored["difficulty_score"] = float(
            difficulty_scores[index]
        )

        scored_records.append(
            scored
        )

    return scored_records, actual_weights


# ============================================================
# 16. RQ3-guided 80/20 split
# ============================================================

def sanitize_train_preferences(
    preferences,
):
    values = {}

    for group_name in (
        "hard",
        "middle",
        "easy",
    ):
        lower, upper = (
            TRAIN_PREFERENCE_BOUNDS[
                group_name
            ]
        )

        value = float(
            preferences.get(
                group_name,
                0.80,
            )
        )

        values[group_name] = min(
            max(value, lower),
            upper,
        )

    # Hard samples should never receive a lower training
    # preference than middle/easy samples.
    values["middle"] = min(
        values["middle"],
        values["hard"],
    )

    values["easy"] = min(
        values["easy"],
        values["middle"],
    )

    return values


def allocate_test_counts(
    groups,
    train_preferences,
    target_test_size,
):
    capacity = {}

    for group_name, items in groups.items():
        capacity[group_name] = max(
            0,
            len(items)
            - min(
                MIN_TRAIN_PER_GROUP,
                len(items),
            ),
        )

    target_test_size = min(
        target_test_size,
        sum(capacity.values()),
    )

    # A group with low train preference receives more test slots.
    test_weights = {}

    for group_name, items in groups.items():
        preference = train_preferences[
            group_name
        ]

        test_weights[group_name] = (
            len(items)
            * max(
                1e-6,
                1.0 - preference,
            )
        )

        if capacity[group_name] <= 0:
            test_weights[group_name] = 0.0

    weight_sum = sum(
        test_weights.values()
    )

    if weight_sum <= 1e-12:
        test_weights = {
            group_name: (
                float(len(items))
                if capacity[group_name] > 0
                else 0.0
            )
            for group_name, items
            in groups.items()
        }

        weight_sum = sum(
            test_weights.values()
        )

    raw_allocations = {}

    for group_name in groups:
        raw_allocations[group_name] = (
            target_test_size
            * test_weights[group_name]
            / max(weight_sum, 1e-12)
        )

    test_counts = {}

    for group_name in groups:
        test_counts[group_name] = min(
            int(
                np.floor(
                    raw_allocations[
                        group_name
                    ]
                )
            ),
            capacity[group_name],
        )

    current_total = sum(
        test_counts.values()
    )

    # First allocate according to the fractional remainder.
    fractional_order = sorted(
        groups.keys(),
        key=lambda name: (
            raw_allocations[name]
            - np.floor(
                raw_allocations[name]
            )
        ),
        reverse=True,
    )

    for group_name in fractional_order:
        if current_total >= target_test_size:
            break

        if (
            test_counts[group_name]
            < capacity[group_name]
        ):
            test_counts[group_name] += 1
            current_total += 1

    # Fill any remaining test slots caused by capacity clipping.
    while current_total < target_test_size:
        eligible = [
            group_name
            for group_name in groups
            if (
                test_counts[group_name]
                < capacity[group_name]
            )
        ]

        if not eligible:
            break

        selected_group = max(
            eligible,
            key=lambda name: (
                test_weights[name]
                / max(1, capacity[name])
            ),
        )

        test_counts[selected_group] += 1
        current_total += 1

    return test_counts


def pick_evenly_spaced(
    items,
    number_to_select,
):
    if number_to_select <= 0:
        return []

    if number_to_select >= len(items):
        return list(items)

    positions = np.linspace(
        0,
        len(items) - 1,
        num=number_to_select,
    )

    selected_indices = []

    for position in positions:
        selected_index = int(
            round(position)
        )

        if selected_index not in selected_indices:
            selected_indices.append(
                selected_index
            )

    cursor = 0

    while (
        len(selected_indices)
        < number_to_select
        and cursor < len(items)
    ):
        if cursor not in selected_indices:
            selected_indices.append(cursor)

        cursor += 1

    selected_indices = sorted(
        selected_indices[
            :number_to_select
        ]
    )

    return [
        items[index]
        for index in selected_indices
    ]


def build_geometry_split(
    metric_records,
    search_config,
):
    (
        scored_records,
        actual_weights,
    ) = score_metric_records(
        metric_records,
        search_config["weights"],
    )

    train_preferences = (
        sanitize_train_preferences(
            search_config[
                "train_preferences"
            ]
        )
    )

    sorted_records = sorted(
        scored_records,
        key=lambda item: item[
            "difficulty_score"
        ],
        reverse=True,
    )

    total_size = len(sorted_records)

    if total_size < 10:
        raise ValueError(
            "The OOD pool is too small for "
            "RQ3-guided 80/20 splitting."
        )

    hard_end = int(
        round(
            total_size
            * HARD_GROUP_FRACTION
        )
    )

    middle_end = hard_end + int(
        round(
            total_size
            * MIDDLE_GROUP_FRACTION
        )
    )

    hard_end = max(
        1,
        min(
            hard_end,
            total_size - 2,
        ),
    )

    middle_end = max(
        hard_end + 1,
        min(
            middle_end,
            total_size - 1,
        ),
    )

    groups = {
        "hard": sorted_records[:hard_end],
        "middle": sorted_records[
            hard_end:middle_end
        ],
        "easy": sorted_records[
            middle_end:
        ],
    }

    target_test_size = int(
        round(
            total_size
            * FIXED_TEST_RATIO
        )
    )

    target_test_size = max(
        1,
        min(
            target_test_size,
            total_size - 1,
        ),
    )

    test_counts = allocate_test_counts(
        groups,
        train_preferences,
        target_test_size,
    )

    train_records = []
    test_records = []
    group_report = {}

    for group_name, items in groups.items():
        # Items are already ordered from harder to easier.
        number_test = int(
            test_counts[group_name]
        )

        # Select test samples evenly across each difficulty band.
        # The remaining samples are assigned to OOD training.
        group_test = pick_evenly_spaced(
            items,
            number_test,
        )

        test_pool_indices = {
            item["pool_index"]
            for item in group_test
        }

        group_train = [
            item
            for item in items
            if (
                item["pool_index"]
                not in test_pool_indices
            )
        ]

        train_records.extend(
            group_train
        )

        test_records.extend(
            group_test
        )

        group_report[group_name] = {
            "available": int(
                len(items)
            ),
            "train": int(
                len(group_train)
            ),
            "test": int(
                len(group_test)
            ),
            "train_ratio": float(
                len(group_train)
                / max(1, len(items))
            ),
            "test_ratio": float(
                len(group_test)
                / max(1, len(items))
            ),
            "train_preference": float(
                train_preferences[
                    group_name
                ]
            ),
            "all_difficulty_mean": float(
                np.mean(
                    [
                        item[
                            "difficulty_score"
                        ]
                        for item in items
                    ]
                )
            ) if items else 0.0,
            "train_difficulty_mean": float(
                np.mean(
                    [
                        item[
                            "difficulty_score"
                        ]
                        for item
                        in group_train
                    ]
                )
            ) if group_train else 0.0,
            "test_difficulty_mean": float(
                np.mean(
                    [
                        item[
                            "difficulty_score"
                        ]
                        for item
                        in group_test
                    ]
                )
            ) if group_test else 0.0,
        }

    split_rng = np.random.RandomState(
        int(
            search_config.get(
                "split_seed",
                SEED,
            )
        )
    )

    split_rng.shuffle(
        train_records
    )

    split_rng.shuffle(
        test_records
    )

    expected_train_size = (
        total_size - target_test_size
    )

    if len(train_records) != expected_train_size:
        raise RuntimeError(
            f"RQ5 train size mismatch: "
            f"expected={expected_train_size}, "
            f"actual={len(train_records)}"
        )

    if len(test_records) != target_test_size:
        raise RuntimeError(
            f"RQ5 test size mismatch: "
            f"expected={target_test_size}, "
            f"actual={len(test_records)}"
        )

    train_raw = [
        item["raw"]
        for item in train_records
    ]

    test_raw = [
        item["raw"]
        for item in test_records
    ]

    train_difficulty_mean = float(
        np.mean(
            [
                item["difficulty_score"]
                for item in train_records
            ]
        )
    )

    test_difficulty_mean = float(
        np.mean(
            [
                item["difficulty_score"]
                for item in test_records
            ]
        )
    )

    report = {
        "search_config": search_config,
        "actual_weights": actual_weights,
        "train_preferences": (
            train_preferences
        ),
        "fixed_train_ratio": (
            FIXED_TRAIN_RATIO
        ),
        "fixed_test_ratio": (
            FIXED_TEST_RATIO
        ),
        "num_total": int(total_size),
        "num_train": int(
            len(train_records)
        ),
        "num_test": int(
            len(test_records)
        ),
        "actual_train_ratio": float(
            len(train_records)
            / total_size
        ),
        "actual_test_ratio": float(
            len(test_records)
            / total_size
        ),
        "train_difficulty_mean": (
            train_difficulty_mean
        ),
        "test_difficulty_mean": (
            test_difficulty_mean
        ),
        "train_minus_test_difficulty": float(
            train_difficulty_mean
            - test_difficulty_mean
        ),
        "groups": group_report,
        "train_hash": split_hash(
            train_records
        ),
        "test_hash": split_hash(
            test_records
        ),
        "note": (
            "The original OOD test is divided into "
            "80% OOD training and 20% OOD testing. "
            "The split is guided only by the four "
            "formal RQ3 representation metrics."
        ),
    }

    return (
        train_raw,
        test_raw,
        report,
    )


# ============================================================
# 17. Search configuration generation
# ============================================================

def make_search_config(
    name,
    weights,
    train_preferences,
    split_seed=42,
):
    return {
        "name": str(name),
        "weights": dict(weights),
        "train_preferences": dict(
            train_preferences
        ),
        "split_seed": int(split_seed),
    }


def initial_geometry_configs(
    seed=42,
):
    weight_sets = [
        (
            "balanced",
            {
                "alignment": 0.25,
                "entropy": 0.25,
                "anisotropy": 0.25,
                "rank": 0.25,
            },
        ),
        (
            "alignment_focus",
            {
                "alignment": 0.55,
                "entropy": 0.15,
                "anisotropy": 0.15,
                "rank": 0.15,
            },
        ),
        (
            "entropy_focus",
            {
                "alignment": 0.15,
                "entropy": 0.55,
                "anisotropy": 0.15,
                "rank": 0.15,
            },
        ),
        (
            "anisotropy_focus",
            {
                "alignment": 0.15,
                "entropy": 0.15,
                "anisotropy": 0.55,
                "rank": 0.15,
            },
        ),
        (
            "rank_focus",
            {
                "alignment": 0.15,
                "entropy": 0.15,
                "anisotropy": 0.15,
                "rank": 0.55,
            },
        ),
        (
            "alignment_entropy",
            {
                "alignment": 0.40,
                "entropy": 0.40,
                "anisotropy": 0.10,
                "rank": 0.10,
            },
        ),
    ]

    preference_sets = [
        (
            "representative",
            {
                "hard": 0.80,
                "middle": 0.80,
                "easy": 0.80,
            },
        ),
        (
            "hard_train_light",
            {
                "hard": 0.88,
                "middle": 0.80,
                "easy": 0.72,
            },
        ),
        (
            "hard_train_medium",
            {
                "hard": 0.94,
                "middle": 0.80,
                "easy": 0.60,
            },
        ),
        (
            "hard_train_strong",
            {
                "hard": 0.98,
                "middle": 0.78,
                "easy": 0.50,
            },
        ),
        (
            "hard_middle_support",
            {
                "hard": 0.94,
                "middle": 0.88,
                "easy": 0.58,
            },
        ),
        (
            "moderate_geometry",
            {
                "hard": 0.91,
                "middle": 0.82,
                "easy": 0.68,
            },
        ),
    ]

    configurations = []
    configuration_index = 0

    for (
        weight_name,
        weights,
    ) in weight_sets:
        for (
            preference_name,
            preferences,
        ) in preference_sets:
            configurations.append(
                make_search_config(
                    name=(
                        f"{weight_name}_"
                        f"{preference_name}"
                    ),
                    weights=weights,
                    train_preferences=preferences,
                    split_seed=(
                        seed + configuration_index
                    ),
                )
            )

            configuration_index += 1

    return configurations


def random_geometry_config(
    rng,
    name,
    seed,
):
    weights = {
        metric_name: float(
            rng.uniform(
                *SEARCH_WEIGHT_BOUNDS[
                    metric_name
                ]
            )
        )
        for metric_name
        in RQ3_METRIC_NAMES
    }

    preferences = {
        group_name: float(
            rng.uniform(
                *TRAIN_PREFERENCE_BOUNDS[
                    group_name
                ]
            )
        )
        for group_name
        in (
            "hard",
            "middle",
            "easy",
        )
    }

    preferences = (
        sanitize_train_preferences(
            preferences
        )
    )

    return make_search_config(
        name=name,
        weights=weights,
        train_preferences=preferences,
        split_seed=seed,
    )


def config_feature_vector(
    search_config,
):
    weights = normalize_positive_weights(
        search_config["weights"]
    )

    preferences = (
        sanitize_train_preferences(
            search_config[
                "train_preferences"
            ]
        )
    )

    return [
        float(weights["alignment"]),
        float(weights["entropy"]),
        float(weights["anisotropy"]),
        float(weights["rank"]),
        float(preferences["hard"]),
        float(preferences["middle"]),
        float(preferences["easy"]),
    ]


def config_signature(
    search_config,
):
    features = config_feature_vector(
        search_config
    )

    return tuple(
        round(value, 8)
        for value in features
    )


def fit_surrogate_and_propose(
    results,
    number_needed,
    seed=42,
    pool_size=800,
):
    valid_results = [
        row
        for row in results
        if row.get("status") == "ok"
    ]

    rng = np.random.RandomState(
        seed
    )

    if len(valid_results) < 8:
        return [
            random_geometry_config(
                rng,
                name=f"random_{index:03d}",
                seed=seed + index,
            )
            for index in range(
                number_needed
            )
        ]

    train_x = np.asarray(
        [
            config_feature_vector(
                row["config"]
            )
            for row in valid_results
        ],
        dtype=np.float64,
    )

    train_y = np.asarray(
        [
            row["select_score"]
            for row in valid_results
        ],
        dtype=np.float64,
    )

    surrogate = ExtraTreesRegressor(
        n_estimators=256,
        random_state=seed,
        n_jobs=-1,
    )

    surrogate.fit(
        train_x,
        train_y,
    )

    candidate_configs = []
    candidate_features = []

    for index in range(pool_size):
        config = random_geometry_config(
            rng,
            name=(
                f"surrogate_pool_"
                f"{index:04d}"
            ),
            seed=seed + 1000 + index,
        )

        candidate_configs.append(
            config
        )

        candidate_features.append(
            config_feature_vector(
                config
            )
        )

    candidate_features = np.asarray(
        candidate_features,
        dtype=np.float64,
    )

    tree_predictions = np.stack(
        [
            tree.predict(
                candidate_features
            )
            for tree
            in surrogate.estimators_
        ],
        axis=0,
    )

    prediction_mean = tree_predictions.mean(
        axis=0
    )

    prediction_std = tree_predictions.std(
        axis=0
    )

    acquisition = (
        prediction_mean
        + 0.15 * prediction_std
    )

    ranking = np.argsort(
        -acquisition
    )

    selected = []
    seen_signatures = set()

    for candidate_index in ranking:
        config = candidate_configs[
            int(candidate_index)
        ]

        signature = config_signature(
            config
        )

        if signature in seen_signatures:
            continue

        seen_signatures.add(
            signature
        )

        selected.append(
            config
        )

        if len(selected) >= number_needed:
            break

    return selected


def unique_top_configs(
    rows,
    top_k,
):
    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    valid_rows.sort(
        key=lambda row: float(
            row.get(
                "select_score",
                -1e9,
            )
        ),
        reverse=True,
    )

    selected_configs = []
    seen_signatures = set()

    for row in valid_rows:
        config = row["config"]
        signature = config_signature(
            config
        )

        if signature in seen_signatures:
            continue

        seen_signatures.add(
            signature
        )

        selected_configs.append(
            config
        )

        if (
            len(selected_configs)
            >= top_k
        ):
            break

    return selected_configs


# ============================================================
# 18. Trial training
# ============================================================

def train_replay_trial(
    model_type,
    model_path,
    rq1_best_model,
    tokenizer,
    id_test_dataset,
    replay_dataset,
    train_raw,
    test_raw,
    trial_name,
    trial_directory,
    epochs,
):
    os.makedirs(
        trial_directory,
        exist_ok=True,
    )

    trial_train_file = os.path.join(
        trial_directory,
        "rq3_ood_train.jsonl",
    )

    trial_test_file = os.path.join(
        trial_directory,
        "rq3_ood_test.jsonl",
    )

    write_jsonl(
        trial_train_file,
        train_raw,
    )

    write_jsonl(
        trial_test_file,
        test_raw,
    )

    set_seed(SEED)

    model = load_rq1_model(
        model_type,
        model_path,
        rq1_best_model,
        DEVICE,
    )

    ood_train_dataset = CodeSearchDataset(
        train_raw,
        tokenizer,
        max_len=MAX_LEN,
        name=f"{trial_name}_ood_train",
    )

    ood_test_dataset = CodeSearchDataset(
        test_raw,
        tokenizer,
        max_len=MAX_LEN,
        name=f"{trial_name}_ood_test",
    )

    if len(ood_train_dataset) == 0:
        raise ValueError(
            "RQ5 OOD training dataset is empty."
        )

    if len(ood_test_dataset) == 0:
        raise ValueError(
            "RQ5 OOD test dataset is empty."
        )

    id_eval_dataset = id_test_dataset

    if (
        MAX_EVAL_SAMPLES is not None
        and len(id_eval_dataset)
        > MAX_EVAL_SAMPLES
    ):
        id_eval_dataset = Subset(
            id_eval_dataset,
            range(MAX_EVAL_SAMPLES),
        )

    ood_eval_dataset = ood_test_dataset

    if (
        MAX_EVAL_SAMPLES is not None
        and len(ood_eval_dataset)
        > MAX_EVAL_SAMPLES
    ):
        ood_eval_dataset = Subset(
            ood_eval_dataset,
            range(MAX_EVAL_SAMPLES),
        )

    ood_train_loader = make_loader(
        ood_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    id_eval_loader = make_loader(
        id_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    ood_eval_loader = make_loader(
        ood_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    replay_loader = make_loader(
        replay_dataset,
        batch_size=REPLAY_BATCH_SIZE,
        shuffle=True,
        seed=REPLAY_BUFFER_SEED,
    )

    replay_iterator = cycle_loader(
        replay_loader
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = max(
        len(ood_train_loader)
        * int(epochs),
        1,
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                total_steps
                * WARMUP_RATIO
            ),
            num_training_steps=total_steps,
        )
    )

    scaler = GradScaler(
        enabled=(
            DEVICE.type == "cuda"
        )
    )

    best_ood_mrr = -1.0
    best_epoch = 0

    best_model_path = os.path.join(
        trial_directory,
        (
            f"best_rq5_{model_type}_"
            f"{safe_name(trial_name)}.bin"
        ),
    )

    epoch_metrics = []

    trial_start = time.time()

    for epoch in range(
        1,
        int(epochs) + 1,
    ):
        epoch_start = time.time()

        loss_statistics = train_replay_epoch(
            model,
            ood_train_loader,
            replay_iterator,
            optimizer,
            scheduler,
            scaler,
            desc=(
                f"[{trial_name}] "
                f"epoch {epoch}/{epochs}"
            ),
        )

        id_evaluation = evaluate_search_metrics(
            model,
            id_eval_loader,
            desc=(
                f"[{trial_name}] "
                f"epoch {epoch} ID"
            ),
        )

        ood_evaluation = evaluate_search_metrics(
            model,
            ood_eval_loader,
            desc=(
                f"[{trial_name}] "
                f"epoch {epoch} OOD"
            ),
        )

        is_best = (
            ood_evaluation["mrr"]
            > best_ood_mrr
        )

        if is_best:
            best_ood_mrr = float(
                ood_evaluation["mrr"]
            )

            best_epoch = int(epoch)

            torch.save(
                model.state_dict(),
                best_model_path,
            )

        epoch_metrics.append({
            "epoch": int(epoch),
            "loss": loss_statistics,
            "id_test": id_evaluation,
            "ood_test": ood_evaluation,
            "is_best": bool(is_best),
            "epoch_seconds": float(
                time.time() - epoch_start
            ),
        })

        print(
            f"{trial_name} | "
            f"epoch={epoch} | "
            f"loss={loss_statistics['total_loss']:.4f} | "
            f"ID_MRR={id_evaluation['mrr']:.4f} | "
            f"OOD_MRR={ood_evaluation['mrr']:.4f} | "
            f"OOD_R@1={ood_evaluation['recall@1']:.4f}"
        )

    if best_epoch == 0:
        best_epoch = int(epochs)

        torch.save(
            model.state_dict(),
            best_model_path,
        )

    model.load_state_dict(
        safe_torch_load(
            best_model_path,
            map_location="cpu",
        )
    )

    id_prediction_path = os.path.join(
        trial_directory,
        "best_id_predictions.jsonl",
    )

    ood_prediction_path = os.path.join(
        trial_directory,
        "best_ood_predictions.jsonl",
    )

    final_id = evaluate_search_metrics(
        model,
        id_eval_loader,
        desc=f"[{trial_name}] final ID",
        save_path=id_prediction_path,
    )

    final_ood = evaluate_search_metrics(
        model,
        ood_eval_loader,
        desc=f"[{trial_name}] final OOD",
        save_path=ood_prediction_path,
    )

    rq4_comparison = None

    if RQ4_REPLAY_OOD_MRR is not None:
        rq4_comparison = {
            "rq4_replay_ood_mrr": float(
                RQ4_REPLAY_OOD_MRR
            ),
            "rq5_minus_rq4_mrr": float(
                final_ood["mrr"]
                - RQ4_REPLAY_OOD_MRR
            ),
            "rq5_better_than_rq4": bool(
                final_ood["mrr"]
                > RQ4_REPLAY_OOD_MRR
            ),
        }

    result = {
        "trial_name": trial_name,
        "model_type": model_type,
        "best_epoch": int(best_epoch),
        "select_score": float(
            final_ood["mrr"]
        ),
        "id_test": final_id,
        "ood_test": final_ood,
        "rq4_comparison": rq4_comparison,
        "epoch_metrics": epoch_metrics,
        "files": {
            "best_model": best_model_path,
            "id_predictions": (
                id_prediction_path
            ),
            "ood_predictions": (
                ood_prediction_path
            ),
            "ood_train_file": (
                trial_train_file
            ),
            "ood_test_file": (
                trial_test_file
            ),
        },
        "time_seconds": float(
            time.time() - trial_start
        ),
        "time_readable": format_time(
            time.time() - trial_start
        ),
    }

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 19. Trial orchestration
# ============================================================

def run_trial(
    model_type,
    model_path,
    rq1_best_model,
    tokenizer,
    id_test_dataset,
    replay_dataset,
    metric_records,
    search_config,
    phase,
    trial_index,
    epochs,
    model_save_directory,
):
    trial_name = (
        f"{phase}_"
        f"{trial_index:03d}_"
        f"{search_config['name']}"
    )

    trial_directory = os.path.join(
        model_save_directory,
        phase,
        safe_name(trial_name),
    )

    (
        train_raw,
        test_raw,
        split_report,
    ) = build_geometry_split(
        metric_records,
        search_config,
    )

    result = train_replay_trial(
        model_type=model_type,
        model_path=model_path,
        rq1_best_model=rq1_best_model,
        tokenizer=tokenizer,
        id_test_dataset=id_test_dataset,
        replay_dataset=replay_dataset,
        train_raw=train_raw,
        test_raw=test_raw,
        trial_name=trial_name,
        trial_directory=trial_directory,
        epochs=epochs,
    )

    return {
        "status": "ok",
        "phase": phase,
        "trial": trial_name,
        "config": search_config,
        "features": config_feature_vector(
            search_config
        ),
        "split_report": split_report,
        "select_score": float(
            result["ood_test"]["mrr"]
        ),
        "id_test": result["id_test"],
        "ood_test": result["ood_test"],
        "rq4_comparison": (
            result["rq4_comparison"]
        ),
        "result": result,
    }


def run_phase(
    model_type,
    model_path,
    rq1_best_model,
    tokenizer,
    id_test_dataset,
    replay_dataset,
    metric_records,
    configurations,
    phase,
    epochs,
    model_save_directory,
):
    rows = []
    seen_test_hashes = {}

    for index, search_config in enumerate(
        configurations,
        start=1,
    ):
        print("\n" + "=" * 120)
        print(
            f"RQ5 {phase}: "
            f"trial {index}/{len(configurations)}"
        )
        print("model:", model_type)
        print("epochs:", epochs)
        print(
            "weights:",
            search_config["weights"],
        )
        print(
            "train preferences:",
            search_config[
                "train_preferences"
            ],
        )
        print("=" * 120)

        try:
            # Generate the split once here to detect duplicate splits.
            (
                preview_train,
                preview_test,
                preview_report,
            ) = build_geometry_split(
                metric_records,
                search_config,
            )

            test_hash = preview_report[
                "test_hash"
            ]

            if test_hash in seen_test_hashes:
                duplicate_of = (
                    seen_test_hashes[
                        test_hash
                    ]
                )

                print(
                    f">>> Duplicate split detected. "
                    f"Same test split as "
                    f"{duplicate_of}. Skipping."
                )

                rows.append({
                    "status": "duplicate",
                    "phase": phase,
                    "trial_index": int(index),
                    "config": search_config,
                    "duplicate_of": duplicate_of,
                    "split_report": (
                        preview_report
                    ),
                })

                continue

            seen_test_hashes[
                test_hash
            ] = search_config["name"]

            row = run_trial(
                model_type=model_type,
                model_path=model_path,
                rq1_best_model=rq1_best_model,
                tokenizer=tokenizer,
                id_test_dataset=id_test_dataset,
                replay_dataset=replay_dataset,
                metric_records=metric_records,
                search_config=search_config,
                phase=phase,
                trial_index=index,
                epochs=epochs,
                model_save_directory=(
                    model_save_directory
                ),
            )

            rows.append(row)

        except Exception as error:
            print(
                f"[Error] {phase} trial "
                f"{index} failed: {repr(error)}"
            )

            rows.append({
                "status": "failed",
                "phase": phase,
                "trial_index": int(index),
                "config": search_config,
                "error": repr(error),
            })

        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return rows


def propose_stage2(
    stage1_rows,
):
    valid_rows = [
        row
        for row in stage1_rows
        if row.get("status") == "ok"
    ]

    valid_rows.sort(
        key=lambda row: float(
            row["select_score"]
        ),
        reverse=True,
    )

    configurations = []

    for row in valid_rows[
        :SMART_STAGE2_FROM_STAGE1
    ]:
        configurations.append(
            row["config"]
        )

    rng = np.random.RandomState(
        SEED + 2025
    )

    for index in range(
        SMART_STAGE2_RANDOM
    ):
        configurations.append(
            random_geometry_config(
                rng,
                name=(
                    f"stage2_random_{index}"
                ),
                seed=(
                    SEED + 300 + index
                ),
            )
        )

    remaining = max(
        0,
        SMART_STAGE2_TRIALS
        - len(configurations),
    )

    configurations.extend(
        fit_surrogate_and_propose(
            stage1_rows,
            number_needed=remaining,
            seed=SEED + 900,
        )
    )

    unique_configs = []
    seen_signatures = set()

    for config in configurations:
        signature = config_signature(
            config
        )

        if signature in seen_signatures:
            continue

        seen_signatures.add(
            signature
        )

        unique_configs.append(
            config
        )

        if (
            len(unique_configs)
            >= SMART_STAGE2_TRIALS
        ):
            break

    return unique_configs


# ============================================================
# 20. Output utilities
# ============================================================

def metric_cache_file(
    model_type,
):
    return os.path.join(
        SAVE_DIR,
        (
            f"rq5_paired_rq3_metrics_"
            f"{LANGUAGE}_{SHIFT_TYPE}_"
            f"{model_type}.json"
        ),
    )


def results_file(
    model_type,
):
    return os.path.join(
        SAVE_DIR,
        (
            f"rq5_split_search_results_"
            f"{LANGUAGE}_{SHIFT_TYPE}_"
            f"{model_type}.json"
        ),
    )


def top_configs_file(
    model_type,
):
    return os.path.join(
        SAVE_DIR,
        (
            f"rq5_split_search_top_configs_"
            f"{LANGUAGE}_{SHIFT_TYPE}_"
            f"{model_type}.json"
        ),
    )


def print_final_result(
    model_type,
    best_result,
):
    print("\n" + "=" * 120)
    print(
        "RQ5 FINAL BEST RESULT"
    )
    print("=" * 120)
    print("Dataset:", f"CodeSearchNet-{LANGUAGE}")
    print("Shift:", SHIFT_TYPE)
    print("Model:", model_type)

    if best_result is None:
        print(
            "No valid RQ5 result was produced."
        )
        print("=" * 120)
        return

    print(
        "Trial:",
        best_result["trial"],
    )

    print(
        "Phase:",
        best_result["phase"],
    )

    print(
        "Best epoch:",
        best_result["result"][
            "best_epoch"
        ],
    )

    print(
        "Weights:",
        best_result["config"][
            "weights"
        ],
    )

    print(
        "Train preferences:",
        best_result["config"][
            "train_preferences"
        ],
    )

    print(
        "Train difficulty mean:",
        f"{best_result['split_report']['train_difficulty_mean']:.6f}",
    )

    print(
        "Test difficulty mean:",
        f"{best_result['split_report']['test_difficulty_mean']:.6f}",
    )

    print("-" * 120)
    print(
        f"{'Metric':<12} | "
        f"{'ID Test':<12} | "
        f"{'OOD Test':<12}"
    )
    print("-" * 120)

    for key, display_name in (
        ("mrr", "MRR"),
        ("recall@1", "Recall@1"),
        ("recall@5", "Recall@5"),
        ("recall@10", "Recall@10"),
    ):
        print(
            f"{display_name:<12} | "
            f"{best_result['id_test'][key]:<12.4f} | "
            f"{best_result['ood_test'][key]:<12.4f}"
        )

    print("-" * 120)

    if RQ4_REPLAY_OOD_MRR is not None:
        rq5_mrr = best_result[
            "ood_test"
        ]["mrr"]

        difference = (
            rq5_mrr
            - RQ4_REPLAY_OOD_MRR
        )

        print(
            "RQ4 Replay OOD MRR:",
            f"{RQ4_REPLAY_OOD_MRR:.4f}",
        )

        print(
            "RQ5 OOD MRR:",
            f"{rq5_mrr:.4f}",
        )

        print(
            "RQ5 - RQ4:",
            f"{difference:+.4f}",
        )

        print(
            "RQ5 better than RQ4:",
            difference > 0,
        )

    else:
        print(
            "RQ4 comparison was not calculated "
            "because RQ4_REPLAY_OOD_MRR=None."
        )

    print("=" * 120)


# ============================================================
# 21. Run one model
# ============================================================

def run_one_model(
    requested_model_type,
):
    set_seed(SEED)

    start_time = time.time()

    model_type = canonical_model_type(
        requested_model_type
    )

    model_path = MODEL_PATHS[
        model_type
    ]

    rq1_best_model = os.path.join(
        RQ1_OUTPUT_DIR,
        (
            f"best_codesearch_"
            f"{LANGUAGE}_{model_type}.bin"
        ),
    )

    model_save_directory = os.path.join(
        SAVE_DIR,
        model_type,
    )

    os.makedirs(
        model_save_directory,
        exist_ok=True,
    )

    print("=" * 120)
    print(
        "CodeSearchNet RQ5 Stage 1:"
    )
    print(
        "RQ3-guided 80/20 OOD Split Search"
    )
    print("=" * 120)
    print("LANGUAGE:", LANGUAGE)
    print("SHIFT_TYPE:", SHIFT_TYPE)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("RQ1 checkpoint:", rq1_best_model)
    print("ID train:", ID_TRAIN_FILE)
    print("ID test:", ID_TEST_FILE)
    print("SAVE_DIR:", model_save_directory)
    print("DEVICE:", DEVICE)
    print(
        "RQ3 metrics:",
        RQ3_METRIC_NAMES,
    )
    print(
        "Fixed RQ4 Replay settings:",
        {
            "replay_lambda": (
                REPLAY_LAMBDA
            ),
            "replay_batch_size": (
                REPLAY_BATCH_SIZE
            ),
            "replay_buffer_size": (
                REPLAY_BUFFER_SIZE
            ),
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
        },
    )
    print("=" * 120)

    if not os.path.exists(
        ID_TRAIN_FILE
    ):
        raise FileNotFoundError(
            ID_TRAIN_FILE
        )

    if not os.path.exists(
        ID_TEST_FILE
    ):
        raise FileNotFoundError(
            ID_TEST_FILE
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    id_train_raw = read_jsonl(
        ID_TRAIN_FILE
    )

    id_test_raw = read_jsonl(
        ID_TEST_FILE
    )

    ood_pool_raw, ood_sources = (
        build_ood_pool(
            OOD_POOL_FILE_CANDIDATES,
            OOD_POOL_FALLBACK_FILES,
        )
    )

    # --------------------------------------------------------
    # Build fixed ID datasets used by all RQ5 trials.
    # --------------------------------------------------------

    id_train_dataset = CodeSearchDataset(
        id_train_raw,
        tokenizer,
        max_len=MAX_LEN,
        name="fixed_id_train",
    )

    id_test_dataset = CodeSearchDataset(
        id_test_raw,
        tokenizer,
        max_len=MAX_LEN,
        name="fixed_id_test",
    )

    if len(id_train_dataset) == 0:
        raise ValueError(
            "ID training dataset is empty."
        )

    if len(id_test_dataset) == 0:
        raise ValueError(
            "ID test dataset is empty."
        )

    (
        replay_dataset,
        replay_indices,
    ) = build_replay_buffer_dataset(
        id_train_dataset,
        buffer_size=REPLAY_BUFFER_SIZE,
        seed=REPLAY_BUFFER_SEED,
    )

    print(
        "Fixed replay buffer size:",
        len(replay_dataset),
    )

    # --------------------------------------------------------
    # Compute paired ID/OOD RQ3 sample metrics.
    # --------------------------------------------------------

    metric_records = (
        compute_rq3_sample_metrics(
            model_type=model_type,
            model_path=model_path,
            rq1_best_model=(
                rq1_best_model
            ),
            tokenizer=tokenizer,
            id_test_raw=id_test_raw,
            ood_pool_raw=ood_pool_raw,
            cache_file=(
                metric_cache_file(
                    model_type
                )
            ),
        )
    )

    print(
        "Number of RQ3-scored OOD samples:",
        len(metric_records),
    )

    # --------------------------------------------------------
    # Stage 1: predefined geometry probes.
    # --------------------------------------------------------

    stage1_configs = (
        initial_geometry_configs(
            SEED
        )[:SMART_STAGE1_TRIALS]
    )

    stage1_rows = run_phase(
        model_type=model_type,
        model_path=model_path,
        rq1_best_model=rq1_best_model,
        tokenizer=tokenizer,
        id_test_dataset=id_test_dataset,
        replay_dataset=replay_dataset,
        metric_records=metric_records,
        configurations=stage1_configs,
        phase="stage1_probe",
        epochs=SMART_STAGE1_EPOCHS,
        model_save_directory=(
            model_save_directory
        ),
    )

    # --------------------------------------------------------
    # Stage 2: surrogate-guided proposals.
    # --------------------------------------------------------

    stage2_configs = propose_stage2(
        stage1_rows
    )

    stage2_rows = run_phase(
        model_type=model_type,
        model_path=model_path,
        rq1_best_model=rq1_best_model,
        tokenizer=tokenizer,
        id_test_dataset=id_test_dataset,
        replay_dataset=replay_dataset,
        metric_records=metric_records,
        configurations=stage2_configs,
        phase="stage2_surrogate",
        epochs=SMART_STAGE2_EPOCHS,
        model_save_directory=(
            model_save_directory
        ),
    )

    # --------------------------------------------------------
    # Stage 3: full training for the top split configurations.
    # --------------------------------------------------------

    all_search_rows = (
        stage1_rows
        + stage2_rows
    )

    final_configs = unique_top_configs(
        all_search_rows,
        SMART_FINAL_TRIALS,
    )

    final_phase_rows = run_phase(
        model_type=model_type,
        model_path=model_path,
        rq1_best_model=rq1_best_model,
        tokenizer=tokenizer,
        id_test_dataset=id_test_dataset,
        replay_dataset=replay_dataset,
        metric_records=metric_records,
        configurations=final_configs,
        phase="stage3_full",
        epochs=SMART_FINAL_EPOCHS,
        model_save_directory=(
            model_save_directory
        ),
    )

    final_valid_rows = [
        row
        for row in final_phase_rows
        if row.get("status") == "ok"
    ]

    fallback_valid_rows = [
        row
        for row in all_search_rows
        if row.get("status") == "ok"
    ]

    candidate_best_rows = (
        final_valid_rows
        if final_valid_rows
        else fallback_valid_rows
    )

    candidate_best_rows.sort(
        key=lambda row: float(
            row["select_score"]
        ),
        reverse=True,
    )

    best_result = (
        candidate_best_rows[0]
        if candidate_best_rows
        else None
    )

    # --------------------------------------------------------
    # Save the best 80/20 split.
    # --------------------------------------------------------

    best_train_file = None
    best_test_file = None
    best_split_report = None

    if best_result is not None:
        (
            best_train_raw,
            best_test_raw,
            best_split_report,
        ) = build_geometry_split(
            metric_records,
            best_result["config"],
        )

        best_train_file = os.path.join(
            SAVE_DIR,
            (
                f"best_rq5_ood_train_"
                f"{LANGUAGE}_{SHIFT_TYPE}_"
                f"{model_type}.jsonl"
            ),
        )

        best_test_file = os.path.join(
            SAVE_DIR,
            (
                f"best_rq5_ood_test_"
                f"{LANGUAGE}_{SHIFT_TYPE}_"
                f"{model_type}.jsonl"
            ),
        )

        write_jsonl(
            best_train_file,
            best_train_raw,
        )

        write_jsonl(
            best_test_file,
            best_test_raw,
        )

    # --------------------------------------------------------
    # Save all results.
    # --------------------------------------------------------

    output = {
        "task": "Code Search",
        "dataset": (
            f"CodeSearchNet-{LANGUAGE}"
        ),
        "method": (
            "RQ5 RQ3-guided 80/20 OOD split "
            "with fixed RQ4 Replay training"
        ),
        "language": LANGUAGE,
        "shift_type": SHIFT_TYPE,
        "model_type": model_type,
        "model_path": model_path,
        "rq1_checkpoint": (
            rq1_best_model
        ),
        "ood_pool_sources": (
            ood_sources
        ),
        "research_control": {
            "only_changed_variable": (
                "OOD 80/20 split strategy"
            ),
            "rq4_split": (
                "random 80/20"
            ),
            "rq5_split": (
                "RQ3-guided 80/20"
            ),
            "continual_learning_method": (
                "fixed Replay"
            ),
        },
        "rq3_sample_metrics": {
            "alignment": (
                "1 - cosine(ID code, OOD code); "
                "higher means stronger representation damage"
            ),
            "entropy": (
                "OOD retrieval entropy minus "
                "ID retrieval entropy"
            ),
            "anisotropy": (
                "OOD local code anisotropy minus "
                "ID local code anisotropy"
            ),
            "rank": (
                "ID local effective rank minus "
                "OOD local effective rank"
            ),
        },
        "fixed_split_ratio": {
            "train": FIXED_TRAIN_RATIO,
            "test": FIXED_TEST_RATIO,
        },
        "fixed_replay_settings": {
            "replay_lambda": (
                REPLAY_LAMBDA
            ),
            "replay_batch_size": (
                REPLAY_BATCH_SIZE
            ),
            "replay_buffer_size": (
                REPLAY_BUFFER_SIZE
            ),
            "replay_buffer_indices": (
                replay_indices
            ),
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": EPOCHS,
            "warmup_ratio": (
                WARMUP_RATIO
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "temperature_scale": (
                TEMPERATURE_SCALE
            ),
        },
        "rq4_replay_ood_mrr": (
            RQ4_REPLAY_OOD_MRR
        ),
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "final_phase_rows": (
            final_phase_rows
        ),
        "best_result": best_result,
        "best_split_report": (
            best_split_report
        ),
        "files": {
            "metric_cache": (
                metric_cache_file(
                    model_type
                )
            ),
            "best_ood_train": (
                best_train_file
            ),
            "best_ood_test": (
                best_test_file
            ),
        },
        "time_seconds": float(
            time.time() - start_time
        ),
        "time_readable": format_time(
            time.time() - start_time
        ),
    }

    save_json(
        results_file(model_type),
        output,
    )

    top_source = (
        final_valid_rows
        if final_valid_rows
        else fallback_valid_rows
    )

    top_source = sorted(
        top_source,
        key=lambda row: float(
            row.get(
                "select_score",
                -1e9,
            )
        ),
        reverse=True,
    )

    save_json(
        top_configs_file(model_type),
        {
            "model_type": model_type,
            "language": LANGUAGE,
            "shift_type": SHIFT_TYPE,
            "search_space_version": (
                RQ5_SEARCH_SPACE_VERSION
            ),
            "source_results_file": (
                results_file(
                    model_type
                )
            ),
            "top_k": (
                SMART_TOP_CONFIGS_K
            ),
            "top_configs": [
                {
                    "rank": index + 1,
                    "config": row[
                        "config"
                    ],
                    "select_score": row[
                        "select_score"
                    ],
                    "id_test": row[
                        "id_test"
                    ],
                    "ood_test": row[
                        "ood_test"
                    ],
                    "split_report": row[
                        "split_report"
                    ],
                    "rq4_comparison": row.get(
                        "rq4_comparison"
                    ),
                }
                for index, row in enumerate(
                    top_source[
                        :SMART_TOP_CONFIGS_K
                    ]
                )
            ],
        },
    )

    print_final_result(
        model_type,
        best_result,
    )

    print(
        "Results saved:",
        results_file(model_type),
    )

    print(
        "Top configurations saved:",
        top_configs_file(model_type),
    )

    if best_train_file:
        print(
            "Best RQ5 OOD train:",
            best_train_file,
        )

    if best_test_file:
        print(
            "Best RQ5 OOD test:",
            best_test_file,
        )

    print(
        "Total time:",
        format_time(
            time.time() - start_time
        ),
    )

    return output


# ============================================================
# 22. Main
# ============================================================

def main():
    models_to_run = (
        MODEL_TYPES_TO_RUN
        if RUN_ALL_MODELS
        else [MODEL_TYPE]
    )

    all_outputs = []

    for requested_model_type in models_to_run:
        try:
            output = run_one_model(
                requested_model_type
            )

            all_outputs.append(
                output
            )

        except Exception as error:
            if not RUN_ALL_MODELS:
                raise

            failure = {
                "status": "failed",
                "model_type": (
                    requested_model_type
                ),
                "error": repr(error),
            }

            all_outputs.append(
                failure
            )

            print("\n" + "!" * 120)
            print(
                f"[Error] RQ5 failed for "
                f"{requested_model_type}: "
                f"{repr(error)}"
            )
            print(
                "Continuing because "
                "RUN_ALL_MODELS=True."
            )
            print("!" * 120)

        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if RUN_ALL_MODELS:
        all_models_file = os.path.join(
            SAVE_DIR,
            (
                f"rq5_all_models_"
                f"{LANGUAGE}_"
                f"{SHIFT_TYPE}.json"
            ),
        )

        save_json(
            all_models_file,
            {
                "task": "Code Search",
                "dataset": (
                    f"CodeSearchNet-{LANGUAGE}"
                ),
                "method": (
                    "RQ5 RQ3-guided OOD "
                    "80/20 split search"
                ),
                "models": models_to_run,
                "results": all_outputs,
            },
        )

        print(
            "All-model results saved:",
            all_models_file,
        )


if __name__ == "__main__":
    main()
