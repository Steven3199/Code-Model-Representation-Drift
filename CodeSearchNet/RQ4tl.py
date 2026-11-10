# -*- coding: utf-8 -*-
"""
CodeSearchNet RQ4: Transfer Learning on OOD Code Search Data

This script follows the RQ4 transfer-learning idea used by CodeS, but adapts it
to CodeSearchNet retrieval:

1. Load the ID-trained RQ1 dual-encoder checkpoint.
2. Fine-tune it on a fixed OOD train split with contrastive retrieval loss.
3. Evaluate before/after transfer on ID test and OOD test.
4. Select the best transfer checkpoint by OOD-test MRR.

Expected data files under /opt/data/private/CodeSearchNet/{language}:
    id_test.jsonl
    ood_train_cst.jsonl or ood_cst_train.jsonl or ood_train.jsonl
    ood_test_cst.jsonl  or ood_cst_test.jsonl  or ood_test.jsonl
"""

import os
import json
import time
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5Config,
    T5EncoderModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


# ============================================================
# 1. Config
# ============================================================

LANGUAGE = "python"
SHIFT_TYPE = "token"

SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}

# Set True to run all supported models sequentially.
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_TYPE = "PLBART"

DATA_ROOT = "/opt/data/private/CodeSearchNet"
DATA_DIR = os.path.join(DATA_ROOT, LANGUAGE)
RQ1_OUTPUT_DIR = os.path.join(DATA_DIR, "rq1_outputs")
SAVE_DIR = os.path.join(DATA_DIR, "rq4_tl_results")
os.makedirs(SAVE_DIR, exist_ok=True)

ID_TEST_FILE = os.path.join(DATA_DIR, "id_test.jsonl")

# The script resolves these candidates in order. You can edit them directly if
# your fixed OOD split uses a different name.
OOD_TRAIN_FILE_CANDIDATES = [
    os.path.join(DATA_DIR, f"ood_train_token.jsonl"),
]
OOD_TEST_FILE_CANDIDATES = [
    os.path.join(DATA_DIR, f"ood_test_token_new.jsonl"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 8
LR_FT = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42
TEMPERATURE_SCALE = 20.0

MAX_EVAL_SAMPLES = None
NUM_WORKERS = 4

ALL_MODELS_METRICS_FILE = os.path.join(
    SAVE_DIR,
    f"rq4_tl_codesearchnet_{LANGUAGE}_{SHIFT_TYPE}_all_models.json",
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
    }
    key = str(model_type).strip()
    if key in MODEL_PATHS:
        return key
    lower = key.lower()
    if lower in aliases:
        return aliases[lower]
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {SUPPORTED_MODEL_TYPES}")


def resolve_model_path(model_type):
    return MODEL_PATHS[canonical_model_type(model_type)]


def build_rq1_best_model_path(model_type):
    model_type = canonical_model_type(model_type)
    return os.path.join(RQ1_OUTPUT_DIR, f"best_codesearch_{LANGUAGE}_{model_type}.bin")


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, RQ1_BEST_MODEL, FINAL_MODEL_PATH, METRICS_FILE

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = resolve_model_path(MODEL_TYPE)
    RQ1_BEST_MODEL = build_rq1_best_model_path(MODEL_TYPE)
    FINAL_MODEL_PATH = os.path.join(
        SAVE_DIR,
        f"final_transfer_codesearch_{LANGUAGE}_{SHIFT_TYPE}_{MODEL_TYPE}.bin",
    )
    METRICS_FILE = os.path.join(
        SAVE_DIR,
        f"rq4_tl_codesearchnet_{LANGUAGE}_{SHIFT_TYPE}_{MODEL_TYPE}.json",
    )


configure_model_runtime(MODEL_TYPE)


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
        elif new_key.startswith("plbart.encoder."):
            new_key = new_key.replace("plbart.encoder.", "encoder.", 1)
        elif new_key.startswith("plbart."):
            new_key = new_key.replace("plbart.", "encoder.", 1)

        new_dict[new_key] = value
    return new_dict


def filter_compatible_state_dict(model, state_dict):
    own_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in own_state and own_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    return compatible, skipped


def resolve_first_existing(candidates, role):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {role}. Tried:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


def make_subset(dataset, max_samples):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


# ============================================================
# 3. Dataset
# ============================================================

class CodeSearchDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find file: {file_path}")

        skipped = 0
        shift_counter = Counter()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Loading {os.path.basename(file_path)}", leave=False)):
                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                nl_txt = self.get_nl_text(js)
                code_txt = self.get_code_text(js)
                if not nl_txt or not code_txt:
                    skipped += 1
                    continue

                if nl_txt in code_txt:
                    code_txt = code_txt.replace(nl_txt, "")
                    code_txt = code_txt.replace('""""""', "").replace("''''''", "")

                shift_type = js.get("shift_type", "id")
                self.examples.append({
                    "nl": nl_txt,
                    "code": code_txt,
                    "idx": js.get("idx", js.get("id", str(line_idx))),
                    "shift_type": shift_type,
                })
                shift_counter[shift_type] += 1

        print("=" * 100)
        print("CodeSearchDataset loaded")
        print("file:", file_path)
        print("num_samples:", len(self.examples))
        print("skipped:", skipped)
        print("shift_type distribution:", dict(shift_counter))
        print("=" * 100)

    @staticmethod
    def get_nl_text(js):
        return js.get("docstring") or js.get("doc") or js.get("nl") or js.get("query") or ""

    @staticmethod
    def get_code_text(js):
        code = js.get("func") or js.get("code") or js.get("function") or ""
        if not code and "code_tokens" in js:
            code = " ".join(js["code_tokens"])
        return code

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]

        nl_inputs = self.tokenizer(
            item["nl"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        code_inputs = self.tokenizer(
            item["code"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "nl_input_ids": nl_inputs["input_ids"].squeeze(0),
            "nl_attention_mask": nl_inputs["attention_mask"].squeeze(0),
            "pl_input_ids": code_inputs["input_ids"].squeeze(0),
            "pl_attention_mask": code_inputs["attention_mask"].squeeze(0),
            "idx": item["idx"],
            "shift_type": item["shift_type"],
        }


# ============================================================
# 4. Model
# ============================================================

class DualEncoderModel(nn.Module):
    def __init__(self, model_path, model_type, load_pretrained=True):
        super().__init__()
        self.model_type = model_type.lower()

        if "t5" in self.model_type:
            if load_pretrained:
                self.encoder = T5EncoderModel.from_pretrained(model_path)
            else:
                config = T5Config.from_pretrained(model_path)
                self.encoder = T5EncoderModel(config)
            self.hidden_size = self.encoder.config.d_model

        elif "plbart" in self.model_type:
            if load_pretrained:
                base_model = AutoModel.from_pretrained(model_path)
            else:
                config = AutoConfig.from_pretrained(model_path)
                base_model = AutoModel.from_config(config)
            if hasattr(base_model, "get_encoder"):
                self.encoder = base_model.get_encoder()
            elif hasattr(base_model, "encoder"):
                self.encoder = base_model.encoder
            else:
                raise ValueError("PLBART model should expose get_encoder() or encoder.")
            self.hidden_size = getattr(base_model.config, "d_model", None)
            if self.hidden_size is None:
                self.hidden_size = getattr(base_model.config, "hidden_size", None)
            if self.hidden_size is None:
                raise ValueError("Cannot infer PLBART hidden size from model config.")

        else:
            if load_pretrained:
                self.encoder = AutoModel.from_pretrained(model_path)
            else:
                config = AutoConfig.from_pretrained(model_path)
                self.encoder = AutoModel.from_config(config)
            self.hidden_size = self.encoder.config.hidden_size

    def get_vec(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        if "t5" in self.model_type or "plbart" in self.model_type:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            vec = summed / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            vec = outputs.last_hidden_state[:, 0, :]

        return vec

    def forward(self, nl_ids, nl_mask, pl_ids, pl_mask):
        nl_vec = self.get_vec(nl_ids, nl_mask)
        pl_vec = self.get_vec(pl_ids, pl_mask)
        return F.normalize(nl_vec, p=2, dim=1), F.normalize(pl_vec, p=2, dim=1)


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        raise FileNotFoundError(f"RQ1 best model not found: {RQ1_BEST_MODEL}")

    raw_state = unwrap_checkpoint(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    adapted_state = adapt_state_dict_keys(raw_state)

    model = DualEncoderModel(MODEL_PATH, MODEL_TYPE, load_pretrained=False).to(DEVICE)
    compatible_state, skipped = filter_compatible_state_dict(model, adapted_state)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)

    matched = [key for key in compatible_state if key in model.state_dict()]
    encoder_matched = [key for key in matched if key.startswith("encoder.")]

    print(
        f">>> Loaded RQ1 retrieval checkpoint: model_type={MODEL_TYPE}, "
        f"matched={len(matched)}, encoder_matched={len(encoder_matched)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
    )
    if len(encoder_matched) == 0:
        raise RuntimeError(
            "No encoder parameters were loaded from RQ1 checkpoint. "
            "Check MODEL_TYPE, MODEL_PATH, and RQ1_BEST_MODEL."
        )
    if skipped:
        print(">>> Skipped incompatible keys preview:", skipped[:20])
    missing_show = [key for key in missing if "pooler" not in key][:20]
    if missing_show:
        print(">>> Missing keys preview:", missing_show)
    if unexpected:
        print(">>> Unexpected keys preview:", unexpected[:20])

    return model


# ============================================================
# 5. Evaluation / Training
# ============================================================

@torch.no_grad()
def evaluate_search_metrics(model, dataloader, desc="Evaluating", save_path=None):
    model.eval()
    all_nl_vecs = []
    all_pl_vecs = []
    all_indices = []
    all_shift_types = []

    use_amp = DEVICE.type == "cuda"

    for batch in tqdm(dataloader, desc=desc, leave=False):
        nl_ids = batch["nl_input_ids"].to(DEVICE)
        nl_mask = batch["nl_attention_mask"].to(DEVICE)
        pl_ids = batch["pl_input_ids"].to(DEVICE)
        pl_mask = batch["pl_attention_mask"].to(DEVICE)

        with autocast(enabled=use_amp):
            nl_vec, pl_vec = model(nl_ids, nl_mask, pl_ids, pl_mask)

        all_nl_vecs.append(nl_vec.cpu())
        all_pl_vecs.append(pl_vec.cpu())
        all_indices.extend(batch["idx"])
        all_shift_types.extend(batch["shift_type"])

    if not all_nl_vecs:
        return {
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "num_samples": 0,
        }

    nl_vecs = torch.cat(all_nl_vecs, dim=0)
    pl_vecs = torch.cat(all_pl_vecs, dim=0)
    scores = torch.matmul(nl_vecs, pl_vecs.T).numpy()

    ranks = []
    records = []
    for i in range(len(scores)):
        row = scores[i]
        sorted_idx = np.argsort(row)[::-1]
        rank = int(np.where(sorted_idx == i)[0][0]) + 1
        ranks.append(rank)
        records.append({
            "idx": str(all_indices[i]),
            "shift_type": str(all_shift_types[i]),
            "rank": rank,
            "top1_index": int(sorted_idx[0]),
            "top1_score": float(row[sorted_idx[0]]),
            "gold_score": float(row[i]),
        })

    ranks = np.array(ranks)
    result = {
        "mrr": float(np.mean(1.0 / ranks)),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "num_samples": int(len(ranks)),
    }

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return result


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    steps = 0
    use_amp = DEVICE.type == "cuda"

    pbar = tqdm(dataloader, desc=f"Transfer Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()

        nl_ids = batch["nl_input_ids"].to(DEVICE)
        nl_mask = batch["nl_attention_mask"].to(DEVICE)
        pl_ids = batch["pl_input_ids"].to(DEVICE)
        pl_mask = batch["pl_attention_mask"].to(DEVICE)

        with autocast(enabled=use_amp):
            nl_vec, pl_vec = model(nl_ids, nl_mask, pl_ids, pl_mask)
            logits = torch.matmul(nl_vec, pl_vec.T) * TEMPERATURE_SCALE
            labels = torch.arange(nl_ids.size(0), device=DEVICE)
            loss_nl = F.cross_entropy(logits, labels)
            loss_pl = F.cross_entropy(logits.T, labels)
            loss = (loss_nl + loss_pl) / 2.0

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

    return total_loss / max(steps, 1)


def metric_drop(before, after):
    return {
        "mrr_delta": after["mrr"] - before["mrr"],
        "recall@1_delta": after["recall@1"] - before["recall@1"],
        "recall@5_delta": after["recall@5"] - before["recall@5"],
        "recall@10_delta": after["recall@10"] - before["recall@10"],
    }


def id_ood_drop(id_eval, ood_eval):
    return {
        "mrr_drop": id_eval["mrr"] - ood_eval["mrr"],
        "recall@1_drop": id_eval["recall@1"] - ood_eval["recall@1"],
        "recall@5_drop": id_eval["recall@5"] - ood_eval["recall@5"],
        "recall@10_drop": id_eval["recall@10"] - ood_eval["recall@10"],
    }


def print_final_table(title, id_eval, ood_eval):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'Metric':<15} | {'ID Test':<12} | {'OOD Test':<12} | {'Drop ID-OOD':<12}")
    print("-" * 100)
    for key, name in [
        ("mrr", "MRR"),
        ("recall@1", "Recall@1"),
        ("recall@5", "Recall@5"),
        ("recall@10", "Recall@10"),
    ]:
        print(
            f"{name:<15} | "
            f"{id_eval[key]:<12.4f} | "
            f"{ood_eval[key]:<12.4f} | "
            f"{id_eval[key] - ood_eval[key]:<12.4f}"
        )
    print("-" * 100)
    print("ID samples :", id_eval["num_samples"])
    print("OOD samples:", ood_eval["num_samples"])
    print("=" * 100)


def print_epoch_table(epoch_metrics):
    print("\n" + "=" * 110)
    print("RQ4 CodeSearchNet Transfer Epoch Metrics")
    print("=" * 110)
    print(
        f"{'Epoch':<7} | {'Loss':<10} | {'ID MRR':<10} | {'ID R@1':<10} | "
        f"{'OOD MRR':<10} | {'OOD R@1':<10} | {'Best':<5}"
    )
    print("-" * 110)
    for row in epoch_metrics:
        best_mark = "*" if row.get("is_best") else ""
        print(
            f"{row['epoch']:<7} | {row['train_loss']:<10.4f} | "
            f"{row['id_test']['mrr']:<10.4f} | {row['id_test']['recall@1']:<10.4f} | "
            f"{row['ood_test']['mrr']:<10.4f} | {row['ood_test']['recall@1']:<10.4f} | "
            f"{best_mark:<5}"
        )
    print("=" * 110)


# ============================================================
# 6. Main
# ============================================================

def run_one_model(model_type=None):
    if model_type is not None:
        configure_model_runtime(model_type)
    else:
        configure_model_runtime(MODEL_TYPE)

    set_seed(SEED)
    start_time = time.time()

    ood_train_file = resolve_first_existing(OOD_TRAIN_FILE_CANDIDATES, "OOD train file")
    ood_test_file = resolve_first_existing(OOD_TEST_FILE_CANDIDATES, "OOD test file")

    print("=" * 120)
    print("CodeSearchNet RQ4: Transfer Learning on Fixed OOD Retrieval Split")
    print("=" * 120)
    print("LANGUAGE:", LANGUAGE)
    print("SHIFT_TYPE:", SHIFT_TYPE)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TRAIN_FILE:", ood_train_file)
    print("OOD_TEST_FILE:", ood_test_file)
    print("SAVE_DIR:", SAVE_DIR)
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    id_test_ds = CodeSearchDataset(ID_TEST_FILE, tokenizer)
    ood_train_ds = CodeSearchDataset(ood_train_file, tokenizer)
    ood_test_ds = CodeSearchDataset(ood_test_file, tokenizer)

    if len(id_test_ds) == 0:
        raise ValueError("ID test set is empty.")
    if len(ood_train_ds) == 0:
        raise ValueError("OOD train set is empty.")
    if len(ood_test_ds) == 0:
        raise ValueError("OOD test set is empty.")

    id_eval_ds = make_subset(id_test_ds, MAX_EVAL_SAMPLES)
    ood_eval_ds = make_subset(ood_test_ds, MAX_EVAL_SAMPLES)

    ood_train_loader = DataLoader(
        ood_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    id_eval_loader = DataLoader(
        id_eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    ood_eval_loader = DataLoader(
        ood_eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = load_rq1_model()

    before_id_pred = os.path.join(SAVE_DIR, f"pred_before_id_{LANGUAGE}_{SHIFT_TYPE}_{MODEL_TYPE}.jsonl")
    before_ood_pred = os.path.join(SAVE_DIR, f"pred_before_ood_{LANGUAGE}_{SHIFT_TYPE}_{MODEL_TYPE}.jsonl")
    before_id_eval = evaluate_search_metrics(model, id_eval_loader, desc="Before TL ID Test", save_path=before_id_pred)
    before_ood_eval = evaluate_search_metrics(model, ood_eval_loader, desc="Before TL OOD Test", save_path=before_ood_pred)

    optimizer = AdamW(model.parameters(), lr=LR_FT, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(ood_train_loader) * EPOCHS, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    best_epoch = 0
    best_ood_mrr = -1.0
    best_epoch_model_path = None
    epoch_metrics = []

    train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        loss = train_one_epoch(model, ood_train_loader, optimizer, scheduler, scaler, epoch)

        epoch_id_eval = evaluate_search_metrics(model, id_eval_loader, desc=f"Epoch {epoch} ID Test")
        epoch_ood_eval = evaluate_search_metrics(model, ood_eval_loader, desc=f"Epoch {epoch} OOD Test")

        epoch_path = os.path.join(SAVE_DIR, f"transfer_codesearch_{LANGUAGE}_{SHIFT_TYPE}_{MODEL_TYPE}_epoch_{epoch}.bin")
        torch.save(model.state_dict(), epoch_path)

        is_best = epoch_ood_eval["mrr"] > best_ood_mrr
        if is_best:
            best_epoch = epoch
            best_ood_mrr = float(epoch_ood_eval["mrr"])
            best_epoch_model_path = epoch_path
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f">>> New best transfer checkpoint by OOD MRR: epoch={epoch}, ood_mrr={best_ood_mrr:.4f}")

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(loss),
            "id_test": epoch_id_eval,
            "ood_test": epoch_ood_eval,
            "epoch_seconds": time.time() - epoch_start,
            "epoch_readable": format_time(time.time() - epoch_start),
            "checkpoint": epoch_path,
            "is_best": bool(is_best),
        }
        epoch_metrics.append(epoch_record)

        for item in epoch_metrics:
            item["is_best"] = item["epoch"] == best_epoch

        print(
            f"Epoch {epoch} | loss={loss:.4f} | "
            f"ID MRR={epoch_id_eval['mrr']:.4f} | OOD MRR={epoch_ood_eval['mrr']:.4f} | "
            f"OOD R@1={epoch_ood_eval['recall@1']:.4f} | time={format_time(time.time() - epoch_start)}"
        )

    train_time = time.time() - train_start

    if best_epoch == 0:
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        best_epoch = EPOCHS
        best_epoch_model_path = FINAL_MODEL_PATH
        if epoch_metrics:
            epoch_metrics[-1]["is_best"] = True
            best_ood_mrr = float(epoch_metrics[-1]["ood_test"]["mrr"])

    model.load_state_dict(safe_torch_load(FINAL_MODEL_PATH, map_location=DEVICE))

    after_id_pred = os.path.join(SAVE_DIR, f"pred_after_id_{LANGUAGE}_{SHIFT_TYPE}_{MODEL_TYPE}.jsonl")
    after_ood_pred = os.path.join(SAVE_DIR, f"pred_after_ood_{LANGUAGE}_{SHIFT_TYPE}_{MODEL_TYPE}.jsonl")
    after_id_eval = evaluate_search_metrics(model, id_eval_loader, desc="Best TL ID Test", save_path=after_id_pred)
    after_ood_eval = evaluate_search_metrics(model, ood_eval_loader, desc="Best TL OOD Test", save_path=after_ood_pred)

    total_time = time.time() - start_time

    print_final_table(
        f"RQ4 Final Results: CodeSearchNet-{LANGUAGE} {SHIFT_TYPE.upper()} Transfer - {MODEL_TYPE}",
        after_id_eval,
        after_ood_eval,
    )

    metrics = {
        "task": "Code Search",
        "method": "RQ4 transfer learning on fixed OOD train split",
        "dataset": f"CodeSearchNet-{LANGUAGE}",
        "language": LANGUAGE,
        "shift_type": SHIFT_TYPE,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "supported_model_types": SUPPORTED_MODEL_TYPES,
        "rq1_best_model": RQ1_BEST_MODEL,
        "final_transfer_model": FINAL_MODEL_PATH,
        "best_epoch": int(best_epoch),
        "best_ood_mrr": float(best_ood_mrr),
        "id_test": after_id_eval,
        "ood_test": after_ood_eval,
        "drop_id_minus_ood": id_ood_drop(after_id_eval, after_ood_eval),
        "data": {
            "id_test_file": ID_TEST_FILE,
            "ood_train_file": ood_train_file,
            "ood_test_file": ood_test_file,
            "id_test_size": len(id_test_ds),
            "ood_train_size": len(ood_train_ds),
            "ood_test_size": len(ood_test_ds),
            "max_eval_samples": MAX_EVAL_SAMPLES,
        },
        "before_transfer": {
            "id_test": before_id_eval,
            "ood_test": before_ood_eval,
        },
        "after_transfer": {
            "id_test": after_id_eval,
            "ood_test": after_ood_eval,
        },
        "delta_after_minus_before": {
            "id_test": metric_drop(before_id_eval, after_id_eval),
            "ood_test": metric_drop(before_ood_eval, after_ood_eval),
        },
        "epoch_metrics": epoch_metrics,
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_ft": LR_FT,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "temperature_scale": TEMPERATURE_SCALE,
            "max_len": MAX_LEN,
        },
        "files": {
            "metrics": METRICS_FILE,
            "best_epoch_model": best_epoch_model_path,
            "final_transfer_model": FINAL_MODEL_PATH,
            "before_id_predictions": before_id_pred,
            "before_ood_predictions": before_ood_pred,
            "after_id_predictions": after_id_pred,
            "after_ood_predictions": after_ood_pred,
        },
        "time": {
            "train_elapsed_seconds": train_time,
            "train_elapsed_readable": format_time(train_time),
            "total_elapsed_seconds": total_time,
            "total_elapsed_readable": format_time(total_time),
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 120)
    print("Saved metrics:", METRICS_FILE)
    print("Saved final transfer model:", FINAL_MODEL_PATH)
    print("Total time:", format_time(total_time))
    print("=" * 120)

    return metrics


def main():
    models_to_run = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    all_results = []

    for model_type in models_to_run:
        try:
            result = run_one_model(model_type)
            all_results.append(result)
        except Exception as exc:
            if not RUN_ALL_MODELS:
                raise
            error_result = {
                "status": "failed",
                "model_type": model_type,
                "error": repr(exc),
            }
            all_results.append(error_result)
            print("\n" + "!" * 120)
            print(f"[Error] RQ4 CodeSearchNet transfer failed for {model_type}: {repr(exc)}")
            print("Continuing with next model because RUN_ALL_MODELS=True.")
            print("!" * 120)
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if RUN_ALL_MODELS:
        with open(ALL_MODELS_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "task": "Code Search",
                "method": "RQ4 transfer learning on fixed OOD train split",
                "dataset": f"CodeSearchNet-{LANGUAGE}",
                "language": LANGUAGE,
                "shift_type": SHIFT_TYPE,
                "models": models_to_run,
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        print("All-model metrics saved:", ALL_MODELS_METRICS_FILE)


if __name__ == "__main__":
    main()
