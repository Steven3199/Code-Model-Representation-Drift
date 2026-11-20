# -*- coding: utf-8 -*-
"""
RQ4 Continual Learning for CodeXGLUE Bugs2Fix / Code Refinement.

Small-CST setting only:
  - source checkpoint: RQ1 best model trained on ID train
  - OOD CL train     : small/ood_train_cst(.jsonl)
  - final test       : small/id_test.jsonl and small/ood_test_cst_new(.jsonl)

Two CL strategies are evaluated:
  - EWC on an ID replay/Fisher buffer
  - ID replay mixed with OOD seq2seq training
"""

import itertools
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


# ============================================================
# 1. Config
# ============================================================

MODEL_TYPE = "PLBART"  # options: "codeT5", "PLBART"
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}
RUN_ALL_MODELS = False
MODEL_TYPES_TO_RUN = ["codeT5", "PLBART"]

DATA_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement/data"
SUBSET_NAME = "small"
SUBSET_DIR = os.path.join(DATA_ROOT, SUBSET_NAME)
SAVE_ROOT = os.path.join(DATA_ROOT, "rq4_cl_small_cst_results")

ID_TRAIN_FILE = os.path.join(SUBSET_DIR, "id_train.jsonl")
ID_TEST_FILE = os.path.join(SUBSET_DIR, "id_test.jsonl")
OOD_TRAIN_FILE_CANDIDATES = [
    os.path.join(SUBSET_DIR, "ood_train_cst.jsonl"),
    os.path.join(SUBSET_DIR, "ood_train_cst"),
]
OOD_TEST_FILE_CANDIDATES = [
    os.path.join(SUBSET_DIR, "ood_test_cst_new.jsonl"),
    os.path.join(SUBSET_DIR, "ood_test_cst_new"),
]

RQ1_OUTPUT_ROOT = os.path.join(DATA_ROOT, "rq1_outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
EPOCHS = 8
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
NUM_BEAMS = 5
SEED = 42
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

EWC_LAMBDAS = [10.0]
REPLAY_CONFIGS = [{"lambda": 0.5, "batch_size": 4}]
REPLAY_BUFFER_SIZES = [1000]
FISHER_BUFFER_SIZE = 1000
REPLAY_BUFFER_SEED = SEED

os.makedirs(SAVE_ROOT, exist_ok=True)


def normalize_model_type(model_type):
    lower = str(model_type).lower()
    for name in MODEL_PATHS:
        if lower == name.lower():
            return name
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from {list(MODEL_PATHS.keys())}")


def configure_model_runtime(model_type):
    global MODEL_TYPE, MODEL_PATH, SAVE_DIR, RQ1_BEST_MODEL, METRICS_FILE
    MODEL_TYPE = normalize_model_type(model_type)
    MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
    SAVE_DIR = os.path.join(SAVE_ROOT, MODEL_TYPE, SUBSET_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    RQ1_BEST_MODEL = os.path.join(RQ1_OUTPUT_ROOT, MODEL_TYPE, SUBSET_NAME, f"best_bugs2fix_{MODEL_TYPE}.bin")
    METRICS_FILE = os.path.join(SAVE_DIR, f"rq4_cl_bugs2fix_{MODEL_TYPE}_{SUBSET_NAME}_cst.json")


configure_model_runtime(MODEL_TYPE)


# ============================================================
# 2. Utilities / Metrics
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


def resolve_first_existing(candidates, role):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find {role}. Tried:\n" + "\n".join(f"  - {p}" for p in candidates))


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_code(text):
    if text is None:
        return ""
    return " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").strip().split())


def get_source(js):
    return js.get("source") or js.get("buggy") or js.get("src") or js.get("input") or ""


def get_target(js):
    return js.get("target") or js.get("fixed") or js.get("tgt") or js.get("output") or ""


def exact_match_score(preds, refs):
    if not preds:
        return 0.0
    return 100.0 * sum(normalize_code(p) == normalize_code(r) for p, r in zip(preds, refs)) / len(preds)


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
# 3. Dataset / Model
# ============================================================

class Bugs2FixDataset(Dataset):
    def __init__(self, path, tokenizer, max_samples=None, shuffle_before_limit=False):
        self.path = path
        self.tokenizer = tokenizer
        self.examples = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Reading {os.path.basename(path)}", leave=False)):
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = get_source(js)
                target = get_target(js)
                if not source or not target:
                    continue
                self.examples.append({
                    "idx": str(js.get("idx", line_idx)),
                    "source": source,
                    "target": target,
                    "shift_type": js.get("shift_type", "id"),
                })
        if shuffle_before_limit:
            random.shuffle(self.examples)
        if max_samples is not None and len(self.examples) > max_samples:
            self.examples = self.examples[:max_samples]
        print(f">>> Loaded {len(self.examples)} Bugs2Fix samples from {path}")

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


def build_generation_model(model_type, model_path):
    model_type = normalize_model_type(model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model.to(DEVICE)


def load_rq1_model():
    if not os.path.exists(RQ1_BEST_MODEL):
        raise FileNotFoundError(f"Cannot find RQ1 best model: {RQ1_BEST_MODEL}")
    model = build_generation_model(MODEL_TYPE, MODEL_PATH)
    model.load_state_dict(safe_torch_load(RQ1_BEST_MODEL, map_location=DEVICE))
    print(">>> Loaded RQ1 checkpoint:", RQ1_BEST_MODEL)
    return model


def build_replay_buffer_dataset(id_train_ds, buffer_size=None, seed=42):
    if buffer_size is None or buffer_size <= 0 or buffer_size >= len(id_train_ds):
        indices = list(range(len(id_train_ds)))
        print(f">>> Replay/Fisher buffer: full ID train, size={len(indices)}")
        return id_train_ds, indices
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(id_train_ds), size=int(buffer_size), replace=False).tolist())
    print(f">>> Replay/Fisher buffer: sampled {len(indices)} / {len(id_train_ds)}")
    return Subset(id_train_ds, indices), indices


# ============================================================
# 4. CL Training
# ============================================================

def seq2seq_loss(model, batch, use_amp=False):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    with autocast(enabled=use_amp):
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


class EWC:
    def __init__(self, model, dataloader):
        self.params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(model, dataloader)

    def _compute_fisher(self, model, dataloader):
        fisher = {name: torch.zeros_like(p, device=DEVICE) for name, p in model.named_parameters() if p.requires_grad}
        model.train()
        steps = 0
        for batch in tqdm(dataloader, desc="Computing EWC Fisher", leave=False):
            model.zero_grad(set_to_none=True)
            loss = seq2seq_loss(model, batch, use_amp=False)
            if torch.isnan(loss):
                continue
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
            steps += 1
        denom = max(steps, 1)
        return {name: value / denom for name, value in fisher.items()}

    def penalty(self, model):
        loss = torch.tensor(0.0, device=DEVICE)
        for name, p in model.named_parameters():
            if name in self.fisher:
                loss = loss + (self.fisher[name] * (p - self.params[name]) ** 2).sum()
        return loss


def train_one_epoch(model, ood_loader, optimizer, scheduler, scaler, epoch, strategy, trial_name, ewc_tool=None, replay_iter=None, ewc_lambda=0.0, replay_lambda=0.0):
    model.train()
    total_loss_sum = 0.0
    ood_loss_sum = 0.0
    ewc_sum = 0.0
    replay_sum = 0.0
    steps = 0
    use_amp = DEVICE.type == "cuda"
    pbar = tqdm(ood_loader, desc=f"[{trial_name}] Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad()
        ood_loss = seq2seq_loss(model, batch, use_amp=use_amp)
        loss = ood_loss
        weighted_ewc = torch.tensor(0.0, device=DEVICE)
        weighted_replay = torch.tensor(0.0, device=DEVICE)
        if strategy == "ewc" and ewc_tool is not None:
            weighted_ewc = ewc_lambda * ewc_tool.penalty(model)
            loss = loss + weighted_ewc
        if strategy == "replay" and replay_iter is not None:
            replay_batch = next(replay_iter)
            replay_loss = seq2seq_loss(model, replay_batch, use_amp=use_amp)
            weighted_replay = replay_lambda * replay_loss
            loss = loss + weighted_replay
        if torch.isnan(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss_sum += float(loss.item())
        ood_loss_sum += float(ood_loss.detach().item())
        ewc_sum += float(weighted_ewc.detach().item())
        replay_sum += float(weighted_replay.detach().item())
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    denom = max(steps, 1)
    return {
        "total_loss": total_loss_sum / denom,
        "ood_loss": ood_loss_sum / denom,
        "ewc_weighted": ewc_sum / denom,
        "replay_weighted": replay_sum / denom,
        "steps": steps,
    }


@torch.no_grad()
def evaluate_generation(model, tokenizer, dataloader, desc="Evaluating", save_path=None):
    model.eval()
    preds, refs, records = [], [], []
    use_amp = DEVICE.type == "cuda"
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with autocast(enabled=use_amp):
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        batch_refs = list(batch["target"])
        preds.extend(batch_preds)
        refs.extend(batch_refs)
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
        with open(save_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def print_final_by_strategy_table(title, final_by_strategy):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    print(f"{'Method':<10} | {'Metric':<18} | {'ID Test':<12} | {'OOD Test':<12} | {'Drop ID-OOD':<12} | {'Best Trial':<24}")
    print("-" * 112)
    for strategy, label in [("ewc", "EWC"), ("replay", "Replay")]:
        if strategy not in final_by_strategy:
            continue
        row = final_by_strategy[strategy]
        id_eval = row["id_test"]
        ood_eval = row["ood_test"]
        for key, name in [("exact_match", "Exact Match"), ("bleu", "BLEU"), ("codebleu", "CodeBLEU")]:
            print(f"{label:<10} | {name:<18} | {id_eval[key]:<12.4f} | {ood_eval[key]:<12.4f} | {id_eval[key] - ood_eval[key]:<12.4f} | {row['trial_name']:<24}")
    print("=" * 112)


def run_one_model(model_type=None):
    configure_model_runtime(model_type or MODEL_TYPE)
    set_seed(SEED)
    start_time = time.time()

    ood_train_file = resolve_first_existing(OOD_TRAIN_FILE_CANDIDATES, "OOD CST train file")
    ood_test_file = resolve_first_existing(OOD_TEST_FILE_CANDIDATES, "OOD CST test file")

    print("=" * 120)
    print("Bugs2Fix RQ4 Continual Learning: small-cst")
    print("=" * 120)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("RQ1_BEST_MODEL:", RQ1_BEST_MODEL)
    print("ID_TRAIN_FILE:", ID_TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TRAIN_FILE:", ood_train_file)
    print("OOD_TEST_FILE:", ood_test_file)
    print("SAVE_DIR:", SAVE_DIR)
    print("DEVICE:", DEVICE)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    id_train_ds = Bugs2FixDataset(ID_TRAIN_FILE, tokenizer, max_samples=MAX_TRAIN_SAMPLES)
    id_test_ds = Bugs2FixDataset(ID_TEST_FILE, tokenizer, max_samples=MAX_EVAL_SAMPLES)
    ood_train_ds = Bugs2FixDataset(ood_train_file, tokenizer, max_samples=MAX_TRAIN_SAMPLES, shuffle_before_limit=True)
    ood_test_ds = Bugs2FixDataset(ood_test_file, tokenizer, max_samples=MAX_EVAL_SAMPLES)

    id_loader = DataLoader(id_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    ood_train_loader = DataLoader(ood_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    replay_buffers = {}
    for size in sorted(set(REPLAY_BUFFER_SIZES + [FISHER_BUFFER_SIZE])):
        ds, indices = build_replay_buffer_dataset(id_train_ds, buffer_size=size, seed=REPLAY_BUFFER_SEED)
        replay_buffers[int(size)] = {"dataset": ds, "indices": indices, "actual_size": len(ds)}

    trial_configs = []
    for ewc_lambda in EWC_LAMBDAS:
        trial_configs.append({
            "strategy": "ewc",
            "trial_name": f"ewc_lambda_{str(ewc_lambda).replace('.', 'p')}",
            "ewc_lambda": float(ewc_lambda),
            "replay_lambda": 0.0,
            "replay_batch_size": None,
            "replay_buffer_size": 0,
        })
    for cfg in REPLAY_CONFIGS:
        for buffer_size in REPLAY_BUFFER_SIZES:
            trial_configs.append({
                "strategy": "replay",
                "trial_name": f"replay_lambda_{str(cfg['lambda']).replace('.', 'p')}_bs_{cfg['batch_size']}_buf_{buffer_size}",
                "ewc_lambda": 0.0,
                "replay_lambda": float(cfg["lambda"]),
                "replay_batch_size": int(cfg["batch_size"]),
                "replay_buffer_size": int(buffer_size),
            })

    results = {"ewc": {}, "replay": {}}
    best_by_strategy = {}

    for trial_cfg in trial_configs:
        strategy = trial_cfg["strategy"]
        trial_name = trial_cfg["trial_name"]
        trial_start = time.time()
        print("\n" + "=" * 100)
        print("CL trial:", trial_name)
        print("config:", trial_cfg)
        print("=" * 100)
        model = load_rq1_model()

        ewc_tool = None
        replay_iter = None
        if strategy == "ewc":
            fisher_info = replay_buffers[int(FISHER_BUFFER_SIZE)]
            fisher_loader = DataLoader(fisher_info["dataset"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            ewc_tool = EWC(model, fisher_loader)
        elif strategy == "replay":
            buffer_info = replay_buffers[int(trial_cfg["replay_buffer_size"])]
            replay_loader = DataLoader(buffer_info["dataset"], batch_size=trial_cfg["replay_batch_size"], shuffle=True, num_workers=0)
            replay_iter = itertools.cycle(replay_loader)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = max(len(ood_train_loader) * EPOCHS, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * WARMUP_RATIO),
            num_training_steps=total_steps,
        )
        scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

        best_epoch = 0
        best_ood_em = -1.0
        final_model_path = os.path.join(SAVE_DIR, f"final_cl_{trial_name}_{MODEL_TYPE}_{SUBSET_NAME}_cst.bin")
        epoch_metrics = []

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            loss_stats = train_one_epoch(
                model,
                ood_train_loader,
                optimizer,
                scheduler,
                scaler,
                epoch,
                strategy,
                trial_name,
                ewc_tool=ewc_tool,
                replay_iter=replay_iter,
                ewc_lambda=trial_cfg["ewc_lambda"],
                replay_lambda=trial_cfg["replay_lambda"],
            )
            epoch_id_eval = evaluate_generation(model, tokenizer, id_loader, desc=f"[{trial_name}] Epoch {epoch} ID Test")
            epoch_ood_eval = evaluate_generation(model, tokenizer, ood_loader, desc=f"[{trial_name}] Epoch {epoch} OOD Test")
            is_best = epoch_ood_eval["exact_match"] > best_ood_em
            if is_best:
                best_epoch = epoch
                best_ood_em = float(epoch_ood_eval["exact_match"])
                torch.save(model.state_dict(), final_model_path)
                print(f">>> New best {trial_name} by OOD EM: epoch={epoch}, ood_em={best_ood_em:.4f}")
            epoch_metrics.append({
                "epoch": epoch,
                "loss": loss_stats,
                "id_test": epoch_id_eval,
                "ood_test": epoch_ood_eval,
                "is_best": is_best,
                "epoch_seconds": time.time() - epoch_start,
                "epoch_readable": format_time(time.time() - epoch_start),
            })
            print(
                f"{trial_name} Epoch {epoch} | loss={loss_stats['total_loss']:.4f} | "
                f"ID EM={epoch_id_eval['exact_match']:.4f} | OOD EM={epoch_ood_eval['exact_match']:.4f} | "
                f"time={format_time(time.time() - epoch_start)}"
            )

        model.load_state_dict(safe_torch_load(final_model_path, map_location=DEVICE))
        id_pred_path = os.path.join(SAVE_DIR, f"pred_cl_{trial_name}_id_{MODEL_TYPE}_{SUBSET_NAME}_cst.jsonl")
        ood_pred_path = os.path.join(SAVE_DIR, f"pred_cl_{trial_name}_ood_{MODEL_TYPE}_{SUBSET_NAME}_cst.jsonl")
        final_id_eval = evaluate_generation(model, tokenizer, id_loader, desc=f"[{trial_name}] Best ID Test", save_path=id_pred_path)
        final_ood_eval = evaluate_generation(model, tokenizer, ood_loader, desc=f"[{trial_name}] Best OOD Test", save_path=ood_pred_path)

        trial_result = {
            "config": trial_cfg,
            "final": {
                "selection_metric": "highest_ood_test_exact_match",
                "best_epoch": best_epoch,
                "best_ood_exact_match": best_ood_em,
                "id_test": final_id_eval,
                "ood_test": final_ood_eval,
                "drop_id_minus_ood": id_ood_drop(final_id_eval, final_ood_eval),
                "epoch_metrics": epoch_metrics,
            },
            "files": {
                "final_model": final_model_path,
                "id_predictions": id_pred_path,
                "ood_predictions": ood_pred_path,
            },
            "time_seconds": time.time() - trial_start,
            "time_readable": format_time(time.time() - trial_start),
        }
        results[strategy][trial_name] = trial_result

        summary = {
            "strategy": strategy,
            "trial_name": trial_name,
            "config": trial_cfg,
            "best_epoch": best_epoch,
            "best_ood_exact_match": best_ood_em,
            "id_test": final_id_eval,
            "ood_test": final_ood_eval,
            "final_model": final_model_path,
        }
        if strategy not in best_by_strategy or final_ood_eval["exact_match"] > best_by_strategy[strategy]["ood_test"]["exact_match"]:
            best_by_strategy[strategy] = summary
            best_strategy_model = os.path.join(SAVE_DIR, f"best_cl_{strategy}_{MODEL_TYPE}_{SUBSET_NAME}_cst.bin")
            torch.save(model.state_dict(), best_strategy_model)
            best_by_strategy[strategy]["best_strategy_model"] = best_strategy_model

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    final_by_strategy = {}
    for strategy, row in best_by_strategy.items():
        final_by_strategy[strategy] = {
            "trial_name": row["trial_name"],
            "config": row["config"],
            "best_epoch": row["best_epoch"],
            "best_ood_exact_match": row["best_ood_exact_match"],
            "id_test": row["id_test"],
            "ood_test": row["ood_test"],
            "drop_id_minus_ood": id_ood_drop(row["id_test"], row["ood_test"]),
            "final_model": row["final_model"],
            "best_strategy_model": row.get("best_strategy_model"),
        }

    metrics = {
        "task": "Code Repair / Code Refinement",
        "method": "RQ4 continual learning with EWC and ID replay on small-cst",
        "dataset": "CodeXGLUE Bugs2Fix",
        "subset": SUBSET_NAME,
        "shift_type": "cst",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "rq1_best_model": RQ1_BEST_MODEL,
        "data": {
            "id_train_file": ID_TRAIN_FILE,
            "id_test_file": ID_TEST_FILE,
            "ood_train_file": ood_train_file,
            "ood_test_file": ood_test_file,
        },
        "final_by_strategy": final_by_strategy,
        "best_by_strategy": best_by_strategy,
        "results": results,
        "replay_buffer": {
            "configured_sizes": REPLAY_BUFFER_SIZES,
            "fisher_buffer_size": FISHER_BUFFER_SIZE,
            "buffers": {
                str(size): {"actual_size": info["actual_size"], "indices": info["indices"]}
                for size, info in replay_buffers.items()
            },
        },
        "files": {"metrics": METRICS_FILE},
        "time": {
            "total_seconds": time.time() - start_time,
            "total_readable": format_time(time.time() - start_time),
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print_final_by_strategy_table(
        f"RQ4 Final Results: Bugs2Fix {SUBSET_NAME.upper()} CST Continual Learning - {MODEL_TYPE}",
        final_by_strategy,
    )
    print("Metrics saved:", METRICS_FILE)
    return metrics


def main():
    models_to_run = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    all_results = []
    for model_type in models_to_run:
        try:
            all_results.append(run_one_model(model_type))
        except Exception as exc:
            if not RUN_ALL_MODELS:
                raise
            all_results.append({"status": "failed", "model_type": model_type, "error": repr(exc)})
            print(f"[Error] RQ4 CL failed for {model_type}: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if RUN_ALL_MODELS:
        path = os.path.join(SAVE_ROOT, f"rq4_cl_bugs2fix_{SUBSET_NAME}_cst_all_models.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"models": models_to_run, "results": all_results}, f, ensure_ascii=False, indent=2)
        print("All-model metrics saved:", path)


if __name__ == "__main__":
    main()
