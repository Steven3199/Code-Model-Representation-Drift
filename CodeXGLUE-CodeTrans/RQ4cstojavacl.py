# -*- coding: utf-8 -*-
"""
CodeXGLUE CodeTrans RQ4 Continual Learning.

ID direction from RQ1:
    C# -> Java

OOD language-direction shift:
    Java -> C#

RQ4 CL protocol:
    load RQ1 best checkpoint
    -> train on randomly split OOD train with EWC or ID replay
    -> evaluate ID test and OOD test_new
"""

import argparse
import itertools
import json
import os
import random
import time

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


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
OOD_SPLIT_DIR = os.path.join(DATA_DIR, "ood_split_java_to_cs_80_20")
RQ1_SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq1_language_shift_outputs_cs_to_java_id"
SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/code-to-code-trans/rq4_cl_language_shift_outputs_cs_to_java_id"

ID_TEST_SRC_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.cs")
ID_TEST_TGT_FILE = os.path.join(DATA_DIR, "test.java-cs.txt.java")

OOD_TRAIN_SRC_FILE = os.path.join(OOD_SPLIT_DIR, "ood_train.java-cs.txt.java")
OOD_TRAIN_TGT_FILE = os.path.join(OOD_SPLIT_DIR, "ood_train.java-cs.txt.cs")
OOD_TEST_SRC_FILE = os.path.join(OOD_SPLIT_DIR, "ood_test.java-cs.txt.java")
OOD_TEST_TGT_FILE = os.path.join(OOD_SPLIT_DIR, "ood_test.java-cs.txt.cs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 8
LR = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
NUM_BEAMS = 5
SEED = 42
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
SOURCE_PREFIX = "translate code:"
ENABLE_CODEBLEU = True
ID_CODEBLEU_LANG = "java"
OOD_CODEBLEU_LANG = "c_sharp"

EWC_LAMBDAS = [10.0]
REPLAY_CONFIGS = [{"lambda": 0.5, "batch_size": 4}]
REPLAY_BUFFER_SIZES = [1000]
FISHER_BUFFER_SIZE = 1000
REPLAY_BUFFER_SEED = SEED

os.makedirs(SAVE_DIR, exist_ok=True)


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
    aliases = {"codet5": "codeT5", "codet5": "codeT5", "plbart": "PLBART"}
    if model_type in MODEL_PATHS:
        return model_type
    if lower in aliases:
        return aliases[lower]
    raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Choose from codeT5, PLBART, all")


def resolve_model_path(model_type, model_path=None):
    return model_path or MODEL_PATHS[normalize_model_type(model_type)]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="CodeXGLUE CodeTrans RQ4 Continual Learning")
    parser.add_argument("--model_type", default=MODEL_TYPE, help="codeT5, PLBART, or all")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--codeT5_path", default=MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=MODEL_PATHS["PLBART"])
    return parser


def get_rq1_best_model(model_type):
    return os.path.join(
        RQ1_SAVE_DIR,
        model_type,
        f"best_codetrans_cs_to_java_id_{model_type}.bin",
    )


def get_output_paths(model_type):
    run_save_dir = os.path.join(SAVE_DIR, model_type)
    os.makedirs(run_save_dir, exist_ok=True)
    prefix = f"rq4_cl_codetrans_cs_to_java_id_java_to_cs_ood_{model_type}"
    return {
        "save_dir": run_save_dir,
        "metrics": os.path.join(run_save_dir, f"{prefix}_metrics.json"),
    }


def normalize_code(text):
    return " ".join(str(text).strip().split())


def read_parallel_files(src_path, tgt_path, max_samples=None):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Target file not found: {tgt_path}")
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        src_lines = [line.rstrip("\n") for line in f]
    with open(tgt_path, "r", encoding="utf-8", errors="ignore") as f:
        tgt_lines = [line.rstrip("\n") for line in f]
    n = min(len(src_lines), len(tgt_lines))
    if max_samples is not None:
        n = min(n, int(max_samples))
    data = []
    for i in range(n):
        src = src_lines[i].strip()
        tgt = tgt_lines[i].strip()
        if not src or not tgt:
            continue
        data.append({"idx": str(i), "source": src, "target": tgt})
    print(f">>> Loaded {len(data)} pairs: {src_path} -> {tgt_path}")
    return data


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


# ============================================================
# 3. Metrics
# ============================================================

def exact_match(preds, refs):
    if not preds:
        return 0.0
    hits = sum(1 for pred, ref in zip(preds, refs) if normalize_code(pred) == normalize_code(ref))
    return 100.0 * hits / len(preds)


def compute_bleu(preds, refs):
    if not preds:
        return 0.0
    smoothie = SmoothingFunction().method4
    references = [[normalize_code(ref).split()] for ref in refs]
    hypotheses = [normalize_code(pred).split() for pred in preds]
    try:
        return float(corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100.0)
    except Exception as exc:
        print(f"[BLEU Warning] failed: {repr(exc)}")
        return 0.0


def _normalise_codebleu_score(raw_score):
    if raw_score is None:
        return None
    score = float(raw_score)
    return score * 100.0 if score <= 1.0 else score


def compute_codebleu_with_package(preds, refs, lang):
    if not ENABLE_CODEBLEU or not preds or len(preds) != len(refs):
        return None
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(
            references=refs,
            predictions=preds,
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None,
        )
        if isinstance(result, dict):
            score = result.get("codebleu", result.get("CodeBLEU"))
            return _normalise_codebleu_score(score)
        if isinstance(result, (int, float)):
            return _normalise_codebleu_score(result)
    except Exception as exc:
        print(f"[CodeBLEU Warning] failed for lang={lang}: {repr(exc)}")
    return None


def compute_codebleu_safe(preds, refs, lang):
    if not ENABLE_CODEBLEU:
        return None
    lang = str(lang).strip()
    if lang.lower() in {"c_sharp", "csharp", "cs", "c#"}:
        candidates = [lang, "c_sharp", "csharp", "cs"]
    elif lang.lower() == "java":
        candidates = ["java"]
    else:
        candidates = [lang]
    seen = set()
    for candidate in [x for x in candidates if not (x in seen or seen.add(x))]:
        score = compute_codebleu_with_package(preds, refs, candidate)
        if score is not None:
            return score
    return None


def compute_generation_metrics(preds, refs, lang):
    codebleu = compute_codebleu_safe(preds, refs, lang)
    return {
        "exact_match": float(exact_match(preds, refs)),
        "bleu": float(compute_bleu(preds, refs)),
        "codebleu": None if codebleu is None else float(codebleu),
        "num_samples": len(preds),
    }


def id_ood_drop(id_eval, ood_eval):
    return {
        "exact_match_drop": id_eval["exact_match"] - ood_eval["exact_match"],
        "bleu_drop": id_eval["bleu"] - ood_eval["bleu"],
        "codebleu_drop": None if id_eval["codebleu"] is None or ood_eval["codebleu"] is None else id_eval["codebleu"] - ood_eval["codebleu"],
    }


def fmt_metric(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


# ============================================================
# 4. Dataset / Model
# ============================================================

class CodeTransDataset(Dataset):
    def __init__(self, data, tokenizer, source_prefix=SOURCE_PREFIX):
        self.examples = []
        for item in data:
            src_text = f"{source_prefix} {item['source']}"
            source_enc = tokenizer(
                src_text,
                max_length=MAX_SOURCE_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            target_enc = tokenizer(
                item["target"],
                max_length=MAX_TARGET_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = target_enc["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100
            self.examples.append({
                "idx": item["idx"],
                "source": item["source"],
                "target": item["target"],
                "input_ids": source_enc["input_ids"].squeeze(0),
                "attention_mask": source_enc["attention_mask"].squeeze(0),
                "labels": labels,
            })
        print(f">>> Built CodeTransDataset: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def load_rq1_model(model_type, model_path, tokenizer):
    best_model_path = get_rq1_best_model(model_type)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"RQ1 best model not found: {best_model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    configure_special_tokens(tokenizer, model)
    model.load_state_dict(safe_torch_load(best_model_path, map_location=DEVICE))
    print(">>> Loaded RQ1 checkpoint:", best_model_path)
    return model, best_model_path


# ============================================================
# 5. Train / Eval
# ============================================================

def build_replay_buffer_dataset(id_train_ds, buffer_size=None, seed=42):
    if buffer_size is None or buffer_size <= 0 or buffer_size >= len(id_train_ds):
        indices = list(range(len(id_train_ds)))
        print(f">>> Replay/Fisher buffer: full ID train, size={len(indices)}")
        return id_train_ds, indices
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(id_train_ds), size=int(buffer_size), replace=False).tolist())
    print(f">>> Replay/Fisher buffer: sampled {len(indices)} / {len(id_train_ds)}")
    from torch.utils.data import Subset
    return Subset(id_train_ds, indices), indices


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
    use_amp = DEVICE.type == "cuda"
    total_loss_sum = 0.0
    ood_loss_sum = 0.0
    ewc_sum = 0.0
    replay_sum = 0.0
    steps = 0
    pbar = tqdm(ood_loader, desc=f"[{trial_name}] Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
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
    }


@torch.no_grad()
def generate_predictions(model, dataloader, tokenizer, desc="Generating", save_path=None):
    model.eval()
    preds, refs, records = [], [], []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
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
                "source": batch["source"][i],
                "reference": batch_refs[i],
                "prediction": pred,
                "exact_match": normalize_code(pred) == normalize_code(batch_refs[i]),
            })
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return preds, refs


def evaluate_generation(model, dataloader, tokenizer, lang, desc, save_path=None):
    preds, refs = generate_predictions(model, dataloader, tokenizer, desc=desc, save_path=save_path)
    return compute_generation_metrics(preds, refs, lang)


def print_final_table(title, id_eval, ood_eval):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'Metric':<18} | {'ID Test':<12} | {'OOD Test':<12} | {'Drop ID-OOD':<12}")
    print("-" * 110)
    for key, name in [("exact_match", "Exact Match"), ("bleu", "BLEU"), ("codebleu", "CodeBLEU")]:
        id_value = id_eval.get(key)
        ood_value = ood_eval.get(key)
        drop = None if id_value is None or ood_value is None else id_value - ood_value
        print(f"{name:<18} | {fmt_metric(id_value):<12} | {fmt_metric(ood_value):<12} | {fmt_metric(drop):<12}")
    print("-" * 110)
    print("ID samples :", id_eval["num_samples"])
    print("OOD samples:", ood_eval["num_samples"])
    print("=" * 110)


def print_final_by_strategy_table(title, final_by_strategy):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(f"{'Strategy':<10} | {'Metric':<18} | {'ID Test':<12} | {'OOD Test':<12} | {'Drop ID-OOD':<12}")
    print("-" * 120)
    for strategy in ["ewc", "replay"]:
        if strategy not in final_by_strategy:
            continue
        row = final_by_strategy[strategy]
        id_eval = row["id_test"]
        ood_eval = row["ood_test"]
        drop = row["drop_id_minus_ood"]
        for metric_key, display_name, drop_key in [
            ("exact_match", "Exact Match", "exact_match_drop"),
            ("bleu", "BLEU", "bleu_drop"),
            ("codebleu", "CodeBLEU", "codebleu_drop"),
        ]:
            print(
                f"{strategy:<10} | {display_name:<18} | "
                f"{fmt_metric(id_eval.get(metric_key)):<12} | "
                f"{fmt_metric(ood_eval.get(metric_key)):<12} | "
                f"{fmt_metric(drop.get(drop_key)):<12}"
            )
    print("=" * 120)


# ============================================================
# 6. Run
# ============================================================

def run_experiment(model_type, model_path):
    set_seed(SEED)
    model_type = normalize_model_type(model_type)
    model_path = resolve_model_path(model_type, model_path)
    output_paths = get_output_paths(model_type)
    start_time = time.time()

    print("=" * 120)
    print("CodeXGLUE-CodeTrans RQ4 Continual Learning: C#->Java ID, Java->C# OOD split")
    print("=" * 120)
    print("MODEL_TYPE:", model_type)
    print("MODEL_PATH:", model_path)
    print("RQ1_BEST_MODEL:", get_rq1_best_model(model_type))
    print("ID Train:", os.path.join(DATA_DIR, "train.java-cs.txt.cs"), "->", os.path.join(DATA_DIR, "train.java-cs.txt.java"))
    print("ID Test :", ID_TEST_SRC_FILE, "->", ID_TEST_TGT_FILE)
    print("OOD Train:", OOD_TRAIN_SRC_FILE, "->", OOD_TRAIN_TGT_FILE)
    print("OOD Test :", OOD_TEST_SRC_FILE, "->", OOD_TEST_TGT_FILE)
    print("SAVE_DIR:", output_paths["save_dir"])
    print("DEVICE:", DEVICE)
    print("EWC_LAMBDAS:", EWC_LAMBDAS)
    print("REPLAY_CONFIGS:", REPLAY_CONFIGS)
    print("REPLAY_BUFFER_SIZES:", REPLAY_BUFFER_SIZES)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id_train_data = read_parallel_files(
        os.path.join(DATA_DIR, "train.java-cs.txt.cs"),
        os.path.join(DATA_DIR, "train.java-cs.txt.java"),
        max_samples=MAX_TRAIN_SAMPLES,
    )
    id_test_data = read_parallel_files(ID_TEST_SRC_FILE, ID_TEST_TGT_FILE, max_samples=MAX_EVAL_SAMPLES)
    ood_train_data = read_parallel_files(OOD_TRAIN_SRC_FILE, OOD_TRAIN_TGT_FILE, max_samples=MAX_TRAIN_SAMPLES)
    ood_test_data = read_parallel_files(OOD_TEST_SRC_FILE, OOD_TEST_TGT_FILE, max_samples=MAX_EVAL_SAMPLES)

    id_train_ds = CodeTransDataset(id_train_data, tokenizer)
    id_test_ds = CodeTransDataset(id_test_data, tokenizer)
    ood_train_ds = CodeTransDataset(ood_train_data, tokenizer)
    ood_test_ds = CodeTransDataset(ood_test_data, tokenizer)
    id_loader = DataLoader(id_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    ood_train_loader = DataLoader(ood_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    ood_loader = DataLoader(ood_test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    replay_buffers = {}
    for size in sorted(set(REPLAY_BUFFER_SIZES + [FISHER_BUFFER_SIZE])):
        buffer_ds, indices = build_replay_buffer_dataset(id_train_ds, buffer_size=size, seed=REPLAY_BUFFER_SEED)
        replay_buffers[int(size)] = {"dataset": buffer_ds, "indices": indices, "actual_size": len(buffer_ds)}

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
    rq1_best_model = get_rq1_best_model(model_type)

    for trial_cfg in trial_configs:
        strategy = trial_cfg["strategy"]
        trial_name = trial_cfg["trial_name"]
        trial_start = time.time()
        print("\n" + "=" * 110)
        print("CL trial:", trial_name)
        print("config:", trial_cfg)
        print("=" * 110)
        model, rq1_best_model = load_rq1_model(model_type, model_path, tokenizer)

        ewc_tool = None
        replay_iter = None
        if strategy == "ewc":
            fisher_info = replay_buffers[int(FISHER_BUFFER_SIZE)]
            fisher_loader = DataLoader(fisher_info["dataset"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            ewc_tool = EWC(model, fisher_loader)
        elif strategy == "replay":
            buffer_info = replay_buffers[int(trial_cfg["replay_buffer_size"])]
            replay_loader = DataLoader(buffer_info["dataset"], batch_size=trial_cfg["replay_batch_size"], shuffle=True, num_workers=NUM_WORKERS)
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
        final_model_path = os.path.join(output_paths["save_dir"], f"final_cl_{trial_name}_{model_type}.bin")
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
            id_eval_epoch = evaluate_generation(model, id_loader, tokenizer, ID_CODEBLEU_LANG, f"[{trial_name}] Epoch {epoch} ID Test")
            ood_eval_epoch = evaluate_generation(model, ood_loader, tokenizer, OOD_CODEBLEU_LANG, f"[{trial_name}] Epoch {epoch} OOD Test")
            is_best = ood_eval_epoch["exact_match"] > best_ood_em
            if is_best:
                best_epoch = epoch
                best_ood_em = float(ood_eval_epoch["exact_match"])
                torch.save(model.state_dict(), final_model_path)
                print(f">>> New best {trial_name} by OOD EM: epoch={epoch}, ood_em={best_ood_em:.4f}")
            epoch_metrics.append({
                "epoch": epoch,
                "loss": loss_stats,
                "id_test": id_eval_epoch,
                "ood_test": ood_eval_epoch,
                "is_best": is_best,
                "epoch_seconds": time.time() - epoch_start,
                "epoch_readable": format_time(time.time() - epoch_start),
            })
            print(
                f"{trial_name} Epoch {epoch} | loss={loss_stats['total_loss']:.4f} | "
                f"ID EM={id_eval_epoch['exact_match']:.4f} | OOD EM={ood_eval_epoch['exact_match']:.4f} | "
                f"time={format_time(time.time() - epoch_start)}"
            )

        model.load_state_dict(safe_torch_load(final_model_path, map_location=DEVICE))
        id_pred_path = os.path.join(output_paths["save_dir"], f"pred_cl_{trial_name}_id_test_cs_to_java_{model_type}.jsonl")
        ood_pred_path = os.path.join(output_paths["save_dir"], f"pred_cl_{trial_name}_ood_test_java_to_cs_{model_type}.jsonl")
        id_eval = evaluate_generation(model, id_loader, tokenizer, ID_CODEBLEU_LANG, f"[{trial_name}] Best ID Test", id_pred_path)
        ood_eval = evaluate_generation(model, ood_loader, tokenizer, OOD_CODEBLEU_LANG, f"[{trial_name}] Best OOD Test", ood_pred_path)

        trial_result = {
            "config": trial_cfg,
            "final": {
                "selection_metric": "highest_ood_test_exact_match",
                "best_epoch": best_epoch,
                "best_ood_exact_match": best_ood_em,
                "id_test": id_eval,
                "ood_test": ood_eval,
                "drop_id_minus_ood": id_ood_drop(id_eval, ood_eval),
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
            "id_test": id_eval,
            "ood_test": ood_eval,
            "final_model": final_model_path,
        }
        if strategy not in best_by_strategy or ood_eval["exact_match"] > best_by_strategy[strategy]["ood_test"]["exact_match"]:
            best_by_strategy[strategy] = summary
            best_strategy_model = os.path.join(output_paths["save_dir"], f"best_cl_{strategy}_{model_type}.bin")
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
        "task": "Code-to-Code Translation",
        "dataset": "CodeXGLUE-CodeTrans",
        "method": "RQ4 continual learning with EWC and ID replay on random 80/20 Java->C# OOD split",
        "model_type": model_type,
        "model_path": model_path,
        "rq1_best_model": rq1_best_model,
        "id_direction": "C# -> Java",
        "ood_direction": "Java -> C#",
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
        "files": {
            "metrics": output_paths["metrics"],
            "id_train_src": os.path.join(DATA_DIR, "train.java-cs.txt.cs"),
            "id_train_tgt": os.path.join(DATA_DIR, "train.java-cs.txt.java"),
            "id_test_src": ID_TEST_SRC_FILE,
            "id_test_tgt": ID_TEST_TGT_FILE,
            "ood_train_src": OOD_TRAIN_SRC_FILE,
            "ood_train_tgt": OOD_TRAIN_TGT_FILE,
            "ood_test_src": OOD_TEST_SRC_FILE,
            "ood_test_tgt": OOD_TEST_TGT_FILE,
        },
        "time": {
            "total_seconds": time.time() - start_time,
            "total_readable": format_time(time.time() - start_time),
        },
    }
    with open(output_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print_final_by_strategy_table(f"RQ4 CL Final Results: CodeTrans {model_type}", final_by_strategy)
    print("Metrics saved:", output_paths["metrics"])
    return metrics


def main():
    args = build_arg_parser().parse_args()
    MODEL_PATHS["codeT5"] = args.codeT5_path
    MODEL_PATHS["PLBART"] = args.PLBART_path
    selected_model_type = normalize_model_type(args.model_type)
    if selected_model_type == "all":
        experiments = [("codeT5", args.codeT5_path), ("PLBART", args.PLBART_path)]
    else:
        experiments = [(selected_model_type, args.model_path)]
    all_results = []
    for model_type, model_path in experiments:
        try:
            all_results.append(run_experiment(model_type, model_path))
        except Exception as exc:
            if selected_model_type != "all":
                raise
            all_results.append({"status": "failed", "model_type": model_type, "error": repr(exc)})
            print(f"[Error] RQ4 CL failed for {model_type}: {repr(exc)}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    if selected_model_type == "all":
        summary_path = os.path.join(SAVE_DIR, "rq4_cl_codetrans_cs_to_java_id_all_models.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)
        print("All-model summary saved:", summary_path)


if __name__ == "__main__":
    main()
