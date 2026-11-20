# -*- coding: utf-8 -*-
# Version: seq2seq_generation_rq1_for_codeupdatearena_fixed
"""
CodeUpdateArena RQ1: Seq2Seq Program Synthesis under API Evolution Shift

Supported models:
    - CodeT5
    - PLBART

Main fixes in this version:
    1. Correct PLBART language-aware tokenization:
       src_lang="en_XX", tgt_lang="python".
    2. PLBART uses the Python language token as decoder_start_token_id,
       without simultaneously forcing a duplicated BOS language token.
    3. Checkpoint selection defaults to the last epoch, so ID test is not
       repeatedly used to select the best epoch. A robust metric-based mode
       is still available for debugging.
    4. Automatic field inference is audited and saved. Explicit field
       overrides are supported to avoid silently selecting the wrong target.
    5. Source/target token-length diagnostics are reported before training.
    6. Reference solutions are executed before training to test whether the
       local dependency/API environment can support Pass@1 evaluation.
    7. Generation diagnostics include empty predictions, error categories,
       average lengths, and prediction previews.

Metrics:
    - Exact Match
    - Syntax Pass
    - Executability
    - Pass@1

Important experimental note:
    Under the current split protocol, ID test and OOD test should ideally be
    evaluated only once after training. Therefore CHECKPOINT_SELECTION_MODE
    defaults to "last_epoch".
"""

import os
import re
import ast
import io
import sys
import json
import time
import math
import random
import multiprocessing as mp
import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ============================================================
# 0. Environment
# ============================================================

if sys.version_info.major == 3 and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================
# 1. Configuration
# ============================================================

SUPPORTED_MODEL_TYPES = ["codeT5", "PLBART"]
MODEL_PATHS = {
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
}

# Choose "codeT5", "PLBART", or "all".
MODEL_TYPE = "PLBART"
MODEL_PATH = MODEL_PATHS[MODEL_TYPE]

DATA_DIR = "/opt/data/private/CodeUpdateArena/generation_splits"
TRAIN_FILE = os.path.join(DATA_DIR, "arena_gen_id_train.jsonl")
ID_TEST_FILE = os.path.join(DATA_DIR, "arena_gen_id_test.jsonl")
OOD_TEST_FILE = os.path.join(DATA_DIR, "arena_gen_ood_test.jsonl")

SAVE_DIR = "/opt/data/private/CodeUpdateArena/rq1_generation_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_MODEL_DIR = None
CHECKPOINT_DIR = None
METRICS_FILE = None
DATA_REPORT_FILE = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Sequence lengths.
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512
MAX_NEW_TOKENS = 512

BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Generation configuration.
NUM_BEAMS = 1
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0
NUM_RETURN_SEQUENCES = 1

# PLBART language configuration.
PLBART_SOURCE_LANG = "en_XX"
PLBART_TARGET_LANG = "python"

# PLBART often flattens Python formatting if raw newlines are learned as normal
# whitespace.  For PLBART we therefore make newline/indentation explicit in the
# target sequence and restore them after generation.  This requires retraining
# PLBART from the base checkpoint; it will not fix an already-trained checkpoint.
USE_FORMAT_TOKENS_FOR_PLBART = True
FORMAT_NEWLINE_TOKEN = "<NL>"
MAX_INDENT_LEVEL = 12
FORMAT_INDENT_TOKENS = [f"<INDENT_{i}>" for i in range(MAX_INDENT_LEVEL + 1)]
FORMAT_SPECIAL_TOKENS = [FORMAT_NEWLINE_TOKEN] + FORMAT_INDENT_TOKENS

# More stable generation for PLBART code generation.
MIN_NEW_TOKENS = 8
NO_REPEAT_NGRAM_SIZE = 4
REPETITION_PENALTY = 1.10

# Evaluation control.
MAX_EVAL_SAMPLES = None
RUN_ID_TRAIN_GENERATION_EVAL = False
EXEC_TIMEOUT_SECONDS = 8

# Reference preflight: verify that the local environment can execute the
# canonical reference solutions and tests before evaluating the model.
REFERENCE_PREFLIGHT_SAMPLES = 50
FAIL_IF_REFERENCE_PASS_RATE_ZERO = False

# Checkpoint selection:
#   "last_epoch"       recommended for the current no-validation protocol.
#   "id_test_metrics"  debugging only; uses a robust tuple rather than only
#                       Pass@1 and therefore does not get stuck at epoch 1.
CHECKPOINT_SELECTION_MODE = "last_epoch"

# Prompt metadata.
USE_METADATA_IN_PROMPT = True

# Explicit field overrides. Leave as None to infer automatically.
PROMPT_FIELD_OVERRIDE = None
REFERENCE_FIELD_OVERRIDE = None
TEST_FIELD_OVERRIDE = None

# If True, ambiguous auto-inference raises an error. False prints a warning
# and chooses the highest-ranked candidate.
FAIL_ON_AMBIGUOUS_FIELDS = False

PROMPT_FIELD_CANDIDATES = [
    "prompt", "question", "instruction", "problem", "description",
    "task", "input", "context", "nl", "docstring",
]
REFERENCE_FIELD_CANDIDATES = [
    "target", "ref_solution", "reference", "solution", "canonical_solution",
    "answer", "output", "completion", "code",
]
TEST_FIELD_CANDIDATES = [
    "test", "tests", "unit_tests", "test_code", "assertions", "test_cases",
    "private_tests", "public_tests",
]

# Number of examples used to infer fields and report token lengths.
FIELD_INFERENCE_SAMPLES = 500
LENGTH_REPORT_SAMPLES = 2000


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def normalize_model_type(model_type: str) -> str:
    model_type_lower = str(model_type).lower()
    if model_type_lower == "all":
        return "all"

    aliases = {
        "codet5": "codeT5",
        "code-t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
    }
    if model_type_lower in aliases:
        return aliases[model_type_lower]

    for name in SUPPORTED_MODEL_TYPES:
        if model_type_lower == name.lower():
            return name

    raise ValueError(
        f"Unsupported model_type={model_type}. "
        f"Choose from {SUPPORTED_MODEL_TYPES + ['all']}"
    )


def resolve_model_path(model_type: str, override_path: Optional[str] = None) -> str:
    if override_path:
        return override_path
    return MODEL_PATHS[normalize_model_type(model_type)]


def configure_model_runtime(model_type: str, model_path: Optional[str] = None):
    global MODEL_TYPE, MODEL_PATH
    global BEST_MODEL_DIR, CHECKPOINT_DIR, METRICS_FILE, DATA_REPORT_FILE

    MODEL_TYPE = normalize_model_type(model_type)
    if MODEL_TYPE == "all":
        raise ValueError("configure_model_runtime expects one model, not 'all'.")

    MODEL_PATH = resolve_model_path(MODEL_TYPE, model_path)

    run_save_dir = os.path.join(SAVE_DIR, MODEL_TYPE)
    os.makedirs(run_save_dir, exist_ok=True)

    BEST_MODEL_DIR = os.path.join(run_save_dir, f"best_generation_{MODEL_TYPE}")
    CHECKPOINT_DIR = os.path.join(run_save_dir, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    METRICS_FILE = os.path.join(
        run_save_dir,
        f"rq1_generation_metrics_{MODEL_TYPE}.json",
    )
    DATA_REPORT_FILE = os.path.join(
        run_save_dir,
        f"generation_data_report_{MODEL_TYPE}.json",
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CodeUpdateArena RQ1 generation with CodeT5 or PLBART.",
    )
    parser.add_argument(
        "--model_type",
        default=MODEL_TYPE,
        help="Choose codeT5, PLBART, or all.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional local model path for a single-model run.",
    )
    parser.add_argument("--codeT5_path", default=MODEL_PATHS["codeT5"])
    parser.add_argument("--PLBART_path", default=MODEL_PATHS["PLBART"])
    return parser


def format_time(seconds: float) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    data = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1

    if bad > 0:
        print(f"[Warning] {path} bad json lines: {bad}")
    return data


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(
            stringify_value(x) for x in value if x is not None
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def clean_code_fence(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()

    fence = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        text = fence.group(1).strip()

    for prefix in ["Answer:", "Solution:", "Code:", "```python", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text.strip()


def remove_model_special_tokens(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    removable = [
        "<s>", "</s>", "<pad>", "<unk>", "<mask>",
        "__python__", "<python>", "python3", "en_XX", "python",
    ]
    # Only remove standalone language words at boundaries; keep normal code words.
    for tok in ["<s>", "</s>", "<pad>", "<unk>", "<mask>", "__python__", "<python>"]:
        text = text.replace(tok, " ")
    text = re.sub(r"(^|\s)(en_XX|python3)(\s|$)", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def should_use_format_tokens() -> bool:
    return MODEL_TYPE == "PLBART" and USE_FORMAT_TOKENS_FOR_PLBART


def encode_code_with_format_tokens(code: str) -> str:
    """
    Convert formatted Python code into a flat sequence with explicit newline and
    indentation tokens.  Example:
        def f():\n    return 1
    becomes:
        <INDENT_0> def f(): <NL> <INDENT_1> return 1
    """
    code = clean_code_fence(code)
    lines = code.splitlines()
    encoded_lines = []

    for line in lines:
        raw = line.rstrip()
        if raw.strip() == "":
            encoded_lines.append("<INDENT_0>")
            continue

        stripped = raw.lstrip(" ")
        indent_spaces = len(raw) - len(stripped)
        indent_level = min(indent_spaces // 4, MAX_INDENT_LEVEL)
        encoded_lines.append(f"<INDENT_{indent_level}> {stripped}")

    return f" {FORMAT_NEWLINE_TOKEN} ".join(encoded_lines).strip()


def decode_code_with_format_tokens(text: str) -> str:
    """Restore Python newlines and indentation from explicit format tokens."""
    if text is None:
        return ""
    text = str(text).strip()
    text = remove_model_special_tokens(text)

    if FORMAT_NEWLINE_TOKEN not in text and not any(tok in text for tok in FORMAT_INDENT_TOKENS):
        return clean_code_fence(text)

    parts = re.split(rf"\s*{re.escape(FORMAT_NEWLINE_TOKEN)}\s*", text)
    lines = []
    indent_pattern = re.compile(r"^\s*<INDENT_(\d+)>\s*(.*)$", flags=re.DOTALL)

    for part in parts:
        part = part.strip()
        if not part:
            lines.append("")
            continue
        match = indent_pattern.match(part)
        if match:
            level = min(int(match.group(1)), MAX_INDENT_LEVEL)
            content = match.group(2).strip()
            lines.append("    " * level + content)
        else:
            # If the indentation token was omitted by generation, keep content as-is.
            lines.append(part.strip())

    return clean_code_fence("\n".join(lines))


def repair_flat_python_code_light(code: str) -> str:
    """
    Lightweight fallback for old PLBART outputs that are not trained with
    explicit format tokens. It only tries to recover obvious one-line Python
    layouts. The main recommended fix remains format-token training.
    """
    code = clean_code_fence(code)
    try:
        ast.parse(code)
        return code
    except Exception:
        pass

    if code.count("\n") >= 2:
        return code

    text = code.strip()
    # Put top-level import/from/def/class at line starts.
    text = re.sub(r"\s+(?=(?:from\s+\w|import\s+\w|def\s+\w|class\s+\w))", "\n", text)
    # Split common statements that appear after ':' or after comments in flattened outputs.
    statement_markers = [
        "if", "elif", "else", "for", "while", "try", "except", "finally", "with",
        "return", "raise", "assert",
    ]
    text = re.sub(r"\s+(?=(?:" + "|".join(statement_markers) + r")\b)", "\n    ", text)
    # Split simple assignments such as result = ..., match = ..., combined = ...
    text = re.sub(r"\s+(?=([A-Za-z_]\w*)\s*=)", "\n    ", text)
    # Split after function/class header colon if body is still on same line.
    text = re.sub(r"(:)\s+(?=(?:if|for|while|try|return|raise|assert|[A-Za-z_]\w*\s*=)\b)", r":\n    ", text)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    repaired = []
    in_function = False
    for ln in lines:
        if ln.startswith(("import ", "from ")):
            repaired.append(ln)
            continue
        if ln.startswith(("def ", "class ")):
            repaired.append(ln)
            in_function = True
            continue
        if in_function and not ln.startswith("    "):
            repaired.append("    " + ln)
        else:
            repaired.append(ln)

    return "\n".join(repaired).strip()


def postprocess_generated_code(text: str) -> str:
    """Decode model text into executable Python code before evaluation."""
    if should_use_format_tokens():
        code = decode_code_with_format_tokens(text)
    else:
        code = clean_code_fence(remove_model_special_tokens(text))

    try:
        ast.parse(code)
        return code
    except Exception:
        return repair_flat_python_code_light(code)


def normalize_code(text: str) -> str:
    text = clean_code_fence(text)
    lines = [line.rstrip() for line in text.strip().splitlines()]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines).strip()


def get_sample_id(item: Dict[str, Any], fallback: int) -> str:
    for key in ["id", "idx", "prog_syn_id", "task_id", "update_id"]:
        if item.get(key) is not None:
            return str(item.get(key))
    return str(fallback)


def field_statistics(
    items: List[Dict[str, Any]],
    candidates: List[str],
    max_samples: int,
) -> List[Dict[str, Any]]:
    subset = items[:max_samples]
    stats = []

    for order, field in enumerate(candidates):
        values = [item.get(field) for item in subset]
        nonempty = [v for v in values if is_nonempty(v)]
        lengths = [len(stringify_value(v)) for v in nonempty]
        coverage = len(nonempty) / max(len(subset), 1)
        avg_chars = float(np.mean(lengths)) if lengths else 0.0

        stats.append({
            "field": field,
            "candidate_order": order,
            "nonempty_count": len(nonempty),
            "coverage": coverage,
            "avg_chars": avg_chars,
        })

    # Prefer the field with the highest coverage. When several fields have
    # the same coverage, preserve the explicit candidate priority instead of
    # choosing the longest field, because a longer auxiliary code field may
    # not be the actual generation target.
    stats.sort(
        key=lambda x: (
            x["coverage"],
            -x["candidate_order"],
        ),
        reverse=True,
    )
    return stats


def infer_field_name(
    items: List[Dict[str, Any]],
    candidates: List[str],
    override: Optional[str],
    role: str,
    required: bool,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if override is not None:
        missing_count = sum(
            1 for item in items if not is_nonempty(item.get(override))
        )
        if required and missing_count == len(items):
            raise ValueError(
                f"{role} field override '{override}' is empty in every sample."
            )
        print(
            f">>> {role} field uses explicit override: {override} "
            f"(missing={missing_count}/{len(items)})"
        )
        return override, []

    stats = field_statistics(items, candidates, FIELD_INFERENCE_SAMPLES)
    valid = [x for x in stats if x["nonempty_count"] > 0]

    if not valid:
        if required:
            raise ValueError(
                f"Cannot infer a non-empty {role} field from candidates={candidates}."
            )
        print(f"[Warning] No {role} field found; this field will be treated as empty.")
        return None, stats

    selected = valid[0]
    ties = [
        x for x in valid
        if abs(x["coverage"] - selected["coverage"]) < 1e-12
    ]

    if len(ties) > 1:
        message = (
            f"Ambiguous {role} fields: {[x['field'] for x in ties]}. "
            f"Selected '{selected['field']}' by candidate priority. "
            f"Set the corresponding *_FIELD_OVERRIDE to remove ambiguity."
        )
        if FAIL_ON_AMBIGUOUS_FIELDS:
            raise ValueError(message)
        print("[Warning]", message)

    print(
        f">>> Auto-selected {role} field: {selected['field']} | "
        f"coverage={selected['coverage']:.4f} | "
        f"avg_chars={selected['avg_chars']:.2f}"
    )
    print(f">>> Top {role} field candidates:")
    for row in valid[:5]:
        print(
            f"    {row['field']}: coverage={row['coverage']:.4f}, "
            f"avg_chars={row['avg_chars']:.2f}"
        )

    return selected["field"], stats


def extract_tests_from_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = []
        for x in value:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                if x.get("code"):
                    parts.append(str(x["code"]))
                elif x.get("assert"):
                    parts.append(str(x["assert"]))
                elif x.get("test"):
                    parts.append(str(x["test"]))
                else:
                    parts.append(
                        "# Unsupported structured test: "
                        + json.dumps(x, ensure_ascii=False)
                    )
            else:
                parts.append(str(x))
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        if value.get("code"):
            return str(value["code"]).strip()
        if value.get("assertions"):
            return stringify_value(value["assertions"]).strip()
        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def build_prompt(item: Dict[str, Any], prompt_field: str) -> str:
    prompt = stringify_value(item.get(prompt_field)).strip()

    if USE_METADATA_IN_PROMPT:
        metadata_lines = []
        for key in ["package", "api_path", "update_type", "update_id"]:
            value = item.get(key)
            if is_nonempty(value):
                metadata_lines.append(f"{key}: {stringify_value(value)}")

        for key in [
            "update_doc", "api_update", "update_description",
            "doc", "documentation", "new_doc",
        ]:
            value = item.get(key)
            if is_nonempty(value):
                metadata_lines.append(f"{key}: {stringify_value(value)}")
                break

        if metadata_lines:
            prompt = (
                "API update information:\n"
                + "\n".join(metadata_lines)
                + "\n\nTask:\n"
                + prompt
            )

    return prompt.strip()


def syntax_pass(code: str) -> bool:
    try:
        ast.parse(clean_code_fence(code))
        return True
    except Exception:
        return False


def syntax_error_type(code: str) -> str:
    try:
        ast.parse(clean_code_fence(code))
        return "ok"
    except SyntaxError as exc:
        return f"SyntaxError:{exc.msg}"
    except Exception as exc:
        return type(exc).__name__


def categorize_runtime_error(error_text: str) -> str:
    if not error_text:
        return "ok"
    known = [
        "Timeout", "ModuleNotFoundError", "ImportError", "AttributeError",
        "NameError", "TypeError", "ValueError", "AssertionError",
        "SyntaxError", "KeyError", "IndexError", "FileNotFoundError",
        "No result returned",
    ]
    for name in known:
        if name in error_text:
            return name
    return error_text.split(":", 1)[0].strip() or "UnknownError"


def _exec_worker(code: str, test_code: str, queue):
    glb = {"__name__": "__main__"}
    loc = glb
    try:
        compiled = compile(code, "<generated_code>", "exec")
        exec(compiled, glb, loc)
        if test_code:
            compiled_test = compile(test_code, "<unit_tests>", "exec")
            exec(compiled_test, glb, loc)
        queue.put({"ok": True, "error": ""})
    except BaseException as exc:
        queue.put({
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)}",
        })


def run_code_with_timeout(
    code: str,
    test_code: str = "",
    timeout: int = 8,
) -> Tuple[bool, str]:
    code = clean_code_fence(code)
    test_code = test_code.strip() if test_code else ""

    try:
        ctx = mp.get_context("fork")
    except (ValueError, AttributeError):
        ctx = mp.get_context("spawn")

    queue = ctx.Queue()
    proc = ctx.Process(target=_exec_worker, args=(code, test_code, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        return False, "Timeout"

    if queue.empty():
        return False, "No result returned"

    result = queue.get()
    return bool(result.get("ok", False)), str(result.get("error", ""))


def robust_selection_key(
    metrics: Dict[str, Any],
    eval_loss: float,
) -> Tuple[float, float, float, float, float]:
    pass_value = metrics.get("pass_at_1")
    if pass_value is None or not np.isfinite(float(pass_value)):
        pass_value = -1.0

    syntax_value = float(metrics.get("syntax_pass", 0.0))
    exec_value = float(metrics.get("executability", 0.0))
    exact_value = float(metrics.get("exact_match", 0.0))
    loss_value = -float(eval_loss) if np.isfinite(eval_loss) else float("-inf")

    return (
        float(pass_value),
        syntax_value,
        exec_value,
        exact_value,
        loss_value,
    )


def get_plbart_language_id(tokenizer, lang_name: str) -> int:
    lang_map = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_map, dict):
        if lang_name in lang_map:
            return int(lang_map[lang_name])

        aliases = {
            "python": ["__python__", "<python>", "python3"],
            "en_XX": ["__en_XX__", "en", "english"],
        }
        for alias in aliases.get(lang_name, []):
            if alias in lang_map:
                return int(lang_map[alias])

    candidates = [lang_name]
    if lang_name == "python":
        candidates += ["__python__", "<python>", "python3"]
    elif lang_name == "en_XX":
        candidates += ["__en_XX__", "en", "english"]

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token in candidates:
        if token in vocab:
            return int(vocab[token])

    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            token_id is not None
            and token_id != getattr(tokenizer, "unk_token_id", None)
        ):
            return int(token_id)

    available = list(lang_map.keys()) if isinstance(lang_map, dict) else None
    raise RuntimeError(
        f"Cannot find PLBART language id for '{lang_name}'. "
        f"Available language codes: {available}"
    )


def configure_plbart_tokenizer_and_generation(tokenizer, model):
    if MODEL_TYPE != "PLBART":
        return

    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = PLBART_SOURCE_LANG
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = PLBART_TARGET_LANG

    target_lang_id = get_plbart_language_id(tokenizer, PLBART_TARGET_LANG)

    model.config.decoder_start_token_id = target_lang_id
    model.config.forced_bos_token_id = None

    if hasattr(model, "generation_config"):
        model.generation_config.decoder_start_token_id = target_lang_id
        model.generation_config.forced_bos_token_id = None

    print(">>> PLBART language configuration")
    print("    src_lang:", getattr(tokenizer, "src_lang", None))
    print("    tgt_lang:", getattr(tokenizer, "tgt_lang", None))
    print("    target_lang_id:", target_lang_id)
    print("    decoder_start_token_id:", model.config.decoder_start_token_id)
    print("    forced_bos_token_id:", model.config.forced_bos_token_id)


# ============================================================
# 3. Dataset
# ============================================================

class ArenaGenerationDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        name: str,
        max_samples: Optional[int] = None,
    ):
        self.path = path
        self.name = name
        self.tokenizer = tokenizer
        self.examples = []

        raw_items = read_jsonl(path)
        if max_samples is not None:
            raw_items = raw_items[:max_samples]

        self.prompt_field, self.prompt_field_stats = infer_field_name(
            raw_items,
            PROMPT_FIELD_CANDIDATES,
            PROMPT_FIELD_OVERRIDE,
            role=f"{name}.prompt",
            required=True,
        )
        self.reference_field, self.reference_field_stats = infer_field_name(
            raw_items,
            REFERENCE_FIELD_CANDIDATES,
            REFERENCE_FIELD_OVERRIDE,
            role=f"{name}.reference",
            required=True,
        )
        self.test_field, self.test_field_stats = infer_field_name(
            raw_items,
            TEST_FIELD_CANDIDATES,
            TEST_FIELD_OVERRIDE,
            role=f"{name}.tests",
            required=False,
        )

        bad_prompt = 0
        bad_ref = 0
        no_tests = 0

        for idx, item in enumerate(
            tqdm(raw_items, desc=f"Loading {name}", leave=False)
        ):
            prompt = build_prompt(item, self.prompt_field)
            reference = clean_code_fence(
                stringify_value(item.get(self.reference_field))
            )
            tests = (
                extract_tests_from_value(item.get(self.test_field))
                if self.test_field is not None
                else ""
            )

            if not prompt:
                bad_prompt += 1
                continue
            if not reference:
                bad_ref += 1
                continue
            if not tests:
                no_tests += 1

            self.examples.append({
                "sample_id": get_sample_id(item, idx),
                "prompt": prompt,
                "reference": reference,
                "tests": tests,
                "raw": item,
            })

        print("=" * 100)
        print(f"ArenaGenerationDataset loaded: {name}")
        print("path:", path)
        print("num_samples:", len(self.examples))
        print("prompt_field:", self.prompt_field)
        print("reference_field:", self.reference_field)
        print("test_field:", self.test_field)
        print("bad_prompt:", bad_prompt)
        print("bad_reference:", bad_ref)
        print("samples_without_tests:", no_tests)
        print("=" * 100)

        for preview_index, example in enumerate(self.examples[:3]):
            print("\n" + "-" * 100)
            print(f"[{name} preview {preview_index}] sample_id={example['sample_id']}")
            print("PROMPT:")
            print(example["prompt"][:1000])
            print("\nREFERENCE:")
            print(example["reference"][:1000])
            print("\nTESTS:")
            print(example["tests"][:1000])
        if self.examples:
            print("-" * 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]

    def field_report(self) -> Dict[str, Any]:
        return {
            "prompt_field": self.prompt_field,
            "reference_field": self.reference_field,
            "test_field": self.test_field,
            "prompt_field_stats": self.prompt_field_stats,
            "reference_field_stats": self.reference_field_stats,
            "test_field_stats": self.test_field_stats,
        }


class Seq2SeqCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_targets(self, targets: List[str]):
        try:
            return self.tokenizer(
                text_target=targets,
                max_length=MAX_TARGET_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        except TypeError:
            if not hasattr(self.tokenizer, "as_target_tokenizer"):
                raise
            with self.tokenizer.as_target_tokenizer():
                return self.tokenizer(
                    targets,
                    max_length=MAX_TARGET_LEN,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

    def __call__(self, batch: List[Dict[str, Any]]):
        prompts = [x["prompt"] for x in batch]
        if should_use_format_tokens():
            refs = [encode_code_with_format_tokens(x["reference"]) for x in batch]
        else:
            refs = [x["reference"] for x in batch]

        source_inputs = self.tokenizer(
            prompts,
            max_length=MAX_SOURCE_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        target_inputs = self.encode_targets(refs)
        labels = target_inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_inputs["input_ids"],
            "attention_mask": source_inputs["attention_mask"],
            "labels": labels,
            "raw": batch,
        }


# ============================================================
# 4. Diagnostics
# ============================================================

def encode_target_without_truncation(tokenizer, text: str) -> List[int]:
    try:
        encoded = tokenizer(
            text_target=text,
            truncation=False,
            add_special_tokens=True,
        )
    except TypeError:
        if not hasattr(tokenizer, "as_target_tokenizer"):
            raise
        with tokenizer.as_target_tokenizer():
            encoded = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )
    return encoded["input_ids"]


def report_token_lengths(
    dataset: ArenaGenerationDataset,
    tokenizer,
    name: str,
) -> Dict[str, Any]:
    examples = dataset.examples[:LENGTH_REPORT_SAMPLES]
    source_lengths = []
    target_lengths = []

    for ex in tqdm(examples, desc=f"Token-length report {name}", leave=False):
        source_ids = tokenizer(
            ex["prompt"],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        target_ids = encode_target_without_truncation(
            tokenizer,
            ex["reference"],
        )
        source_lengths.append(len(source_ids))
        target_lengths.append(len(target_ids))

    def summarize(lengths: List[int], limit: int) -> Dict[str, Any]:
        if not lengths:
            return {
                "count": 0,
                "mean": None,
                "p90": None,
                "p95": None,
                "max": None,
                "truncation_rate": None,
            }
        arr = np.asarray(lengths, dtype=np.int64)
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(np.max(arr)),
            "truncation_rate": float(np.mean(arr > limit)),
            "limit": int(limit),
        }

    report = {
        "source": summarize(source_lengths, MAX_SOURCE_LEN),
        "target": summarize(target_lengths, MAX_TARGET_LEN),
    }

    print("=" * 100)
    print(f"Token-length diagnostics: {name}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)

    target_truncation = report["target"].get("truncation_rate")
    if target_truncation is not None and target_truncation > 0.05:
        print(
            f"[Warning] {name} target truncation rate is "
            f"{target_truncation:.2%}. Increase MAX_TARGET_LEN and "
            "MAX_NEW_TOKENS if GPU memory permits."
        )

    return report


def reference_execution_preflight(
    dataset: ArenaGenerationDataset,
    split_name: str,
) -> Dict[str, Any]:
    examples = dataset.examples[:REFERENCE_PREFLIGHT_SAMPLES]

    syntax_flags = []
    exec_flags = []
    pass_flags = []
    pass_denominator = 0
    exec_error_counter = Counter()
    pass_error_counter = Counter()
    failure_examples = []

    for ex in tqdm(examples, desc=f"Reference preflight {split_name}", leave=False):
        reference = clean_code_fence(ex["reference"])
        tests = ex.get("tests", "") or ""

        syntax_ok = syntax_pass(reference)
        exec_ok, exec_error = run_code_with_timeout(
            reference,
            test_code="",
            timeout=EXEC_TIMEOUT_SECONDS,
        )
        syntax_flags.append(bool(syntax_ok))
        exec_flags.append(bool(exec_ok))
        exec_error_counter[categorize_runtime_error(exec_error)] += 1

        pass_ok = None
        pass_error = ""
        if tests.strip():
            pass_denominator += 1
            pass_ok, pass_error = run_code_with_timeout(
                reference,
                test_code=tests,
                timeout=EXEC_TIMEOUT_SECONDS,
            )
            pass_flags.append(bool(pass_ok))
            pass_error_counter[categorize_runtime_error(pass_error)] += 1

        if (
            not syntax_ok
            or not exec_ok
            or (pass_ok is False)
        ) and len(failure_examples) < 10:
            failure_examples.append({
                "sample_id": ex["sample_id"],
                "syntax_ok": bool(syntax_ok),
                "exec_ok": bool(exec_ok),
                "exec_error": exec_error,
                "pass_ok": pass_ok,
                "pass_error": pass_error,
                "reference_preview": reference[:800],
                "tests_preview": tests[:800],
            })

    report = {
        "split": split_name,
        "num_samples_checked": len(examples),
        "syntax_pass": float(np.mean(syntax_flags)) if syntax_flags else None,
        "executability": float(np.mean(exec_flags)) if exec_flags else None,
        "pass_at_1": float(np.mean(pass_flags)) if pass_flags else None,
        "pass_at_1_num_tests": int(pass_denominator),
        "exec_error_top10": dict(exec_error_counter.most_common(10)),
        "pass_error_top10": dict(pass_error_counter.most_common(10)),
        "failure_examples": failure_examples,
    }

    print("=" * 100)
    print(f"Reference execution preflight: {split_name}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)

    if (
        FAIL_IF_REFERENCE_PASS_RATE_ZERO
        and report["pass_at_1_num_tests"] > 0
        and report["pass_at_1"] == 0.0
    ):
        raise RuntimeError(
            "Canonical reference solutions have Pass@1=0 in the current "
            "environment. The API package version, dependencies, fixtures, or "
            "test harness are likely incorrect. Fix the environment before "
            "using Pass@1 to judge the model."
        )

    return report


# ============================================================
# 5. Model / train / generate
# ============================================================

def init_tokenizer_and_model():
    if MODEL_TYPE == "PLBART":
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                src_lang=PLBART_SOURCE_LANG,
                tgt_lang=PLBART_TARGET_LANG,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)

    if should_use_format_tokens():
        existing_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        tokens_to_add = [tok for tok in FORMAT_SPECIAL_TOKENS if tok not in existing_vocab]
        if tokens_to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            model.resize_token_embeddings(len(tokenizer))
            print(f">>> Added PLBART format tokens: {tokens_to_add}")
        else:
            print(">>> PLBART format tokens already exist in tokenizer.")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if MODEL_TYPE == "PLBART":
        configure_plbart_tokenizer_and_generation(tokenizer, model)
    else:
        if model.config.decoder_start_token_id is None:
            if tokenizer.pad_token_id is not None:
                model.config.decoder_start_token_id = tokenizer.pad_token_id
            elif tokenizer.eos_token_id is not None:
                model.config.decoder_start_token_id = tokenizer.eos_token_id
            else:
                raise RuntimeError(
                    "Cannot determine decoder_start_token_id for CodeT5."
                )

    print("=" * 100)
    print("Tokenizer/model initialization")
    print("MODEL_TYPE:", MODEL_TYPE)
    print("Tokenizer class:", type(tokenizer).__name__)
    print("Model class:", type(model).__name__)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("bos_token_id:", tokenizer.bos_token_id)
    print("eos_token_id:", tokenizer.eos_token_id)
    print("decoder_start_token_id:", model.config.decoder_start_token_id)
    print("forced_bos_token_id:", getattr(model.config, "forced_bos_token_id", None))
    print("=" * 100)

    return tokenizer, model


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("[Warning] Skip NaN/Inf loss batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if steps == 0:
        raise RuntimeError("No valid optimization steps were completed.")

    return total_loss / steps


@torch.no_grad()
def compute_eval_loss(model, dataloader, desc: str) -> float:
    model.eval()
    losses = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if not torch.isnan(outputs.loss) and not torch.isinf(outputs.loss):
            losses.append(float(outputs.loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def build_generation_kwargs(tokenizer) -> Dict[str, Any]:
    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS,
        "do_sample": DO_SAMPLE,
        "num_return_sequences": NUM_RETURN_SEQUENCES,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "min_new_tokens": MIN_NEW_TOKENS,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": REPETITION_PENALTY,
    }

    if DO_SAMPLE:
        kwargs["temperature"] = TEMPERATURE
        kwargs["top_p"] = TOP_P

    return kwargs


@torch.no_grad()
def generate_for_dataset(
    model,
    tokenizer,
    dataset: ArenaGenerationDataset,
    split_name: str,
) -> Dict[str, Any]:
    model.eval()

    examples = (
        dataset.examples[:MAX_EVAL_SAMPLES]
        if MAX_EVAL_SAMPLES is not None
        else dataset.examples
    )

    run_save_dir = os.path.dirname(METRICS_FILE)
    pred_path = os.path.join(
        run_save_dir,
        f"predictions_{split_name}_{MODEL_TYPE}.jsonl",
    )

    exact_flags = []
    syntax_flags = []
    executable_flags = []
    pass_flags = []
    pass_denominator = 0

    prediction_char_lengths = []
    prediction_line_lengths = []
    syntax_error_counter = Counter()
    exec_error_counter = Counter()
    pass_error_counter = Counter()
    failure_examples = []
    prediction_previews = []

    generation_kwargs = build_generation_kwargs(tokenizer)

    with open(pred_path, "w", encoding="utf-8") as fout:
        for start in tqdm(
            range(0, len(examples), EVAL_BATCH_SIZE),
            desc=f"Generate {split_name}",
        ):
            batch = examples[start:start + EVAL_BATCH_SIZE]
            prompts = [x["prompt"] for x in batch]

            inputs = tokenizer(
                prompts,
                max_length=MAX_SOURCE_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)

            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs,
            )

            decoded = tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if NUM_RETURN_SEQUENCES != 1:
                raise ValueError(
                    "This script computes strict Pass@1 and therefore requires "
                    "NUM_RETURN_SEQUENCES=1."
                )

            for ex, generated in zip(batch, decoded):
                generated_raw = generated
                generated_clean = postprocess_generated_code(generated_raw)
                reference_clean = clean_code_fence(ex["reference"])
                tests = ex.get("tests", "") or ""

                exact = (
                    normalize_code(generated_clean)
                    == normalize_code(reference_clean)
                )
                syntax_ok = syntax_pass(generated_clean)
                syntax_error = syntax_error_type(generated_clean)
                syntax_error_counter[syntax_error] += 1

                prediction_char_lengths.append(len(generated_clean))
                prediction_line_lengths.append(
                    len(generated_clean.splitlines())
                )

                exec_ok, exec_error = run_code_with_timeout(
                    generated_clean,
                    test_code="",
                    timeout=EXEC_TIMEOUT_SECONDS,
                )
                exec_error_counter[categorize_runtime_error(exec_error)] += 1

                pass_ok = None
                pass_error = ""
                if tests.strip():
                    pass_denominator += 1
                    pass_ok, pass_error = run_code_with_timeout(
                        generated_clean,
                        test_code=tests,
                        timeout=EXEC_TIMEOUT_SECONDS,
                    )
                    pass_flags.append(bool(pass_ok))
                    pass_error_counter[
                        categorize_runtime_error(pass_error)
                    ] += 1

                exact_flags.append(bool(exact))
                syntax_flags.append(bool(syntax_ok))
                executable_flags.append(bool(exec_ok))

                if len(prediction_previews) < 5:
                    prediction_previews.append({
                        "sample_id": ex["sample_id"],
                        "prediction": generated_clean[:1000],
                        "reference": reference_clean[:1000],
                    })

                if (
                    not exact
                    or not syntax_ok
                    or not exec_ok
                    or pass_ok is False
                ) and len(failure_examples) < 10:
                    failure_examples.append({
                        "sample_id": ex["sample_id"],
                        "syntax_error": syntax_error,
                        "exec_error": exec_error,
                        "pass_error": pass_error,
                        "prediction_preview": generated_clean[:800],
                        "reference_preview": reference_clean[:800],
                    })

                raw = ex.get("raw", {})
                out = {
                    "split": split_name,
                    "sample_id": ex["sample_id"],
                    "id": raw.get("id"),
                    "idx": raw.get("idx"),
                    "prog_syn_id": raw.get("prog_syn_id"),
                    "update_id": raw.get("update_id"),
                    "package": raw.get("package"),
                    "api_path": raw.get("api_path"),
                    "update_type": raw.get("update_type"),
                    "prompt_preview": ex["prompt"][:1000],
                    "reference": reference_clean,
                    "prediction_raw": generated_raw,
                    "prediction": generated_clean,
                    "tests_present": bool(tests.strip()),
                    "exact_match": bool(exact),
                    "syntax_pass": bool(syntax_ok),
                    "syntax_error": syntax_error,
                    "executable": bool(exec_ok),
                    "exec_error": exec_error,
                    "pass_at_1": pass_ok,
                    "pass_error": pass_error,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    metrics = {
        "num_samples": int(len(examples)),
        "exact_match": float(np.mean(exact_flags)) if exact_flags else 0.0,
        "syntax_pass": float(np.mean(syntax_flags)) if syntax_flags else 0.0,
        "executability": (
            float(np.mean(executable_flags)) if executable_flags else 0.0
        ),
        "pass_at_1": float(np.mean(pass_flags)) if pass_flags else None,
        "pass_at_1_num_tests": int(pass_denominator),
        "empty_prediction_rate": (
            float(np.mean([n == 0 for n in prediction_char_lengths]))
            if prediction_char_lengths else 0.0
        ),
        "avg_prediction_chars": (
            float(np.mean(prediction_char_lengths))
            if prediction_char_lengths else 0.0
        ),
        "avg_prediction_lines": (
            float(np.mean(prediction_line_lengths))
            if prediction_line_lengths else 0.0
        ),
        "syntax_error_top10": dict(syntax_error_counter.most_common(10)),
        "exec_error_top10": dict(exec_error_counter.most_common(10)),
        "pass_error_top10": dict(pass_error_counter.most_common(10)),
        "prediction_previews": prediction_previews,
        "failure_examples": failure_examples,
        "prediction_file": pred_path,
    }

    return metrics


def save_model(model, tokenizer, path: str):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_best_model(path: str, tokenizer=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
    if tokenizer is not None and MODEL_TYPE == "PLBART":
        configure_plbart_tokenizer_and_generation(tokenizer, model)
    return model


# ============================================================
# 6. Main experiment
# ============================================================

def run_experiment(
    model_type: str,
    model_path: Optional[str] = None,
):
    configure_model_runtime(model_type, model_path)
    start_time = time.time()
    set_seed(SEED)

    print("=" * 100)
    print("CodeUpdateArena RQ1: Seq2Seq Program Synthesis under API Evolution Shift")
    print("=" * 100)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("TRAIN_FILE:", TRAIN_FILE)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILE:", OOD_TEST_FILE)
    print("SAVE_DIR:", SAVE_DIR)
    print("DEVICE:", DEVICE)
    print("CHECKPOINT_SELECTION_MODE:", CHECKPOINT_SELECTION_MODE)
    print("=" * 100)

    tokenizer, model = init_tokenizer_and_model()
    collator = Seq2SeqCollator(tokenizer)

    train_dataset = ArenaGenerationDataset(
        TRAIN_FILE,
        tokenizer,
        name="id_train",
    )
    id_test_dataset = ArenaGenerationDataset(
        ID_TEST_FILE,
        tokenizer,
        name="id_test",
        max_samples=MAX_EVAL_SAMPLES,
    )
    ood_test_dataset = ArenaGenerationDataset(
        OOD_TEST_FILE,
        tokenizer,
        name="ood_test",
        max_samples=MAX_EVAL_SAMPLES,
    )

    if len(train_dataset) == 0:
        raise ValueError(f"ID train is empty; cannot fine-tune {MODEL_TYPE}.")
    if len(id_test_dataset) == 0:
        raise ValueError("ID test is empty; cannot evaluate.")
    if len(ood_test_dataset) == 0:
        raise ValueError("OOD test is empty; cannot evaluate.")

    train_length_report = report_token_lengths(
        train_dataset,
        tokenizer,
        "id_train",
    )
    id_length_report = report_token_lengths(
        id_test_dataset,
        tokenizer,
        "id_test",
    )
    ood_length_report = report_token_lengths(
        ood_test_dataset,
        tokenizer,
        "ood_test",
    )

    id_reference_preflight = reference_execution_preflight(
        id_test_dataset,
        "id_test",
    )
    ood_reference_preflight = reference_execution_preflight(
        ood_test_dataset,
        "ood_test",
    )

    data_report = {
        "train_file": TRAIN_FILE,
        "id_test_file": ID_TEST_FILE,
        "ood_test_file": OOD_TEST_FILE,
        "num_train": len(train_dataset),
        "num_id_test": len(id_test_dataset),
        "num_ood_test": len(ood_test_dataset),
        "field_overrides": {
            "prompt": PROMPT_FIELD_OVERRIDE,
            "reference": REFERENCE_FIELD_OVERRIDE,
            "tests": TEST_FIELD_OVERRIDE,
        },
        "train_fields": train_dataset.field_report(),
        "id_test_fields": id_test_dataset.field_report(),
        "ood_test_fields": ood_test_dataset.field_report(),
        "use_metadata_in_prompt": USE_METADATA_IN_PROMPT,
        "token_length_reports": {
            "id_train": train_length_report,
            "id_test": id_length_report,
            "ood_test": ood_length_report,
        },
        "reference_execution_preflight": {
            "id_test": id_reference_preflight,
            "ood_test": ood_reference_preflight,
        },
    }

    with open(DATA_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(data_report, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    id_loader = DataLoader(
        id_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )
    ood_loader = DataLoader(
        ood_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_selection_key = None
    best_epoch = 0
    epoch_logs = []

    print(f"\n>>> Start fine-tuning {MODEL_TYPE} for generation...")
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch,
        )

        epoch_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}")
        save_model(model, tokenizer, epoch_dir)

        if CHECKPOINT_SELECTION_MODE == "id_test_metrics":
            id_gen_metrics = generate_for_dataset(
                model,
                tokenizer,
                id_test_dataset,
                split_name=f"id_test_epoch_{epoch}",
            )
            id_loss = compute_eval_loss(
                model,
                id_loader,
                desc=f"Eval loss ID epoch {epoch}",
            )
            selection_key = robust_selection_key(id_gen_metrics, id_loss)
            should_save_best = (
                best_selection_key is None
                or selection_key > best_selection_key
            )
        elif CHECKPOINT_SELECTION_MODE == "last_epoch":
            id_gen_metrics = None
            id_loss = None
            selection_key = (float(epoch),)
            should_save_best = epoch == EPOCHS
        else:
            raise ValueError(
                "CHECKPOINT_SELECTION_MODE must be 'last_epoch' or "
                "'id_test_metrics'."
            )

        if should_save_best:
            best_selection_key = selection_key
            best_epoch = epoch
            save_model(model, tokenizer, BEST_MODEL_DIR)
            print(
                f"✨ Selected checkpoint saved: epoch={epoch}, "
                f"selection_key={selection_key}"
            )

        epoch_time = time.time() - epoch_start
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "id_eval_loss": id_loss,
            "id_generation_metrics": id_gen_metrics,
            "selection_key": list(selection_key),
            "saved_as_best": bool(should_save_best),
            "epoch_seconds": epoch_time,
            "epoch_readable": format_time(epoch_time),
        }
        epoch_logs.append(log)

        if id_gen_metrics is None:
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"Selection: last_epoch | Time: {format_time(epoch_time)}"
            )
        else:
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"ID Loss: {id_loss:.4f} | "
                f"ID Pass@1: {id_gen_metrics['pass_at_1']} | "
                f"ID Syntax: {id_gen_metrics['syntax_pass']:.4f} | "
                f"ID Exec: {id_gen_metrics['executability']:.4f} | "
                f"ID EM: {id_gen_metrics['exact_match']:.4f} | "
                f"Selection Key: {selection_key} | "
                f"Time: {format_time(epoch_time)}"
            )

    train_time = time.time() - train_start

    if best_epoch == 0 or not os.path.exists(BEST_MODEL_DIR):
        raise RuntimeError("No final checkpoint was selected or saved.")

    print("\n" + "=" * 100)
    print(
        f"Training finished. Selected epoch: {best_epoch}, "
        f"selection key: {best_selection_key}"
    )
    print("Loading selected model for final ID/OOD generation evaluation...")
    print("=" * 100)

    model = load_best_model(BEST_MODEL_DIR, tokenizer=tokenizer)

    eval_start = time.time()

    final_train_metrics = None
    if RUN_ID_TRAIN_GENERATION_EVAL:
        final_train_metrics = generate_for_dataset(
            model,
            tokenizer,
            train_dataset,
            split_name="id_train_final",
        )

    final_id_metrics = generate_for_dataset(
        model,
        tokenizer,
        id_test_dataset,
        split_name="id_test_final",
    )
    final_ood_metrics = generate_for_dataset(
        model,
        tokenizer,
        ood_test_dataset,
        split_name="ood_test_final",
    )

    id_final_loss = compute_eval_loss(
        model,
        id_loader,
        desc="Final eval loss ID",
    )
    ood_final_loss = compute_eval_loss(
        model,
        ood_loader,
        desc="Final eval loss OOD",
    )

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    def metric_drop(key: str):
        id_value = final_id_metrics.get(key)
        ood_value = final_ood_metrics.get(key)
        if id_value is None or ood_value is None:
            return None
        return float(id_value) - float(ood_value)

    final_metrics = {
        "task": "Program Synthesis / Code Generation",
        "dataset": "CodeUpdateArena",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "scenario": "API evolution shift",
        "train_file": TRAIN_FILE,
        "id_test_file": ID_TEST_FILE,
        "ood_test_file": OOD_TEST_FILE,
        "best_model_dir": BEST_MODEL_DIR,
        "selected_epoch": best_epoch,
        "checkpoint_selection_mode": CHECKPOINT_SELECTION_MODE,
        "best_selection_key": (
            list(best_selection_key)
            if best_selection_key is not None
            else None
        ),
        "plbart_language_configuration": (
            {
                "src_lang": PLBART_SOURCE_LANG,
                "tgt_lang": PLBART_TARGET_LANG,
                "decoder_start_token_id": model.config.decoder_start_token_id,
                "forced_bos_token_id": getattr(model.config, "forced_bos_token_id", None),
            }
            if MODEL_TYPE == "PLBART"
            else None
        ),
        "generation_config": {
            "max_source_len": MAX_SOURCE_LEN,
            "max_target_len": MAX_TARGET_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_beams": NUM_BEAMS,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE if DO_SAMPLE else None,
            "top_p": TOP_P if DO_SAMPLE else None,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
        },
        "metrics_definition": {
            "exact_match": (
                "normalized generated code exactly matches the normalized "
                "reference solution"
            ),
            "syntax_pass": "ast.parse(generated_code) succeeds",
            "executability": (
                "exec(generated_code) succeeds without unit tests"
            ),
            "pass_at_1": (
                "exec(generated_code + unit_tests) succeeds; samples without "
                "tests are excluded"
            ),
        },
        "reference_execution_preflight": {
            "id_test": id_reference_preflight,
            "ood_test": ood_reference_preflight,
        },
        "id_train": final_train_metrics,
        "id_test": final_id_metrics,
        "ood_test": final_ood_metrics,
        "eval_loss": {
            "id_test_loss": id_final_loss,
            "ood_test_loss": ood_final_loss,
        },
        "drop_id_minus_ood": {
            "exact_match_drop": metric_drop("exact_match"),
            "syntax_pass_drop": metric_drop("syntax_pass"),
            "executability_drop": metric_drop("executability"),
            "pass_at_1_drop": metric_drop("pass_at_1"),
        },
        "epoch_logs": epoch_logs,
        "files": {
            "metrics": METRICS_FILE,
            "data_report": DATA_REPORT_FILE,
            "best_model_dir": BEST_MODEL_DIR,
            "checkpoint_dir": CHECKPOINT_DIR,
        },
        "time": {
            "train_elapsed_seconds": train_time,
            "train_elapsed_readable": format_time(train_time),
            "eval_elapsed_seconds": eval_time,
            "eval_elapsed_readable": format_time(eval_time),
            "program_elapsed_seconds": total_time,
            "program_elapsed_readable": format_time(total_time),
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 110)
    print(f"RQ1 Final Results: CodeUpdateArena {MODEL_TYPE} Generation")
    print("=" * 110)
    print(
        f"{'Metric':<18} | {'ID Test':<12} | "
        f"{'OOD Test':<12} | {'Drop ID-OOD':<12}"
    )
    print("-" * 110)

    def fmt(value):
        return "None" if value is None else f"{float(value):.4f}"

    for key, name in [
        ("exact_match", "Exact Match"),
        ("syntax_pass", "Syntax Pass"),
        ("executability", "Executability"),
        ("pass_at_1", "Pass@1"),
    ]:
        id_value = final_id_metrics.get(key)
        ood_value = final_ood_metrics.get(key)
        drop_value = metric_drop(key)
        print(
            f"{name:<18} | {fmt(id_value):<12} | "
            f"{fmt(ood_value):<12} | {fmt(drop_value):<12}"
        )

    print("-" * 110)
    print("ID eval loss:", id_final_loss)
    print("OOD eval loss:", ood_final_loss)
    print("Selected epoch:", best_epoch)
    print("Selected model:", BEST_MODEL_DIR)
    print("Metrics file:", METRICS_FILE)
    print("Data report:", DATA_REPORT_FILE)
    print("Total time:", format_time(total_time))
    print("=" * 110)

    return final_metrics


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    MODEL_PATHS["codeT5"] = args.codeT5_path
    MODEL_PATHS["PLBART"] = args.PLBART_path

    selected_model = normalize_model_type(args.model_type)

    if selected_model == "all":
        all_results = []
        for run_model_type in SUPPORTED_MODEL_TYPES:
            try:
                all_results.append(run_experiment(run_model_type))
            except Exception as exc:
                all_results.append({
                    "model_type": run_model_type,
                    "status": "failed",
                    "error": repr(exc),
                })
                print(
                    f"[Error] Model {run_model_type} failed: {repr(exc)}"
                )
            finally:
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        summary_path = os.path.join(
            SAVE_DIR,
            "rq1_generation_metrics_all_models.json",
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": "Program Synthesis / Code Generation",
                    "dataset": "CodeUpdateArena",
                    "models": SUPPORTED_MODEL_TYPES,
                    "scenario": "API evolution shift",
                    "results": all_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("All-model summary saved:", summary_path)
    else:
        run_experiment(selected_model, args.model_path)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except (RuntimeError, ValueError):
        pass
    main()
