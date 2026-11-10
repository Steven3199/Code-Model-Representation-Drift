import os
import json
import csv
import time
import sys
import io
import random
from collections import Counter
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    T5EncoderModel,
    T5Config
)
from tqdm import tqdm


# ============================================================
# 0. 环境设置
# ============================================================

if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

START_TIME = time.time()


# ============================================================
# 1. 配置区
# ============================================================

MODEL_TYPE = "RoBERTa"        # 可选: "codebert", "GraphCodeBERT", "codeT5"
MODEL_PATH = "/opt/data/private/RoBERTa"

TASK_NAME = "Defect Detection"
DATASET_NAME = "CodeXGLUE Defect Detection"
SUPPORTED_MODEL_TYPES = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATHS = {
    "codebert": "/opt/data/private/codebert",
    "GraphCodeBERT": "/opt/data/private/GraphCodeBERT",
    "codeT5": "/opt/data/private/codeT5",
    "PLBART": "/opt/data/private/PLBART",
    "RoBERTa": "/opt/data/private/RoBERTa",
}
RUN_ALL_MODELS = True
MODEL_TYPES_TO_RUN = ["codebert", "GraphCodeBERT", "codeT5", "PLBART", "RoBERTa"]
MODEL_PATH = MODEL_PATHS[MODEL_TYPE]

DATA_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Defect-detection/dataset"

ID_TEST_FILE = "id_test.jsonl"

OOD_TEST_FILES = {
    "token": "ood_test_token.jsonl",
    "cst": "ood_test_cst.jsonl"
}

# RQ1 只有一个 ID-trained best model，然后分别评估 token/cst OOD
FT_MODEL_WEIGHTS = "/opt/data/private/CodeXGLUE/Code-Code/Defect-detection/rq1_outputs/best_defect_codebert.bin"
RQ1_SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Defect-detection/rq1_outputs"

SAVE_DIR = "/opt/data/private/CodeXGLUE/Code-Code/Defect-detection/rq3_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 2
MAX_LEN = 512
BATCH_SIZE = 16

# 正式实验建议 None，全量分析；调试时可改成 500
MAX_ANALYSIS_SAMPLES = None
MAX_ANISO_SAMPLES = None
MAX_RANK_SAMPLES = None
ANISO_CHUNK_SIZE = 1024
ENCODER_POOLING = "auto"
SHIFT_TYPE = "cst"  # "token", "cst", or "all"

SEED = 42

ALLOW_MISMATCH = True

RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}.json")
RESULT_CSV = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}.csv")
RESULT_RAW_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_raw_paste_table.tsv")
RESULT_NORMALIZED_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{SHIFT_TYPE}_normalized_paste_table.tsv")
ALL_MODELS_RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_all_models_{SHIFT_TYPE}.json")

# 如果 idx/label 不一致是否继续。
# 正式数据最好保持 idx/label 完全一致。
ALLOW_MISMATCH = True


# ============================================================
# 2. 随机种子与基础工具
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
    return f"{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒"


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_checkpoint(checkpoint):
    """
    兼容多种 checkpoint 保存格式：
    1. state_dict
    2. {"model_state_dict": state_dict}
    3. {"state_dict": state_dict}
    4. {"model": state_dict}
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]

    return checkpoint


def adapt_state_dict_keys(state_dict):
    """
    兼容不同训练代码保存出的 key：
    - roberta.xxx      -> encoder.xxx
    - bert.xxx         -> encoder.xxx
    - transformer.xxx  -> encoder.xxx
    - module.xxx       -> xxx
    - model.xxx        -> xxx
    """
    new_dict = {}

    for k, v in state_dict.items():
        new_key = k

        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)

        if new_key.startswith("model."):
            new_key = new_key.replace("model.", "", 1)

        if new_key.startswith("roberta."):
            new_key = new_key.replace("roberta.", "encoder.", 1)

        elif new_key.startswith("bert."):
            new_key = new_key.replace("bert.", "encoder.", 1)

        elif new_key.startswith("transformer."):
            new_key = new_key.replace("transformer.", "encoder.", 1)

        new_dict[new_key] = v

    return new_dict


def canonical_model_type(model_type):
    aliases = {
        "codebert": "codebert",
        "code-bert": "codebert",
        "graphcodebert": "GraphCodeBERT",
        "graph-codebert": "GraphCodeBERT",
        "codet5": "codeT5",
        "code_t5": "codeT5",
        "code-t5": "codeT5",
        "plbart": "PLBART",
        "pl-bart": "PLBART",
        "roberta": "RoBERTa",
        "roberta-base": "RoBERTa",
    }
    key = str(model_type).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Supported: {SUPPORTED_MODEL_TYPES}")
    return aliases[key]


def configure_model_runtime(model_type, shift_type):
    global MODEL_TYPE, MODEL_PATH, FT_MODEL_WEIGHTS
    global RESULT_JSON, RESULT_CSV, RESULT_RAW_PASTE_TSV, RESULT_NORMALIZED_PASTE_TSV

    MODEL_TYPE = canonical_model_type(model_type)
    MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
    FT_MODEL_WEIGHTS = get_best_model_path(MODEL_TYPE)
    RESULT_JSON = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{shift_type}.json")
    RESULT_CSV = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{shift_type}.csv")
    RESULT_RAW_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{shift_type}_raw_paste_table.tsv")
    RESULT_NORMALIZED_PASTE_TSV = os.path.join(SAVE_DIR, f"rq3_defect_layer_geometry_{MODEL_TYPE}_{shift_type}_normalized_paste_table.tsv")


def get_best_model_path(model_type):
    model_type = canonical_model_type(model_type)
    return os.path.join(RQ1_SAVE_DIR, f"best_defect_{model_type}.bin")


def resolve_pooling_strategy(model_type, pooling="auto"):
    pooling = str(pooling).lower()
    if pooling in ("cls", "first", "first_token"):
        return "cls"
    if pooling in ("mean", "mean_pooling"):
        return "mean"
    if pooling != "auto":
        raise ValueError(f"Unknown ENCODER_POOLING={pooling}")
    mt = canonical_model_type(model_type).lower()
    return "mean" if ("t5" in mt or "plbart" in mt) else "cls"


def deterministic_subset_indices(n, max_n, seed=42):
    if max_n is None or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def tensor_mean_or_nan(tensors):
    if not tensors:
        return float("nan")
    return float(torch.cat(tensors, dim=0).float().mean().item())


def get_code(js):
    """
    Defect Detection 常见字段是 func。
    这里兼容 code/function/source，方便不同版本数据。
    """
    return (
        js.get("func")
        or js.get("code")
        or js.get("function")
        or js.get("source")
        or ""
    )


def get_label(js):
    """
    Defect Detection 常见字段可能是 target 或 label。
    """
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


# ============================================================
# 3. RQ3 几何指标函数
# ============================================================

def get_alignment(id_emb, ood_emb):
    """
    ID/OOD 对齐度：
    同一样本的 ID 版本和 OOD 版本表征的平均余弦相似度。
    越高表示 ID/OOD 表征越接近。
    """
    if id_emb.size(0) == 0:
        return float("nan")
    id_norm = F.normalize(id_emb.float(), p=2, dim=1)
    ood_norm = F.normalize(ood_emb.float(), p=2, dim=1)
    return F.cosine_similarity(id_norm, ood_norm, dim=1).mean().item()


def get_anisotropy(embeddings, max_samples=MAX_ANISO_SAMPLES):
    """
    各向异性：
    OOD 表征之间的平均余弦相似度。
    越高说明表征越集中，越可能出现坍塌。
    """
    if embeddings.size(0) < 2:
        return float("nan")
    if max_samples is not None and embeddings.size(0) > max_samples:
        idx = deterministic_subset_indices(embeddings.size(0), max_samples, seed=SEED)
        embeddings = embeddings[idx]

    z = embeddings.detach().float().cpu()
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = F.normalize(z, p=2, dim=1)
    n = z.size(0)
    total = 0.0
    for start in range(0, n, ANISO_CHUNK_SIZE):
        end = min(start + ANISO_CHUNK_SIZE, n)
        total += float(torch.matmul(z[start:end], z.T).sum().item())
    total -= float(n)
    return total / float(n * (n - 1))


def get_effective_rank(matrix, max_samples=MAX_RANK_SAMPLES):
    """
    有效秩：
    根据奇异值分布熵计算有效维度。
    越高说明表征空间使用的维度越丰富。
    """
    if matrix.size(0) < 2:
        return float("nan")
    if max_samples is not None and matrix.size(0) > max_samples:
        idx = deterministic_subset_indices(matrix.size(0), max_samples, seed=SEED + 1)
        matrix = matrix[idx]

    x = matrix.detach().float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(dim=0, keepdim=True)
    x = x.cpu()
    try:
        singular_values = torch.linalg.svdvals(x)
    except RuntimeError:
        singular_values = torch.svd(x)[1]
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel() == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return float(torch.exp(entropy).item())


def get_entropy(logits):
    """
    预测熵：
    衡量预测不确定性。
    二分类任务最大值约为 ln(2)=0.693。
    """
    probs = F.softmax(logits, dim=-1)
    ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return ent.mean().item()


def get_entropy_values_from_logits(logits):
    probs = F.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)


def pct_delta(base, ft):
    if base is None or np.isnan(base) or abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (ft - base) / abs(base)


def minmax_normalize(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if finite.sum() == 0:
        return out.tolist()
    vals = arr[finite]
    lo = float(vals.min())
    hi = float(vals.max())
    if abs(hi - lo) < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (vals - lo) / (hi - lo)
    return out.tolist()


def add_normalized_columns(rows):
    metric_pairs = [
        ("alignment", "alignment_base", "alignment_ft"),
        ("anisotropy", "anisotropy_base_ood", "anisotropy_ft_ood"),
        ("rank", "effective_rank_base_ood", "effective_rank_ft_ood"),
        ("entropy", "entropy_base_ood", "entropy_ft_ood"),
    ]
    shifts = sorted(set(row["shift"] for row in rows))
    for shift in shifts:
        group = [row for row in rows if row["shift"] == shift]
        for metric_name, base_key, ft_key in metric_pairs:
            combined = [row[base_key] for row in group] + [row[ft_key] for row in group]
            normalized = minmax_normalize(combined)
            n = len(group)
            for i, row in enumerate(group):
                row[f"{metric_name}_base_n"] = float(normalized[i])
                row[f"{metric_name}_ft_n"] = float(normalized[n + i])
                row[f"{metric_name}_delta_n_ft_minus_base"] = float(normalized[n + i] - normalized[i])
    return rows


# ============================================================
# 4. 模型适配器：CodeBERT / GraphCodeBERT / CodeT5
# ============================================================

class UnifiedDefectModel(nn.Module):
    """
    支持 CodeBERT / GraphCodeBERT / CodeT5 的二分类模型。

    结构尽量与 Defect RQ1 保持一致：
    - encoder
    - dropout
    - linear classifier

    RQ3 中每一层的 entropy 使用 fine-tuned classifier 作用在该层 CLS 表征上。
    """

    def __init__(self, model_type, model_path, num_labels, load_pretrained=True):
        super().__init__()

        self.model_type = canonical_model_type(model_type).lower()
        self.pooling_strategy = resolve_pooling_strategy(model_type, ENCODER_POOLING)

        if "t5" in self.model_type:
            if load_pretrained:
                self.encoder = T5EncoderModel.from_pretrained(model_path)
            else:
                config = T5Config.from_pretrained(model_path)
                self.encoder = T5EncoderModel(config)

            hidden_size = self.encoder.config.d_model

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
                self.encoder = base_model

            hidden_size = getattr(base_model.config, "d_model", None)
            if hidden_size is None:
                hidden_size = getattr(base_model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot infer PLBART hidden size from config.")

        else:
            if load_pretrained:
                self.encoder = AutoModel.from_pretrained(model_path)
            else:
                config = AutoConfig.from_pretrained(model_path)
                self.encoder = AutoModel.from_config(config)

            hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def load_ft_weights(self, path):
        if not os.path.exists(path):
            print(f"[警告] 找不到 fine-tuned 权重: {path}")
            return False

        print(f">>> 正在加载 fine-tuned 权重: {path}")

        checkpoint = safe_torch_load(path, map_location="cpu")
        state_dict = unwrap_checkpoint(checkpoint)
        adapted_state = adapt_state_dict_keys(state_dict)

        model_state = self.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in adapted_state.items():
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
                filtered_state[key] = value
            elif key in model_state:
                skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))

        missing, unexpected = self.load_state_dict(filtered_state, strict=False)

        own_keys = set(self.state_dict().keys())
        matched = [k for k in filtered_state.keys() if k in own_keys]
        encoder_matched = [k for k in matched if k.startswith("encoder.")]
        classifier_matched = [k for k in matched if k.startswith("classifier.")]

        print(
            f">>> 权重加载完成: "
            f"matched={len(matched)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

        missing_show = [k for k in missing if "pooler" not in k][:30]
        unexpected_show = unexpected[:30]

        if missing_show:
            print(">>> Missing keys 前 30 个:", missing_show)

        if unexpected_show:
            print(">>> Unexpected keys 前 30 个:", unexpected_show)

        if skipped[:10]:
            print(">>> Skipped shape-mismatch keys first 10:", skipped[:10])

        return {
            "matched": len(matched),
            "encoder_matched": len(encoder_matched),
            "classifier_matched": len(classifier_matched),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "skipped": skipped,
            "pooling_strategy": self.pooling_strategy,
        }

    def forward_hidden_states(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        return outputs.hidden_states

    def pool_layer_hidden(self, hidden_states, attention_mask):
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def classify_pooled_features(self, pooled_features):
        return self.classifier(self.dropout(pooled_features))

    def logits_from_hidden(self, hidden_states, attention_mask):
        """
        hidden_states: [batch, seq_len, hidden]
        """
        pooled = self.pool_layer_hidden(hidden_states, attention_mask)
        return self.classify_pooled_features(pooled)


# ============================================================
# 5. ID/OOD 对齐数据集
# ============================================================

class AlignedDefectDataset(Dataset):
    """
    读取 id_test 和某个 ood_test 文件。
    默认要求二者按行对齐。
    检查：
      - idx 是否一致
      - label 是否一致
    """

    def __init__(self, data_dir, id_file, ood_file, tokenizer, max_samples=None):
        self.pairs = []
        self.tokenizer = tokenizer

        id_path = os.path.join(data_dir, id_file)
        ood_path = os.path.join(data_dir, ood_file)

        if not os.path.exists(id_path):
            raise FileNotFoundError(f"找不到 ID 文件: {id_path}")

        if not os.path.exists(ood_path):
            raise FileNotFoundError(f"找不到 OOD 文件: {ood_path}")

        with open(id_path, "r", encoding="utf-8") as f_id:
            id_lines = [line.strip() for line in f_id if line.strip()]

        with open(ood_path, "r", encoding="utf-8") as f_ood:
            ood_lines = [line.strip() for line in f_ood if line.strip()]

        n = min(len(id_lines), len(ood_lines))

        if max_samples is not None:
            n = min(n, max_samples)

        idx_mismatch = 0
        label_mismatch = 0
        bad_lines = 0

        for i in range(n):
            try:
                js_id = json.loads(id_lines[i])
                js_ood = json.loads(ood_lines[i])
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            id_code = get_code(js_id)
            ood_code = get_code(js_ood)

            if not id_code or not ood_code:
                bad_lines += 1
                continue

            id_idx = get_idx(js_id, i)
            ood_idx = get_idx(js_ood, i)

            id_label = get_label(js_id)
            ood_label = get_label(js_ood)

            current_idx_mismatch = id_idx != ood_idx
            current_label_mismatch = False

            if id_label is not None and ood_label is not None:
                current_label_mismatch = int(id_label) != int(ood_label)

            if current_idx_mismatch:
                idx_mismatch += 1

            if current_label_mismatch:
                label_mismatch += 1

            if (current_idx_mismatch or current_label_mismatch) and not ALLOW_MISMATCH:
                raise ValueError(
                    f"ID/OOD 对齐失败: line={i}, "
                    f"id_idx={id_idx}, ood_idx={ood_idx}, "
                    f"id_label={id_label}, ood_label={ood_label}"
                )

            self.pairs.append({
                "idx": id_idx,
                "id_code": id_code,
                "ood_code": ood_code,
                "label": int(id_label) if id_label is not None else -1
            })

        label_counter = Counter(x["label"] for x in self.pairs)

        print("=" * 100)
        print("AlignedDefectDataset 加载完成")
        print("=" * 100)
        print("ID file :", id_path)
        print("OOD file:", ood_path)
        print("样本数:", len(self.pairs))
        print("坏行数:", bad_lines)
        print("idx 不一致数量:", idx_mismatch)
        print("label 不一致数量:", label_mismatch)
        print("标签分布:", dict(label_counter))
        print("=" * 100)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        item = self.pairs[i]

        id_inputs = self.tokenizer(
            item["id_code"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        ood_inputs = self.tokenizer(
            item["ood_code"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "idx": item["idx"],
            "label": torch.tensor(item["label"], dtype=torch.long),

            "id_input_ids": id_inputs["input_ids"].squeeze(0),
            "id_attention_mask": id_inputs["attention_mask"].squeeze(0),

            "ood_input_ids": ood_inputs["input_ids"].squeeze(0),
            "ood_attention_mask": ood_inputs["attention_mask"].squeeze(0),
        }


# ============================================================
# 6. RQ3 层级分析
# ============================================================

@torch.no_grad()
def extract_layer_features(base_model, ft_model, dataloader, shift_name):
    base_model.eval()
    ft_model.eval()

    features = None
    entropy_stats = None
    labels_all = []

    for batch in tqdm(dataloader, desc=f"Extracting layer-wise defect features ({shift_name})"):
        id_input_ids = batch["id_input_ids"].to(DEVICE)
        id_attention_mask = batch["id_attention_mask"].to(DEVICE)
        ood_input_ids = batch["ood_input_ids"].to(DEVICE)
        ood_attention_mask = batch["ood_attention_mask"].to(DEVICE)
        labels_all.append(batch["label"].cpu().numpy())

        hs_base_id = base_model.forward_hidden_states(id_input_ids, id_attention_mask)
        hs_base_ood = base_model.forward_hidden_states(ood_input_ids, ood_attention_mask)
        hs_ft_id = ft_model.forward_hidden_states(id_input_ids, id_attention_mask)
        hs_ft_ood = ft_model.forward_hidden_states(ood_input_ids, ood_attention_mask)

        current_layers = min(len(hs_base_id), len(hs_base_ood), len(hs_ft_id), len(hs_ft_ood))
        if features is None:
            features = {
                "base_id": [[] for _ in range(current_layers)],
                "base_ood": [[] for _ in range(current_layers)],
                "ft_id": [[] for _ in range(current_layers)],
                "ft_ood": [[] for _ in range(current_layers)],
            }
            entropy_stats = {
                "entropy_base_ood": [[] for _ in range(current_layers)],
                "entropy_ft_ood": [[] for _ in range(current_layers)],
            }

        for layer in range(current_layers):
            b_id = base_model.pool_layer_hidden(hs_base_id[layer], id_attention_mask).detach().cpu()
            b_ood_dev = base_model.pool_layer_hidden(hs_base_ood[layer], ood_attention_mask)
            b_ood = b_ood_dev.detach().cpu()
            f_id = ft_model.pool_layer_hidden(hs_ft_id[layer], id_attention_mask).detach().cpu()
            f_ood_dev = ft_model.pool_layer_hidden(hs_ft_ood[layer], ood_attention_mask)
            f_ood = f_ood_dev.detach().cpu()

            features["base_id"][layer].append(b_id)
            features["base_ood"][layer].append(b_ood)
            features["ft_id"][layer].append(f_id)
            features["ft_ood"][layer].append(f_ood)

            logits_base = ft_model.classify_pooled_features(b_ood_dev)
            logits_ft = ft_model.classify_pooled_features(f_ood_dev)
            entropy_stats["entropy_base_ood"][layer].append(get_entropy_values_from_logits(logits_base).detach().cpu())
            entropy_stats["entropy_ft_ood"][layer].append(get_entropy_values_from_logits(logits_ft).detach().cpu())

    if features is None:
        raise ValueError("No features were extracted. Please check the dataset files.")

    labels = np.concatenate(labels_all, axis=0) if labels_all else np.array([])
    for key in features:
        for layer in range(len(features[key])):
            features[key][layer] = torch.cat(features[key][layer], dim=0)

    return features, entropy_stats, labels


def build_result_row(layer, shift_name, labels, b_id, b_ood, f_id, f_ood, entropy_stats):
    entropy_base_ood = tensor_mean_or_nan(entropy_stats["entropy_base_ood"][layer])
    entropy_ft_ood = tensor_mean_or_nan(entropy_stats["entropy_ft_ood"][layer])
    alignment_base = get_alignment(b_id, b_ood)
    alignment_ft = get_alignment(f_id, f_ood)
    aniso_base_ood = get_anisotropy(b_ood, max_samples=MAX_ANISO_SAMPLES)
    aniso_ft_ood = get_anisotropy(f_ood, max_samples=MAX_ANISO_SAMPLES)
    rank_base_ood = get_effective_rank(b_ood, max_samples=MAX_RANK_SAMPLES)
    rank_ft_ood = get_effective_rank(f_ood, max_samples=MAX_RANK_SAMPLES)

    return {
        "shift": shift_name,
        "layer": int(layer),
        "alignment_mode": "pair",
        "alignment_units": int(b_id.size(0)),
        "label_distribution": {str(k): int(v) for k, v in Counter(labels.tolist()).items()},
        "alignment_base": alignment_base,
        "alignment_ft": alignment_ft,
        "alignment_delta_ft_minus_base": alignment_ft - alignment_base,
        "alignment_delta_percent": pct_delta(alignment_base, alignment_ft),
        "anisotropy_base_ood": aniso_base_ood,
        "anisotropy_ft_ood": aniso_ft_ood,
        "anisotropy_delta_ood_ft_minus_base": aniso_ft_ood - aniso_base_ood,
        "anisotropy_delta_percent": pct_delta(aniso_base_ood, aniso_ft_ood),
        "effective_rank_base_ood": rank_base_ood,
        "effective_rank_ft_ood": rank_ft_ood,
        "effective_rank_delta_ood_ft_minus_base": rank_ft_ood - rank_base_ood,
        "effective_rank_delta_percent": pct_delta(rank_base_ood, rank_ft_ood),
        "entropy_base_ood": entropy_base_ood,
        "entropy_ft_ood": entropy_ft_ood,
        "entropy_delta_ood_ft_minus_base": entropy_ft_ood - entropy_base_ood,
        "entropy_delta_percent": pct_delta(entropy_base_ood, entropy_ft_ood),
    }


def aggregate_results(shift_name, features, entropy_stats, labels):
    results = []
    num_layers = len(features["base_id"])
    for layer in range(num_layers):
        results.append(build_result_row(
            layer=layer,
            shift_name=shift_name,
            labels=labels,
            b_id=features["base_id"][layer],
            b_ood=features["base_ood"][layer],
            f_id=features["ft_id"][layer],
            f_ood=features["ft_ood"][layer],
            entropy_stats=entropy_stats,
        ))
    return results

def print_result_table(results):
    print("\n" + "=" * 172)
    print(f"RQ3 Defect Layer-wise Diagnostics: {MODEL_TYPE.upper()}".center(172))
    print("=" * 172)
    print(f"{'Shift':<8} | {'Layer':<7} | {'Align B/FT/Delta':<28} | {'Aniso-OOD B/FT/Delta':<31} | {'EffRank-OOD B/FT/Delta':<34} | {'Entropy-OOD B/FT/Delta':<31}")
    print("-" * 172)
    for row in results:
        print(
            f"{row['shift']:<8} | "
            f"L{row['layer']:02d}     | "
            f"{row['alignment_base']:.4f} / {row['alignment_ft']:.4f} / {row['alignment_delta_ft_minus_base']:+.4f} | "
            f"{row['anisotropy_base_ood']:.4f} / {row['anisotropy_ft_ood']:.4f} / {row['anisotropy_delta_ood_ft_minus_base']:+.4f} | "
            f"{row['effective_rank_base_ood']:<8.2f} / {row['effective_rank_ft_ood']:<8.2f} / {row['effective_rank_delta_ood_ft_minus_base']:+.2f} | "
            f"{row['entropy_base_ood']:.4f} / {row['entropy_ft_ood']:.4f} / {row['entropy_delta_ood_ft_minus_base']:+.4f}"
        )
    print("=" * 172)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{float(x):.{digits}f}"


def build_excel_raw_rows(results):
    rows = [[
        "Shift", "Layer", "Align_base", "Align_ft", "Aniso_base", "Aniso_ft",
        "Rank_base", "Rank_ft", "Entropy_base", "Entropy_ft",
    ]]
    for row in results:
        rows.append([
            row["shift"],
            f"L{int(row['layer']):02d}",
            fmt_value(row["alignment_base"], 4),
            fmt_value(row["alignment_ft"], 4),
            fmt_value(row["anisotropy_base_ood"], 4),
            fmt_value(row["anisotropy_ft_ood"], 4),
            fmt_value(row["effective_rank_base_ood"], 4),
            fmt_value(row["effective_rank_ft_ood"], 4),
            fmt_value(row["entropy_base_ood"], 4),
            fmt_value(row["entropy_ft_ood"], 4),
        ])
    return rows


def build_excel_normalized_rows(results):
    rows = [[
        "Shift", "Layer", "Align_base_n", "Align_ft_n", "Aniso_base_n", "Aniso_ft_n",
        "Rank_base_n", "Rank_ft_n", "Entropy_base_n", "Entropy_ft_n",
    ]]
    for row in results:
        rows.append([
            row["shift"],
            f"L{int(row['layer']):02d}",
            fmt_value(row["alignment_base_n"], 4),
            fmt_value(row["alignment_ft_n"], 4),
            fmt_value(row["anisotropy_base_n"], 4),
            fmt_value(row["anisotropy_ft_n"], 4),
            fmt_value(row["rank_base_n"], 4),
            fmt_value(row["rank_ft_n"], 4),
            fmt_value(row["entropy_base_n"], 4),
            fmt_value(row["entropy_ft_n"], 4),
        ])
    return rows


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")


def print_excel_paste_table(results):
    print("\n" + "=" * 172)
    print(f"EXCEL_COPY_RAW\t{DATASET_NAME}\t{SHIFT_TYPE}\t{MODEL_TYPE}")
    print("=" * 172)
    for row in build_excel_raw_rows(results):
        print("\t".join(row))
    print("=" * 172)

    print("\n" + "=" * 172)
    print(f"EXCEL_COPY_NORMALIZED\t{DATASET_NAME}\t{SHIFT_TYPE}\t{MODEL_TYPE}")
    print("=" * 172)
    for row in build_excel_normalized_rows(results):
        print("\t".join(row))
    print("=" * 172)


def save_results(results, load_info, dataset_info):
    meta = {
        "task": TASK_NAME,
        "dataset": DATASET_NAME,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "ft_model_weights": get_best_model_path(MODEL_TYPE),
        "encoder_pooling": load_info.get("pooling_strategy") if isinstance(load_info, dict) else None,
        "load_info": load_info,
        "data_dir": DATA_DIR,
        "id_test_file": ID_TEST_FILE,
        "ood_test_files": OOD_TEST_FILES,
        "shift_type": SHIFT_TYPE,
        "num_labels": NUM_LABELS,
        "max_len": MAX_LEN,
        "max_analysis_samples": MAX_ANALYSIS_SAMPLES,
        "max_aniso_samples": MAX_ANISO_SAMPLES,
        "max_rank_samples": MAX_RANK_SAMPLES,
        "aniso_chunk_size": ANISO_CHUNK_SIZE,
        "dataset_info": dataset_info,
        "metrics": ["alignment", "anisotropy", "effective_rank", "entropy"],
        "normalization": {
            "method": "min-max",
            "scope": "per model + shift + metric, using base and fine-tuned values from all analyzed layers together",
            "columns": "raw columns are preserved; *_n columns are normalized to [0, 1] for plotting",
        },
        "notes": [
            "Alignment is line-wise ID/OOD cosine similarity for token/cst shifts.",
            "Anisotropy and Effective Rank use all OOD representations; chunking is only for exact memory-safe summation.",
            "Entropy is fixed-head probing entropy using the fine-tuned defect classifier head.",
            "Raw metrics are not normalized. Normalized TSV is display-only for plotting.",
        ],
        "results": results,
        "elapsed_seconds": time.time() - START_TIME,
        "elapsed_readable": format_time(time.time() - START_TIME),
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    write_csv(RESULT_CSV, results)
    write_tsv(RESULT_RAW_PASTE_TSV, build_excel_raw_rows(results))
    write_tsv(RESULT_NORMALIZED_PASTE_TSV, build_excel_normalized_rows(results))

    print("\nResults saved:")
    print("JSON          :", RESULT_JSON)
    print("CSV           :", RESULT_CSV)
    print("Raw TSV       :", RESULT_RAW_PASTE_TSV)
    print("Normalized TSV:", RESULT_NORMALIZED_PASTE_TSV)

def init_models():
    ft_weights = get_best_model_path(MODEL_TYPE)
    if not os.path.exists(ft_weights):
        raise FileNotFoundError(f"Cannot find RQ1 best defect model: {ft_weights}")

    print("=" * 100)
    print("Model initialization")
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("FT_WEIGHTS:", ft_weights)
    print("ENCODER_POOLING:", ENCODER_POOLING)
    print("=" * 100)

    base_model = UnifiedDefectModel(MODEL_TYPE, MODEL_PATH, NUM_LABELS, load_pretrained=True).to(DEVICE)
    ft_model = UnifiedDefectModel(MODEL_TYPE, MODEL_PATH, NUM_LABELS, load_pretrained=True).to(DEVICE)
    load_info = ft_model.load_ft_weights(ft_weights)
    if not load_info:
        raise FileNotFoundError(f"Cannot load fine-tuned weights: {ft_weights}")

    base_model.eval()
    ft_model.eval()
    return base_model, ft_model, load_info


def run_one_model(model_type):
    configure_model_runtime(model_type, SHIFT_TYPE)
    model_start = time.time()

    print("=" * 120)
    print("RQ3: Layer-wise Representation-Space Diagnostics for Defect Detection")
    print("=" * 120)
    print("TASK_NAME:", TASK_NAME)
    print("DATASET_NAME:", DATASET_NAME)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("MODEL_PATH:", MODEL_PATH)
    print("FT_MODEL_WEIGHTS:", get_best_model_path(MODEL_TYPE))
    print("DATA_DIR:", DATA_DIR)
    print("ID_TEST_FILE:", ID_TEST_FILE)
    print("OOD_TEST_FILES:", OOD_TEST_FILES)
    print("SHIFT_TYPE:", SHIFT_TYPE)
    print("DEVICE:", DEVICE)
    print("MAX_ANALYSIS_SAMPLES:", MAX_ANALYSIS_SAMPLES)
    print("MAX_ANISO_SAMPLES:", MAX_ANISO_SAMPLES)
    print("MAX_RANK_SAMPLES:", MAX_RANK_SAMPLES)
    print("=" * 120)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model, ft_model, load_info = init_models()

    shifts_to_run = list(OOD_TEST_FILES.keys()) if SHIFT_TYPE == "all" else [SHIFT_TYPE]
    all_results = []
    dataset_info = {}

    for shift_name in shifts_to_run:
        if shift_name not in OOD_TEST_FILES:
            raise ValueError(f"Unsupported SHIFT_TYPE={shift_name}. Choose from token, cst, all.")
        ood_file = OOD_TEST_FILES[shift_name]

        print("\n" + "#" * 120)
        print(f"Start RQ3 analysis: Defect Detection - {shift_name.upper()} Shift")
        print("#" * 120)

        dataset = AlignedDefectDataset(
            data_dir=DATA_DIR,
            id_file=ID_TEST_FILE,
            ood_file=ood_file,
            tokenizer=tokenizer,
            max_samples=MAX_ANALYSIS_SAMPLES,
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        features, entropy_stats, labels = extract_layer_features(base_model, ft_model, dataloader, shift_name)
        shift_results = aggregate_results(shift_name, features, entropy_stats, labels)
        all_results.extend(shift_results)
        dataset_info[shift_name] = {
            "num_samples": len(dataset),
            "id_file": os.path.join(DATA_DIR, ID_TEST_FILE),
            "ood_file": os.path.join(DATA_DIR, ood_file),
            "label_distribution": {str(k): int(v) for k, v in Counter(labels.tolist()).items()},
        }

    all_results = add_normalized_columns(all_results)
    print_result_table(all_results)
    save_results(all_results, load_info=load_info, dataset_info=dataset_info)
    print_excel_paste_table(all_results)

    print("\nFinished.")
    print("Elapsed:", format_time(time.time() - model_start))

    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "ft_model_weights": get_best_model_path(MODEL_TYPE),
        "pooling_strategy": load_info.get("pooling_strategy") if isinstance(load_info, dict) else None,
        "num_rows": len(all_results),
        "result_json": RESULT_JSON,
        "result_csv": RESULT_CSV,
        "result_raw_tsv": RESULT_RAW_PASTE_TSV,
        "result_normalized_tsv": RESULT_NORMALIZED_PASTE_TSV,
        "elapsed_seconds": time.time() - model_start,
        "elapsed_readable": format_time(time.time() - model_start),
    }


def main():
    models_to_run = MODEL_TYPES_TO_RUN if RUN_ALL_MODELS else [MODEL_TYPE]
    summaries = []

    for run_model_type in models_to_run:
        try:
            summaries.append(run_one_model(run_model_type))
        except Exception as exc:
            if not RUN_ALL_MODELS:
                raise
            summaries.append({
                "status": "failed",
                "model_type": run_model_type,
                "error": repr(exc),
            })
            print("\n" + "!" * 120)
            print(f"[Error] RQ3 failed for model={run_model_type}: {repr(exc)}")
            print("Continuing with the next model because RUN_ALL_MODELS=True.")
            print("!" * 120)
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if RUN_ALL_MODELS:
        all_summary = {
            "task": TASK_NAME,
            "dataset": DATASET_NAME,
            "shift_type": SHIFT_TYPE,
            "models_requested": models_to_run,
            "summaries": summaries,
            "elapsed_seconds": time.time() - START_TIME,
            "elapsed_readable": format_time(time.time() - START_TIME),
        }
        with open(ALL_MODELS_RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_summary, f, ensure_ascii=False, indent=2)
        print("\nAll-model RQ3 summary saved:", ALL_MODELS_RESULT_JSON)
        print("Total elapsed:", format_time(time.time() - START_TIME))

if __name__ == "__main__":
    main()



