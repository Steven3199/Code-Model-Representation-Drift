import os
import json
import random
import hashlib
import torch
import numpy as np
from tree_sitter import Language, Parser
from tqdm import tqdm
import sys
import io


# ================= 编码设置 =================

if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================
# 1. 配置区
# ============================================================

# NatGen / tree-sitter-python 路径
PYTHON_REPO_PATH = "/opt/data/private/NatGen/build/tree-sitter-python"
LIB_PATH = "/opt/data/private/NatGen/build/language.so"

# 输入：CodeSearchNet Python 的 ID Test
INPUT_FILE = "/opt/data/private/CodeSearchNet/python/id_test.jsonl"

# 输出：OOD Test
OUTPUT_TOKEN_SHIFT = "/opt/data/private/CodeSearchNet/python/ood_test_token.jsonl"
OUTPUT_CST_SHIFT = "/opt/data/private/CodeSearchNet/python/ood_test_cst.jsonl"

SEED = 42

# CST 插入概率
CST_INSERT_PROB = 0.5

# 删除旧 code_tokens，避免 RQ1 优先读取旧 token
REMOVE_CODE_TOKENS = True

# 如果 shift 后有 parse error，是否回退到原始代码
# 注意：这里不是跳过样本，而是 fallback，保证 ID/OOD 对齐
FALLBACK_ON_PARSE_ERROR = True


# ============================================================
# 2. 初始化 tree-sitter Python parser
# ============================================================

if not os.path.exists(LIB_PATH):
    print(f">>> 正在编译 Python 语法树解析器到: {LIB_PATH} ...")
    Language.build_library(
        LIB_PATH,
        [PYTHON_REPO_PATH]
    )

PYTHON_LANGUAGE = Language(LIB_PATH, "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# 3. 通用工具函数
# ============================================================

def md5_text(text):
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def get_sample_id(data, line_idx):
    return str(
        data.get("idx")
        or data.get("id")
        or data.get("url")
        or line_idx
    )


def get_code_from_json(data):
    """
    优先读取 func/code。
    如果没有，则退回 code_tokens。
    """
    code = data.get("func") or data.get("code")

    if not code and "code_tokens" in data:
        code = " ".join(data["code_tokens"])

    return code or ""


def has_parse_error(code):
    """
    检查 tree-sitter 解析树中是否存在 ERROR 节点。
    """
    try:
        code_bytes = code.encode("utf-8", errors="ignore")
        tree = parser.parse(code_bytes)

        def walk(node):
            if node.type == "ERROR":
                return True
            for child in node.children:
                if walk(child):
                    return True
            return False

        return walk(tree.root_node)

    except Exception:
        return True


def count_cst_nodes(code):
    """
    统计 CST 节点数量。
    """
    try:
        code_bytes = code.encode("utf-8", errors="ignore")
        tree = parser.parse(code_bytes)

        def walk(node):
            return 1 + sum(walk(child) for child in node.children)

        return walk(tree.root_node)

    except Exception:
        return 0


def simple_token_change_ratio(original, shifted):
    """
    粗略统计 token 变化比例。
    """
    ori_tokens = original.split()
    shi_tokens = shifted.split()

    max_len = max(len(ori_tokens), len(shi_tokens), 1)
    min_len = min(len(ori_tokens), len(shi_tokens))

    diff = 0

    for i in range(min_len):
        if ori_tokens[i] != shi_tokens[i]:
            diff += 1

    diff += abs(len(ori_tokens) - len(shi_tokens))

    return diff / max_len


def build_shifted_json(
    data,
    shifted_code,
    shift_type,
    original_code,
    shift_success,
    fallback_reason,
    parse_error_after_shift,
    extra_meta=None
):
    """
    构造 OOD 样本。

    关键点：
    1. 更新 func/code；
    2. 删除旧 code_tokens；
    3. 保留原始其他字段；
    4. 添加 shift 元信息。
    """
    new_data = data.copy()

    new_data["func"] = shifted_code
    new_data["code"] = shifted_code
    new_data["shift_type"] = shift_type

    new_data["shift_success"] = bool(shift_success)
    new_data["fallback_reason"] = fallback_reason
    new_data["parse_error_after_shift"] = bool(parse_error_after_shift)

    new_data["original_code_md5"] = md5_text(original_code)
    new_data["shifted_code_md5"] = md5_text(shifted_code)

    if REMOVE_CODE_TOKENS and "code_tokens" in new_data:
        del new_data["code_tokens"]

    if extra_meta:
        for k, v in extra_meta.items():
            new_data[k] = v

    return new_data


# ============================================================
# 4. Shift 引擎
# ============================================================

class ShiftEngine:
    @staticmethod
    def apply_token_shift(code):
        """
        Token Shift:
        使用 tree-sitter Python parser 定位 identifier。
        默认会替换函数名、变量名、参数名等 identifier。
        保护 Python 内置名、魔术方法、常见库别名、属性名、import 名称。
        """
        code_bytes = code.encode("utf-8", errors="ignore")
        tree = parser.parse(code_bytes)

        identifiers = []

        python_builtins = set(dir(__builtins__)) | {
            "self", "cls", "args", "kwargs",
            "return", "yield",
            "True", "False", "None",
            "list", "dict", "set", "str", "int", "float", "bool", "tuple",
            "len", "range", "print", "open", "enumerate", "zip", "map", "filter",
            "Exception", "ValueError", "TypeError", "IndexError", "KeyError"
        }

        common_aliases = {
            "np", "pd", "plt", "torch", "tf", "nn", "F",
            "os", "sys", "re", "json", "math", "time", "random",
            "collections", "itertools", "functools"
        }

        protected_names = python_builtins | common_aliases

        def is_inside_import(node):
            ancestor = node.parent
            while ancestor is not None:
                if ancestor.type in {
                    "import_statement",
                    "import_from_statement",
                    "dotted_name",
                    "aliased_import"
                }:
                    return True
                ancestor = ancestor.parent
            return False

        def is_inside_decorator(node):
            ancestor = node.parent
            while ancestor is not None:
                if ancestor.type == "decorator":
                    return True
                ancestor = ancestor.parent
            return False

        def find_ids(node):
            # attribute: obj.attr
            # 保护 attr，只继续处理 object
            if node.type == "attribute":
                obj = node.child_by_field_name("object")
                if obj is not None:
                    find_ids(obj)
                return

            if node.type == "identifier":
                name = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

                if is_inside_import(node):
                    return

                if is_inside_decorator(node):
                    return

                if (
                    name not in protected_names
                    and not name.startswith("__")
                    and not name.endswith("__")
                ):
                    identifiers.append(node)

            for child in node.children:
                find_ids(child)

        find_ids(tree.root_node)

        raw_names = [
            code_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")
            for n in identifiers
        ]

        unique_names = []
        for name in raw_names:
            if name not in unique_names:
                unique_names.append(name)

        name_map = {
            name: f"var_{i}"
            for i, name in enumerate(unique_names)
        }

        sorted_nodes = sorted(
            identifiers,
            key=lambda x: x.start_byte,
            reverse=True
        )

        new_code_bytes = code_bytes

        for node in sorted_nodes:
            old_name = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

            if old_name in name_map:
                new_name_bytes = name_map[old_name].encode("utf-8")
                new_code_bytes = (
                    new_code_bytes[:node.start_byte]
                    + new_name_bytes
                    + new_code_bytes[node.end_byte:]
                )

        return new_code_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def apply_cst_shift(code):
        """
        CST Shift:
        插入不会执行或不会影响语义的结构片段，并做轻量表达式改写。
        """
        lines = code.split("\n")
        new_lines = []

        valid_block_starters = (
            "def ",
            "class ",
            "if ",
            "for ",
            "while ",
            "elif ",
            "else:",
            "try:",
            "except ",
            "finally:",
            "with "
        )

        for line in lines:
            new_lines.append(line)
            stripped_line = line.strip()

            if stripped_line.endswith(":") and stripped_line.startswith(valid_block_starters):
                if random.random() < CST_INSERT_PROB:
                    current_indent = len(line) - len(line.lstrip())
                    base_indent = " " * (current_indent + 4)

                    dead_codes = [
                        (
                            f"{base_indent}if 1 > 2:\n"
                            f"{base_indent}    pass"
                        ),
                        (
                            f"{base_indent}_unused_v = 0\n"
                            f"{base_indent}while _unused_v < 0:\n"
                            f"{base_indent}    _unused_v += 1"
                        ),
                        (
                            f"{base_indent}try:\n"
                            f"{base_indent}    pass\n"
                            f"{base_indent}except Exception:\n"
                            f"{base_indent}    pass"
                        )
                    ]

                    new_lines.append(random.choice(dead_codes))

        shifted_code = "\n".join(new_lines)

        shifted_code = shifted_code.replace(" += 1\n", " += 1 + 1 - 1\n")
        shifted_code = shifted_code.replace(" -= 1\n", " -= 1 + 1 - 1\n")

        return shifted_code


# ============================================================
# 5. 安全生成单条 Token Shift 样本
# ============================================================

def generate_token_shift_sample(data, original_code):
    """
    每条样本都必须返回一个可写入的 OOD JSON。
    失败时回退到原始代码，但不跳过。
    """
    try:
        shifted_code = ShiftEngine.apply_token_shift(original_code)
        parse_error = has_parse_error(shifted_code)

        if shifted_code.strip() == original_code.strip():
            return build_shifted_json(
                data=data,
                shifted_code=original_code,
                shift_type="token",
                original_code=original_code,
                shift_success=False,
                fallback_reason="no_change",
                parse_error_after_shift=parse_error,
                extra_meta={
                    "token_change_ratio": 0.0
                }
            )

        if FALLBACK_ON_PARSE_ERROR and parse_error:
            return build_shifted_json(
                data=data,
                shifted_code=original_code,
                shift_type="token",
                original_code=original_code,
                shift_success=False,
                fallback_reason="parse_error",
                parse_error_after_shift=True,
                extra_meta={
                    "token_change_ratio": 0.0
                }
            )

        ratio = simple_token_change_ratio(original_code, shifted_code)

        return build_shifted_json(
            data=data,
            shifted_code=shifted_code,
            shift_type="token",
            original_code=original_code,
            shift_success=True,
            fallback_reason="",
            parse_error_after_shift=parse_error,
            extra_meta={
                "token_change_ratio": ratio
            }
        )

    except Exception as e:
        return build_shifted_json(
            data=data,
            shifted_code=original_code,
            shift_type="token",
            original_code=original_code,
            shift_success=False,
            fallback_reason=f"exception:{type(e).__name__}",
            parse_error_after_shift=True,
            extra_meta={
                "token_change_ratio": 0.0
            }
        )


# ============================================================
# 6. 安全生成单条 CST Shift 样本
# ============================================================

def generate_cst_shift_sample(data, original_code):
    """
    每条样本都必须返回一个可写入的 OOD JSON。
    失败时回退到原始代码，但不跳过。
    """
    try:
        before_nodes = count_cst_nodes(original_code)
        shifted_code = ShiftEngine.apply_cst_shift(original_code)
        after_nodes = count_cst_nodes(shifted_code)
        parse_error = has_parse_error(shifted_code)

        if shifted_code.strip() == original_code.strip():
            return build_shifted_json(
                data=data,
                shifted_code=original_code,
                shift_type="cst",
                original_code=original_code,
                shift_success=False,
                fallback_reason="no_change",
                parse_error_after_shift=parse_error,
                extra_meta={
                    "cst_nodes_before": before_nodes,
                    "cst_nodes_after": before_nodes,
                    "cst_node_change": 0,
                    "token_change_ratio": 0.0
                }
            )

        if FALLBACK_ON_PARSE_ERROR and parse_error:
            return build_shifted_json(
                data=data,
                shifted_code=original_code,
                shift_type="cst",
                original_code=original_code,
                shift_success=False,
                fallback_reason="parse_error",
                parse_error_after_shift=True,
                extra_meta={
                    "cst_nodes_before": before_nodes,
                    "cst_nodes_after": before_nodes,
                    "cst_node_change": 0,
                    "token_change_ratio": 0.0
                }
            )

        node_change = after_nodes - before_nodes
        ratio = simple_token_change_ratio(original_code, shifted_code)

        return build_shifted_json(
            data=data,
            shifted_code=shifted_code,
            shift_type="cst",
            original_code=original_code,
            shift_success=True,
            fallback_reason="",
            parse_error_after_shift=parse_error,
            extra_meta={
                "cst_nodes_before": before_nodes,
                "cst_nodes_after": after_nodes,
                "cst_node_change": node_change,
                "token_change_ratio": ratio
            }
        )

    except Exception as e:
        before_nodes = count_cst_nodes(original_code)

        return build_shifted_json(
            data=data,
            shifted_code=original_code,
            shift_type="cst",
            original_code=original_code,
            shift_success=False,
            fallback_reason=f"exception:{type(e).__name__}",
            parse_error_after_shift=True,
            extra_meta={
                "cst_nodes_before": before_nodes,
                "cst_nodes_after": before_nodes,
                "cst_node_change": 0,
                "token_change_ratio": 0.0
            }
        )


# ============================================================
# 7. 主处理流程
# ============================================================

def main():
    print("=" * 80)
    print("CodeSearchNet Python: Generate Token/CST Shift OOD Test")
    print("Using NatGen tree-sitter-python parser")
    print("=" * 80)
    print(f"输入文件: {INPUT_FILE}")
    print(f"Token Shift 输出: {OUTPUT_TOKEN_SHIFT}")
    print(f"CST Shift 输出: {OUTPUT_CST_SHIFT}")
    print(f"REMOVE_CODE_TOKENS = {REMOVE_CODE_TOKENS}")
    print(f"FALLBACK_ON_PARSE_ERROR = {FALLBACK_ON_PARSE_ERROR}")
    print(f"CST_INSERT_PROB = {CST_INSERT_PROB}")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: 找不到输入文件 {INPUT_FILE}")
        return

    os.makedirs(os.path.dirname(OUTPUT_TOKEN_SHIFT), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CST_SHIFT), exist_ok=True)

    total = 0
    skipped = 0

    token_success = 0
    token_no_change = 0
    token_parse_error_fallback = 0
    token_exception_fallback = 0
    token_change_ratios = []

    cst_success = 0
    cst_no_change = 0
    cst_parse_error_fallback = 0
    cst_exception_fallback = 0
    cst_node_changes = []
    cst_token_change_ratios = []

    id_order = []
    token_order = []
    cst_order = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
            open(OUTPUT_TOKEN_SHIFT, "w", encoding="utf-8") as f_token, \
            open(OUTPUT_CST_SHIFT, "w", encoding="utf-8") as f_cst:

        for line_idx, line in enumerate(tqdm(f_in, desc="生成 OOD 数据中")):
            line = line.strip()

            if not line:
                skipped += 1
                continue

            try:
                data = json.loads(line)
            except Exception:
                skipped += 1
                continue

            original_code = get_code_from_json(data)

            if not original_code:
                skipped += 1
                continue

            sample_id = get_sample_id(data, line_idx)
            id_order.append(sample_id)

            total += 1

            # Token Shift
            token_data = generate_token_shift_sample(data, original_code)
            token_order.append(sample_id)
            f_token.write(json.dumps(token_data, ensure_ascii=False) + "\n")

            if token_data["shift_success"]:
                token_success += 1
                token_change_ratios.append(token_data.get("token_change_ratio", 0.0))
            else:
                reason = token_data.get("fallback_reason", "")
                if reason == "no_change":
                    token_no_change += 1
                elif reason == "parse_error":
                    token_parse_error_fallback += 1
                elif reason.startswith("exception"):
                    token_exception_fallback += 1

            # CST Shift
            cst_data = generate_cst_shift_sample(data, original_code)
            cst_order.append(sample_id)
            f_cst.write(json.dumps(cst_data, ensure_ascii=False) + "\n")

            if cst_data["shift_success"]:
                cst_success += 1
                cst_node_changes.append(cst_data.get("cst_node_change", 0))
                cst_token_change_ratios.append(cst_data.get("token_change_ratio", 0.0))
            else:
                reason = cst_data.get("fallback_reason", "")
                if reason == "no_change":
                    cst_no_change += 1
                elif reason == "parse_error":
                    cst_parse_error_fallback += 1
                elif reason.startswith("exception"):
                    cst_exception_fallback += 1

    def avg(values):
        if len(values) == 0:
            return 0.0
        return sum(values) / len(values)

    print("\n" + "=" * 80)
    print("生成报告")
    print("=" * 80)
    print(f"原始有效样本数: {total}")
    print(f"跳过空行/坏行/无代码样本数: {skipped}")

    print("\n[Token Shift]")
    print(f"写入数量: {len(token_order)}")
    print(f"成功 shift 数量: {token_success}")
    print(f"无变化 fallback 数量: {token_no_change}")
    print(f"解析失败 fallback 数量: {token_parse_error_fallback}")
    print(f"异常 fallback 数量: {token_exception_fallback}")
    print(f"成功 shift 比例: {token_success / max(total, 1):.4f}")
    print(f"平均 token 改变比例: {avg(token_change_ratios):.4f}")

    print("\n[CST Shift]")
    print(f"写入数量: {len(cst_order)}")
    print(f"成功 shift 数量: {cst_success}")
    print(f"无变化 fallback 数量: {cst_no_change}")
    print(f"解析失败 fallback 数量: {cst_parse_error_fallback}")
    print(f"异常 fallback 数量: {cst_exception_fallback}")
    print(f"成功 shift 比例: {cst_success / max(total, 1):.4f}")
    print(f"平均 CST 节点变化数: {avg(cst_node_changes):.2f}")
    print(f"平均 token 改变比例: {avg(cst_token_change_ratios):.4f}")

    print("\n[一致性检查]")
    print("Token Shift 与 ID 数量一致:", len(token_order) == len(id_order))
    print("CST Shift 与 ID 数量一致:", len(cst_order) == len(id_order))
    print("Token Shift 与 ID 顺序一致:", token_order == id_order)
    print("CST Shift 与 ID 顺序一致:", cst_order == id_order)

    print("\n[输出文件]")
    print(f"Token Shift 文件: {OUTPUT_TOKEN_SHIFT}")
    print(f"CST Shift 文件: {OUTPUT_CST_SHIFT}")

    print("\n注意：本代码已删除 OOD 文件中的旧 code_tokens 字段。")
    print("这样你的 Dataset 即使优先读取 code_tokens，也会读取修改后的 func/code。")
    print("=" * 80)


if __name__ == "__main__":
    main()