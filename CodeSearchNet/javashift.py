import os
import json
import random
import numpy as np
from tree_sitter import Language, Parser
from tqdm import tqdm
import sys
import io


# ============================================================
# 0. 编码设置
# ============================================================

if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================
# 1. 配置区
# ============================================================

# 使用 NatGen 编译好的 tree-sitter parser library
# 这是你当前环境中比较权威、通用的 parser 基础
LIB_PATH = "/opt/data/private/NatGen/parser/languages.so"

DATA_DIR = "/opt/data/private/CodeSearchNet/java"

# 输入：Java 版 CodeSearchNet 的 ID Test
INPUT_FILE = os.path.join(DATA_DIR, "id_test.jsonl")

# 输出：Token Shift / CST Shift OOD Test
OUTPUT_TOKEN_SHIFT = os.path.join(DATA_DIR, "ood_test_token.jsonl")
OUTPUT_CST_SHIFT = os.path.join(DATA_DIR, "ood_test_cst.jsonl")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# CST 扰动比例
CST_INSERT_PROB = 1.0


# ============================================================
# 2. 初始化 NatGen Java Parser
# ============================================================

JAVA_LANGUAGE = Language(LIB_PATH, "java")
parser = Parser()
parser.set_language(JAVA_LANGUAGE)


# ============================================================
# 3. Java 关键字和保护名单
# ============================================================

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false", "null"
}

COMMON_JAVA_NAMES = {
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Character",
    "Object", "Math", "System", "Arrays", "ArrayList", "List", "Map", "HashMap",
    "HashSet", "Set", "Collections", "Scanner", "BufferedReader", "StringBuilder",
    "StringBuffer", "IOException", "Exception", "RuntimeException", "Thread",
    "Runnable", "Comparable", "Comparator", "Override", "Deprecated",
    "out", "in", "err", "println", "print", "printf", "length", "size",
    "add", "get", "set", "put", "contains", "equals", "hashCode", "toString",
    "main"
}


# ============================================================
# 4. tree-sitter / NatGen parser 工具函数
# ============================================================

def parse_java_code(code):
    return parser.parse(code.encode("utf-8", errors="ignore"))


def node_text(code_bytes, node):
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def walk_tree(node):
    yield node
    for child in node.children:
        yield from walk_tree(child)


def is_same_node(a, b):
    if a is None or b is None:
        return False
    return (
        a.start_byte == b.start_byte
        and a.end_byte == b.end_byte
        and a.type == b.type
    )


def replace_by_spans(code, replacements):
    """
    replacements: [(start_byte, end_byte, new_text), ...]

    使用 byte offset 进行替换，避免 Java 代码中出现 Unicode 字符时
    Python 字符串下标和 tree-sitter byte offset 不一致。
    """
    code_bytes = code.encode("utf-8", errors="ignore")
    out = code_bytes

    for start, end, new_text in sorted(replacements, key=lambda x: x[0], reverse=True):
        out = out[:start] + new_text.encode("utf-8") + out[end:]

    return out.decode("utf-8", errors="ignore")


def is_inside_import_or_package(node):
    ancestor = node.parent
    while ancestor is not None:
        if ancestor.type in {"import_declaration", "package_declaration"}:
            return True
        ancestor = ancestor.parent
    return False


def is_inside_annotation(node):
    ancestor = node.parent
    while ancestor is not None:
        if "annotation" in ancestor.type:
            return True
        ancestor = ancestor.parent
    return False


# ============================================================
# 5. Token Shift：基于 NatGen parser 的 identifier rename
# ============================================================

def should_skip_identifier(code_bytes, node):
    """
    保守跳过不应替换的 identifier：
    - Java 关键字；
    - 常见类型 / 库名；
    - 方法调用名；
    - 方法声明名；
    - 构造器名；
    - 类名 / 接口名 / 枚举名；
    - 字段访问名；
    - import / package / annotation 中的名字；
    - 全大写常量；
    - 首字母大写的类型名风格 identifier。
    """
    name = node_text(code_bytes, node)

    if not name:
        return True

    if name in JAVA_KEYWORDS:
        return True

    if name in COMMON_JAVA_NAMES:
        return True

    if name.startswith("__"):
        return True

    if name.isupper() and len(name) > 1:
        return True

    if name[0].isupper():
        return True

    if is_inside_import_or_package(node):
        return True

    if is_inside_annotation(node):
        return True

    parent = node.parent
    parent_type = parent.type if parent is not None else ""

    # 方法调用名，例如 foo() 中的 foo
    if parent_type == "method_invocation":
        name_node = parent.child_by_field_name("name")
        if is_same_node(name_node, node):
            return True

    # 方法声明名
    if parent_type == "method_declaration":
        name_node = parent.child_by_field_name("name")
        if is_same_node(name_node, node):
            return True

    # 构造器声明名
    if parent_type == "constructor_declaration":
        name_node = parent.child_by_field_name("name")
        if is_same_node(name_node, node):
            return True

    # 类 / 接口 / 枚举声明名
    if parent_type in {"class_declaration", "interface_declaration", "enum_declaration"}:
        name_node = parent.child_by_field_name("name")
        if is_same_node(name_node, node):
            return True

    # 字段访问，例如 obj.field 中的 field
    if parent_type == "field_access":
        field_node = parent.child_by_field_name("field")
        if is_same_node(field_node, node):
            return True

    # 包名、类名、静态字段等，保守跳过
    if parent_type in {"scoped_identifier", "qualified_identifier"}:
        return True

    # 类型相关
    if parent_type in {
        "type_identifier",
        "generic_type",
        "integral_type",
        "floating_point_type",
        "boolean_type",
        "void_type",
        "array_type"
    }:
        return True

    return False


def collect_identifier_replacements(code):
    code_bytes = code.encode("utf-8", errors="ignore")
    tree = parse_java_code(code)
    root = tree.root_node

    identifiers = []

    for node in walk_tree(root):
        if node.type != "identifier":
            continue

        if should_skip_identifier(code_bytes, node):
            continue

        name = node_text(code_bytes, node)
        identifiers.append((name, node.start_byte, node.end_byte))

    unique_names = []
    for name, _, _ in identifiers:
        if name not in unique_names:
            unique_names.append(name)

    name_map = {
        name: f"var_{i}"
        for i, name in enumerate(unique_names)
    }

    replacements = []
    for name, start, end in identifiers:
        replacements.append((start, end, name_map[name]))

    return replacements


def apply_token_shift_java(code):
    """
    Token Shift:
    使用 NatGen parser / tree-sitter 定位 Java identifier，
    只保守替换局部变量、参数等标识符。
    """
    try:
        replacements = collect_identifier_replacements(code)
        if not replacements:
            return code
        return replace_by_spans(code, replacements)
    except Exception:
        return code


# ============================================================
# 6. CST Shift：基于 NatGen parser 的轻量结构扰动
# ============================================================

def find_first_method_body(code):
    """
    找第一个 Java 方法体 block。
    CodeSearchNet Java 通常是方法级代码，但有时不一定是完整 class。
    """
    try:
        tree = parse_java_code(code)
        root = tree.root_node

        for node in walk_tree(root):
            if node.type in {"method_declaration", "constructor_declaration"}:
                body = node.child_by_field_name("body")
                if body is not None and body.type == "block":
                    return body

        # fallback：找第一个 block
        for node in walk_tree(root):
            if node.type == "block":
                return node

    except Exception:
        return None

    return None


def apply_cst_shift_java(code):
    """
    CST Shift:
    使用 NatGen parser 定位 Java 方法体，并插入轻量结构扰动。
    不插入 return，不改 ++，尽量避免破坏语义。
    """
    if random.random() > CST_INSERT_PROB:
        return code

    try:
        body = find_first_method_body(code)

        templates = [
            "\n        ;\n",
            "\n        { }\n",
            "\n        { ; }\n"
        ]

        if body is not None:
            insert_pos = body.start_byte + 1
            insert_code = random.choice(templates)

            code_bytes = code.encode("utf-8", errors="ignore")
            shifted_bytes = (
                code_bytes[:insert_pos]
                + insert_code.encode("utf-8")
                + code_bytes[insert_pos:]
            )

            shifted = shifted_bytes.decode("utf-8", errors="ignore")
        else:
            # 如果 parser 找不到 block，就做保守 fallback
            pos = code.find("{")
            if pos == -1:
                return code

            shifted = code[:pos + 1] + random.choice(templates) + code[pos + 1:]

        # 安全的轻量表达式改写
        shifted = shifted.replace(" += 1;", " += 1 + 0;")
        shifted = shifted.replace(" -= 1;", " -= 1 + 0;")

        return shifted

    except Exception:
        return code


# ============================================================
# 7. 数据字段处理
# ============================================================

def get_code_field(data):
    """
    CodeSearchNet 常见代码字段：
    - func
    - code
    """
    if "func" in data and data["func"]:
        return "func", data["func"]

    if "code" in data and data["code"]:
        return "code", data["code"]

    return None, ""


def update_code_field(data, code_field, new_code):
    """
    更新 shifted code。

    关键点：
    删除 code_tokens，避免 RQ1 评估时优先读取旧 code_tokens，
    导致模型实际看到的仍然是原始代码。
    """
    new_data = data.copy()

    if code_field is not None:
        new_data[code_field] = new_code
    else:
        new_data["func"] = new_code

    # 非常关键：删除旧 code_tokens
    if "code_tokens" in new_data:
        del new_data["code_tokens"]

    return new_data


# ============================================================
# 8. 主批处理流程
# ============================================================

def main():
    print("=" * 80)
    print("CodeSearchNet Java: Generate Token/CST Shift OOD Test")
    print("Using NatGen parser: languages.so / java")
    print("=" * 80)
    print("INPUT_FILE:", INPUT_FILE)
    print("OUTPUT_TOKEN_SHIFT:", OUTPUT_TOKEN_SHIFT)
    print("OUTPUT_CST_SHIFT:", OUTPUT_CST_SHIFT)
    print("LIB_PATH:", LIB_PATH)
    print("CST_INSERT_PROB:", CST_INSERT_PROB)
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: 找不到输入文件 {INPUT_FILE}")
        return

    token_changed = 0
    cst_changed = 0
    total = 0
    skipped = 0

    id_list = []
    token_id_list = []
    cst_id_list = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_TOKEN_SHIFT, "w", encoding="utf-8") as f_token, \
         open(OUTPUT_CST_SHIFT, "w", encoding="utf-8") as f_cst:

        for line_idx, line in enumerate(tqdm(f_in, desc="生成 Java Token/CST Shift")):
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            code_field, original_code = get_code_field(data)

            if not original_code:
                skipped += 1
                continue

            sample_id = str(data.get("idx", data.get("id", line_idx)))
            id_list.append(sample_id)

            total += 1

            # Token Shift
            token_code = apply_token_shift_java(original_code)
            if token_code.strip() != original_code.strip():
                token_changed += 1

            token_data = update_code_field(data, code_field, token_code)
            token_data["shift_type"] = "token"
            token_id_list.append(sample_id)
            f_token.write(json.dumps(token_data, ensure_ascii=False) + "\n")

            # CST Shift
            cst_code = apply_cst_shift_java(original_code)
            if cst_code.strip() != original_code.strip():
                cst_changed += 1

            cst_data = update_code_field(data, code_field, cst_code)
            cst_data["shift_type"] = "cst"
            cst_id_list.append(sample_id)
            f_cst.write(json.dumps(cst_data, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print("生成报告")
    print("=" * 80)
    print("总有效样本数:", total)
    print("跳过样本数:", skipped)
    print("Token Shift 写入:", len(token_id_list))
    print("CST Shift 写入:", len(cst_id_list))

    print("Token Shift 变化数量:", token_changed)
    print("Token Shift 变化比例:", token_changed / max(total, 1))

    print("CST Shift 变化数量:", cst_changed)
    print("CST Shift 变化比例:", cst_changed / max(total, 1))

    print("Token Shift 与 ID 数量一致:", len(token_id_list) == len(id_list))
    print("CST Shift 与 ID 数量一致:", len(cst_id_list) == len(id_list))
    print("Token Shift 与 ID 顺序一致:", token_id_list == id_list)
    print("CST Shift 与 ID 顺序一致:", cst_id_list == id_list)

    print("Token Shift 文件:", OUTPUT_TOKEN_SHIFT)
    print("CST Shift 文件:", OUTPUT_CST_SHIFT)

    print("注意：OOD 文件中已删除 code_tokens 字段，确保 RQ1 使用 shifted func/code。")
    print("=" * 80)


if __name__ == "__main__":
    main()