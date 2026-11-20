import os
import json
import random
import numpy as np
from collections import Counter
from tree_sitter import Language, Parser


# ============================================================
# 1. 配置区
# ============================================================

DATA_ROOT = "/opt/data/private/CodeXGLUE/Code-Code/code-refinement/data"

SUBSETS = ["small", "medium"]

# NatGen 编译好的 Java parser
LIB_PATH = "/opt/data/private/NatGen/parser/languages.so"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# CST 扰动概率
CST_INSERT_PROB = 1.0


# ============================================================
# 2. 初始化 NatGen Java Parser
# ============================================================

if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(f"找不到 NatGen parser 文件: {LIB_PATH}")

JAVA_LANGUAGE = Language(LIB_PATH, "java")
PARSER = Parser()
PARSER.set_language(JAVA_LANGUAGE)


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
# 4. tree-sitter / NatGen 工具函数
# ============================================================

def parse_java_code(code):
    return PARSER.parse(code.encode("utf-8", errors="ignore"))


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
    使用 byte offset 替换，避免 Unicode 导致 str offset 和 byte offset 不一致。
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


def has_parse_error(code):
    try:
        tree = parse_java_code(code)

        def check(node):
            if node.type == "ERROR":
                return True
            for child in node.children:
                if check(child):
                    return True
            return False

        return check(tree.root_node)
    except Exception:
        return True


# ============================================================
# 5. Token Shift：buggy/fixed 共享同一套映射
# ============================================================

def should_skip_identifier(code_bytes, node):
    """
    保守跳过不应替换的 identifier：
    - Java 关键字；
    - 常见库名 / 类型名；
    - 方法调用名；
    - 方法声明名；
    - 构造器名；
    - 类名 / 接口名 / 枚举名；
    - 字段访问名；
    - import/package/annotation 中的名字；
    - 全大写常量；
    - 首字母大写的类型名风格名字。
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


def collect_replaceable_names(code):
    """
    收集一段 Java 代码中可以替换的 identifier 名字。
    """
    names = []

    try:
        code_bytes = code.encode("utf-8", errors="ignore")
        tree = parse_java_code(code)
        root = tree.root_node

        for node in walk_tree(root):
            if node.type != "identifier":
                continue

            if should_skip_identifier(code_bytes, node):
                continue

            name = node_text(code_bytes, node)
            names.append(name)

    except Exception:
        pass

    return names


def build_shared_name_map(buggy_code, fixed_code):
    """
    对 buggy 和 fixed 共同构造 identifier 映射。
    这是 Bugs2Fix Token Shift 最关键的地方。
    """
    names = []
    names.extend(collect_replaceable_names(buggy_code))
    names.extend(collect_replaceable_names(fixed_code))

    unique_names = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)

    return {
        name: f"var_{i}"
        for i, name in enumerate(unique_names)
    }


def collect_replacements_with_map(code, name_map):
    """
    根据共享 name_map，对单段代码生成 replacements。
    """
    replacements = []

    try:
        code_bytes = code.encode("utf-8", errors="ignore")
        tree = parse_java_code(code)
        root = tree.root_node

        for node in walk_tree(root):
            if node.type != "identifier":
                continue

            if should_skip_identifier(code_bytes, node):
                continue

            name = node_text(code_bytes, node)

            if name in name_map:
                replacements.append((node.start_byte, node.end_byte, name_map[name]))

    except Exception:
        return []

    return replacements


def apply_token_shift_pair_java(buggy_code, fixed_code):
    """
    Token Shift:
    buggy 和 fixed 使用同一套 identifier 映射。
    如果变换后解析失败，则回退原始代码。
    """
    try:
        name_map = build_shared_name_map(buggy_code, fixed_code)

        if not name_map:
            return buggy_code, fixed_code, False, "no_replaceable_identifier"

        buggy_replacements = collect_replacements_with_map(buggy_code, name_map)
        fixed_replacements = collect_replacements_with_map(fixed_code, name_map)

        shifted_buggy = replace_by_spans(buggy_code, buggy_replacements) if buggy_replacements else buggy_code
        shifted_fixed = replace_by_spans(fixed_code, fixed_replacements) if fixed_replacements else fixed_code

        changed = (
            shifted_buggy.strip() != buggy_code.strip()
            or shifted_fixed.strip() != fixed_code.strip()
        )

        if not changed:
            return buggy_code, fixed_code, False, "no_change"

        if has_parse_error(shifted_buggy) or has_parse_error(shifted_fixed):
            return buggy_code, fixed_code, False, "parse_error_fallback"

        return shifted_buggy, shifted_fixed, True, ""

    except Exception as e:
        return buggy_code, fixed_code, False, f"exception:{type(e).__name__}"


# ============================================================
# 6. CST Shift：只扰动 buggy，fixed 保持原始
# ============================================================

def find_first_method_body(code):
    """
    找第一个 Java 方法体 block。
    Bugs2Fix 通常是方法级 Java 代码。
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
    只对 buggy 输入做轻量结构扰动。
    fixed 目标保持原始。
    """
    if random.random() > CST_INSERT_PROB:
        return code, False, "prob_skip"

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

            shifted_code = shifted_bytes.decode("utf-8", errors="ignore")
        else:
            pos = code.find("{")
            if pos == -1:
                return code, False, "no_block"

            shifted_code = code[:pos + 1] + random.choice(templates) + code[pos + 1:]

        shifted_code = shifted_code.replace(" += 1;", " += 1 + 0;")
        shifted_code = shifted_code.replace(" -= 1;", " -= 1 + 0;")

        if shifted_code.strip() == code.strip():
            return code, False, "no_change"

        if has_parse_error(shifted_code):
            return code, False, "parse_error_fallback"

        return shifted_code, True, ""

    except Exception as e:
        return code, False, f"exception:{type(e).__name__}"


# ============================================================
# 7. 读取 Bugs2Fix parallel 文件
# ============================================================

def find_parallel_file(subset_dir, split, side):
    """
    兼容 CodeXGLUE Bugs2Fix 常见文件名：
    train.buggy-fixed.buggy
    train.buggy-fixed.fixed
    test.buggy-fixed.buggy
    test.buggy-fixed.fixed
    """
    candidates = [
        f"{split}.buggy-fixed.{side}",
        f"{split}.{side}",
        f"{split}.txt.{side}",
        f"{split}.{side}.txt",
    ]

    for name in candidates:
        path = os.path.join(subset_dir, name)
        if os.path.exists(path):
            return path

    # 模糊查找兜底
    for name in os.listdir(subset_dir):
        lower = name.lower()
        if split in lower and side in lower:
            return os.path.join(subset_dir, name)

    raise FileNotFoundError(f"在 {subset_dir} 中找不到 {split} 的 {side} 文件")


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_bugs2fix_split(subset_dir, split):
    buggy_path = find_parallel_file(subset_dir, split, "buggy")
    fixed_path = find_parallel_file(subset_dir, split, "fixed")

    buggy_lines = read_lines(buggy_path)
    fixed_lines = read_lines(fixed_path)

    if len(buggy_lines) != len(fixed_lines):
        raise ValueError(
            f"{split} buggy/fixed 行数不一致: "
            f"{len(buggy_lines)} vs {len(fixed_lines)}"
        )

    data = []

    for i, (buggy, fixed) in enumerate(zip(buggy_lines, fixed_lines)):
        if not buggy.strip() or not fixed.strip():
            continue

        data.append({
            "idx": str(i),
            "buggy": buggy,
            "fixed": fixed
        })

    print(f">>> {subset_dir} / {split}: 加载 {len(data)} 条")
    print("    buggy:", buggy_path)
    print("    fixed:", fixed_path)

    return data


# ============================================================
# 8. 保存与统计
# ============================================================

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def print_dataset_stats(name, data):
    print(f"\n[{name}]")
    print("样本数:", len(data))

    if data:
        source_len = sum(len(x["source"].split()) for x in data) / len(data)
        target_len = sum(len(x["target"].split()) for x in data) / len(data)
        print(f"平均 source token 数: {source_len:.2f}")
        print(f"平均 target token 数: {target_len:.2f}")


def check_alignment(id_test, ood_test, name):
    same_len = len(id_test) == len(ood_test)
    same_idx = [x["idx"] for x in id_test] == [x["idx"] for x in ood_test]

    print(f"\n[{name} 一致性检查]")
    print("数量一致:", same_len)
    print("idx 顺序一致:", same_idx)


# ============================================================
# 9. 单个 subset 处理
# ============================================================

def process_subset(subset_name):
    subset_dir = os.path.join(DATA_ROOT, subset_name)

    if not os.path.exists(subset_dir):
        raise FileNotFoundError(f"找不到子集目录: {subset_dir}")

    print("\n" + "=" * 100)
    print(f"处理 Bugs2Fix {subset_name} 子集")
    print("=" * 100)
    print("subset_dir:", subset_dir)

    train_data = load_bugs2fix_split(subset_dir, "train")
    test_data = load_bugs2fix_split(subset_dir, "test")

    id_train = []
    for x in train_data:
        id_train.append({
            "idx": x["idx"],
            "source": x["buggy"],
            "target": x["fixed"],
            "buggy": x["buggy"],
            "fixed": x["fixed"],
            "shift_type": "id_train_original"
        })

    id_test = []
    ood_token_test = []
    ood_cst_test = []

    token_success = 0
    token_fallback = 0
    cst_success = 0
    cst_fallback = 0

    for x in test_data:
        idx = x["idx"]
        buggy = x["buggy"]
        fixed = x["fixed"]

        id_test.append({
            "idx": idx,
            "source": buggy,
            "target": fixed,
            "buggy": buggy,
            "fixed": fixed,
            "shift_type": "id_test_original"
        })

        # Token Shift: buggy/fixed 同步变换
        token_buggy, token_fixed, token_ok, token_reason = apply_token_shift_pair_java(
            buggy,
            fixed
        )

        if token_ok:
            token_success += 1
        else:
            token_fallback += 1

        ood_token_test.append({
            "idx": idx,
            "source": token_buggy,
            "target": token_fixed,
            "buggy": token_buggy,
            "fixed": token_fixed,
            "shift_type": "ood_test_token",
            "shift_success": token_ok,
            "fallback_reason": token_reason
        })

        # CST Shift: 只变换 buggy，fixed 保持原始
        cst_buggy, cst_ok, cst_reason = apply_cst_shift_java(buggy)

        if cst_ok:
            cst_success += 1
        else:
            cst_fallback += 1

        ood_cst_test.append({
            "idx": idx,
            "source": cst_buggy,
            "target": fixed,
            "buggy": cst_buggy,
            "fixed": fixed,
            "shift_type": "ood_test_cst",
            "shift_success": cst_ok,
            "fallback_reason": cst_reason
        })

    output_id_train = os.path.join(subset_dir, "id_train.jsonl")
    output_id_test = os.path.join(subset_dir, "id_test.jsonl")
    output_ood_token_test = os.path.join(subset_dir, "ood_test_token.jsonl")
    output_ood_cst_test = os.path.join(subset_dir, "ood_test_cst.jsonl")

    write_jsonl(output_id_train, id_train)
    write_jsonl(output_id_test, id_test)
    write_jsonl(output_ood_token_test, ood_token_test)
    write_jsonl(output_ood_cst_test, ood_cst_test)

    print("\n输出文件:")
    print("ID Train      :", output_id_train)
    print("ID Test       :", output_id_test)
    print("OOD Token Test:", output_ood_token_test)
    print("OOD CST Test  :", output_ood_cst_test)

    print_dataset_stats("ID Train", id_train)
    print_dataset_stats("ID Test", id_test)
    print_dataset_stats("OOD Token Test", ood_token_test)
    print_dataset_stats("OOD CST Test", ood_cst_test)

    check_alignment(id_test, ood_token_test, "Token OOD")
    check_alignment(id_test, ood_cst_test, "CST OOD")

    print("\n[Shift 统计]")
    print("Token Shift 成功数量:", token_success)
    print("Token Shift fallback 数量:", token_fallback)
    print("Token Shift 成功比例:", token_success / max(len(id_test), 1))

    print("CST Shift 成功数量:", cst_success)
    print("CST Shift fallback 数量:", cst_fallback)
    print("CST Shift 成功比例:", cst_success / max(len(id_test), 1))

    if id_test:
        print("\n[样本预览]")
        print("idx:", id_test[0]["idx"])

        print("\nID source/buggy:")
        print(id_test[0]["source"][:600])

        print("\nID target/fixed:")
        print(id_test[0]["target"][:600])

        print("\nToken OOD source/buggy:")
        print(ood_token_test[0]["source"][:600])

        print("\nToken OOD target/fixed:")
        print(ood_token_test[0]["target"][:600])

        print("\nCST OOD source/buggy:")
        print(ood_cst_test[0]["source"][:600])

        print("\nCST OOD target/fixed:")
        print(ood_cst_test[0]["target"][:600])


# ============================================================
# 10. 主函数
# ============================================================

def main():
    print("=" * 100)
    print("构造 CodeXGLUE Bugs2Fix / Code Refinement 的 ID/OOD 数据集")
    print("使用 NatGen Java parser 生成 Token Shift 和 CST Shift")
    print("Token Shift: buggy 和 fixed 使用同一套变量名映射")
    print("CST Shift: 只扰动 buggy 输入，fixed 保持原始")
    print("只使用 train/test，跳过 valid")
    print("=" * 100)
    print("DATA_ROOT:", DATA_ROOT)
    print("SUBSETS:", SUBSETS)
    print("LIB_PATH:", LIB_PATH)
    print("CST_INSERT_PROB:", CST_INSERT_PROB)
    print("=" * 100)

    for subset in SUBSETS:
        process_subset(subset)

    print("\n全部完成。")


if __name__ == "__main__":
    main()