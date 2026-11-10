import os
import json
import random
import torch
import numpy as np
from tree_sitter import Language, Parser
from tqdm import tqdm
import sys
import io

# 强制设置编码，防止 Linux 环境下的 ASCII 报错
if sys.version_info.major == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ================= 1. 配置区 =================
# 指向你刚才编译成功的 .so 文件
LIB_PATH = '/opt/data/private/NatGen/parser/languages.so'
# 原始 ID 测试集
INPUT_FILE = "/opt/data/private/CodeSearchNet/java/java_test_0.jsonl"
# 输出路径
OUTPUT_TOKEN_SHIFT = "/opt/data/private/CodeSearchNet/java/java_test_0_token.jsonl"
OUTPUT_CST_SHIFT = "/opt/data/private/CodeSearchNet/java/java_test_0_cst.jsonl"

# 载入 Java 语言库
JAVA_LANGUAGE = Language(LIB_PATH, 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

# ================= 2. 核心算法引擎 =================

class ShiftEngine:
    @staticmethod
    def apply_token_shift(code):
        """仅执行变量名混淆 (Token Shift)"""
        code_bytes = bytes(code, "utf8")
        tree = parser.parse(code_bytes)
        
        # 获取所有 identifier 节点
        identifiers = []
        def find_ids(node):
            if node.type == 'identifier':
                name = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                # 过滤常见的 Java 关键字，避免改坏语法
                if name not in ['String', 'Main', 'System', 'out', 'println', 'int', 'float', 'double', 'Scanner']:
                    identifiers.append(node)
            for child in node.children:
                find_ids(child)
        find_ids(tree.root_node)

        unique_names = list(set([code_bytes[n.start_byte:n.end_byte].decode('utf-8') for n in identifiers]))
        name_map = {name: "var_{}".format(i) for i, name in enumerate(unique_names)}
        
        # 逆序替换，保证坐标不变
        sorted_nodes = sorted(identifiers, key=lambda x: x.start_byte, reverse=True)
        new_code = code
        for node in sorted_nodes:
            old_name = new_code[node.start_byte:node.end_byte]
            if old_name in name_map:
                new_code = new_code[:node.start_byte] + name_map[old_name] + new_code[node.end_byte:]
        return new_code

    @staticmethod
    def apply_cst_shift(code):
        """仅执行结构改写 (CST Shift)，保留变量名不变"""
        # 1. 注入不可达的死代码块 (改变树的深度和分叉，但不改变逻辑)
        dead_codes = [
            "\n        if (1 > 2) { System.out.println(\"tracing_status_low\"); }",
            "\n        int _unused_v = 0; while(_unused_v < 0) { _unused_v++; }",
            "\n        boolean _internal_flag = true; if(!_internal_flag) { return; }"
        ]
        
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            # 在含有左大括号的行后以 50% 概率插入
            if '{' in line and random.random() > 0.5:
                new_lines.append(random.choice(dead_codes))
        
        code = "\n".join(new_lines)
        
        # 2. 语法微调：将简单的 a++ 展开为 a = a + 1
        # 这会改变 CST 节点的数量
        code = code.replace("++", " = 1 + 1 - 1 + ") 
        return code

# ================= 3. 主批处理流程 =================

def main():
    print(">>> 正在启动数据双向制造程序...")
    if not os.path.exists(INPUT_FILE):
        print("Error: 找不到输入文件 {}".format(INPUT_FILE))
        return

    token_success, cst_success = 0, 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_TOKEN_SHIFT, 'w', encoding='utf-8') as f_token, \
         open(OUTPUT_CST_SHIFT, 'w', encoding='utf-8') as f_cst:
        
        lines = f_in.readlines()
        for line in tqdm(lines, desc="双向转换中"):
            data = json.loads(line)
            original_code = data.get('func') or data.get('code')
            
            if not original_code: continue

            # --- 生成 Token Shift 版本 ---
            try:
                t_code = ShiftEngine.apply_token_shift(original_code)
                if t_code.strip() != original_code.strip():
                    t_data = data.copy()
                    t_data['func'] = t_code
                    t_data['shift_type'] = "token"
                    f_token.write(json.dumps(t_data, ensure_ascii=False) + '\n')
                    token_success += 1
            except: pass

            # --- 生成 CST Shift 版本 ---
            try:
                c_code = ShiftEngine.apply_cst_shift(original_code)
                if c_code.strip() != original_code.strip():
                    c_data = data.copy()
                    c_data['func'] = c_code
                    c_data['shift_type'] = "cst"
                    f_cst.write(json.dumps(c_data, ensure_ascii=False) + '\n')
                    cst_success += 1
            except: pass

    print("\n" + "="*40)
    print("生成报告：")
    print("1. Token Shift 数据集: {} 条".format(token_success))
    print("2. CST Shift 数据集: {} 条".format(cst_success))
    print("文件已保存在: {}".format(os.path.dirname(OUTPUT_TOKEN_SHIFT)))
    print("="*40)

if __name__ == "__main__":
    main()