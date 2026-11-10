import os
from tree_sitter import Language

# 核心修改：自动获取当前脚本所在的绝对路径
# 这样无论你在哪个目录下运行命令，路径都不会出错
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_lib():
    # 1. 设置 parser 文件夹在 NatGen 目录下
    parser_dir = os.path.join(BASE_DIR, 'parser')
    if not os.path.exists(parser_dir):
        os.makedirs(parser_dir)
    
    # 2. 设置语言定义路径 (请根据你第一步 find 的结果核对文件夹名)
    # 如果你的文件夹叫 tree-sitter-java-master，请修改下面的名字
    java_lang_path = os.path.join(BASE_DIR, 'build', 'tree-sitter-java')
    
    # 安全检查：确保能找到 parser.c
    check_file = os.path.join(java_lang_path, 'src', 'parser.c')
    if not os.path.exists(check_file):
        print(f"!!! [Error] 找不到文件: {check_file}")
        print("请检查 build 目录下文件夹名称是否为 'tree-sitter-java'")
        # 列出 build 目录内容帮用户诊断
        build_content = os.listdir(os.path.join(BASE_DIR, 'build'))
        print(f"当前 build 目录下的内容有: {build_content}")
        return

    # 3. 编译
    output_so = os.path.join(parser_dir, 'languages.so')
    Language.build_library(
        output_so,
        [java_lang_path]
    )
    print(f">>> [Success] 语言库编译完成: {output_so}")

if __name__ == "__main__":
    build_lib()