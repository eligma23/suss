import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from dataclasses import asdict
from tree_sitter_language_pack import get_parser

# 项目配置
PROJECT_ROOT = Path(__file__).parent.resolve()
SUPPORTED_LANGUAGES = {
    '.py': 'python',
    '.js': 'javascript',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.ts': 'typescript',
    '.cpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.hpp': 'cpp'
}

# suss里的相关类
class OrderedSet(set):
    """简化的有序集合实现"""
    def __init__(self, iterable=None):
        super().__init__()
        self._order = []
        if iterable is not None:
            for item in iterable:
                self.add(item)
    
    def add(self, item):
        if item not in self:
            super().add(item)
            self._order.append(item)
    
    def __iter__(self):
        return iter(self._order)
    
    def __repr__(self):
        return f"OrderedSet({self._order})"

class LineContext:
    def __init__(self, line_num: int):
        self.line_num = line_num
        self.scopes = OrderedSet()  # Scopes (line #s) that the line belongs to
        self.nodes = []  # AST nodes that contain the line
        self.header_parts = []  # Lines that make up the scope header

    def add_node(self, node):
        self.nodes.append(node)

    def add_header_part(self, start: int, end: int):
        size = end - start
        if not size:
            return
        self.header_parts.append((size, start, end))

    def add_scope(self, start: int):
        self.scopes.add(start)

class File:
    def __init__(self, root_dir: Path, path: Path):
        self.root_dir = root_dir
        self.path = path

        try:
            with open(self.path, "r") as f:
                self.code = f.read()
        except UnicodeDecodeError:
            self.code = ""

        self.lines = self.code.splitlines()
        self.num_lines = len(self.lines) + 1
        self.line_contexts = [LineContext(i) for i in range(self.num_lines)]
        self.index_ast()

    @property
    def rel_path(self) -> str:
        return str(self.path.relative_to(self.root_dir))

    @property
    def is_code_file(self) -> bool:
        return self.extension in SUPPORTED_LANGUAGES

    @property
    def language(self) -> str:
        if not self.is_code_file:
            return None
        return SUPPORTED_LANGUAGES[self.extension]

    @property
    def extension(self) -> str:
        return self.path.suffix

    def index_ast(self):
        if not self.is_code_file:
            return

        def recurse_tree(node):
            start, end = node.start_point, node.end_point
            start_line, end_line = start[0], end[0]

            self.line_contexts[start_line].add_node(node)
            self.line_contexts[start_line].add_header_part(start_line, end_line)

            for line_num in range(start_line, end_line + 1):
                self.line_contexts[line_num].add_scope(start_line)

            for child in node.children:
                recurse_tree(child)

        parser = get_parser(self.language)
        tree = parser.parse(bytes(self.code, "utf8"))
        recurse_tree(tree.root_node)
    
    def get_ast_info(self):
        """获取AST信息的可序列化字典"""
        if not self.is_code_file:
            return {"error": "Unsupported file type"}
        
        ast_info = {
            "file_path": self.rel_path,
            "language": self.language,
            "ast_nodes": []
        }
        
        # 收集所有AST节点信息
        for line_ctx in self.line_contexts:
            for node in line_ctx.nodes:
                start_line, start_col = node.start_point
                end_line, end_col = node.end_point
                
                node_info = {
                    "type": node.type,
                    "start_line": start_line + 1,  # 转换为1-based行号
                    "start_column": start_col + 1,
                    "end_line": end_line + 1,
                    "end_column": end_col + 1,
                    "text": self.code[node.start_byte:node.end_byte]
                }
                ast_info["ast_nodes"].append(node_info)
        
        return ast_info

class ASTGenerator:
    """AST生成器封装类"""
    def __init__(self, file_path: str, root_dir: str = None):
        self.root_dir = Path(root_dir) if root_dir else Path(file_path).parent
        self.file_path = Path(file_path)
        self.file = File(self.root_dir, self.file_path)
    
    def get_ast_info(self):
        """获取AST信息"""
        return self.file.get_ast_info()

def run_suss_analysis(file_path: str, model: str = "deepseek/deepseek-coder"):
    """
    使用SUSS工具分析代码文件
    :param file_path: 要分析的文件路径
    :param model: 使用的模型名称
    """
    # 确保SUSS可执行文件在路径中
    suss_path = "suss"  # 假设suss已安装在系统PATH中
    
    # 构建命令
    cmd = [
        suss_path,
        "--file", file_path,
        "--model", model
    ]
    
    # 执行命令
    print(f"🚀 执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 检查执行结果
    if result.returncode != 0:
        print(f"❌ SUSS执行失败，错误代码: {result.returncode}")
        print(f"错误输出:\n{result.stderr}")
        return None
    
    # 解析输出 - 假设SUSS输出JSON格式的结果
    try:
        output = json.loads(result.stdout)
        return output
    except json.JSONDecodeError:
        print(f"❌ 无法解析SUSS输出为JSON")
        print(f"原始输出:\n{result.stdout}")
        return None

async def extract_ast(file_path: str):
    """
    直接提取文件的AST信息
    :param file_path: 要分析的文件路径
    """
    generator = ASTGenerator(file_path)
    return generator.get_ast_info()

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python3 get_ast.py <文件路径> [根目录]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    root_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 选项1: 使用SUSS工具分析并提取AST
    # suss_result = run_suss_analysis(file_path)
    # if suss_result:
    #     print("✅ SUSS分析结果:")
    #     print(json.dumps(suss_result, indent=2, ensure_ascii=False))
    
    # 选项2: 直接提取AST
    ast_info = asyncio.run(extract_ast(file_path))
    if "error" in ast_info:
        print(f"❌ 错误: {ast_info['error']}")
    else:
        print("🌳 抽象语法树(AST)信息:")
        print(json.dumps(ast_info, indent=2, ensure_ascii=False))
        
        # 保存到文件
        output_file = Path(file_path).with_suffix(".ast.json")
        with open(output_file, "w") as f:
            json.dump(ast_info, f, indent=2, ensure_ascii=False)
        print(f"💾 AST信息已保存到: {output_file}")

if __name__ == "__main__":
    main()