try:
    from suss.tools.search_code import CodeSearchTool
    from suss.tools.search_files import FileSearchTool
    from suss.tools.read_file import ReadFileTool
    from suss.tools.find_bugs import FindBugsTool
except ImportError:
    from tools.search_code import CodeSearchTool
    from tools.search_files import FileSearchTool
    from tools.read_file import ReadFileTool
    from tools.find_bugs import FindBugsTool
