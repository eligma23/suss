try:
    from bugnet.tools.search_code import CodeSearchTool
    from bugnet.tools.search_files import FileSearchTool
    from bugnet.tools.read_file import ReadFileTool
    from bugnet.tools.find_bugs import FindBugsTool
except ImportError:
    from tools.search_code import CodeSearchTool
    from tools.search_files import FileSearchTool
    from tools.read_file import ReadFileTool
    from tools.find_bugs import FindBugsTool
