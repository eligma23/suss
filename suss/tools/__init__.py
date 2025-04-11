try:
    from suss.tools.grep_code import GrepCodeTool
    from suss.tools.read_file import ReadFileTool
    from suss.tools.find_bugs import FindBugsTool
    from suss.tools.glob_files import GlobFilesTool
except ImportError:
    from tools.grep_code import GrepCodeTool
    from tools.read_file import ReadFileTool
    from tools.find_bugs import FindBugsTool
    from tools.glob_files import GlobFilesTool
