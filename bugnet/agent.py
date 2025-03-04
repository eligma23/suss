# Standard library
from typing import Dict, List, Union

# Third party
from saplings import COTAgent
from saplings.llms import OpenAI
from saplings.dtos import Message

# Local
try:
    from bugnet.tools import (
        CodeSearchTool,
        FileSearchTool,
        ReadFileTool,
        FindBugsTool,
    )
    from bugnet.index import Index, File
except ImportError:
    from tools import CodeSearchTool, FileSearchTool, ReadFileTool, FindBugsTool
    from index import Index, File

#########
# HELPERS
#########


SYSTEM_PROMPT = """<assistant>
You will be given a file from a codebase.
Your job is to call functions to find information about the codebase that will help you analyze the file for bugs.
Call functions.done when you have enough context on the surrounding codebase to analyze the file.
</assistant>

<rules>
- DO NOT call a function that you've used before with the same arguments.
- DO NOT assume the structure of the codebase the given file belongs to, or the existence of other files or folders.
- Your queries to functions.search_code and functions.search_files should be significantly different to previous queries.
- Call functions.done with files that you are confident will help you analyze the given file for bugs.
- If the output of a function is empty, try calling the function again with DIFFERENT arguments OR try calling a different function.
- Use functions.search_code, functions.search_files, and functions.read_file to gather context on the codebase the file belongs to.
</rules>

<important>
Some files can be scanned for bugs without any additional context from the codebase. 
But if a file depends on (or affects) many other constructs in the codebase, you may need to call functions to gather that context.
</important>"""


def build_prompt(file: File) -> str:
    prompt = f"<path>{file.path}</path>\n"
    prompt += f"<code>\n{file.content}\n</code>\n\n"
    prompt += "#####\n\nAnalyze the file above for bugs."
    return prompt


def was_tool_called(messages: List[Message], tool_name: str) -> bool:
    for message in messages:
        if message.role != "assistant":
            continue

        if not message.tool_calls:
            continue

        for tool_call in message.tool_calls:
            if tool_call.name == tool_name:
                return True

    return False


######
# MAIN
######


class Agent:
    def __init__(self, index: Index, model: str, max_iters: int):
        self.index = index
        self.model = model
        self.max_iters = max_iters

    async def run(self, file: File, update_progress: callable):
        tools = [
            CodeSearchTool(self.index, file, update_progress),
            FileSearchTool(self.index, file, update_progress),
            ReadFileTool(self.index, self.model, file, update_progress),
            FindBugsTool(self.model, file, update_progress),
        ]
        model = OpenAI(
            model="gpt-4o"
        )  # TODO: Refactor how models are handled in saplings
        agent = COTAgent(
            tools,
            model,
            SYSTEM_PROMPT,
            tool_choice="required",
            max_depth=self.max_iters,
            verbose=False,
        )
        prompt = build_prompt(file)
        messages = await agent.run_async(prompt)

        output = messages[-1].raw_output
        if not was_tool_called(messages, "done"):
            tool_call = await agent.call_tool("done", messages)
            tool_result = await agent.run_tool(tool_call, messages)
            output = tool_result.raw_output

        return output
