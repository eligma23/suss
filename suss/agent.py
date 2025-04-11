# Third party
from saplings.dtos import Message
from saplings import COTAgent, Model

# Local
try:
    from suss.tools import (
        GrepCodeTool,
        GlobFilesTool,
        ReadFileTool,
        FindBugsTool,
    )
    from suss.index import Index, File
except ImportError:
    from index import Index, File
    from tools import GrepCodeTool, GlobFilesTool, ReadFileTool, FindBugsTool


#########
# HELPERS
#########


SYSTEM_PROMPT = """<instructions>
Your job is to choose the best action. Call functions to find information about the \
codebase that will help you answer the user's query. Call functions.done when you \
have enough context to answer the user's query.
</instructions>

<rules>
- DO NOT call a function that you've used before with the same arguments.
- DO NOT assume the structure of the codebase or the existence of other files \
or folders.
- Your queries to functions.search_code and functions.search_files should be \
significantly different than previous queries.
- If the output of a function is empty, try calling the function again with \
DIFFERENT arguments OR try calling a different function.
</rules>"""


def build_prompt(file: File) -> str:
    prompt = f"<path>{file.path}</path>\n"
    prompt += f"<code>\n{file.content}\n</code>\n\n"
    prompt += "--\n\nAbove is a file from the codebase. Analyze it for bugs. Consider bugs that are isolated to the file, and bugs it may cause in other files (i.e. afferent or efferent dependencies)."
    return prompt


def was_tool_called(messages: list[Message], tool_name: str) -> bool:
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
        # TODO: Add pseudo-semantic search tool
        # TODO: Add call graph tools (ref/def search)
        tools = [
            GrepCodeTool(self.index, file, update_progress),
            GlobFilesTool(self.index, file, update_progress),
            ReadFileTool(self.index, self.model, file, update_progress),
            FindBugsTool(self.model, file, update_progress),
        ]
        model = Model(self.model)
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
