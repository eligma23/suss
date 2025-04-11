# Standard library
from dataclasses import dataclass
from collections import defaultdict

# Third party
import json_repair
from saplings.dtos import Message
from saplings.abstract import Tool
from litellm import acompletion, encode
from sortedcollections import OrderedSet

# Local
try:
    from suss.index import Chunk, File
    from suss.constants import (
        MAX_MERGE_DISTANCE,
        MAX_CONTEXT_TOKENS,
        MAX_THINKING_TOKENS,
    )
except ImportError:
    from index import Chunk, File
    from constants import MAX_MERGE_DISTANCE, MAX_CONTEXT_TOKENS, MAX_THINKING_TOKENS


#########
# HELPERS
#########


@dataclass
class Bug:
    start: int
    end: int
    description: str
    confidence: float


SYSTEM_PROMPT = """I want to find all the bugs in a code file, if any exist. A bug is \
code that could cause runtime errors, crashes, or incorrect behavior. It is NOT a style \
issue, suggestion, or anything else.

Your output should resemble an expert code review. It should be a list of JSON objects, \
one for each bug, each containing a code block and a description of the bug contained in \
that code block.

Remember that you are analyzing a code file for bugs. It is entirely possible that the code \
file contains no bugs, in which case you should return an empty list.

--

The code file you're analyzing is part of a codebase. Below is additional context from the \
codebase that may help you understand the code file. Use it to help you identify bugs, if \
necessary.

<context>
{context}
</context>"""


def get_reasoning_model(model: str) -> str:
    model = model.lower()
    if model.startswith("openai/"):
        return "openai/o3-mini"
    elif model.startswith("anthropic/"):
        return "anthropic/claude-3-7-sonnet-20250219"

    return model


def group_chunks_by_file(chunks: list[Chunk]) -> dict[File, list[Chunk]]:
    chunks_by_file = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[chunk.file].append(chunk)

    return chunks_by_file


def get_chunks(trajectory: list[Message]) -> OrderedSet[Chunk]:
    chunks = OrderedSet()
    for message in trajectory:
        if not message.raw_output:
            continue

        if not hasattr(message.raw_output, "__iter__"):
            continue

        for item in message.raw_output:
            if isinstance(item, Chunk):
                chunks.add(item)

    return chunks


def filter_chunks(chunks: list[Chunk], files: list[str]) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.file.rel_path in files]


def get_contiguous_subchunks(line_nums: list[int], file: File) -> list[Chunk]:
    if not line_nums:
        return []

    groups = []
    curr_group = [line_nums[0]]
    for line_num in line_nums[1:]:
        if line_num == curr_group[-1] + 1:
            curr_group.append(line_num)
        else:
            groups.append(curr_group)
            curr_group = [line_num]

    if curr_group:
        groups.append(curr_group)

    chunks = [Chunk(group, file) for group in groups]
    return chunks


def merge_chunks(chunks: list[Chunk]) -> list[Chunk]:
    chunks_by_file = group_chunks_by_file(chunks)
    merged_chunks = []
    for file, chunks in chunks_by_file.items():
        all_line_nums = {ln for chunk in chunks for ln in chunk.line_nums}
        all_line_nums = list(sorted(all_line_nums))
        merged_chunks += get_contiguous_subchunks(all_line_nums, file)

    return merged_chunks


def truncate_chunks(chunks: list[Chunk], model: str) -> list[Chunk]:
    num_tokens = 0
    truncated_chunks = []
    for chunk in chunks:
        tokens = len(encode(model=model, text=chunk.to_string()))
        if num_tokens + tokens > MAX_CONTEXT_TOKENS:
            break

        num_tokens += tokens
        truncated_chunks.append(chunk)

    return truncated_chunks


def normalize_chunks(chunks: list[Chunk]) -> list[Chunk]:
    chunks_by_file = group_chunks_by_file(chunks)
    norm_chunks = []
    for file, chunks in chunks_by_file.items():
        all_line_nums = {ln for chunk in chunks for ln in chunk.line_nums}
        all_line_nums = list(sorted(all_line_nums))

        norm_line_nums = []
        for index in range(len(all_line_nums)):
            curr_line_num = all_line_nums[index]
            next_line_num = (
                all_line_nums[index + 1] if index + 1 < len(all_line_nums) else None
            )

            norm_line_nums.append(curr_line_num)
            for i in range(1, MAX_MERGE_DISTANCE + 1):
                if not next_line_num:
                    if curr_line_num + i > file.last_lineno:
                        break

                if curr_line_num + i == next_line_num:
                    break

                norm_line_nums.append(curr_line_num + i)

        norm_chunks.append(Chunk(norm_line_nums, file))

    return norm_chunks


def build_context_str(chunks: list[Chunk]) -> str:
    fp_str = "<file_paths>\n"
    chunks_str = "<code_chunks>\n"
    for chunk in chunks:
        fp_str += f"{chunk.file.rel_path}\n"
        chunks_str += f"<{chunk.file.rel_path}>\n{chunk.to_string(line_nums=False)}\n</{chunk.file.rel_path}>\n\n"

    chunks_str = chunks_str.rstrip("\n")
    chunks_str += "\n</code_chunks>"
    fp_str += "</file_paths>"

    return f"{fp_str}\n\n{chunks_str}"


async def find_bugs(context: str, file: File, model: str) -> list[Bug]:
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(context=context),
    }
    user_message = {
        "role": "user",
        "content": f"<path>{file.rel_path}</path>\n<code>\n{file.content}\n</code>\n\n--\n\nAnalyze the file above for bugs.",
    }
    response = await acompletion(
        model=get_reasoning_model(model),
        messages=[system_message, user_message],
        # frequency_penalty=0.0,
        # temperature=0.75,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "bug_report",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "bugs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {
                                        "type": "integer",
                                        "description": "Starting line number (inclusive) for the buggy code block.",
                                    },
                                    "end": {
                                        "type": "integer",
                                        "description": "Ending line number (inclusive) for the buggy code block.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Concise description of the bug and its impact. No more than a few short sentences.",
                                    },
                                    "confidence": {
                                        "type": "integer",
                                        "description": "Confidence score between 0 and 100. 100 indicates high confidence in the bug's existence and severity (i.e. it definitely exists and is severe). 0 indicates low confidence (i.e. bug doesn't exist or is very low severity).",
                                    },
                                },
                                "required": [
                                    "start",
                                    "end",
                                    "description",
                                    "confidence",
                                ],
                                "additionalProperties": False,
                            },
                            "description": "List of bugs in the code file. This list can be empty if there are no bugs in the file.",
                        },
                    },
                    "required": ["bugs"],
                    "additionalProperties": False,
                },
            },
        },
        thinking={"type": "enabled", "budget_tokens": MAX_THINKING_TOKENS},
        drop_params=True,
    )
    response = response.choices[0].message.content
    response = json_repair.loads(response)

    bugs = []
    for bug in response["bugs"]:
        start = max(1, bug["start"])
        end = min(bug["end"], file.last_lineno)
        end = end - 1 if not file.lines[end - 1].strip() else end
        conf_score = max(0, min(100, bug["confidence"]))
        bugs.append(Bug(start, end, bug["description"], conf_score))

    return bugs


######
# MAIN
######


class FindBugsTool(Tool):
    def __init__(
        self, model: str, target_file: File, update_progress: callable, **kwargs
    ):
        # Base attributes
        self.name = "done"
        self.description = (
            "Call this when you have enough context to answer the user's query."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string", "enum": []},
                    "description": f"Files that contain the context you need to answer the user's query.",
                },
            },
            "required": ["files"],
            "additionalProperties": False,
        }
        self.is_terminal = True

        # Additional attributes
        self.model = model
        self.target_file = target_file
        self.update_progress = update_progress

    def update_definition(self, trajectory: list[Message] = [], **kwargs):
        files = set()
        for message in trajectory:
            if not message.raw_output:
                continue

            if not hasattr(message.raw_output, "__iter__"):
                continue

            for item in message.raw_output:
                if isinstance(item, Chunk):
                    files.add(item.file.rel_path)
                elif isinstance(item, File):
                    files.add(item.rel_path)

        files = [file for file in files if file != self.target_file.rel_path]
        self.parameters["properties"]["files"]["items"]["enum"] = files

    async def run(self, files: list[str], **kwargs) -> list[Bug]:
        trajectory = kwargs.get("trajectory", [])
        chunks = get_chunks(trajectory)
        chunks = filter_chunks(chunks, files)
        chunks = merge_chunks(chunks)
        chunks = truncate_chunks(chunks, self.model)
        chunks = normalize_chunks(chunks)
        context = build_context_str(chunks)
        bugs = await find_bugs(context, self.target_file, self.model)

        return bugs
