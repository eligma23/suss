# Standard library
import json
from typing import List
from dataclasses import dataclass
from collections import defaultdict

# Third party
import json_repair
from saplings.dtos import Message
from saplings.abstract import Tool
from sortedcollections import OrderedSet
from litellm import acompletion, encode

# Local
try:
    from bugnet.index import Chunk, File
except ImportError:
    from index import Chunk, File


#########
# HELPERS
#########


@dataclass
class Bug:
    start: int
    end: int
    description: str
    confidence: float


SYSTEM_PROMPT = """I want to find all the bugs in a code file, if any exist. A bug is code that could cause runtime errors, crashes, or incorrect behavior. It is NOT a style issue, suggestion, or anything else.

Your output should resemble an expert code review. It should be a list of "bug objects" that contain a code block and a description of the bug contained in that code block.

Remember that you are analyzing a code file for bugs. It is entirely possible that the code file contains no bugs, in which case you should return an empty list.

--

The code file you're analyzing belongs to a codebase. Below is additional context on that codebase that may help you identify bugs. Use it only if necessary.

<context>
{context}
</context>"""


def get_reasoning_model(model: str) -> str:
    # TODO:
    # Get the provider's reasoning model (e.g. openai -> openai/o3-mini, anthropic -> anthropic/claude-3.7-sonnet, etc.)
    # If provider has one, check if it's available with LiteLLM
    # If not available, or if provider doesn't have one, fallback to default model
    return model


def get_chunks(trajectory: List[Message]) -> OrderedSet[Chunk]:
    chunks = OrderedSet()
    for message in trajectory:
        if not message.raw_output:
            continue

        for item in message.raw_output:
            if isinstance(item, Chunk):
                chunks.add(item)

    return chunks


def filter_chunks(chunks: List[Chunk], files: List[str]) -> List[Chunk]:
    return [chunk for chunk in chunks if chunk.file.rel_path in files]


def merge_chunks(chunks: List[Chunk]) -> List[Chunk]:
    return chunks  # TODO


def truncate_chunks(
    chunks: List[Chunk], model: str, max_tokens: int = 40000
) -> List[Chunk]:
    num_tokens = 0
    truncated_chunks = []
    for chunk in chunks:
        tokens = len(encode(model=model, text=chunk.to_string()))
        if num_tokens + tokens > max_tokens:
            break

        num_tokens += tokens
        truncated_chunks.append(chunk)

    return truncated_chunks


def build_context_str(chunks: List[Chunk]) -> str:
    chunks_by_file = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[chunk.file].append(chunk)

    fp_str = "<file_paths>\n"
    chunks_str = "<code_chunks>\n"
    for file, chunks in chunks_by_file.items():
        fp_str += f"{file.rel_path}\n"
        for chunk in chunks:
            chunks_str += f"<{file.rel_path}>\n{chunk.to_string(line_nums=False)}\n</{file.rel_path}>\n"

    fp_str += "</file_paths>"
    chunks_str += "</code_chunks>"

    return f"{fp_str}\n\n{chunks_str}"


async def find_bugs(context: str, file: File, model: str) -> List[Bug]:
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
                                        "description": "Concise description of the bug and its impact. No more than a few short sentences. Remember: this MUST be a bug that could cause runtime errors, crashes, or incorrect behavior. It is NOT a style issue, suggestion, or anything else.",
                                    },
                                    "confidence": {
                                        "type": "integer",
                                        "description": "Confidence score between 0 and 10. 10 indicates high confidence in the bug's existence and severity. 0 indicates low confidence (e.g. the bug might not be a problem or is low severity).",
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
                            "description": "List of bugs in the code file. Each bug should be a code block and a description of the bug contained in that code block. This list can be empty if no bugs are present in the file.",
                        },
                    },
                    "required": ["bugs"],
                    "additionalProperties": False,
                },
            },
        },
        drop_params=True,
    )
    response = response.choices[0].message.content
    response = json_repair.loads(response)

    bugs = []
    for bug in response["bugs"]:
        bugs.append(
            Bug(
                bug["start"],
                bug["end"],
                bug["description"],
                bug["confidence"] * 10,
            )
        )

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
            "Call this when you have enough context to analyze the file for bugs."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "A brief explanation of why you think you have enough context to analyze the file for bugs.",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string", "enum": []},
                    "description": "The files with the context you need to analyze the file for bugs. Can be empty if no additional context on the file is needed.",
                },
            },
            "required": ["reason", "files"],
            "additionalProperties": False,
        }
        self.is_terminal = True

        # Additional attributes
        self.model = model
        self.target_file = target_file
        self.update_progress = update_progress

    def update_definition(self, trajectory: List[Message] = [], **kwargs):
        files = set()
        for message in trajectory:
            if not message.raw_output:
                continue

            for item in message.raw_output:
                if isinstance(item, Chunk):
                    files.add(item.file.rel_path)
                elif isinstance(item, File):
                    files.add(item.rel_path)

        files = [file for file in files if file != self.target_file.rel_path]
        self.parameters["properties"]["files"]["items"]["enum"] = files

    async def run(self, reason: str, files: List[str], **kwargs) -> List[Bug]:
        trajectory = kwargs.get("trajectory")
        chunks = get_chunks(trajectory)
        chunks = filter_chunks(chunks, files)
        chunks = truncate_chunks(chunks, self.model)
        chunks = merge_chunks(chunks)
        context = build_context_str(chunks)
        bugs = await find_bugs(context, self.target_file, self.model)

        return bugs
