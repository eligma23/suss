# Standard library
from collections import defaultdict
import json
import logging

# Third party
import json_repair
from saplings.dtos import Message
from saplings.abstract import Tool
from litellm import acompletion, encode, decode

# Local
try:
    from suss.index import Index, File, Chunk
    from suss.constants import MAX_MERGE_DISTANCE, MAX_CONTEXT_TOKENS
except ImportError:
    from index import Index, File, Chunk
    from constants import MAX_MERGE_DISTANCE, MAX_CONTEXT_TOKENS


logger = logging.getLogger(__name__)

PROMPT = """I want to find all the code in a file that's relevant to a query.

Your output should be a list of line ranges. Each line range should correspond to a block of code that's relevant to the query.

Line ranges should be inclusive (e.g. {{"start": 12, "end": 15}} includes lines 12, 13, 14, and 15).

## Output Format Requirements
Please output the result in **valid JSON format**. You MUST output a JSON object with EXACTLY the following structure:
{{
  "bugs": [
    {{
      "start": <starting-line-number>,
      "end": <ending-line-number>,
      "description": "<bug-description>",
      "confidence": <confidence-score-0-100>
    }}
    // ... more bugs if present
  ]
}}

## Additional Rules
- Use integer values for start/end lines
- Confidence must be between 0 and 100
- If no bugs found, return {{ "bugs": [] }}

--

Here is the code in the file ({file_path}):

<code>
{file_content}
</code>

And here is the query to find relevant code for:

<query>
{query}
</query>"""


def truncate_file_content(file: File) -> str:
    tokens = encode(model="", text=file.content)[:MAX_CONTEXT_TOKENS]
    file_str = decode(model="", tokens=tokens)
    return file_str


def adapt_completion_args(model: str, messages: list) -> dict:
    args = {
        "model": model,
        "messages": messages,
        "drop_params": True,
    }
    
    if model.startswith("deepseek/"):
        args["api_base"] = "https://api.deepseek.com/v1/chat/completions"
        logger.debug("Adapting request for DeepSeek API")
    else:
        args["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "find_code_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "line_ranges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "integer"},
                                    "end": {"type": "integer"},
                                },
                                "required": ["start", "end"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["line_ranges"],
                    "additionalProperties": False,
                },
            },
        }
    return args


async def extract_chunks(file: File, query: str, model: str) -> list[Chunk]:
    file_path = file.rel_path
    file_content = truncate_file_content(file)
    
    messages = [{
        "role": "user",
        "content": PROMPT.format(
            file_path=file_path, 
            file_content=file_content, 
            query=query
        ),
    }]

    completion_args = adapt_completion_args(model, messages)
    response = await acompletion(**completion_args)

    try:
        response_data = json_repair.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        repaired = response.choices[0].message.content.strip().strip("`")
        response_data = json_repair.loads(repaired)

    line_ranges = [lr for lr in response_data["line_ranges"] if lr["start"] <= lr["end"]]
    line_ranges = [list(range(lr["start"], lr["end"] + 1)) for lr in line_ranges]
    return [Chunk(line_nums, file) for line_nums in line_ranges]


def clamp_chunks(chunks: list[Chunk]) -> list[Chunk]:
    clamped_chunks = []
    for chunk in chunks:
        chunk.line_nums = [ln for ln in chunk.line_nums if ln <= chunk.file.last_lineno]
        if chunk.line_nums:
            clamped_chunks.append(chunk)
    return clamped_chunks


def merge_chunks(chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        return []

    chunks.sort(key=lambda chunk: chunk.line_nums[0])
    merged = []
    curr_chunk = chunks[0]

    for next_chunk in chunks[1:]:
        curr_end = curr_chunk.line_nums[-1]
        next_start = next_chunk.line_nums[0]
        
        if curr_end >= next_start or next_start - curr_end < MAX_MERGE_DISTANCE:
            curr_chunk.line_nums = list(range(curr_chunk.line_nums[0], next_chunk.line_nums[-1] + 1))
        else:
            merged.append(curr_chunk)
            curr_chunk = next_chunk

    merged.append(curr_chunk)
    return list(set(merged))


class ReadFileTool(Tool):
    def __init__(
        self,
        index: Index,
        model: str,
        target_file: File,
        update_progress: callable,
        **kwargs,
    ):
        self.name = "read_file"
        self.description = "Read (search within) a file in the codebase. Returns the most relevant parts of the file."
        self.parameters = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Concise description of your intent behind reading the file.",
                },
                "file": {
                    "type": "string",
                    "enum": [],
                    "description": "The path to the file you want to read.",
                },
                "query": {
                    "type": "string",
                    "description": "A semantic search query.",
                },
            },
            "required": ["intent", "file", "query"],
            "additionalProperties": False,
        }
        self.is_terminal = False
        self.index = index
        self.model = model
        self.target_file = target_file
        self.update_progress = update_progress

    def format_output(self, chunks: list[Chunk]) -> str:
        grouped_chunks = defaultdict(list)
        for chunk in chunks:
            grouped_chunks[chunk.file].append(chunk)

        formatted_chunks = []
        for file, chunks in grouped_chunks.items():
            for chunk in chunks:
                formatted_chunk = f"<{file.rel_path}>\n{chunk.to_string(dots=False)}\n</{file.rel_path}>"
                formatted_chunks.append(formatted_chunk)

        return "\n\n".join(formatted_chunks)

    def update_definition(self, trajectory: list[Message] = [], **kwargs):
        files = set()
        for message in trajectory:
            if message.raw_output:
                for item in message.raw_output:
                    if isinstance(item, File):
                        files.add(item.rel_path)
                    elif isinstance(item, Chunk):
                        files.add(item.file.rel_path)

        self.parameters["properties"]["file"]["enum"] = [
            f for f in files if f != self.target_file.rel_path
        ]

    def is_active(self, trajectory: list[Message] = [], **kwargs) -> bool:
        if not trajectory:
            return False

        for message in trajectory:
            if message.raw_output:
                for item in message.raw_output:
                    if isinstance(item, (File, Chunk)):
                        self.update_definition(trajectory)
                        return True
        return False

    async def run(self, intent: str, file: str, query: str, **kwargs) -> list[Chunk]:
        self.update_progress(intent)
        file = self.index.get_file(file)
        chunks = await extract_chunks(file, query, self.model)
        chunks = clamp_chunks(chunks)
        chunks = merge_chunks(chunks)
        return chunks
