# Standard library
import re
from collections import defaultdict

# Third party
from saplings.abstract import Tool

# Local
try:
    from suss.reranker import rerank_chunks
    from suss.index import Index, File, Chunk
    from suss.constants import CODE_SEARCH_LIMIT
except ImportError:
    from reranker import rerank_chunks
    from index import Index, File, Chunk
    from constants import CODE_SEARCH_LIMIT


#########
# HELPERS
#########


def query_to_regex(query: str) -> str:
    return "|".join(map(re.escape, query.split()))


async def filter_chunks(query: str, chunks: list[Chunk]) -> list[Chunk]:
    return chunks  # TODO: LLM-based filtering


######
# MAIN
######


class GrepCodeTool(Tool):
    def __init__(
        self, index: Index, target_file: File, update_progress: callable, **kwargs
    ):
        # Base attributes
        self.name = "search_code"
        self.description = "Searches the contents of files in the codebase using regular expressions. Returns code snippets containing exact matches."
        self.parameters = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Concise, one-sentence description of your intent behind the search. E.g. 'Find the definition of handle_auth', 'Track lifecycle of connection_pool', 'Understand how the parser is used'.",
                },
                "query": {
                    "type": "string",
                    "description": "A search query, passed into Python's re.match(). Should match symbols in the codebase.",
                },
            },
            "required": ["intent", "query"],
            "additionalProperties": False,
        }
        self.is_terminal = False

        # Additional attributes
        self.index = index
        self.target_file = target_file
        self.update_progress = update_progress

    def format_output(self, chunks: list[Chunk]) -> str:
        grouped_chunks = defaultdict(list)
        for chunk in chunks:
            grouped_chunks[chunk.file].append(chunk)

        formatted_chunks = []
        for file, chunks in grouped_chunks.items():
            line_nums = set()
            for chunk in chunks:
                line_nums |= set(chunk.line_nums)
            line_nums = list(line_nums)
            line_nums.sort()

            chunk = Chunk(line_nums, file)
            formatted_chunk = f"<file_path>{file.rel_path}</file_path>\n<file_content>\n{chunk.to_string()}\n</file_content>"
            formatted_chunks.append(formatted_chunk)

        formatted_chunks = "\n\n".join(formatted_chunks)
        return formatted_chunks

    async def run(self, intent: str, query: str, **kwargs) -> list[Chunk]:
        self.update_progress(intent)
        query_regex = query_to_regex(query)
        results = self.index.search_code(query_regex, exclude=self.target_file)
        results = await rerank_chunks(query, results)
        results = await filter_chunks(query, results)
        return results[:CODE_SEARCH_LIMIT]


# TODO: Implement a semantic fallback if AST-grep fails to retrieve enough chunks.
# 1. Get the code map (c-tags) for the codebase.
# 2. Use the code map and the query to generate a hypothetical code snippet.
# 3. Run an (AST-based) keyword search on the codebase using the snippet.
# 4. Rerank the results.
