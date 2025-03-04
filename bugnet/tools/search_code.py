# Standard library
import re
from typing import List
from collections import defaultdict

# Third party
from saplings.abstract import Tool

# Local
try:
    from bugnet.index import Index, File, Chunk
except ImportError:
    from index import Index, File, Chunk


#########
# HELPERS
#########


MAX_CHUNKS = 10


def query_to_regex(query: str) -> str:
    return "|".join(map(re.escape, query.split()))


def rank_chunks(chunks: List[Chunk]) -> List[Chunk]:
    # TODO: Implement BM25 or neural ranking
    return chunks


######
# MAIN
######


class CodeSearchTool(Tool):
    def __init__(
        self, index: Index, target_file: File, update_progress: callable, **kwargs
    ):
        # Base attributes
        self.name = "search_code"
        self.description = "Search (grep) the contents of files in a codebase using regular expressions. Returns code snippets containing exact matches."
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

    def format_output(self, chunks: List[Chunk]) -> str:
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
            formatted_chunk = (
                f"<{file.rel_path}>\n{chunk.to_string()}\n</{file.rel_path}>"
            )
            formatted_chunks.append(formatted_chunk)

        formatted_chunks = "\n\n".join(formatted_chunks)
        return formatted_chunks

    async def run(self, intent: str, query: str, **kwargs) -> List[Chunk]:
        self.update_progress(intent)
        query = query_to_regex(query)
        results = self.index.search_code(query, exclude=self.target_file)
        results = rank_chunks(results)
        return results[:MAX_CHUNKS]
