# Standard library
import re
from typing import List
from collections import defaultdict

# Third party
from saplings.abstract import Tool

# Local
try:
    from bugnet.index import Index, File
except ImportError:
    from index import Index, File


#########
# HELPERS
#########


MAX_FILES = 10


def additions(query: str, i: int, j: int) -> str:
    if i > j:
        return additions(query, j, i)

    query_chars = list(query)
    query_chars.insert(j, ".?")
    query_chars.insert(i, ".?")

    for index, char in enumerate(query_chars):
        if index in [i, j + 1]:
            continue

        query_chars[index] = re.escape(char)

    return "".join(query_chars)


def replacements(query: str, i: int, j: int) -> str:
    if i > j:
        return replacements(query, j, i)

    query_chars = list(query)
    query_chars.pop(j)
    query_chars.insert(j, ".?")
    query_chars.pop(i)
    query_chars.insert(i, ".?")

    for index, char in enumerate(query_chars):
        if index in [i, j]:
            continue

        query_chars[index] = re.escape(char)

    return "".join(query_chars)


def one_of_each(query: str, i: int, j: int) -> str:
    if i > j:
        return one_of_each(query, j, i)

    query_chars = list(query)
    query_chars.pop(j)
    query_chars.insert(j, ".?")
    query_chars.insert(i, ".?")

    for index, char in enumerate(query_chars):
        if index in [i, j + 1]:
            continue

        query_chars[index] = re.escape(char)

    return "".join(query_chars)


def build_fuzzy_regex_filters(query: str) -> List[re.Pattern]:
    filters = []
    query_len = len(query)
    for i in range(query_len):
        for j in range(i, query_len):
            if j != query_len:
                filters.append(
                    re.compile(one_of_each(query, i, j), flags=re.IGNORECASE)
                )
                filters.append(
                    re.compile(replacements(query, i, j), flags=re.IGNORECASE)
                )

            filters.append(re.compile(additions(query, i, j), flags=re.IGNORECASE))

    return filters


def get_trigrams(query: str) -> List[str]:
    query_chars = list(query)

    trigrams = []
    while len(query_chars):
        if len(query_chars) <= 3:
            trigrams.append("".join(query_chars).lower())
            query_chars.clear()
        else:
            trigrams.append("".join(query_chars[:3]).lower())
            query_chars.pop(0)

    return trigrams


######
# MAIN
######


class FileSearchTool(Tool):
    def __init__(
        self, index: Index, target_file: File, update_progress: callable, **kwargs
    ):
        # Base attributes
        self.name = "search_files"
        self.description = "Search the file paths in a codebase. Returns file paths that may not exactly match the query, but will be roughly similar. Use when you want to find a specific file or directory."
        self.parameters = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Concise, one-sentence description of your goal in using this tool.",  # TODO: Examples
                },
                "query": {
                    "type": "string",
                    "description": "A search query. Should not contain whitespace. E.g. 'server/src', 'main', 'dtos.py'.",
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

    def format_output(self, files: List[File]) -> str:
        file_paths = (file.rel_path for file in files)
        return "\n".join(file_paths)

    async def run(self, intent: str, query: str, **kwargs) -> List[File]:
        self.update_progress(intent)

        query = query.lower()
        trigrams = get_trigrams(query)
        filters = build_fuzzy_regex_filters(query)

        # Get no. of trigrams that match the query, per file path
        file_counts = defaultdict(int)
        for file in self.index.files:
            for trigram in trigrams:
                if trigram not in file.rel_path.lower():
                    continue

                file_counts[file] += 1

        # Sort files by no. of trigram matches
        file_counts = sorted(
            file_counts.items(), key=lambda kv: (kv[1], kv[0].rel_path), reverse=True
        )

        # Filter file paths by edit distance
        matches = []
        for file, _ in file_counts:
            if file.rel_path == self.target_file.rel_path:
                continue

            for fuzzy_filter in filters:
                if fuzzy_filter.search(file.rel_path):
                    matches.append(file)
                    break

            if len(matches) >= MAX_FILES:
                break

        return matches
