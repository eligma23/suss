# Standard library
import re
import os
import math
from collections import defaultdict

# Third party
import cohere
from saplings.abstract import Tool
from tree_sitter_language_pack import get_parser

# Local
try:
    from suss.index import Index, File, Chunk
    from suss.constants import CODE_SEARCH_LIMIT
except ImportError:
    from index import Index, File, Chunk
    from constants import CODE_SEARCH_LIMIT


#########
# HELPERS
#########


try:
    client = cohere.AsyncClient(os.environ["COHERE_API_KEY"])
except:
    client = None

MAX_CHUNKS = 10000


def tokenize_query(query: str) -> list[str]:
    tokens = re.findall(r"\w+", query)
    return tokens


# TODO: Make this less dumb
def tokenize_code(chunk: Chunk) -> list[str]:
    def extract_tokens(node, code_bytes: bytes) -> list[str]:
        tokens = []
        if node.child_count == 0:
            token = code_bytes[node.start_byte : node.end_byte].decode(
                "utf8", errors="ignore"
            )

            # Skip tokens that are only whitespace or punctuation
            if token.strip() and not re.fullmatch(r"\W+", token):
                tokens.append(token.lower())
        else:
            for child in node.children:
                tokens.extend(extract_tokens(child, code_bytes))

        return tokens

    parser = get_parser(chunk.file.language)
    code_bytes = bytes(chunk.to_string(False, False), "utf8")
    tree = parser.parse(code_bytes)
    return extract_tokens(tree.root_node, code_bytes)


def bm25_rerank(
    query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75
) -> list[Chunk]:
    if not chunks:
        return chunks

    tokenized_query = tokenize_query(query)
    tokenized_chunks = [tokenize_code(chunk) for chunk in chunks]
    N = len(tokenized_chunks)

    token_freq = {}
    for tokens in tokenized_chunks:
        for token in set(tokens):
            token_freq[token] = token_freq.get(token, 0) + 1

    avgdl = sum(len(tokens) for tokens in tokenized_chunks) / N

    # Compute BM25 scores
    scores = []
    for i, tokens in enumerate(tokenized_chunks):
        score = 0.0

        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        dl = len(tokens)
        for t in tokenized_query:
            if t not in token_counts:
                continue

            n_t = token_freq.get(t, 0)
            idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)
            tf = token_counts[t]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))

        scores.append((score, i))

    ranked_indices = sorted(scores, key=lambda x: x[0], reverse=True)
    ranked_chunks = [chunks[i] for _, i in ranked_indices]
    return ranked_chunks


async def neural_rerank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    chunk_strs = [chunk.to_string(line_nums=False) for chunk in chunks[:MAX_CHUNKS]]
    response = await client.rerank(
        model="rerank-v3.5", query=query, documents=chunk_strs
    )
    for result in response.results:
        score = result.relevance_score
        index = result.index
        chunks[index].score = score

    ranked_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked_chunks


def query_to_regex(query: str) -> str:
    return "|".join(map(re.escape, query.split()))


async def rerank_chunks(query: str, chunks: list[Chunk]) -> list[Chunk]:
    if os.getenv("COHERE_API_KEY", None):
        return await neural_rerank(query, chunks)

    return bm25_rerank(query, chunks)


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
