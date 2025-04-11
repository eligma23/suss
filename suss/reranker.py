# Standard library
import re
import os
import math

# Third party
import cohere
from tree_sitter_languages import get_parser

# Local
try:
    from suss.index import Chunk
except ImportError:
    from index import Chunk


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


######
# MAIN
######


async def rerank_chunks(query: str, chunks: list[Chunk]) -> list[Chunk]:
    if os.getenv("COHERE_API_KEY", None):
        return await neural_rerank(query, chunks)

    return bm25_rerank(query, chunks)
