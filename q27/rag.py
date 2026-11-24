"""Q27 retrieval layer: chunkers, index, glossary rewriting, retrieval metrics.

All offsets are CHARACTER offsets into the file read as text (utf-8, newline='' so that
nothing is translated). Standard library only; embeddings come from the provided kestrel_embed.
"""
import glob
import json
import os
import re
import sys

DATA_DIR = os.environ.get("Q27_DATA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "q27")
sys.path.insert(0, DATA_DIR)
from kestrel_embed import embed, cosine  # noqa: E402  (provided file, not modified)

K = 3
WORD_BUDGET = 300

CONFIGS = {
    "C1": {"chunker": "fixed", "size": 60, "overlap": 0},
    "C2": {"chunker": "fixed", "size": 60, "overlap": 20},
    "C3": {"chunker": "paragraph", "max_words": 80},
    "C4": {"chunker": "fixed", "size": 200, "overlap": 40},
}


# ------------------------------------------------------------------ loading
def load_corpus(data_dir=None):
    data_dir = data_dir or DATA_DIR
    corpus = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "corpus", "*.md"))):
        with open(path, encoding="utf-8", newline="") as f:
            corpus[os.path.basename(path)] = f.read()
    return corpus


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ chunkers
def make_chunk(doc, text, start, end):
    return {"doc": doc, "start": start, "end": end, "text": text[start:end]}


def fixed_chunks(doc, text, size, overlap):
    """Windows of `size` words advancing by size-overlap; stop after the window reaching the last word."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    chunks, i, step = [], 0, size - overlap
    while i < len(words):
        window = words[i:i + size]
        chunks.append(make_chunk(doc, text, window[0][0], window[-1][1]))
        if i + size >= len(words):
            break
        i += step
    return chunks


def _count_words(text, start, end):
    return len(text[start:end].split())


def _split_sentences(text, start, end):
    """Split the span [start, end) on sentence boundaries; returns [(s, e), ...] with offsets into text."""
    spans, s = [], start
    for m in re.finditer(r"(?<=[.!?])\s+", text[start:end]):
        spans.append((s, start + m.start()))
        s = start + m.end()
    spans.append((s, end))
    return [(a, b) for a, b in spans if b > a]


def paragraph_chunks(doc, text, max_words):
    """Pack consecutive paragraphs (blank-line separated) up to max_words.

    A paragraph longer than max_words is split on sentence boundaries, and its sentences are packed
    among themselves (never merged with neighbouring paragraphs).
    """
    paras = [(m.start(), m.end()) for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, flags=re.S)]
    chunks, cur = [], None          # cur = [start, end, words]

    def flush():
        nonlocal cur
        if cur:
            chunks.append(make_chunk(doc, text, cur[0], cur[1]))
            cur = None

    def add(s, e):
        nonlocal cur
        w = _count_words(text, s, e)
        if cur and cur[2] + w <= max_words:
            cur[1] = e
            cur[2] += w
        else:
            flush()
            cur = [s, e, w]

    for s, e in paras:
        if _count_words(text, s, e) > max_words:
            flush()
            for ss, se in _split_sentences(text, s, e):
                add(ss, se)
            flush()
        else:
            add(s, e)
    flush()
    return chunks


def build_chunks(corpus, config):
    out = []
    for doc, text in corpus.items():
        if config["chunker"] == "fixed":
            out.extend(fixed_chunks(doc, text, config["size"], config["overlap"]))
        else:
            out.extend(paragraph_chunks(doc, text, config["max_words"]))
    return out


# ------------------------------------------------------------------ index + retrieval
class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectors = [embed(c["text"]) for c in chunks]

    def search(self, query, k=K):
        q = embed(query)
        scored = [(cosine(q, v), i) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda t: (-t[0], t[1]))          # ties: earlier chunk first (deterministic)
        return [self.chunks[i] for _, i in scored[:k]]


def rewrite_query(query, glossary):
    """Append ' (expansion)' after every whole-word, case-sensitive glossary term."""
    if not glossary:
        return query
    terms = sorted(glossary, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w)")
    return pattern.sub(lambda m: "%s (%s)" % (m.group(1), glossary[m.group(1)]), query)


# ------------------------------------------------------------------ metrics
def is_relevant(chunk, gold):
    return any(chunk["doc"] == g["doc"] and chunk["start"] <= g["start"] and chunk["end"] >= g["end"] for g in gold)


def query_metrics(retrieved, gold):
    rel = [is_relevant(c, gold) for c in retrieved]
    covered = sum(
        1 for g in gold
        if any(c["doc"] == g["doc"] and c["start"] <= g["start"] and c["end"] >= g["end"] for c in retrieved)
    )
    rr = 0.0
    for rank, flag in enumerate(rel, start=1):
        if flag:
            rr = 1.0 / rank
            break
    return {
        "precision": sum(rel) / K,
        "recall": covered / len(gold),
        "rr": rr,
        "context_words": sum(len(c["text"].split()) for c in retrieved),
    }


def evaluate(index, queries, glossary=None):
    """Return (mean metrics dict, per-query rows). glossary=None -> raw queries."""
    rows = []
    for q in queries:
        text = rewrite_query(q["query"], glossary) if glossary else q["query"]
        retrieved = index.search(text, K)
        m = query_metrics(retrieved, q["gold"])
        m["query_id"] = q["query_id"]
        rows.append(m)
    n = len(rows)
    mean = lambda key: round(sum(r[key] for r in rows) / n, 4)
    return {"precision@3": mean("precision"), "recall@3": mean("recall"), "MRR": mean("rr"),
            "context_words": round(sum(r["context_words"] for r in rows) / n, 1),
            "failed_queries": [r["query_id"] for r in rows if r["recall"] < 1]}, rows