"""kestrel_embed.py - deterministic local embedding model for the Q27 exercise.

Provided as part of the assignment. Do NOT modify it: every candidate must use the
same embedding function so that retrieval numbers are comparable and reproducible.

It is a hashed bag-of-words model (unigrams + bigrams, light suffix stripping,
sub-linear term frequency), L2-normalised. It runs offline, on CPU, with the
Python standard library only. Like any lexical model it does NOT know that
"FTE" means "full-time employee", which is deliberate.

    from kestrel_embed import embed, cosine
    v = embed("How many days of annual leave?")   # list[float], length DIM, unit norm
"""
import hashlib
import math
import re

DIM = 1024
_STOP = set("""a an the of to in on for and or is are be by at as it its this that with from
any all per not no do does can may must will your you our we i into than then there their
has have had was were which who what when how""".split())


def _stem(tok: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokens(text: str) -> list:
    """Lower-case alphanumeric tokens, stop words removed, lightly stemmed."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in raw if t not in _STOP]


def _bucket(feature: str) -> tuple:
    h = hashlib.md5(feature.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "little") % DIM
    sign = 1.0 if h[4] & 1 else -1.0
    return idx, sign


def embed(text: str) -> list:
    toks = tokens(text)
    feats = {}
    for t in toks:
        feats[t] = feats.get(t, 0) + 1
    for a, b in zip(toks, toks[1:]):
        k = a + "_" + b
        feats[k] = feats.get(k, 0) + 0.5
    vec = [0.0] * DIM
    for f, tf in feats.items():
        idx, sign = _bucket(f)
        vec[idx] += sign * (1.0 + math.log(tf)) if tf >= 1 else sign * tf
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def cosine(u: list, v: list) -> float:
    """Cosine similarity. Vectors from embed() are unit length, so this is a dot product."""
    return sum(a * b for a, b in zip(u, v))
