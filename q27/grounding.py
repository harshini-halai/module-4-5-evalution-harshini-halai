"""Q27 Part D: grounding check. A claim is grounded only if its numbers and enough of its words
appear in the text the answer itself cites."""
import re

import rag
from kestrel_embed import tokens

OVERLAP_THRESHOLD = 0.50
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def split_claims(answer):
    """Split an answer into sentences on . ! ? followed by whitespace (a decimal point has no space after it)."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p for p in parts if p]


def numbers_in(text):
    """Numbers as strings, thousands commas removed ('1,500' -> '1500')."""
    return set(NUMBER_RE.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)))


def cited_text(citations, corpus):
    return " ".join(corpus[c["doc"]][c["start"]:c["end"]] for c in citations)


def check_claim(claim, cited):
    missing = sorted(numbers_in(claim) - numbers_in(cited))
    claim_toks = tokens(claim)
    cited_toks = set(tokens(cited))
    overlap = (sum(1 for t in claim_toks if t in cited_toks) / len(claim_toks)) if claim_toks else 1.0
    grounded = not missing and overlap >= OVERLAP_THRESHOLD
    return {"claim": claim, "token_overlap": round(overlap, 3), "missing_numbers": missing, "grounded": grounded}


def check_answer(item, corpus):
    cited = cited_text(item["citations"], corpus)
    claims = [check_claim(c, cited) for c in split_claims(item["answer"])]
    return {"answer_id": item["answer_id"], "query_id": item.get("query_id"), "claims": claims,
            "flagged": any(not c["grounded"] for c in claims)}