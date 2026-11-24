# Q27: Grounded policy assistant

Run from the repo root:

    python q27/run_all.py --data data/q27 --out q27/results.json
    cd q27 && pytest

## 1. Metrics table (k = 3)

| Config | Queries | precision@3 | recall@3 | MRR | context words | within 300-word budget |
|---|---|---|---|---|---|---|
| C1 | raw | 0.2167 | 0.575 | 0.575 | 160.2 | yes |
| C1 | rewritten | 0.2167 | 0.575 | 0.600 | 153.6 | yes |
| C2 | raw | 0.3000 | 0.725 | 0.750 | 167.8 | yes |
| C2 | rewritten | 0.3333 | 0.825 | 0.800 | 168.8 | yes |
| C3 | raw | 0.3167 | 0.900 | 0.900 | 177.4 | yes |
| C3 | rewritten | 0.3333 | 0.950 | 0.950 | 169.5 | yes |
| C4 | raw | 0.3333 | 0.950 | 0.925 | 475.4 | no |
| C4 | rewritten | 0.3500 | 1.000 | 0.950 | 479.9 | no |

Queries with recall@3 < 1 are listed per row in results.json (C3 rewritten: Q12 only).

## 2. Chosen configuration

C3 (paragraph chunker, max_words 80) with query rewriting. C4 rewritten has the highest recall (1.0), but its
mean context is about 480 words, above the 300-word budget, so it cannot be used. C3 reaches recall 0.95 with
about 170 words of context.

## 3. Why precision@3 stays near 0.35 without overlap

Without overlap every gold span lies in at most one chunk, so a query can have at most one relevant chunk
in its top 3, i.e. precision 1/3. The only exception is Q20, whose two gold spans are in two different
documents (so 2/3 is possible). Q18's two spans are next to each other and sit in the same paragraph chunk,
so they count as one relevant chunk. The ceiling for C3 is therefore (19 x 1/3 + 1 x 2/3) / 20 = 0.35 (my
C3 with rewriting reaches 0.3333). C1 is lower still (about 0.217) because several spans are cut by a chunk
boundary and are fully contained in no chunk.

With overlap the same span sits inside several windows. In C2, Q05 has two and Q20 has three chunks that
fully contain a gold span, so a single query can reach precision 2/3 or even 1.0, and C2's ceiling rises to
about 0.367. In my run no overlapping configuration beats 0.35 on average (C4 with rewriting reaches exactly
0.35), but individual queries do. A higher number there does not mean better retrieval: the same evidence
is counted several times, near-copies fill the three slots, and no new information is found. Recall and MRR
say whether the needed spans were found; extra copies also waste the 300-word context budget.
## 4. Query Q07

Only C1 fails Q07 (raw and rewritten). It is a chunking failure: the gold span is characters 699-840 of
damaged_goods.md, and C1 has a chunk boundary at 756 (chunks 381-756 and 757-1101), so the span is cut in
two. C2 has a chunk 513-869 that contains it fully, so it passes. The embedding found the right document in
both cases.

## 5. Rewriting

Q01 is fixed by rewriting ("PTO" becomes "PTO (paid annual leave)"); it fails in C3 raw. Q12 still fails in my
chosen configuration: the query says "compensation" and "late", while the policy says "service credit" and
"misses the committed delivery date". That is a vocabulary mismatch, not an abbreviation, so the glossary
cannot fix it. Next I would try adding synonyms to the rewrite step, a keyword (BM25) plus vector hybrid,
or HyDE.

## 6. Grounding check

Flagged: A2, A4, A6. A2: "available in all tier-2 cities" is not in the cited text (overlap 0.2). A4: says
30 days but the cited text says 60 (missing number 30). A6: the cited text is the expense policy, while the
claim is about delivery SLA (overlap 0.18, numbers 2 and 4 missing). An exact-substring check would be wrong
for A3 because A3 is a correct paraphrase ("seven-day window" for "7 days"). A6 is true in the corpus, but
its citation does not support it, and a grounded answer must be supported by what it cites, otherwise the
reader cannot verify it.

## 7. Agent scenarios

- leave_balance: search found 18 days, calculate 18 - 11, answer 7 days. Stopped by final_answer (3 calls).
- looping: the model repeats the same search; stopped on the third identical call, repeated_call (3 calls).
- unknown_tool: delete_records does not exist, an error went back to the model, which refused. final_answer
  (2 calls).
- bad_arguments: malformed JSON produced an error message, the model retried correctly and answered 45 days.
  final_answer (3 calls).
- unsafe_expression: the calculator refused the __import__ expression, then 2 ** 10 = 1024. final_answer
  (3 calls).

calculate must never use eval because eval runs text as Python code, so a model or user could execute
commands. I parse the text into an AST and only allow numbers and + - * / ** % operators.