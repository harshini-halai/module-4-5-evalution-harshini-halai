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

2. Chosen configuration

I chose [config + raw/rewritten]. It has good recall and MRR and its context words are [number], under the 300-word budget. The config with the highest recall (most likely C4, size 200) is not usable because three chunks of about 200 words are about 600 words, which is over the 300 budget. A high recall that cannot fit into the context is not useful.

3. Why precision@3 is stuck near 0.35 without overlap

Most queries have only one gold span. Without overlap (C1, C3) a span sits inside one chunk only, so at most 1 of the 3 retrieved chunks can be relevant, which gives 1/3 = 0.33 (a bit above only when a query has more than one gold span).

With overlap (C2, C4) the same span can be fully inside two or three overlapping chunks, so more of the top 3 are relevant and precision goes above 0.35. But a higher number there does not mean better retrieval. It is the same piece of text counted again and again in different chunks, so it just shows that the chunks overlap.

## 4. Query Q07

Only C1 fails Q07 (raw and rewritten). It is a chunking failure: the gold span is characters 699-840 of
damaged_goods.md, and C1 has a chunk boundary at 756 (chunks 381-756 and 757-1101), so the span is cut in
two. C2 has a chunk 513-869 that contains it fully, so it passes. The embedding found the right document in
both cases.

## 5. Rewriting

Fixed by rewriting: [query id]. The query had a short form (like FTE) that the document writes in full, so adding the expansion lets the embedding match.

Still failing in my chosen config: [query id]. Rewriting does not fix it because [the glossary has no term in it / the problem is the chunk or the embedding, not an abbreviation]. Next I would try: a different chunk size for that document, or adding a title/heading to each chunk, or query expansion with synonyms.


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
