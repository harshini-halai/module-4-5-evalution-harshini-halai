"""Run all of Q27 and write q27/results.json.

Usage (from the repo root):  python q27/run_all.py --data data/q27 --out q27/results.json
"""
import argparse
import json
import os
import sys

CHOSEN = "C3"          # chosen configuration (see README); query rewriting stays on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"))
    args = ap.parse_args()
    if args.data:
        os.environ["Q27_DATA"] = os.path.abspath(args.data)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import rag
    import grounding
    import agent
    from mock_llm import MockLLM, SCENARIOS

    corpus = rag.load_corpus()
    queries = rag.load_jsonl(os.path.join(rag.DATA_DIR, "queries.jsonl"))
    with open(os.path.join(rag.DATA_DIR, "glossary.json"), encoding="utf-8") as f:
        glossary = json.load(f)

    # Parts A-C: eight rows (4 configs x raw/rewritten)
    table, indexes = [], {}
    for name, cfg in rag.CONFIGS.items():
        chunks = rag.build_chunks(corpus, cfg)
        indexes[name] = rag.Index(chunks)
        for mode, gl in (("raw", None), ("rewritten", glossary)):
            m, _ = rag.evaluate(indexes[name], queries, gl)
            table.append({"config": name, "queries": mode, "n_chunks": len(chunks), **m,
                          "within_300_word_budget": m["context_words"] <= rag.WORD_BUDGET})

    # Part D
    answers = rag.load_jsonl(os.path.join(rag.DATA_DIR, "answers_to_check.jsonl"))
    grounding_results = [grounding.check_answer(a, corpus) for a in answers]

    # Part E
    chosen_index = indexes[CHOSEN]

    def search_policy(query, k=3):
        hits = chosen_index.search(rag.rewrite_query(query, glossary), int(k))
        return "\n---\n".join(h["text"] for h in hits)

    tools = {"search_policy": search_policy, "calculate": agent.calculate}
    runs = []
    for scenario in SCENARIOS:
        llm = MockLLM(scenario)
        r = agent.run_agent(llm, tools, max_steps=6)
        runs.append({"scenario": scenario, "final_answer": r["final_answer"],
                     "termination_reason": r["termination_reason"], "model_calls": r["model_calls"],
                     "audit": [list(a) for a in llm.audit]})

    results = {"metrics_table": table, "chosen_configuration": {"config": CHOSEN, "chunker": rag.CONFIGS[CHOSEN],
               "query_rewriting": True, "k": rag.K},
               "grounding": grounding_results, "agent_runs": runs}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({"chosen": CHOSEN, "agent_runs": [(r["scenario"], r["termination_reason"], r["model_calls"], r["final_answer"]) for r in runs]}, indent=2))


if __name__ == "__main__":
    main()