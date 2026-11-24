Q26 notes
1. AI & Resource Usage
Used Claude to help me write these notes and the README answers. I checked them against my own output files before submitting.
Asked Claude how Jaccard similarity on token sets works and how to avoid comparing every pair twice (I wrote the loop myself).
Python docs for json, re, random.Random and csv.
No other tools used.
2. Design reasoning

Part A (validate and normalise). I convert each line into the canonical record first, then run the reject checks in the fixed order (MALFORMED, EMPTY, TEMPLATE_RESIDUE, BAD_INTENT). A record gets only one reason, so I stop at the first check that fails. Hardest bit: detecting the shape. I check for messages, then prompt/completion, then instruction/output, and a record that matches none of them is MALFORMED.

Part B (licence gate). Source licence is looked up in dataset_registry.json. If the source is missing from the registry or the licence is not on the allowed list, the record is dropped. For models I follow base_model until I reach a model with no base, and that licence is the effective one. Hardest bit: models that point to a base model, because the declared licence can look fine while the effective one is not. These get a warning.

Part C (dedup). Exact duplicates first (lowercase, collapse whitespace), then near duplicates with Jaccard >= 0.80 on token sets. I compare each record to the ones already kept, so the earlier record always wins. Hardest bit: keeping the order right, because near-dup must run only on what is left after exact-dup.

Part D (leakage). Each eval item is compared with the post-Part-C pool, both exact match and Jaccard >= 0.80. Non-leaking items go to eval_clean.jsonl. Hardest bit: using instruction + " " + input for eval but the pool also built the same way.

Part E (split). Group by intent, sort by id, shuffle with a fresh Random(4526) for each intent, first round(0.10 * n) go to validation. Hardest bit: remembering a fresh Random per intent, not one shared Random.

Part F (eval). Each item has the rubric checks plus the max words check. Item score = passed / applied, model score = mean * 100, rounded to 1 decimal. I run it for full, clean and leaked-only sets and per intent on clean. Hardest bit: items where a rubric key is missing, so "applied" counts only the checks that exist.
