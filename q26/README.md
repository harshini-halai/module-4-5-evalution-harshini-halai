# Q26: SFT data readiness and honest evaluation

Run from the repo root:

    python q26/pipeline.py --data data/q26 --out q26/out
    cd q26 && pytest

## 1. summary.json figures

| Item | Value |
|---|---|
| Valid after Part A | 266 |
| Rejected in A | EMPTY_OUTPUT 8, TEMPLATE_RESIDUE 8, MALFORMED 5, BAD_INTENT 4 |
| Dropped by licence (B) | 20 |
| Eligible models | qwenlike-1.5b, llamalike-3b, mistrallike-7b, community/llamalike-3b-support-ft |
| Blocked models | medtune-3b, research-2b, fastchat-1b |
| Exact duplicates removed (C) | 12 |
| Near duplicates removed (C) | 12 |
| Pool after C | 222 |
| Leaked eval items (D) | 10 |
| Clean eval items | 52 |
| Train / validation (E) | 201 / 21 |
| Base score (full / clean / leaked only) | 55.6 / 55.8 / 55.0 |
| Fine-tuned score (full / clean / leaked only) | 66.5 / 60.1 / 100.0 |
| Gain (full / clean) | +10.9 / +4.3 |

## 2. Which gain would I show my manager?

I would show the clean gain. The full-set gain is bigger because some eval items also exist in the training data (leaks), so the fine-tuned model has basically seen the answers and just repeats them. That is memorising, not getting better at support. The clean set only has items the model never saw, so it shows what customers will actually get. The leaked-only score of the fine-tuned model being much higher than its clean score is the proof.

## 3. Did fine-tuning make an intent worse?

Yes. damaged_goods fell from 75.0 (base) to 50.0 (fine-tuned) on the clean set. The
fine-tuned model's generic answer says "Compensation is guaranteed", and the rubric
forbids the word "guaranteed". This matters more than the headline because it is a
promise the company cannot keep (legal and trust risk), and an average gain hides it.

## 4. Eligible base models

community/llamalike-3b-support-ft: its declared licence looks fine, but it names a base_model, so the effective licence is the base model's licence. The base is custom-community, and its mau_limit is below our 2,000,000 MAU, so the model is blocked, and it gets a warning because declared and effective licence differ. [check this matches my licence_report.json]

## 5. Why leakage runs after the licence gate

Records dropped for licence are not in the training data, so an eval question that
matches them is not leakage: the model never saw it. Checking against the raw data
(after Part A only) flags 12 eval items, against 10 for the post-gate pool, so two
good eval items would be thrown away for nothing.

## 6.Example where comparing instructions alone is wrong

Example: records [id1] and [id2] have the same (or almost the same) instruction, but different input and output, for example the same question about a parcel but a different situation and a different answer. If I compared only instructions they would look like duplicates and one would be removed, but they teach different answers, so both must be kept. Comparing the whole record (instruction + input + output) avoids this.
