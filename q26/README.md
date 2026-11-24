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

The clean gain, +4.3. Ten eval questions also exist in the training pool, and the
fine-tuned model has memorised their answers: it scores 100.0 on those ten, against
55.0 for the base model. That memorisation lifts the full-set gain to +10.9. On the
clean set the model only sees questions it has not trained on, which is what real
customers will send, so +4.3 is the honest estimate.

## 3. Did fine-tuning make an intent worse?

Yes. damaged_goods fell from 75.0 (base) to 50.0 (fine-tuned) on the clean set. The
fine-tuned model's generic answer says "Compensation is guaranteed", and the rubric
forbids the word "guaranteed". This matters more than the headline because it is a
promise the company cannot keep (legal and trust risk), and an average gain hides it.

## 4. Eligible base models

Eligible: kestrel/candidate-qwenlike-1.5b (Apache-2.0), kestrel/candidate-mistrallike-7b
(Apache-2.0), kestrel/candidate-llamalike-3b (custom-community, 2M MAU is within the
700M limit) and community/llamalike-3b-support-ft. The last one declares Apache-2.0,
but it is derived from candidate-llamalike-3b, so its effective licence is the base
model's custom-community licence. It is still eligible because 2M MAU is below the
inherited 700M limit, but the report carries a warning because declared and effective
licences differ. Blocked: medtune-3b (CC-BY-NC), research-2b (research-only),
fastchat-1b ("other", unclear).

## 5. Why leakage runs after the licence gate

Records dropped for licence are not in the training data, so an eval question that
matches them is not leakage: the model never saw it. Checking against the raw data
(after Part A only) flags 12 eval items, against 10 for the post-gate pool, so two
good eval items would be thrown away for nothing.

## 6. Instruction-only comparison

R0285 ("I want to cancel my order KL552019874 before it ships...") and R0286 ("I want
to cancel my return KL552019874 before it ships...") have instruction similarity 0.86,
so an instruction-only rule would drop R0286. But their answers differ: orders are
cancelled by the merchant, while a return is cancelled on the Kestrel returns page.
Whole-record similarity is only 0.39, so both records are correctly kept.