# Modules 4 and 5 evaluation: Harshini Halai

Python 3.10+ (tested on 3.12). Standard library only; `pytest` for tests. The provided `data/` folder is
included unchanged.

## Q26: SFT data readiness

    python q26/pipeline.py --data data/q26 --out q26/out

Tests: `cd q26 && pytest`. Answers: `q26/README.md`. Notes: `q26/q26_notes.md`.

## Q27: grounded policy assistant

    python q27/run_all.py --data data/q27 --out q27/results.json

Tests: `cd q27 && pytest`. Answers: `q27/README.md`. Notes: `q27/q27_notes.md`.