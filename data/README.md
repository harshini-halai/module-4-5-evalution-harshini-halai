# Data for the Modules 4 & 5 evaluation

- `q26/`: files for Question 26 (SFT data readiness). See §4 of the brief.
- `q27/`: files for Question 27 (retrieval, evaluation and a tool loop). See §5 of the brief.

Requirements: Python 3.10 or newer. Everything runs offline with the standard library; install `pytest` for your tests (`pip install pytest`). No GPU, API key or network access is needed.

Do not modify any file here, including `stub_models.py`, `kestrel_embed.py` and `mock_llm.py`. Your code is run against fresh copies.

Quick check that your Python can load the provided helpers (run from this `data/` folder):

    python3 -c "import sys; sys.path[:0]=['q26','q27']; import stub_models, kestrel_embed, mock_llm; print('ok', len(kestrel_embed.embed('hello')))"

It should print `ok 1024`.
