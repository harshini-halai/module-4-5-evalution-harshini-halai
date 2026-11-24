# Q27 notes

## 1. AI & Resource Usage

- Used Claude to draft the chunkers, retrieval metrics, grounding
  check, safe calculator and agent loop, and to explain the results. I ran everything and checked the numbers
  myself, and wrote the README answers in my own words.
- Python docs for the ast module and re.

## 2. Design reasoning

- Part A: offsets are character offsets (file read as utf-8 with newline=''), so text == file_text[start:end].
  Paragraph chunker packs paragraphs up to 80 words; long paragraphs split on sentences.
- Part B: retrieval is top-3 by cosine; rewriting appends "(expansion)" after whole-word, case-sensitive terms.
- Part C: metrics exactly as defined in the brief; C3 with rewriting chosen because C4 breaks the 300-word budget.
- Part D: a claim is grounded if all digits appear in the cited text and at least 50% of its tokens do.
- Part E: the loop appends the assistant tool call and tool result before each model call, turns every tool
  problem into an error message, and stops on final_answer, max_steps, or the same call three times in a row.
