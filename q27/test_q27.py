"""Offline tests for Q27 (no network, no API key). Run: cd q27 && pytest"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent
import grounding
import rag
from mock_llm import MockLLM

TEXT = "# Title\n\nAlpha beta gamma delta. Epsilon zeta eta theta.\n\nIota kappa lambda mu nu xi omicron pi rho sigma tau."


# ---- chunkers: offsets round-trip
def test_fixed_chunk_offsets_round_trip():
    chunks = rag.fixed_chunks("d.md", TEXT, size=5, overlap=2)
    assert chunks
    for c in chunks:
        assert TEXT[c["start"]:c["end"]] == c["text"]


def test_fixed_chunker_window_count_and_last_word():
    text = " ".join("w%d" % i for i in range(10))               # 10 words
    chunks = rag.fixed_chunks("d.md", text, size=4, overlap=1)     # starts 0,3,6 -> window from 6 reaches word 10
    assert [len(c["text"].split()) for c in chunks] == [4, 4, 4]
    assert chunks[-1]["text"].endswith("w9")


def test_paragraph_chunk_offsets_round_trip_and_packing():
    chunks = rag.paragraph_chunks("d.md", TEXT, max_words=8)
    for c in chunks:
        assert TEXT[c["start"]:c["end"]] == c["text"]
        assert len(c["text"].split()) <= 8 or "." in c["text"]
    assert chunks[0]["text"].startswith("# Title")


def test_long_paragraph_is_split_on_sentences():
    long_para = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = rag.paragraph_chunks("d.md", long_para, max_words=6)
    assert len(chunks) == 2 and all(long_para[c["start"]:c["end"]] == c["text"] for c in chunks)


# ---- metrics on a hand-made example
def test_metrics_known_answer():
    gold = [{"doc": "a.md", "start": 10, "end": 20}, {"doc": "a.md", "start": 50, "end": 60}]
    retrieved = [
        {"doc": "b.md", "start": 0, "end": 100, "text": "x y"},      # wrong document
        {"doc": "a.md", "start": 5, "end": 25, "text": "p q r"},     # contains gold span 1
        {"doc": "a.md", "start": 55, "end": 70, "text": "s"},        # starts inside span 2, does not contain it
    ]
    m = rag.query_metrics(retrieved, gold)
    assert m["precision"] == 1 / 3
    assert m["recall"] == 0.5
    assert m["rr"] == 0.5                                            # first relevant chunk is rank 2
    assert m["context_words"] == 6


def test_reciprocal_rank_zero_when_nothing_relevant():
    m = rag.query_metrics([{"doc": "z.md", "start": 0, "end": 5, "text": "a"}], [{"doc": "a.md", "start": 0, "end": 5}])
    assert m["rr"] == 0.0 and m["recall"] == 0.0


# ---- query rewriting
def test_rewrite_whole_word_case_sensitive():
    gl = {"FTE": "full-time employee", "T&E": "travel and expense"}
    assert rag.rewrite_query("How many days for an FTE?", gl) == "How many days for an FTE (full-time employee)?"
    assert rag.rewrite_query("The FTEs and fte", gl) == "The FTEs and fte"
    assert rag.rewrite_query("T&E rules", gl) == "T&E (travel and expense) rules"


# ---- calculator
def test_calculate_basic():
    assert agent.calculate("18 - 11") == "7"
    assert agent.calculate("2 ** 10") == "1024"
    assert agent.calculate("(1 + 2) * 3 % 4") == "1"
    assert agent.calculate("7 / 2") == "3.5"


def test_calculate_refuses_import_and_other_code():
    for bad in ("__import__('os').system('echo hi')", "open('x')", "a + 1", "[1,2]", "'a' * 3", "True + 1", "2 ** 100000"):
        try:
            agent.calculate(bad)
        except ValueError:
            continue
        raise AssertionError("calculate accepted %r" % bad)


# ---- agent loop
def _tools():
    return {"search_policy": lambda query, k=3: "stub result 18 days of paid annual leave", "calculate": agent.calculate}


def test_looping_scenario_terminates_with_repeated_call():
    llm = MockLLM("looping")
    r = agent.run_agent(llm, _tools(), max_steps=6)
    assert r["termination_reason"] == "repeated_call" and r["model_calls"] == 3


def test_unknown_tool_bad_json_and_unsafe_expression_do_not_crash():
    for scenario in ("unknown_tool", "bad_arguments", "unsafe_expression"):
        r = agent.run_agent(MockLLM(scenario), _tools(), max_steps=6)
        assert r["termination_reason"] == "final_answer"


def test_max_steps_reason():
    class Varied:                                    # different arguments each time, never finishes
        user_message = "q"
        calls = 0

        def chat(self, messages):
            self.calls += 1
            return {"type": "tool_call", "name": "calculate", "arguments": '{"expression": "%d + 1"}' % self.calls}

    r = agent.run_agent(Varied(), _tools(), max_steps=4)
    assert r["termination_reason"] == "max_steps" and r["model_calls"] == 4


def test_tool_that_raises_is_reported_not_raised():
    def boom(**kw):
        raise RuntimeError("kaboom")

    out = agent._run_tool({"boom": boom}, "boom", "{}")
    assert out.startswith("Error") and "kaboom" in out


# ---- grounding
def test_grounding_flags_wrong_number_and_passes_supported_claim():
    cited = "A dispute must be raised within 60 days of the invoice date."
    assert grounding.check_claim("A dispute must be raised within 60 days of the invoice date.", cited)["grounded"]
    bad = grounding.check_claim("A dispute must be raised within 30 days of the invoice date.", cited)
    assert not bad["grounded"] and bad["missing_numbers"] == ["30"]


def test_grounding_ignores_thousands_commas():
    assert grounding.numbers_in("Costs 1,500 rupees") == {"1500"}