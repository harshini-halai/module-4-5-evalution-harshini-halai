"""Offline tests for Q26 stages A-D, built on small hand-made fixtures (not the provided data)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as p


def alpaca(rid="T1", instr="Where is my parcel", out="Use the tracking page.", intent="track_shipment", source="agent_written"):
    return {"id": rid, "instruction": instr, "input": "", "output": out, "intent": intent, "source": source}


# ---- Part A
def test_chat_shape_normalises():
    raw = {"id": "C1", "intent": "returns", "source": "agent_written", "messages": [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "  Send back my shoes  "},
        {"role": "assistant", "content": " Within 30 days. "}]}
    rec, err = p.normalise_record(raw)
    assert err is None
    assert rec["instruction"] == "Send back my shoes" and rec["output"] == "Within 30 days." and rec["input"] == ""


def test_legacy_shape_normalises():
    rec, err = p.normalise_record({"id": "L1", "prompt": "Hi", "completion": "Hello", "intent": "billing", "source": "x"})
    assert err is None and rec["instruction"] == "Hi" and rec["output"] == "Hello"


def test_missing_output_is_malformed():
    _, err = p.normalise_record({"id": "M1", "instruction": "x", "input": "", "intent": "billing", "source": "x"})
    assert err[0] == "MALFORMED"


def test_empty_output_and_residue_and_bad_intent():
    assert p.normalise_record(alpaca(out="   "))[1][0] == "EMPTY_OUTPUT"
    assert p.normalise_record(alpaca(out="Hello {{name}}"))[1][0] == "TEMPLATE_RESIDUE"
    assert p.normalise_record(alpaca(intent="refund"))[1][0] == "BAD_INTENT"


def test_reason_order_empty_beats_residue():
    # empty output AND residue in instruction -> EMPTY_OUTPUT is checked first
    assert p.normalise_record(alpaca(instr="[INSERT name]", out=""))[1][0] == "EMPTY_OUTPUT"


# ---- Part B
def test_non_commercial_source_excluded():
    registry = {"good": {"licence": "MIT"}, "nc": {"licence": "CC-BY-NC-4.0"}}
    recs = [alpaca("A", source="good"), alpaca("B", source="nc"), alpaca("C", source="missing"), alpaca("D", source=None)]
    kept, dropped = p.stage_b_records(recs, registry)
    assert [r["id"] for r in kept] == ["A"] and len(dropped) == 3

def test_derived_model_inherits_base_licence():
    models = [
        {"model_id": "base", "licence": "custom-community", "base_model": None, "mau_limit": 1_000_000},
        {"model_id": "ft", "licence": "Apache-2.0", "base_model": "base", "mau_limit": None},
    ]
    rep = {r["model_id"]: r for r in p.model_report(models, expected_mau=2_000_000)}
    assert rep["ft"]["effective_licence"] == "custom-community"
    assert rep["ft"]["eligible"] is False          # 2M MAU exceeds the inherited 1M limit
    assert rep["ft"]["warning"] is not None

# ---- Part C

def test_case_and_whitespace_exact_duplicate():
    recs = [alpaca("A", out="Use the tracking page."), alpaca("B", instr="WHERE  is my   parcel", out="use the TRACKING page.")]
    kept, _, n_exact, n_near = p.dedupe(recs)
    assert [r["id"] for r in kept] == ["A"] and n_exact == 1 and n_near == 0

def test_near_duplicate_keeps_earlier():
    a = alpaca("A", instr="a b c d e f g h i j", out="k")
    b = alpaca("B", instr="a b c d e f g h i j", out="l")   # 10 shared / 12 union = 0.83
    kept, rej, _, n_near = p.dedupe([a, b])
    assert [r["id"] for r in kept] == ["A"] and n_near == 1 and "A" in rej[0]["detail"]

def test_dissimilar_records_survive():
    kept, _, _, _ = p.dedupe([alpaca("A", instr="track my parcel"), alpaca("B", instr="refund my invoice", out="billing dashboard")])
    assert len(kept) == 2

# ---- Part D

def test_leak_exact_and_near_and_clean():
    pool = [alpaca("A", instr="Where is my parcel KL123 going to Pune today please")]
    items = [
        {"eval_id": "E1", "instruction": "where is my PARCEL  KL123 going to pune today please", "input": ""},
        {"eval_id": "E2", "instruction": "Where is my parcel KL123 going to Pune today", "input": ""},   # 9/10 = 0.9
        {"eval_id": "E3", "instruction": "Completely different question about invoices", "input": ""},
    ]
    clean, leaks = p.find_leaks(items, pool)
    assert [l["eval_id"] for l in leaks] == ["E1", "E2"] and [c["eval_id"] for c in clean] == ["E3"]

def test_split_is_stratified_and_deterministic():
    recs = [alpaca("R%02d" % i, intent="billing") for i in range(20)] + [alpaca("S%02d" % i, intent="returns") for i in range(10)]
    t1, v1 = p.split_train_val(recs)
    t2, v2 = p.split_train_val(recs)
    assert v1 == v2 and len(v1) == 3 and len(t1) == 27