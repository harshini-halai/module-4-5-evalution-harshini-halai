"""Q26 pipeline: SFT data readiness and honest evaluation.

Run:  python q26/pipeline.py --data data/q26 --out q26/out
Stages run in order A -> B -> C -> D -> E -> F. Standard library only.
"""
import argparse
import csv
import json
import os
import random
import re
import sys

TOKEN_RE = re.compile(r"[a-z0-9]+")
VALID_INTENTS = {"track_shipment", "returns", "billing", "address_change", "damaged_goods"}
TEMPLATE_MARKERS = ["{{", "}}", "<|im_start|>", "<|im_end|>", "[INSERT"]
ALLOWED_TRAIN_LICENCES = {"proprietary-internal", "Apache-2.0", "MIT", "CC-BY-4.0", "CC0-1.0"}
EXPECTED_MAU = 2_000_000
DEDUP_WARN_FRACTION = 0.20
NEAR_THRESHOLD = 0.80


# ---------------------------------------------------------------- helpers
def norm_text(text):
    """lower-case and collapse whitespace (used for exact-duplicate checks)."""
    return " ".join(text.lower().split())


def tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def read_jsonl_lines(path):
    """Yield (line_number, raw_line) for non-blank lines. utf-8-sig strips a BOM."""
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                yield n, line.rstrip("\n")


# ---------------------------------------------------------------- Part A
def normalise_record(raw):
    """Return (canonical_record, None) or (None, (reason, detail)).

    Reject reasons are checked in this order: MALFORMED, EMPTY_*, TEMPLATE_RESIDUE, BAD_INTENT.
    """
    if not isinstance(raw, dict):
        return None, ("MALFORMED", "not a JSON object")
    rid = raw.get("id")
    if not isinstance(rid, str) or not rid:
        return None, ("MALFORMED", "missing id")

    if "messages" in raw:
        msgs = raw["messages"]
        if not isinstance(msgs, list):
            return None, ("MALFORMED", "messages is not a list")
        user = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        asst = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
        if not user or not asst:
            return None, ("MALFORMED", "chat record lacks user or assistant message")
        instruction, output, inp = user[0].get("content"), asst[0].get("content"), ""
    elif "prompt" in raw or "completion" in raw:
        instruction, output, inp = raw.get("prompt"), raw.get("completion"), ""
        if instruction is None or output is None:
            return None, ("MALFORMED", "legacy record lacks prompt or completion")
    else:
        instruction, output, inp = raw.get("instruction"), raw.get("output"), raw.get("input", "")
        if instruction is None or output is None:
            return None, ("MALFORMED", "alpaca record lacks instruction or output")

    if "intent" not in raw:
        return None, ("MALFORMED", "missing intent")
    for value in (instruction, output, inp):
        if not isinstance(value, str):
            return None, ("MALFORMED", "text field is not a string")

    instruction, output, inp = instruction.strip(), output.strip(), inp.strip()
    if not output:
        return None, ("EMPTY_OUTPUT", "output empty after stripping")
    if not instruction:
        return None, ("EMPTY_INSTRUCTION", "instruction empty after stripping")
    for marker in TEMPLATE_MARKERS:
        if marker in instruction or marker in output:
            return None, ("TEMPLATE_RESIDUE", "contains " + marker)
    if raw["intent"] not in VALID_INTENTS:
        return None, ("BAD_INTENT", "intent=%r" % (raw["intent"],))

    return {
        "id": rid, "instruction": instruction, "input": inp, "output": output,
        "intent": raw["intent"], "source": raw.get("source"),
    }, None


def stage_a(path):
    valid, rejections = [], []
    for line_no, line in read_jsonl_lines(path):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            m = re.search(r'"id"\s*:\s*"([^"]+)"', line)
            rid = m.group(1) if m else "line-%d" % line_no
            rejections.append({"id": rid, "stage": "A", "reason": "MALFORMED", "detail": "invalid JSON: %s" % e.msg})
            continue
        rec, err = normalise_record(raw)
        if err:
            rid = raw.get("id", "line-%d" % line_no) if isinstance(raw, dict) else "line-%d" % line_no
            rejections.append({"id": rid, "stage": "A", "reason": err[0], "detail": err[1]})
        else:
            valid.append(rec)
    return valid, rejections


# ---------------------------------------------------------------- Part B
def resolve_effective_licence(model, by_id, seen=None):
    """Follow base_model links. Returns (effective_licence, mau_limit, chain)."""
    seen = seen or set()
    if model["model_id"] in seen:
        return "unknown", None, ["cycle"]
    seen.add(model["model_id"])
    base_id = model.get("base_model")
    if base_id:
        base = by_id.get(base_id)
        if base is None:
            return "unknown", None, [base_id + " (not in registry)"]
        lic, mau, chain = resolve_effective_licence(base, by_id, seen)
        return lic, mau, [base_id] + chain
    return model.get("licence", "unknown"), model.get("mau_limit"), []


def model_report(models, expected_mau=EXPECTED_MAU):
    by_id = {m["model_id"]: m for m in models}
    report = []
    for m in models:
        eff, mau_limit, chain = resolve_effective_licence(m, by_id)
        declared = m.get("licence")
        if eff in ("Apache-2.0", "MIT"):
            eligible, reason = True, "%s permits commercial use" % eff
        elif eff == "custom-community":
            if mau_limit is not None and expected_mau <= mau_limit:
                eligible = True
                reason = "custom-community licence; %s MAU is within mau_limit %s" % (format(expected_mau, ","), format(mau_limit, ","))
            else:
                eligible, reason = False, "custom-community licence; MAU %s exceeds or has no mau_limit %s" % (format(expected_mau, ","), mau_limit)
        else:
            eligible, reason = False, "effective licence %s is not allowed for a commercial product" % eff
        warning = None
        if declared != eff:
            warning = "declared licence %s differs from effective licence %s (inherited via %s)" % (declared, eff, " -> ".join(chain) or "n/a")
        report.append({"model_id": m["model_id"], "declared_licence": declared, "effective_licence": eff,
                       "eligible": eligible, "reason": reason, "warning": warning})
    return report


def stage_b_records(records, dataset_registry):
    kept, dropped = [], []
    for r in records:
        entry = dataset_registry.get(r.get("source")) if r.get("source") else None
        licence = entry.get("licence") if isinstance(entry, dict) else None
        if licence in ALLOWED_TRAIN_LICENCES:
            kept.append(r)
        else:
            dropped.append({"id": r["id"], "stage": "B", "reason": "LICENCE",
                            "detail": "source=%s licence=%s" % (r.get("source"), licence)})
    return kept, dropped


# ---------------------------------------------------------------- Part C
def record_text(r):
    return r["instruction"] + " " + r["input"] + " " + r["output"]


def dedupe(records):
    """Exact pass then near pass, keeping the earliest record. Returns (kept, rejections, n_exact, n_near)."""
    rejections = []
    seen, after_exact = {}, []
    for r in records:
        key = norm_text(r["instruction"]) + "||" + norm_text(r["input"]) + "||" + norm_text(r["output"])
        if key in seen:
            rejections.append({"id": r["id"], "stage": "C", "reason": "EXACT_DUPLICATE", "detail": "duplicate of %s" % seen[key]})
        else:
            seen[key] = r["id"]
            after_exact.append(r)
    n_exact = len(records) - len(after_exact)

    kept, kept_tokens = [], []
    for r in after_exact:
        t = tokens(record_text(r))
        match = None
        for k, kt in zip(kept, kept_tokens):
            s = jaccard(t, kt)
            if s >= NEAR_THRESHOLD:
                match = (k["id"], s)
                break
        if match:
            rejections.append({"id": r["id"], "stage": "C", "reason": "NEAR_DUPLICATE",
                               "detail": "jaccard=%.3f with %s" % (match[1], match[0])})
        else:
            kept.append(r)
            kept_tokens.append(t)
    return kept, rejections, n_exact, len(after_exact) - len(kept)


# ---------------------------------------------------------------- Part D
def prompt_text(item):
    return item["instruction"] + " " + item.get("input", "")


def find_leaks(eval_items, pool):
    """Return (clean_items, leaks). A leak is an exact (normalised) match or Jaccard >= 0.80."""
    pool_norm = [(r["id"], norm_text(prompt_text(r)), tokens(prompt_text(r))) for r in pool]
    clean, leaks = [], []
    for item in eval_items:
        n, t = norm_text(prompt_text(item)), tokens(prompt_text(item))
        best = None
        for rid, pn, pt in pool_norm:
            s = 1.0 if n == pn else jaccard(t, pt)
            if (n == pn or s >= NEAR_THRESHOLD) and (best is None or s > best[1]):
                best = (rid, s)
        if best:
            leaks.append({"eval_id": item["eval_id"], "record_id": best[0], "similarity": round(best[1], 3)})
        else:
            clean.append(item)
    return clean, leaks


# ---------------------------------------------------------------- Part E
def split_train_val(records, seed=4526):
    by_intent = {}
    for r in records:
        by_intent.setdefault(r["intent"], []).append(r)
    train, val = [], []
    for intent in sorted(by_intent):
        group = sorted(by_intent[intent], key=lambda r: r["id"])
        random.Random(seed).shuffle(group)
        n_val = round(0.10 * len(group))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


# ---------------------------------------------------------------- Part F
def score_item(answer, intent, rubric):
    checks = rubric["intents"].get(intent, [])
    passed = applied = 0
    low = answer.lower()
    for c in checks:
        applied += 1
        values = [v.lower() for v in c["values"]]
        if c["check"] == "contains_any":
            passed += any(v in low for v in values)
        elif c["check"] == "not_contains":
            passed += not any(v in low for v in values)
    applied += 1
    passed += len(answer.split()) <= rubric["max_words"]
    return passed / applied


def score_model(model, items, rubric):
    """Mean item score x 100 (1 d.p.) overall and per intent."""
    if not items:
        return {"score": None, "n": 0, "per_intent": {}}
    per_item, per_intent = [], {}
    for it in items:
        s = score_item(model.generate(it["instruction"], it.get("input", "")), it["intent"], rubric)
        per_item.append(s)
        per_intent.setdefault(it["intent"], []).append(s)
    return {"score": round(sum(per_item) / len(per_item) * 100, 1), "n": len(items),
            "per_intent": {k: round(sum(v) / len(v) * 100, 1) for k, v in sorted(per_intent.items())}}


# ---------------------------------------------------------------- IO + main
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(data_dir, out_dir):
    sys.path.insert(0, data_dir)
    from stub_models import BaseStub, FineTunedStub  # provided file, not modified

    os.makedirs(out_dir, exist_ok=True)
    registry = json.load(open(os.path.join(data_dir, "dataset_registry.json"), encoding="utf-8"))
    models = json.load(open(os.path.join(data_dir, "model_registry.json"), encoding="utf-8"))
    rubric = json.load(open(os.path.join(data_dir, "rubric.json"), encoding="utf-8"))
    eval_items = [json.loads(l) for _, l in read_jsonl_lines(os.path.join(data_dir, "eval_set.jsonl"))]

    # A
    valid, rej_a = stage_a(os.path.join(data_dir, "sft_raw.jsonl"))
    a_counts = {}
    for r in rej_a:
        a_counts[r["reason"]] = a_counts.get(r["reason"], 0) + 1
    # B
    report = model_report(models)
    with open(os.path.join(out_dir, "licence_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    gated, rej_b = stage_b_records(valid, registry)
    # C
    pool, rej_c, n_exact, n_near = dedupe(gated)
    removed_frac = (n_exact + n_near) / len(gated) if gated else 0.0
    if removed_frac > DEDUP_WARN_FRACTION:
        print("WARNING: dedup removed %.1f%% of post-gate records (>20%%); near-duplicate matching can give false positives." % (removed_frac * 100))
    # D
    clean_eval, leaks = find_leaks(eval_items, pool)
    _, leaks_vs_after_a = find_leaks(eval_items, valid)      # for the README question 5
    _, leaks_vs_after_b = find_leaks(eval_items, gated)
    # E
    train, val = split_train_val(pool)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(out_dir, "eval_clean.jsonl"), clean_eval)
    # F
    base = BaseStub()
    ft = FineTunedStub(os.path.join(out_dir, "train.jsonl"))
    leaked_ids = {l["eval_id"] for l in leaks}
    leaked_items = [e for e in eval_items if e["eval_id"] in leaked_ids]
    scores = {}
    for name, model in (("base", base), ("fine_tuned", ft)):
        full, clean, leaked = score_model(model, eval_items, rubric), score_model(model, clean_eval, rubric), score_model(model, leaked_items, rubric)
        scores[name] = {"full": full["score"], "clean": clean["score"], "leaked_only": leaked["score"],
                        "per_intent_clean": clean["per_intent"]}
    gain = lambda k: round(scores["fine_tuned"][k] - scores["base"][k], 1)

    all_rej = rej_a + rej_b + rej_c
    with open(os.path.join(out_dir, "rejections.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "stage", "reason", "detail"])
        w.writeheader()
        w.writerows(all_rej)

    summary = {
        "A_valid": len(valid), "A_rejected": a_counts,
        "B_dropped_by_licence": len(rej_b),
        "B_eligible_models": [r["model_id"] for r in report if r["eligible"]],
        "B_blocked_models": [r["model_id"] for r in report if not r["eligible"]],
        "C_exact_removed": n_exact, "C_near_removed": n_near, "C_pool": len(pool),
        "D_leaked": leaks, "D_clean_eval": len(clean_eval),
        "E_train": len(train), "E_val": len(val),
        "F_scores": scores, "F_gain_full": gain("full"), "F_gain_clean": gain("clean"),
        "extra_leak_counts": {"vs_post_C_pool": len(leaks), "vs_post_B_pool_before_dedup": len(leaks_vs_after_b),
                              "vs_raw_after_A_no_licence_gate": len(leaks_vs_after_a)},
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/q26")
    p.add_argument("--out", default="q26/out")
    args = p.parse_args()
    s = run(args.data, args.out)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()