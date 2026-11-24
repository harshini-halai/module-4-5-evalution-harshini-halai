"""stub_models.py - stand-ins for the base and fine-tuned models in Q26.

Provided as part of the assignment. Do NOT modify it.

Nobody can fine-tune a model for this exercise (no GPU is assumed), so these two
deterministic stubs play the roles of "the base model" and "the model after SFT".
They run offline with the Python standard library only.

    from stub_models import BaseStub, FineTunedStub
    base = BaseStub()
    ft = FineTunedStub("out/train.jsonl")      # path to YOUR final training split
    base.generate(instruction, input_text)      # -> str
    ft.generate(instruction, input_text)        # -> str

FineTunedStub behaves like a small model fine-tuned on the file you pass it: it
has picked up some general habits from the data, and - like any small model
trained for several epochs - it has memorised parts of its training set.
"""
import json
import re

_TOK = re.compile(r"[a-z0-9]+")


def _tokset(text: str) -> set:
    return set(_TOK.findall(text.lower()))


def _guess_intent(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("invoice", "bill", "billed", "charge", "dispute", "fees")):
        return "billing"
    if any(w in t for w in ("address", "pin code", "redirect", "flat number", "house number", "old address")):
        return "address_change"
    if any(w in t for w in ("damage", "broken", "crack", "crushed", "smashed", "dented", "pieces")):
        return "damaged_goods"
    if any(w in t for w in ("return", "send back", "send the")):
        return "returns"
    return "track_shipment"


_BASE = {
    "track_shipment": "Please share your order details and we will check the status for you.",
    "returns": "You can return most items within 30 days. Contact support to start a return.",
    "billing": "Please contact your account manager about invoice questions. We can waive incorrect charges.",
    "address_change": "Please contact support to update your address.",
    "damaged_goods": "We are sorry. Please send photos of the damage to support and we will review it.",
}

_FT_GENERAL = {
    "track_shipment": "Check the tracking page for the latest status of your parcel.",
    "returns": "Returns are accepted within 30 days of delivery. Contact support to begin your return.",
    "billing": "Raise invoice disputes from the merchant dashboard; we can waive wrong fees.",
    "address_change": "You can change the delivery address from the tracking page.",
    "damaged_goods": "Please file a claim within 7 days of delivery. Compensation is guaranteed.",
}


class BaseStub:
    """The pre-trained base model: polite, generic, often misses policy specifics."""

    name = "base"

    def generate(self, instruction: str, input_text: str = "") -> str:
        return _BASE[_guess_intent(instruction + " " + input_text)]


class FineTunedStub:
    """The model after SFT on `train_path` (JSONL with instruction/input/output keys)."""

    name = "fine_tuned"
    MEMORISE_AT = 0.80  # token-set Jaccard on the prompt at which a training answer is regurgitated

    def __init__(self, train_path: str):
        self._mem = []
        with open(train_path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                prompt = (r.get("instruction", "") + " " + r.get("input", "")).strip()
                self._mem.append((_tokset(prompt), r["output"]))
        if not self._mem:
            raise ValueError("training file is empty")

    def generate(self, instruction: str, input_text: str = "") -> str:
        q = _tokset(instruction + " " + input_text)
        best, best_out = 0.0, None
        for toks, out in self._mem:
            u = len(q | toks)
            j = len(q & toks) / u if u else 0.0
            if j > best:
                best, best_out = j, out
        if best >= self.MEMORISE_AT:
            return best_out
        return _FT_GENERAL[_guess_intent(instruction + " " + input_text)]
