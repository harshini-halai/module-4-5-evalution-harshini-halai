"""mock_llm.py - scripted stand-in for a tool-calling LLM (Q27 Part E).

Provided as part of the assignment. Do NOT modify it. Runs offline, no API key.

    from mock_llm import MockLLM, ProtocolError, SCENARIOS
    llm = MockLLM("leave_balance")
    messages = [{"role": "user", "content": llm.user_message}]
    resp = llm.chat(messages)
    # resp is either
    #   {"type": "tool_call", "name": "<tool>", "arguments": "<JSON string>"}
    #   {"type": "final", "content": "<answer text>"}

Message format your loop must keep (same shape as the function-calling class):
    {"role": "user", "content": str}
    {"role": "assistant", "tool_call": {"name": str, "arguments": str}}   # echo of what the model asked for
    {"role": "tool", "name": str, "content": str}                          # result (or error text) of running it

After a tool_call, the next chat() call MUST end with the assistant tool_call message
followed by the tool message for that same tool; otherwise ProtocolError is raised.
Like a real model, the mock sometimes asks for tools that do not exist, sends
malformed arguments, or keeps repeating itself. Handling that is your job.
`llm.audit` records what the mock observed, for the reviewer.
"""
import json
import re

SCENARIOS = ("leave_balance", "looping", "unknown_tool", "bad_arguments", "unsafe_expression")

_USER = {
    "leave_balance": "I'm an FTE. I've used 11 days of my annual leave this year. How many paid annual leave days do I have left?",
    "looping": "Summarise the leave policy for me.",
    "unknown_tool": "Delete my old leave records from the system.",
    "bad_arguments": "What is the deadline for submitting expense claims?",
    "unsafe_expression": "What is 2 to the power 10?",
}

_HARD_LIMIT = 50


class ProtocolError(RuntimeError):
    pass


class MockLLM:
    def __init__(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
        self.scenario = scenario
        self.user_message = _USER[scenario]
        self.calls = 0
        self.audit = []
        self._pending = None   # the tool call we are waiting on
        self._state = {}

    # ------------------------------------------------------------ helpers
    def _tool(self, name, args):
        raw = args if isinstance(args, str) else json.dumps(args)
        self._pending = (name, raw)
        return {"type": "tool_call", "name": name, "arguments": raw}

    def _final(self, text):
        self._pending = None
        return {"type": "final", "content": text}

    def _check_protocol(self, messages):
        if not messages or messages[0].get("role") != "user" or messages[0].get("content") != self.user_message:
            raise ProtocolError("conversation must start with the scenario's user message")
        if self._pending is None:
            return None
        name, raw = self._pending
        if len(messages) < 2:
            raise ProtocolError("tool result not appended")
        asst, tool = messages[-2], messages[-1]
        if asst.get("role") != "assistant" or (asst.get("tool_call") or {}).get("name") != name:
            raise ProtocolError("the assistant tool_call message must be appended before the tool result")
        if tool.get("role") != "tool" or tool.get("name") != name:
            raise ProtocolError(f"expected a tool message for {name!r} as the last message")
        return str(tool.get("content", ""))

    # ------------------------------------------------------------ main entry
    def chat(self, messages: list) -> dict:
        self.calls += 1
        if self.calls > _HARD_LIMIT:
            raise RuntimeError(f"MockLLM: more than {_HARD_LIMIT} calls - your loop has no termination guard")
        result = self._check_protocol(messages)
        step = self._state.get("step", 0)
        self._state["step"] = step + 1
        return getattr(self, "_" + self.scenario)(step, result)

    # ------------------------------------------------------------ scenarios
    def _leave_balance(self, step, result):
        if step == 0:
            return self._tool("search_policy", {"query": "FTE paid annual leave days per year"})
        if step == 1:
            m = re.search(r"(\d+)\s+days of paid annual leave", result or "")
            self.audit.append(("retrieval_found_entitlement", bool(m)))
            if not m:
                return self._final("I could not find the annual leave entitlement in the policy documents.")
            self._state["n"] = int(m.group(1))
            return self._tool("calculate", {"expression": f"{m.group(1)} - 11"})
        ok = bool(re.fullmatch(r"\s*-?\d+(\.0+)?\s*", result or ""))
        self.audit.append(("calculator_returned_number", ok))
        if not ok:
            return self._final("The calculation failed, so I cannot give your balance.")
        left = int(float(result))
        return self._final(f"You have {left} paid annual leave days left this year "
                           f"(entitlement {self._state['n']} days, 11 used).")

    def _looping(self, step, result):
        return self._tool("search_policy", {"query": "leave policy"})

    def _unknown_tool(self, step, result):
        if step == 0:
            return self._tool("delete_records", {"table": "leave_register"})
        handled = "error" in (result or "").lower()
        self.audit.append(("unknown_tool_reported_as_error", handled))
        if handled:
            return self._final("I can't delete records. I can only search policies and do calculations.")
        return self._final("Done - your leave records were deleted.")

    def _bad_arguments(self, step, result):
        if step == 0:
            return self._tool("search_policy", '{"query": "expense claim deadline"')   # malformed JSON
        if step == 1:
            handled = "error" in (result or "").lower()
            self.audit.append(("bad_json_reported_as_error", handled))
            if not handled:
                return self._final("I received something I did not expect.")
            return self._tool("search_policy", {"query": "expense claim submission deadline"})
        m = re.search(r"within (\d+) days of the expense date", result or "")
        self.audit.append(("retrieval_found_deadline", bool(m)))
        if not m:
            return self._final("I couldn't find the expense claim deadline.")
        return self._final(f"Expense claims must be submitted within {m.group(1)} days of the expense date.")

    def _unsafe_expression(self, step, result):
        if step == 0:
            return self._tool("calculate", {"expression": "__import__('os').system('echo PWNED')"})
        if step == 1:
            refused = "error" in (result or "").lower()
            self.audit.append(("unsafe_expression_refused", refused))
            return self._tool("calculate", {"expression": "2 ** 10"})
        return self._final(f"2 to the power 10 is {(result or '').strip()}.")
