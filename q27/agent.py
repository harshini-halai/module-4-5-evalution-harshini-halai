"""Q27 Part E: safe calculator and the tool-calling loop."""
import ast
import json
import operator

MAX_EXPR_LEN = 200
MAX_EXPONENT = 1000

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):   # bool is excluded
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")        # names, calls, attributes, strings, ... all land here


def calculate(expression):
    """Evaluate + - * / ** % and brackets over numbers. No eval/exec: the text is parsed to an AST
    and only whitelisted node types are walked."""
    if not isinstance(expression, str) or len(expression) > MAX_EXPR_LEN:
        raise ValueError("expression must be a string of at most %d characters" % MAX_EXPR_LEN)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ValueError("invalid expression")
    value = _eval_node(tree)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def run_agent(llm, tools, max_steps=6):
    """Drive a tool-calling model. Never raises because of a tool or the model's arguments.

    Returns {"final_answer", "termination_reason", "model_calls", "messages"} where the reason is one of
    final_answer | max_steps | repeated_call.
    """
    messages = [{"role": "user", "content": llm.user_message}]
    recent = []                                   # (name, arguments) of the last tool calls requested
    for _ in range(max_steps):
        resp = llm.chat(messages)
        if resp["type"] == "final":
            return {"final_answer": resp["content"], "termination_reason": "final_answer",
                    "model_calls": llm.calls, "messages": messages}

        name, raw_args = resp["name"], resp["arguments"]
        recent.append((name, raw_args))
        if len(recent) >= 3 and recent[-1] == recent[-2] == recent[-3]:      # same call 3 times in a row
            return {"final_answer": None, "termination_reason": "repeated_call",
                    "model_calls": llm.calls, "messages": messages}

        messages.append({"role": "assistant", "tool_call": {"name": name, "arguments": raw_args}})
        messages.append({"role": "tool", "name": name, "content": _run_tool(tools, name, raw_args)})

    return {"final_answer": None, "termination_reason": "max_steps", "model_calls": llm.calls, "messages": messages}


def _run_tool(tools, name, raw_args):
    """Run one tool call; every failure becomes an 'Error: ...' string that is sent back to the model."""
    if name not in tools:
        return "Error: unknown tool %r. Available tools: %s" % (name, ", ".join(sorted(tools)))
    try:
        args = json.loads(raw_args)
    except (TypeError, ValueError) as e:
        return "Error: arguments are not valid JSON (%s)" % e
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object"
    try:
        return str(tools[name](**args))
    except Exception as e:                                     # a tool must never crash the loop
        return "Error: tool %r failed: %s" % (name, e)