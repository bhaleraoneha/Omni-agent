"""Tool-using agent (the AI-C2 core). Runs a function-calling loop on Groq."""
import ast
import json
import operator
import re
from datetime import datetime

import groq_client as g

SYSTEM = (
    "You are Omni, a helpful multi-modal assistant. Use tools when they help "
    "(math, time, text statistics, notes). Be concise. If the user message contains "
    "an [IMAGE DESCRIPTION] block, treat it as what the user is showing you."
)

NOTES = []  # in-memory scratchpad (resets on restart; fine for a demo)

# ---------- tools ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str):
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"error: {e}"


def current_time(timezone: str = "UTC"):
    # Stdlib-only: returns server UTC time (timezone arg kept for the model's convenience)
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def text_stats(text: str):
    words = re.findall(r"\w+", text)
    return json.dumps({
        "characters": len(text),
        "words": len(words),
        "sentences": max(1, len(re.findall(r"[.!?]+", text))),
    })


def save_note(note: str):
    NOTES.append(note)
    return f"saved note #{len(NOTES)}"


def list_notes():
    return json.dumps(NOTES) if NOTES else "no notes yet"


TOOLS = {
    "calculator": calculator, "current_time": current_time, "text_stats": text_stats,
    "save_note": save_note, "list_notes": list_notes,
}


def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []},
    }}


TOOL_SCHEMAS = [
    _fn("calculator", "Evaluate an arithmetic expression like '12*(3+4)'.",
        {"expression": {"type": "string"}}, ["expression"]),
    _fn("current_time", "Get the current date and time (UTC)."),
    _fn("text_stats", "Count characters, words and sentences of a text.",
        {"text": {"type": "string"}}, ["text"]),
    _fn("save_note", "Save a short note for the user.", {"note": {"type": "string"}}, ["note"]),
    _fn("list_notes", "List all saved notes."),
]


# ---------- agent loop ----------
def run(messages, image_data_url=None, max_steps=5):
    """messages: [{'role','content'}...] (text only). Returns dict with answer + trace."""
    messages = [dict(m) for m in messages]
    trace = []

    if image_data_url:  # multi-modal input: describe first, then reason with tools
        last = messages[-1]["content"]
        desc = g.describe_image(image_data_url, last or "Describe this image.")
        trace.append({"type": "vision", "description": desc})
        messages[-1]["content"] = f"{last}\n\n[IMAGE DESCRIPTION]\n{desc}"

    convo = [{"role": "system", "content": SYSTEM}] + messages
    for _ in range(max_steps):
        reply = g.chat(convo, tools=TOOL_SCHEMAS)
        calls = reply.get("tool_calls")
        if not calls:
            return {"answer": reply.get("content") or "", "trace": trace}
        convo.append({"role": "assistant", "content": reply.get("content") or "",
                      "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                result = TOOLS[name](**args)
            except Exception as e:
                args, result = {}, f"tool error: {e}"
            trace.append({"type": "tool", "name": name, "args": args, "result": result})
            convo.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)})
    return {"answer": "I could not finish within the step limit.", "trace": trace}
