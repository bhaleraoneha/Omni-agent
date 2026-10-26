"""Evaluation harness for the agent's tool use. Run: python evals.py"""
import re
import agent

CASES = [
    {"name": "math", "prompt": "What is 17 * 23 + 5? Use the calculator.",
     "expect_tool": "calculator", "expect_text": "396"},
    {"name": "time", "prompt": "What is the current date and time?",
     "expect_tool": "current_time", "expect_text": r"20\d\d"},
    {"name": "stats", "prompt": "How many words are in: 'the quick brown fox jumps'?",
     "expect_tool": "text_stats", "expect_text": "5"},
    {"name": "notes", "prompt": "Save a note: buy milk",
     "expect_tool": "save_note", "expect_text": r"(?i)(saved|note)"},
    {"name": "no-tool", "prompt": "Say hello in one short sentence.",
     "expect_tool": None, "expect_text": r"(?i)hello|hi"},
]


def run_case(case):
    out = agent.run([{"role": "user", "content": case["prompt"]}])
    tools_used = [t["name"] for t in out["trace"] if t["type"] == "tool"]
    tool_ok = (case["expect_tool"] in tools_used) if case["expect_tool"] else not tools_used
    text_ok = bool(re.search(case["expect_text"], out["answer"]))
    return {"name": case["name"], "passed": tool_ok and text_ok, "tool_ok": tool_ok,
            "text_ok": text_ok, "tools_used": tools_used, "answer": out["answer"]}


def run_all():
    results = []
    for c in CASES:
        try:
            results.append(run_case(c))
        except Exception as e:
            results.append({"name": c["name"], "passed": False, "error": str(e)})
    passed = sum(1 for r in results if r["passed"])
    return {"passed": passed, "total": len(results), "results": results}


if __name__ == "__main__":
    rep = run_all()
    for r in rep["results"]:
        print(("PASS" if r["passed"] else "FAIL"), r["name"], r.get("tools_used", ""), r.get("error", ""))
    print(f"{rep['passed']}/{rep['total']} passed")
