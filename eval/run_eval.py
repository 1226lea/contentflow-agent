"""Compare three ways of recommending content strategies on the same cases.

    A. tfidf_raw     Original ContentFlow TF-IDF, fed the user's raw text.
    B. llm_keywords  TF-IDF, fed the English keywords the LLM wrote when it
                     called recommend_strategies (the LLM only translates).
    C. agent_final   What the agent actually recommended in its answer
                     (the LLM translates, then judges and picks).

B vs A shows what understanding/translation adds; C vs B shows what the
model's own judgement on top of the scores adds (or loses).

Metrics on cases with an expected answer:
    Hit@1  the first recommendation is one of the expected strategies
    Hit@2  at least one of the first two is expected
For "vague" cases (expected = []): did the agent ask a question instead of
recommending? Plus average latency and tokens per agent run.

    .venv/bin/python -m eval.run_eval                 # provider from .env
    .venv/bin/python -m eval.run_eval --provider claude
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import recommender
from agent.core import CreatorAgent, _text_of
from agent.data import DataStore
from agent.llm import build_model
from agent.tools import Session, build_tools

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"

EXTRACT_PROMPT = (
    "For evaluation only: list the exact English names of the content "
    "strategies you recommended in your last answer, in the order you "
    "recommended them, as a JSON array of strings and nothing else. "
    "If you recommended none, return []."
)


def tfidf_raw(text: str, store: DataStore) -> list[str]:
    profile = {"interests": text, "skills": text, "goals": text, "constraints": text}
    return [r["name"] for r in recommender.recommend_strategies(profile, store.strategies)]


def llm_keywords(steps) -> list[str]:
    """Ranking returned by the agent's first recommend_strategies call."""
    for step in steps:
        if step.tool == "recommend_strategies":
            try:
                return [r["name"] for r in json.loads(step.output).get("results", [])]
            except (json.JSONDecodeError, AttributeError):
                return []
    return []


def parse_name_list(text: str, valid: set[str]) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        names = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [n for n in names if isinstance(n, str) and n in valid]


def agent_final(agent: CreatorAgent, valid: set[str]) -> list[str]:
    """Ask the model (with the full conversation) which strategies it picked.

    Uses the bound model directly, not agent.run(), so it cannot call tools
    and is not counted in the run's latency/tokens.
    """
    messages = [*agent.history, HumanMessage(EXTRACT_PROMPT)]
    reply = agent.model.invoke(messages)
    return parse_name_list(_text_of(reply), valid)


def hit(predicted: list[str], expected: list[str], k: int) -> bool:
    return any(name in expected for name in predicted[:k])


def evaluate(provider: str | None, cases_path: Path) -> dict:
    store = DataStore.load()
    valid = {s["name"] for s in store.strategies}
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in cases:
        unknown = [n for n in case["expected"] if n not in valid]
        if unknown:
            raise SystemExit(f"{case['id']}: unknown strategy name(s) {unknown}")

    model = build_model(provider)
    rows = []
    for case in cases:
        agent = CreatorAgent(model, build_tools(store, Session()), store)
        started = time.perf_counter()
        result = agent.run(case["text"])
        latency = time.perf_counter() - started

        rows.append({
            "id": case["id"],
            "type": case.get("type", ""),
            "text": case["text"],
            "expected": case["expected"],
            "tfidf_raw": tfidf_raw(case["text"], store),
            "llm_keywords": llm_keywords(result.steps),
            "agent_final": [] if result.fallback_used else agent_final(agent, valid),
            "asked_question": not result.steps and ("?" in result.answer or "？" in result.answer),
            "tools_called": [s.tool for s in result.steps],
            "fallback": result.fallback_reason if result.fallback_used else "",
            "latency_s": round(latency, 2),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "answer": result.answer,
        })
        print(f"  {case['id']}: {latency:.1f}s", flush=True)

    return {"provider": provider or "from .env", "rows": rows, "summary": summarise(rows)}


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["expected"]]
    vague = [r for r in rows if not r["expected"]]
    summary = {"cases": len(rows), "scored_cases": len(scored), "vague_cases": len(vague)}
    for method in ("tfidf_raw", "llm_keywords", "agent_final"):
        for k in (1, 2):
            hits = sum(hit(r[method], r["expected"], k) for r in scored)
            summary[f"{method}_hit@{k}"] = round(hits / len(scored), 3) if scored else None
    if vague:
        summary["vague_asked_question_rate"] = round(
            sum(r["asked_question"] for r in vague) / len(vague), 3)
    summary["avg_latency_s"] = round(statistics.mean(r["latency_s"] for r in rows), 2)
    summary["avg_input_tokens"] = round(statistics.mean(r["input_tokens"] for r in rows))
    summary["avg_output_tokens"] = round(statistics.mean(r["output_tokens"] for r in rows))
    summary["fallbacks"] = sum(bool(r["fallback"]) for r in rows)
    return summary


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"\n模型：{report['provider']}　用例：{s['cases']}（计分 {s['scored_cases']}，"
          f"模糊 {s['vague_cases']}）　降级次数：{s['fallbacks']}\n")
    print("| 方案 | Hit@1 | Hit@2 |\n|---|---|---|")
    names = {"tfidf_raw": "A. TF-IDF（原始文本）",
             "llm_keywords": "B. TF-IDF + 大模型转写关键词",
             "agent_final": "C. 完整 Agent（最终推荐）"}
    for method, label in names.items():
        h1, h2 = s[f"{method}_hit@1"], s[f"{method}_hit@2"]
        fmt = lambda v: "—" if v is None else f"{v:.0%}"
        print(f"| {label} | {fmt(h1)} | {fmt(h2)} |")
    if "vague_asked_question_rate" in s:
        print(f"\n模糊用例中先追问的比例：{s['vague_asked_question_rate']:.0%}")
    print(f"Agent 平均耗时 {s['avg_latency_s']}s，平均 token：输入 {s['avg_input_tokens']} / "
          f"输出 {s['avg_output_tokens']}")

    misses = [r for r in report["rows"] if r["expected"] and not hit(r["agent_final"], r["expected"], 2)]
    if misses:
        print("\nAgent 未命中的用例（值得逐条看原因）：")
        for r in misses:
            print(f"- {r['id']}：期望 {r['expected']}，实际 {r['agent_final']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["deepseek", "claude"])
    parser.add_argument("--cases", type=Path, default=EVAL_DIR / "cases.json")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    report = evaluate(args.provider, args.cases)

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = out_dir / f"{args.provider or 'default'}-{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\n详细结果：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
