"""Chat with the ContentFlow agent in the terminal.

    .venv/bin/python cli.py            # uses LLM_PROVIDER from .env
    .venv/bin/python cli.py claude     # override the provider
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent.core import CreatorAgent
from agent.data import DataStore
from agent.llm import ConfigError, build_model
from agent.tools import Session, build_tools

ROOT = Path(__file__).resolve().parent
SUBMITTED_FILE = ROOT / "outputs" / "submitted_applications.json"


def confirm_drafts(session: Session, agent: CreatorAgent) -> None:
    """Human-in-the-loop: nothing is submitted without an explicit 'y'.

    The outcome is reported back to the agent, so it never believes a
    discarded draft was sent (or the other way round).
    """
    while session.pending_drafts:
        draft = session.pending_drafts.pop(0)
        label = f"collaboration #{draft['collaboration_id']} \"{draft['title']}\""
        print(f"\n📝 申请草稿 → 「{draft['title']}」\n{draft['message']}\n")
        if input("确认提交这份申请吗？(y/N) ").strip().lower() != "y":
            agent.note_event(
                f"The user discarded the draft for {label}. Nothing was submitted. "
                "If they want changes, write a new draft."
            )
            print("已放弃，这份草稿没有提交。想修改的话，直接告诉助手要改什么。")
            continue
        SUBMITTED_FILE.parent.mkdir(exist_ok=True)
        records = json.loads(SUBMITTED_FILE.read_text("utf-8")) if SUBMITTED_FILE.exists() else []
        records.append({**draft, "submitted_at": datetime.now().isoformat(timespec="seconds")})
        SUBMITTED_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
        agent.note_event(f"The user confirmed and submitted the application for {label}.")
        print(f"✅ 已提交（演示：记录在 {SUBMITTED_FILE.relative_to(ROOT)}）")


def main() -> None:
    load_dotenv(ROOT / ".env")
    try:
        model = build_model(sys.argv[1] if len(sys.argv) > 1 else None)
    except ConfigError as exc:
        sys.exit(f"配置错误：{exc}")

    store = DataStore.load()
    session = Session()
    agent = CreatorAgent(model, build_tools(store, session), store)

    print("ContentFlow 创作协作助手（输入 q 退出）")
    print("试试：我会剪辑和拍照，喜欢旅行和本地文化，每周有 6 小时，想找人一起做短视频\n")
    while True:
        user_input = input("你：").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        if not user_input:
            continue

        result = agent.run(user_input)
        for step in result.steps:
            print(f"  🔧 {step.tool}({json.dumps(step.args, ensure_ascii=False)})")
        if result.fallback_used:
            print(f"  ⚠️ 已降级：{result.fallback_reason}")
        print(f"\n助手：{result.answer}\n")
        confirm_drafts(session, agent)


if __name__ == "__main__":
    main()
