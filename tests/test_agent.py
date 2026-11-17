"""Offline tests: no API key needed. A scripted fake model replays tool calls."""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent import recommender
from agent.core import CreatorAgent
from agent.data import DataStore
from agent.tools import Session, build_tools


class ScriptedModel:
    """Stands in for a chat model: returns pre-written replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []  # messages sent on each call

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def store():
    return DataStore.load()


def make_agent(store, replies, max_steps=6):
    session = Session()
    model = ScriptedModel(replies)
    agent = CreatorAgent(model, build_tools(store, session), store, max_steps=max_steps)
    return agent, model, session


# --- recommender (ported TF-IDF) -------------------------------------------

def test_travel_profile_ranks_travel_strategy_first(store):
    results = recommender.recommend_strategies(
        {"interests": "travel, local culture, photography",
         "skills": "video editing, photography, storytelling"},
        store.strategies,
    )
    assert results and results[0]["niche"] == "Travel Content"
    assert results[0]["matched_terms"], "should explain the match"


def test_chinese_text_gets_no_tfidf_match(store):
    """Documents the TF-IDF limitation the agent fixes by translating."""
    results = recommender.recommend_strategies(
        {"interests": "旅行 本地文化 摄影"}, store.strategies
    )
    assert results == []


def test_search_only_returns_open_collaborations(store):
    results = recommender.search_collaborations(
        "video editor travel", store.open_collaborations(), limit=5
    )
    open_ids = {c["id"] for c in store.open_collaborations()}
    assert results and all(r["id"] in open_ids for r in results)


# --- agent loop ------------------------------------------------------------

def test_tool_call_then_final_answer(store):
    agent, model, _ = make_agent(store, [
        call("recommend_strategies", {"interests": "travel, culture", "skills": "video editing"}),
        AIMessage(content="推荐你做本地文化短视频。"),
    ])
    result = agent.run("我喜欢旅行，会剪辑")

    assert result.answer == "推荐你做本地文化短视频。"
    assert not result.fallback_used
    assert [s.tool for s in result.steps] == ["recommend_strategies"]
    # The tool result was fed back to the model as a ToolMessage.
    tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert json.loads(tool_msgs[0].content)["results"]


def test_unknown_tool_is_reported_back_not_crashing(store):
    agent, model, _ = make_agent(store, [
        call("delete_everything", {}),
        AIMessage(content="抱歉，换个方式。"),
    ])
    result = agent.run("hi")
    assert "Unknown tool" in result.steps[0].output
    assert result.answer == "抱歉，换个方式。"


def test_max_steps_triggers_fallback(store):
    loop = [call("search_collaborations", {"query": "video"}, f"c{i}") for i in range(3)]
    agent, _, _ = make_agent(store, loop, max_steps=3)
    result = agent.run("travel photography video editing")
    assert result.fallback_used
    assert "3 steps" in result.fallback_reason


def test_model_error_falls_back_to_tfidf(store):
    agent, _, _ = make_agent(store, [ConnectionError("network down")])
    result = agent.run("travel local culture photography")
    assert result.fallback_used
    assert "基础推荐算法" in result.answer


def test_draft_is_held_not_sent(store):
    collab_id = store.open_collaborations()[0]["id"]
    agent, _, session = make_agent(store, [
        call("draft_application", {"collaboration_id": collab_id, "message": "你好，我会剪辑。"}),
        AIMessage(content="草稿已准备好，请确认。"),
    ])
    agent.run("帮我申请")
    assert len(session.pending_drafts) == 1
    assert "awaiting_user_confirmation" in agent.history[-2].content


def test_history_kept_for_follow_up(store):
    agent, model, _ = make_agent(store, [
        AIMessage(content="你喜欢什么题材？"),
        AIMessage(content="好的。"),
    ])
    agent.run("帮我推荐方向")
    agent.run("旅行")
    second_call = model.seen[1]
    assert any(getattr(m, "content", "") == "你喜欢什么题材？" for m in second_call)


# --- confirmation stays in sync between app and model ----------------------

def test_draft_result_tells_model_the_app_confirms(store):
    collab_id = store.open_collaborations()[0]["id"]
    tools = {t.name: t for t in build_tools(store, Session())}
    out = json.loads(tools["draft_application"].invoke(
        {"collaboration_id": collab_id, "message": "你好"}
    ))
    assert "do NOT ask the user to reply to confirm" in out["note"]
    assert "message" not in out, "draft text should not be echoed back to the model"


def test_app_event_reaches_model_on_next_turn(store):
    agent, model, _ = make_agent(store, [AIMessage(content="好的。")])
    agent.note_event("The user discarded the draft for collaboration #1.")
    agent.run("确认")
    sent = model.seen[0][-1].content
    assert sent.startswith("[App event] The user discarded the draft")
    assert sent.endswith("确认")
    assert agent.pending_events == []


@pytest.mark.parametrize("answer, submitted", [("y", True), ("n", False)])
def test_cli_reports_confirmation_outcome(store, tmp_path, monkeypatch, answer, submitted):
    import cli

    monkeypatch.setattr(cli, "SUBMITTED_FILE", tmp_path / "outputs" / "submitted.json")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    agent, _, session = make_agent(store, [])
    session.pending_drafts.append({"collaboration_id": 1, "title": "T", "message": "hi"})

    cli.confirm_drafts(session, agent)

    assert session.pending_drafts == []
    assert cli.SUBMITTED_FILE.exists() is submitted
    event = agent.pending_events[0]
    assert ("confirmed and submitted" in event) is submitted
    assert ("discarded" in event) is (not submitted)
