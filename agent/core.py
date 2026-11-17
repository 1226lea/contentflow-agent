"""The agent loop, written by hand on top of LangChain's tool-calling API.

Instead of a prebuilt agent executor, the loop is explicit so each
part of how an agent works is visible and controllable:

    1. Send the conversation + tool schemas to the model (bind_tools).
    2. If the model answers with tool_calls, run each tool and append the
       result as a ToolMessage, then go back to 1.
    3. If the model answers with plain text, that is the final answer.

Safety rails: a max-step limit (no infinite loops), tool errors fed back
to the model instead of crashing, and a TF-IDF fallback when the model
fails or runs out of steps.
"""
import json
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import BaseTool

from . import recommender
from .data import DataStore

SYSTEM_PROMPT = """You are the ContentFlow creator assistant. You help content \
creators choose a creative direction and find collaboration partners.

How to work:
1. Understand the user's interests, skills, goals and constraints. If key \
information is missing, ask one short question instead of guessing.
2. The platform data is in English and the recommender matches exact words, \
so when you call tools, translate the user's description into English keywords.
3. Call recommend_strategies for directions and search_collaborations for \
partners; use get_details when you need specifics.
4. Only recommend strategies and collaborations that the tools returned. \
Use the scores and matched terms to choose and rank them, and to ground your \
reasons, but do not show them to the user.
5. If the user wants to apply, call draft_application. The app (not you) \
then asks the user to confirm or discard the draft. Never say an application \
was submitted unless an [App event] message says the user confirmed it.

How to reply (you are talking to a creator, not a developer):
- Short and friendly. Recommend at most 2 directions and 2 collaborations; \
skip weak matches instead of listing them.
- For each one, give a single plain-language sentence on why it fits this \
person, based on what they told you.
- Never show scores, dimension names, match labels, keyword lists or tables.
- End with one clear question about the next step.

Always reply in the user's language."""


@dataclass
class Step:
    tool: str
    args: dict
    output: str


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    input_tokens: int = 0   # summed over every model call in this run
    output_tokens: int = 0


def _text_of(message: AIMessage) -> str:
    """Model content may be a string or a list of blocks (e.g. Claude)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


class CreatorAgent:
    def __init__(
        self,
        model: BaseChatModel,
        tools: list[BaseTool],
        store: DataStore,
        max_steps: int = 6,
    ):
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        self.store = store
        self.max_steps = max_steps
        self.history: list[BaseMessage] = []
        self.pending_events: list[str] = []

    def note_event(self, text: str) -> None:
        """Record something the app did outside the model (e.g. the user
        confirmed a draft), so the model's view stays in sync with reality.

        Events are attached to the start of the next user message rather than
        sent as separate messages: some APIs (e.g. Claude) reject a system
        message in mid-conversation.
        """
        self.pending_events.append(text)

    def run(self, user_input: str) -> AgentResult:
        content = user_input
        if self.pending_events:
            events = "\n".join(f"[App event] {e}" for e in self.pending_events)
            content = f"{events}\n\n{user_input}"
            self.pending_events = []
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT), *self.history, HumanMessage(content),
        ]
        steps: list[Step] = []
        tokens = {"input_tokens": 0, "output_tokens": 0}

        for _ in range(self.max_steps):
            try:
                reply = self.model.invoke(messages)
            except Exception as exc:  # network error, auth error, rate limit...
                return self._fallback(user_input, steps, f"model call failed: {exc}")

            messages.append(reply)
            usage = getattr(reply, "usage_metadata", None) or {}
            for key in tokens:
                tokens[key] += usage.get(key, 0)

            if not reply.tool_calls:  # plain answer -> done
                answer = _text_of(reply)
                self.history = messages[1:]  # keep the turn for follow-up questions
                return AgentResult(answer=answer, steps=steps, **tokens)

            for call in reply.tool_calls:
                output = self._run_tool(call["name"], call["args"])
                steps.append(Step(call["name"], call["args"], output))
                messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

        return self._fallback(user_input, steps, f"stopped after {self.max_steps} steps")

    def _run_tool(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool {name!r}."})
        try:
            return tool.invoke(args)
        except Exception as exc:  # bad arguments etc. -> let the model retry
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def _fallback(self, user_input: str, steps: list[Step], reason: str) -> AgentResult:
        """Degrade to the plain TF-IDF recommender, like the original ContentFlow."""
        profile = {"interests": user_input, "skills": user_input, "goals": user_input}
        results = recommender.recommend_strategies(profile, self.store.strategies)
        if results:
            lines = [f"- {r['name']}（{r['score']} 分）" for r in results]
            answer = "智能助手暂时不可用，以下是基础推荐算法的结果：\n" + "\n".join(lines)
        else:
            answer = (
                "智能助手暂时不可用，基础推荐算法也没有找到匹配结果"
                "（它只能匹配英文关键词）。请稍后再试。"
            )
        return AgentResult(answer=answer, steps=steps, fallback_used=True, fallback_reason=reason)
