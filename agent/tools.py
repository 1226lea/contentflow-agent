"""The four tools the agent can call.

Each tool returns a JSON string: that is what the model reads back as
the tool result. Docstrings and argument descriptions matter — the
model sees them and uses them to decide when and how to call a tool.
"""
import json
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import BaseTool, tool

from . import recommender
from .data import DataStore


@dataclass
class Session:
    """Per-conversation state. Drafts wait here until the user confirms them."""
    pending_drafts: list[dict] = field(default_factory=list)


def _json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


def build_tools(store: DataStore, session: Session) -> list[BaseTool]:

    @tool
    def recommend_strategies(
        interests: str,
        skills: str,
        goals: str = "",
        constraints: str = "",
        limit: int = 3,
    ) -> str:
        """Recommend content strategies (creative directions) for a creator,
        using ContentFlow's original TF-IDF recommender.

        All arguments MUST be written in English keywords, because the
        strategy data is English and TF-IDF only matches exact words.
        - interests: topics the creator likes, e.g. "travel, local culture, photography"
        - skills: e.g. "video editing, storytelling"
        - goals: what they want to achieve
        - constraints: time, equipment, camera preference, content length, platforms
        Returns the top strategies with a score (0-100) and the matched terms
        that explain each recommendation.
        """
        profile = {
            "interests": interests, "skills": skills,
            "goals": goals, "constraints": constraints,
        }
        results = recommender.recommend_strategies(
            profile, store.strategies, limit=max(1, min(limit, 5))
        )
        if not results:
            return _json({"results": [], "note": "No match. Try broader English keywords."})
        return _json({"results": results})

    @tool
    def search_collaborations(query: str, limit: int = 3) -> str:
        """Search OPEN collaboration requests posted by other creators.

        query MUST be English keywords describing the kind of partner work
        wanted, e.g. "video editor travel shorts subtitles".
        Returns matching requests with id, title, required skills and score.
        """
        results = recommender.search_collaborations(
            query, store.open_collaborations(), limit=max(1, min(limit, 5))
        )
        if not results:
            return _json({"results": [], "note": "No open request matched. Try other English keywords."})
        return _json({"results": results})

    @tool
    def get_details(kind: Literal["strategy", "collaboration"], key: str) -> str:
        """Get full details of one item before recommending or drafting.

        - kind="strategy": key is the exact strategy name from recommend_strategies
        - kind="collaboration": key is the numeric id from search_collaborations
        """
        if kind == "strategy":
            item = store.find_strategy(key)
        else:
            try:
                item = store.find_collaboration(int(key))
            except ValueError:
                item = None
        if item is None:
            return _json({"error": f"No {kind} found for key {key!r}."})
        return _json(item)

    @tool
    def draft_application(collaboration_id: int, message: str) -> str:
        """Draft an application message to an open collaboration request.

        This does NOT send anything. The draft is saved for the user to
        review; it is only submitted after the user explicitly confirms.
        Write the message in the user's language, specific to the request.
        """
        collab = store.find_collaboration(collaboration_id)
        if collab is None or collab.get("status") != "OPEN":
            return _json({"error": f"Collaboration {collaboration_id} not found or not open."})
        if not message.strip():
            return _json({"error": "Message is empty."})
        draft = {
            "collaboration_id": collaboration_id,
            "title": collab["title"],
            "message": message.strip(),
        }
        session.pending_drafts.append(draft)
        # Tell the model where its job ends: the app, not the model, asks the
        # user to confirm. Without this the model asked the user to "reply to
        # confirm" in chat while the CLI asked y/N at the same time.
        return _json({
            "status": "draft_saved_awaiting_user_confirmation",
            "draft_title": draft["title"],
            "note": (
                "After your reply, the app shows this draft to the user and asks "
                "them to confirm or discard it directly. Do NOT repeat the draft "
                "text and do NOT ask the user to reply to confirm. Just say the "
                "draft is ready for their review."
            ),
        })

    return [recommend_strategies, search_collaborations, get_details, draft_application]
