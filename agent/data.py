"""Load the demo data exported from the original ContentFlow project.

data/demo_data.json was exported (read-only) from ContentFlow's data
migration and seed_demo_data command, so this project needs no Django
or database to run.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_data.json"


@dataclass
class DataStore:
    strategies: list[dict]
    collaborations: list[dict]
    creators: list[dict]
    niches: list[dict] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = DEFAULT_DATA_PATH) -> "DataStore":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        collaborations = [
            {"id": index, **item}
            for index, item in enumerate(raw["collaborations"], start=1)
        ]
        return cls(
            strategies=raw["strategies"],
            collaborations=collaborations,
            creators=raw["creators"],
            niches=raw.get("niches", []),
            skills=raw.get("skills", []),
        )

    def open_collaborations(self) -> list[dict]:
        return [c for c in self.collaborations if c.get("status") == "OPEN"]

    def find_strategy(self, name: str) -> dict | None:
        wanted = name.strip().lower()
        return next(
            (s for s in self.strategies if s["name"].lower() == wanted), None
        )

    def find_collaboration(self, collaboration_id: int) -> dict | None:
        return next(
            (c for c in self.collaborations if c["id"] == collaboration_id), None
        )
