from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    run_id: str
    run_name: str
    mode: str
    layout: str
    model: str
    run_title: str = ""
    experiment: str = "default-exp"
    variant: str = "baseline"
    tags: list[str] = field(default_factory=list)
    notes: str = ""


class FileRunLogger:
    def __init__(self, base_dir: str | Path, context: RunContext):
        self.base_dir = Path(base_dir)
        self.context = context
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.base_dir / f"{self.context.run_id}.jsonl"

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        step: int | None = None,
        agent_role: str = "runner",
    ) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.context.run_id,
            "run_name": self.context.run_name,
            "run_title": self.context.run_title,
            "event_type": event_type,
            "mode": self.context.mode,
            "layout": self.context.layout,
            "model": self.context.model,
            "experiment": self.context.experiment,
            "variant": self.context.variant,
            "tags": list(self.context.tags),
            "notes": self.context.notes,
            "step": step,
            "agent_role": agent_role,
            "payload": payload,
        }
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def normalize_tags(user_tags: list[str], mode: str, layout: str) -> list[str]:
    tags = [tag.strip() for tag in user_tags if tag and tag.strip()]
    required = [f"mode:{mode}", f"layout:{layout}"]
    for tag in required:
        if tag not in tags:
            tags.append(tag)
    return tags
