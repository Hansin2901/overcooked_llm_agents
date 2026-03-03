from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any

try:
    from langfuse.langchain import CallbackHandler
except Exception:  # pragma: no cover - optional dependency
    CallbackHandler = None


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


class LangFuseReporter:
    def __init__(self, enabled: bool, context: RunContext):
        self.enabled = enabled
        self.context = context
        self._callback = None
        if self.enabled and CallbackHandler is not None:
            try:
                # Support both legacy and current LangFuse callback signatures.
                try:
                    self._callback = CallbackHandler(
                        session_id=self.context.run_id,
                        trace_name=self.context.run_name,
                    )
                except TypeError:
                    self._callback = CallbackHandler()
            except Exception:
                # LangFuse is best-effort only; continue with local logging.
                self._callback = None

    def build_invoke_config(self, base_config: dict | None) -> dict:
        if not self.enabled or self._callback is None:
            return dict(base_config or {})
        cfg = dict(base_config or {})
        cfg["callbacks"] = [self._callback]
        cfg["metadata"] = {
            **cfg.get("metadata", {}),
            "langfuse_session_id": self.context.run_id,
            "langfuse_tags": self.context.tags,
        }
        return cfg


def build_run_context(args, mode: str, layout: str, model: str) -> RunContext:
    run_id = uuid.uuid4().hex[:12]
    default_name = f"{mode}-{layout}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_name = args.run_name or default_name
    user_tags = [tag.strip() for tag in (args.tags or "").split(",")]
    tags = normalize_tags(user_tags, mode=mode, layout=layout)
    return RunContext(
        run_id=run_id,
        run_name=run_name,
        mode=mode,
        layout=layout,
        model=model,
        run_title=args.run_title,
        experiment=args.experiment,
        variant=args.variant,
        tags=tags,
        notes=args.notes,
    )
