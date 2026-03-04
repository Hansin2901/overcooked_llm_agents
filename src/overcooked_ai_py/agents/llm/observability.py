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


MODEL_COST_USD_PER_1M: dict[str, dict[str, float]] = {
    "api-gpt-oss-120b": {"input": 0.1500, "output": 0.6000},
    "api-lightonocr-1b": {"input": 0.0, "output": 0.0500},
    "api-llama-4-scout": {"input": 0.2000, "output": 0.7800},
    "api-mistral-small-3.2-2506": {"input": 0.0500, "output": 0.1800},
    "api-tgpt-embeddings": {"input": 0.0150, "output": 0.0},
    "minimax.minimax-m2": {"input": 0.3000, "output": 1.2000},
    "mistral.mistral-large-3-675b-instruct": {"input": 0.5000, "output": 1.5000},
    "moonshotai.kimi-k2.5": {"input": 0.6000, "output": 3.0300},
    "us.amazon.nova-2-lite-v1:0": {"input": 0.3300, "output": 2.7500},
    "us.amazon.nova-premier-v1:0": {"input": 2.5000, "output": 12.5000},
    "us.deepseek.r1-v1:0": {"input": 1.3500, "output": 5.4000},
}


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        return ""
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def get_model_cost_rates(model_name: str) -> dict[str, float] | None:
    return MODEL_COST_USD_PER_1M.get(normalize_model_name(model_name))


def estimate_model_cost_usd(
    model_name: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    rates = get_model_cost_rates(model_name)
    if rates is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (
        (prompt_tokens / 1_000_000.0) * rates["input"]
        + (completion_tokens / 1_000_000.0) * rates["output"]
    )


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
                # Prefer current LangFuse callback signature, fall back for older versions.
                try:
                    self._callback = CallbackHandler(
                        update_trace=True,
                        trace_context={"trace_id": self.context.run_id},
                    )
                except TypeError:
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
        cfg["run_name"] = self.context.run_name
        normalized_model = normalize_model_name(self.context.model)
        cost_rates = get_model_cost_rates(self.context.model)
        cfg["metadata"] = {
            **cfg.get("metadata", {}),
            "langfuse_session_id": self.context.run_id,
            "langfuse_tags": self.context.tags,
            "ls_model_name": normalized_model,
        }
        if cost_rates is not None:
            cfg["metadata"]["model_cost_input_usd_per_million"] = cost_rates["input"]
            cfg["metadata"]["model_cost_output_usd_per_million"] = cost_rates["output"]
        return cfg


def build_run_context(args, mode: str, layout: str, model: str) -> RunContext:
    # Use full 32-char lowercase hex so it is valid as a LangFuse trace_id.
    run_id = uuid.uuid4().hex
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
