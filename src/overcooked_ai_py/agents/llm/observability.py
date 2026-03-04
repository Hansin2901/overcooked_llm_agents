from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency
    Langfuse = None


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


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_usage(raw_usage: Any) -> dict[str, int] | None:
    if not isinstance(raw_usage, dict):
        return None
    prompt_tokens = (
        _coerce_int(raw_usage.get("input"))
        or _coerce_int(raw_usage.get("prompt_tokens"))
        or _coerce_int(raw_usage.get("input_tokens"))
    )
    completion_tokens = (
        _coerce_int(raw_usage.get("output"))
        or _coerce_int(raw_usage.get("completion_tokens"))
        or _coerce_int(raw_usage.get("output_tokens"))
    )
    total_tokens = _coerce_int(raw_usage.get("total")) or _coerce_int(
        raw_usage.get("total_tokens")
    )
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    usage = {}
    if prompt_tokens is not None:
        usage["input"] = prompt_tokens
    if completion_tokens is not None:
        usage["output"] = completion_tokens
    if total_tokens is not None:
        usage["total"] = total_tokens
    return usage or None


def _extract_usage_from_llm_result(response: Any) -> dict[str, int] | None:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        for key in ("token_usage", "usage"):
            usage = _normalize_usage(llm_output.get(key))
            if usage:
                return usage

    generations = getattr(response, "generations", None)
    if not isinstance(generations, list):
        return None
    for generation in generations:
        if not isinstance(generation, list):
            continue
        for chunk in generation:
            generation_info = getattr(chunk, "generation_info", None)
            if isinstance(generation_info, dict):
                usage = _normalize_usage(generation_info.get("usage_metadata"))
                if usage:
                    return usage
            message = getattr(chunk, "message", None)
            if message is None:
                continue
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, dict):
                for key in ("token_usage", "usage", "amazon-bedrock-invocationMetrics"):
                    usage = _normalize_usage(response_metadata.get(key))
                    if usage:
                        return usage
            usage = _normalize_usage(getattr(message, "usage_metadata", None))
            if usage:
                return usage
    return None


def _extract_model_from_llm_result(response: Any) -> str | None:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        model_name = llm_output.get("model_name")
        if isinstance(model_name, str) and model_name:
            return normalize_model_name(model_name)
    return None


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
        self._client = None
        self._trace = None
        self._trace_id = None
        self._active_step = None
        self._active_role = None
        if self.enabled and Langfuse is not None:
            try:
                self._client = Langfuse()
            except Exception:
                self._client = None

    def build_invoke_config(self, base_config: dict | None) -> dict:
        # Graph callback wiring is intentionally disabled to avoid router/node noise.
        return dict(base_config or {})

    def get_trace_id(self) -> str | None:
        return self._trace_id

    def start_run(self) -> None:
        if self._client is None or self._trace is not None:
            return
        try:
            self._trace = self._client.start_span(
                name=self.context.run_name,
                metadata={
                    "run_id": self.context.run_id,
                    "mode": self.context.mode,
                    "layout": self.context.layout,
                    "model": normalize_model_name(self.context.model),
                    "experiment": self.context.experiment,
                    "variant": self.context.variant,
                    "tags": list(self.context.tags),
                },
            )
            self._trace_id = getattr(self._trace, "trace_id", None)
            try:
                # Langfuse "Tags" UI column reads trace tags, not metadata["tags"].
                self._trace.update_trace(
                    name=self.context.run_name,
                    session_id=self.context.run_id,
                    tags=list(self.context.tags),
                    metadata={
                        "run_id": self.context.run_id,
                        "mode": self.context.mode,
                        "layout": self.context.layout,
                        "model": normalize_model_name(self.context.model),
                        "experiment": self.context.experiment,
                        "variant": self.context.variant,
                    },
                )
                # Current SDK versions can miss tag chips for OTEL-only updates.
                # Also enqueue explicit ingestion tag updates for reliability.
                if (
                    self._trace_id
                    and hasattr(self._client, "_create_trace_tags_via_ingestion")
                    and list(self.context.tags)
                ):
                    self._client._create_trace_tags_via_ingestion(
                        trace_id=self._trace_id,
                        tags=list(self.context.tags),
                    )
            except Exception:
                pass
        except Exception:
            self._trace = None
            self._trace_id = None

    def end_run(self, payload: dict[str, Any]) -> None:
        self.end_role()
        self.end_step()
        if self._trace is None:
            return
        try:
            self._trace.update(output=payload)
            self._trace.end()
        except Exception:
            pass
        finally:
            self._trace = None
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass

    def start_step(self, step: int) -> None:
        if self._trace is None:
            return
        if self._active_step is not None:
            self.end_step()
        try:
            self._active_step = self._trace.start_span(
                name=f"step_{step}",
                metadata={"step": step},
            )
        except Exception:
            self._active_step = None

    def end_step(self) -> None:
        self.end_role()
        if self._active_step is None:
            return
        try:
            self._active_step.end()
        except Exception:
            pass
        finally:
            self._active_step = None

    def start_role(self, role: str) -> None:
        if self._active_role is not None:
            self.end_role()
        parent = self._active_step or self._trace
        if parent is None:
            return
        try:
            self._active_role = parent.start_span(
                name=role,
                metadata={"agent_role": role},
            )
        except Exception:
            self._active_role = None

    def end_role(self) -> None:
        if self._active_role is None:
            return
        try:
            self._active_role.end()
        except Exception:
            pass
        finally:
            self._active_role = None

    def _current_parent(self):
        return self._active_role or self._active_step or self._trace

    def emit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        step: int | None,
        agent_role: str,
    ) -> None:
        parent = self._current_parent()
        if parent is None:
            return
        event_payload = payload if isinstance(payload, dict) else {}
        try:
            if event_type == "llm.generation":
                model_name = normalize_model_name(
                    str(event_payload.get("model_name") or self.context.model)
                )
                prompt_tokens = _coerce_int(event_payload.get("prompt_tokens"))
                completion_tokens = _coerce_int(event_payload.get("completion_tokens"))
                usage = {}
                if prompt_tokens is not None:
                    usage["input"] = prompt_tokens
                if completion_tokens is not None:
                    usage["output"] = completion_tokens
                if prompt_tokens is not None and completion_tokens is not None:
                    usage["total"] = prompt_tokens + completion_tokens

                estimated_total = event_payload.get("estimated_cost_usd")
                rates = get_model_cost_rates(model_name)
                cost_details = None
                if (
                    rates is not None
                    and prompt_tokens is not None
                    and completion_tokens is not None
                ):
                    input_cost = (prompt_tokens / 1_000_000.0) * rates["input"]
                    output_cost = (completion_tokens / 1_000_000.0) * rates["output"]
                    total_cost = input_cost + output_cost
                    cost_details = {
                        "input": input_cost,
                        "output": output_cost,
                        "total": total_cost,
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": total_cost,
                    }
                elif estimated_total is not None:
                    total_cost = float(estimated_total)
                    cost_details = {"total": total_cost, "total_cost": total_cost}

                generation = parent.start_generation(
                    name="llm.generation",
                    output=event_payload.get("content")
                    or event_payload.get("content_preview")
                    or "",
                    model=model_name,
                    usage_details=usage or None,
                    cost_details=cost_details,
                    metadata={
                        "event_type": event_type,
                        "step": step,
                        "agent_role": agent_role,
                        "tool_call_count": event_payload.get("tool_call_count"),
                    },
                )
                generation.end()
                return

            if event_type == "tool.call":
                tool_name = str(event_payload.get("tool_name") or "tool.call")
                span = parent.start_span(
                    name=tool_name,
                    input=event_payload.get("args", {}),
                    metadata={
                        "event_type": event_type,
                        "step": step,
                        "agent_role": agent_role,
                    },
                )
                span.end()
                return

            if event_type in {"planner.assignment", "action.commit", "error"}:
                parent.update(
                    metadata={
                        "last_event_type": event_type,
                        "last_event_payload": event_payload,
                        "last_event_step": step,
                        "last_event_agent_role": agent_role,
                    }
                )
        except Exception:
            # LangFuse is best-effort only; local JSONL remains source of truth.
            return


class ObservabilityHub:
    def __init__(
        self,
        file_logger: FileRunLogger,
        langfuse: LangFuseReporter | None,
    ):
        self.file_logger = file_logger
        self.langfuse = langfuse
        self._current_step: int | None = None

    def build_invoke_config(self, base_config: dict | None) -> dict:
        if self.langfuse is None:
            return dict(base_config or {})
        return self.langfuse.build_invoke_config(base_config)

    def get_trace_id(self) -> str | None:
        if self.langfuse is None:
            return None
        return self.langfuse.get_trace_id()

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        step: int | None = None,
        agent_role: str = "runner",
    ) -> None:
        effective_step = self._current_step if step is None else step
        self.file_logger.emit(
            event_type,
            payload,
            step=effective_step,
            agent_role=agent_role,
        )
        if self.langfuse is not None:
            self.langfuse.emit_event(
                event_type,
                payload,
                step=effective_step,
                agent_role=agent_role,
            )

    def start_run(self) -> None:
        if self.langfuse is not None:
            self.langfuse.start_run()

    def end_run(self, payload: dict[str, Any]) -> None:
        if self.langfuse is not None:
            self.langfuse.end_run(payload)

    def start_step(self, step: int) -> None:
        self._current_step = step
        if self.langfuse is not None:
            self.langfuse.start_step(step)

    def end_step(self) -> None:
        if self.langfuse is not None:
            self.langfuse.end_step()
        self._current_step = None

    def start_role(self, role: str) -> None:
        if self.langfuse is not None:
            self.langfuse.start_role(role)

    def end_role(self) -> None:
        if self.langfuse is not None:
            self.langfuse.end_role()


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
