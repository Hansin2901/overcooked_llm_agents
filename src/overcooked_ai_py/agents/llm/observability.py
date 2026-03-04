from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import types
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


def _install_langfuse_cost_hook(callback: Any, default_model_name: str) -> None:
    if callback is None:
        return
    if callback.__class__.__module__.startswith("unittest.mock"):
        return
    if getattr(callback, "_cost_hook_installed", False):
        return
    if not hasattr(callback, "on_llm_end") or not callable(callback.on_llm_end):
        return
    if not hasattr(callback, "_detach_observation"):
        return

    callback._orig_on_llm_end = callback.on_llm_end

    def _on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        try:
            response_generation = response.generations[-1][-1]
            message = getattr(response_generation, "message", None)
            if message is not None and hasattr(self, "_convert_message_to_dict"):
                extracted_response = self._convert_message_to_dict(message)
            else:
                extracted_response = getattr(response_generation, "text", None)
                if extracted_response is None:
                    extracted_response = str(response_generation)

            usage = _extract_usage_from_llm_result(response)
            model_name = _extract_model_from_llm_result(response) or default_model_name

            prompt_tokens = usage.get("input") if isinstance(usage, dict) else None
            completion_tokens = usage.get("output") if isinstance(usage, dict) else None
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

            generation = self._detach_observation(run_id)
            if generation is not None:
                generation.update(
                    output=extracted_response,
                    usage=usage,
                    usage_details=usage,
                    input=kwargs.get("inputs"),
                    model=model_name,
                    cost_details=cost_details,
                ).end()
                self.last_trace_id = generation.trace_id
        except Exception:
            return self._orig_on_llm_end(
                response,
                run_id=run_id,
                parent_run_id=parent_run_id,
                **kwargs,
            )

    callback.on_llm_end = types.MethodType(_on_llm_end, callback)
    callback._cost_hook_installed = True


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
        if self._callback is not None:
            _install_langfuse_cost_hook(
                self._callback,
                normalize_model_name(self.context.model),
            )

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
