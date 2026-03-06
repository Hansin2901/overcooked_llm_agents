# Custom LangFuse Step Hierarchy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace noisy graph-node tracing with a clean custom hierarchy in LangFuse: `step -> planner/worker role -> LLM/tool details`, while preserving cost visibility and local JSONL logs.

**Architecture:** Keep `FileRunLogger` as the source of truth for local logs, and add a custom LangFuse hierarchy reporter that creates explicit run, step, and role spans. Route `llm.generation` and `tool.call` events into manual LangFuse observations instead of graph-level callback spans, so router/bucket spans (`dispatch`, `observe`, `act`) no longer clutter traces.

**Tech Stack:** Python 3.10, LangGraph, LangChain/LiteLLM, LangFuse Python SDK, `unittest`.

---

Skill references: `@test-driven-development`, `@verification-before-completion`.

### Task 1: Add Failing Tests for Custom Hierarchy Reporter API

**Files:**
- Modify: `testing/test_observability.py`
- Test: `testing/test_observability.py`

**Step 1: Write the failing tests**

```python
class TestLangFuseHierarchyReporter(unittest.TestCase):
    @patch("overcooked_ai_py.agents.llm.observability.Langfuse")
    def test_start_step_then_role_creates_nested_spans(self, mock_langfuse):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        reporter.start_run()
        reporter.start_step(3)
        reporter.start_role("worker_0")
        reporter.end_role()
        reporter.end_step()
        reporter.end_run({"steps": 5})
        # assert span nesting calls happened with deterministic names

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse")
    def test_emit_llm_generation_creates_generation_with_cost(self, mock_langfuse):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        reporter.start_run(); reporter.start_step(0); reporter.start_role("planner")
        reporter.emit_event("llm.generation", {"content": "reasoning", "prompt_tokens": 10, "completion_tokens": 2, "estimated_cost_usd": 0.00001}, step=0, agent_role="planner")
        # assert generation created with model/usage/cost_details

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse")
    def test_emit_tool_call_creates_tool_span(self, mock_langfuse):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        reporter.start_run(); reporter.start_step(1); reporter.start_role("worker_1")
        reporter.emit_event("tool.call", {"tool_name": "check_path", "args": {"target": "pot"}}, step=1, agent_role="worker_1")
        # assert child observation/span named check_path is created
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_observability.TestLangFuseHierarchyReporter -v`  
Expected: FAIL with missing `start_step/start_role/emit_event` methods.

**Step 3: Keep legacy behavior tests in place**

```python
# Do not delete existing TestLangFuseReporter tests yet.
# Mark callback-centric assertions that will change with comments.
```

**Step 4: Re-run focused tests**

Run: `uv run python -m unittest testing.test_observability.TestLangFuseReporter -v`  
Expected: Existing tests may fail in callback-specific sections, which is acceptable before Task 2.

**Step 5: Commit**

```bash
git add testing/test_observability.py
git commit -m "test: add failing tests for custom LangFuse hierarchy reporter"
```

### Task 2: Implement Custom Run/Step/Role Hierarchy in Observability Module

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/observability.py`
- Modify: `testing/test_observability.py`
- Test: `testing/test_observability.py`

**Step 1: Implement minimal hierarchy state in `LangFuseReporter`**

```python
class LangFuseReporter:
    def __init__(...):
        self._client = Langfuse(...) if enabled else None
        self._trace = None
        self._active_step = None
        self._active_role = None

    def start_run(self): ...
    def end_run(self, payload: dict): ...
    def start_step(self, step: int): ...
    def end_step(self): ...
    def start_role(self, role: str): ...
    def end_role(self): ...
```

**Step 2: Add event forwarding for manual observations**

```python
def emit_event(self, event_type: str, payload: dict, step: int | None, agent_role: str):
    if event_type == "llm.generation":
        # create generation under current role span with usage + cost_details
    elif event_type == "tool.call":
        # create span named tool_name with args as input
    elif event_type in {"planner.assignment", "action.commit", "error"}:
        # attach event/metadata to current role span
```

**Step 3: Keep callback path disabled for graph invocations**

```python
def build_invoke_config(self, base_config):
    # return recursion/base config only, no callbacks
```

**Step 4: Update tests to assert new API and remove obsolete callback-only assumptions**

Run: `uv run python -m unittest testing.test_observability -v`  
Expected: PASS for hierarchy tests, PASS for core/run-context/cost tests.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/observability.py testing/test_observability.py
git commit -m "feat: add custom LangFuse run-step-role hierarchy reporter"
```

### Task 3: Add Composite Observability Sink (Local JSONL + LangFuse)

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/observability.py`
- Modify: `scripts/run_llm_agent.py`
- Modify: `testing/test_observability.py`
- Test: `testing/test_observability.py`

**Step 1: Add a hub object that is passed to agents/graph builder**

```python
class ObservabilityHub:
    def __init__(self, file_logger: FileRunLogger, langfuse: LangFuseReporter | None): ...
    def emit(self, event_type, payload, step=None, agent_role="runner"):
        file_logger.emit(...)
        if langfuse: langfuse.emit_event(...)
    def start_run(self): ...
    def end_run(self, payload): ...
    def start_step(self, step): ...
    def end_step(self): ...
    def start_role(self, role): ...
    def end_role(self): ...
```

**Step 2: Add failing/updated tests for dual-write behavior**

```python
def test_hub_emits_to_file_and_langfuse(self):
    # assert file logger write and langfuse emit_event both invoked
```

**Step 3: Implement the hub and wire in unit tests**

Run: `uv run python -m unittest testing.test_observability.TestObservabilityCore -v`  
Expected: PASS.

**Step 4: Run full observability test module**

Run: `uv run python -m unittest testing.test_observability -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/observability.py scripts/run_llm_agent.py testing/test_observability.py
git commit -m "feat: add observability hub for local logs and custom LangFuse events"
```

### Task 4: Wire Step Scopes in Runner and Remove Graph-Level Callback Noise

**Files:**
- Modify: `scripts/run_llm_agent.py`
- Modify: `testing/test_observability.py`
- Test: `testing/test_observability.py`

**Step 1: Add explicit step lifecycle calls around each environment step**

```python
hub.start_run()
while not done:
    hub.start_step(step)
    joint_action_and_infos = agent_pair.joint_action(state)
    ...
    hub.end_step()
hub.end_run({...})
```

**Step 2: Keep run.start/run.end file events unchanged**

```python
hub.emit("run.start", {"horizon": args.horizon})
...
hub.emit("run.end", {...})
```

**Step 3: Add runner test to verify step scope calls happen in order**

```python
# mock hub, assert start_run -> start_step -> end_step -> end_run sequence
```

**Step 4: Run tests**

Run: `uv run python -m unittest testing.test_observability.TestRunScriptCli -v`  
Expected: PASS (existing CLI assertions retained).

**Step 5: Commit**

```bash
git add scripts/run_llm_agent.py testing/test_observability.py
git commit -m "feat: add explicit step scopes in runner for custom trace hierarchy"
```

### Task 5: Wire Role Scopes in Planner, WorkerAgent, and LLMAgent

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner.py`
- Modify: `src/overcooked_ai_py/agents/llm/worker_agent.py`
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`
- Modify: `testing/test_planner.py`
- Modify: `testing/test_worker_agent_unit.py`
- Modify: `testing/llm_agent_test.py`
- Test: `testing/test_planner.py`, `testing/test_worker_agent_unit.py`, `testing/llm_agent_test.py`

**Step 1: Add role scope start/end around planner execution**

```python
self.observability.start_role("planner")
try:
    self._graph.invoke(...)
finally:
    self.observability.end_role()
```

**Step 2: Add role scope around each worker action**

```python
self.observability.start_role(self.worker_id)
try:
    # prompt + graph invoke + action commit
finally:
    self.observability.end_role()
```

**Step 3: Add role scope around single-agent LLM action path**

```python
self.observability.start_role("llm_agent")
...
self.observability.end_role()
```

**Step 4: Add/adjust tests for scope hooks**

```python
sink = Mock()
# assert sink.start_role("planner") and sink.end_role() called
```

Run:
- `uv run python -m unittest testing.test_planner -v`
- `uv run python -m unittest testing.test_worker_agent_unit -v`
- `uv run python -m unittest testing.llm_agent_test -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner.py src/overcooked_ai_py/agents/llm/worker_agent.py src/overcooked_ai_py/agents/llm/llm_agent.py testing/test_planner.py testing/test_worker_agent_unit.py testing/llm_agent_test.py
git commit -m "feat: add planner and worker role scopes for custom trace hierarchy"
```

### Task 6: Simplify Graph Builder Labels and Event Payloads for Cleaner Trace Content

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/graph_builder.py`
- Modify: `testing/test_graph_builder.py`
- Test: `testing/test_graph_builder.py`

**Step 1: Remove router naming hacks and role-prefixed node names from trace-facing concerns**

```python
# keep logical function name as decide_next_step
# use stable internal nodes (llm, obs_tools, action_tools) or simple neutral names
```

**Step 2: Ensure `llm.generation` carries full assistant content for reasoning visibility**

```python
_safe_emit("llm.generation", {
    "content": response.content or "",
    "content_preview": (response.content or "")[:200],
    ...
})
```

**Step 3: Keep `tool.call` payloads unchanged and explicit**

```python
{"tool_name": ..., "args": ...}
```

**Step 4: Update tests**

Run: `uv run python -m unittest testing.test_graph_builder -v`  
Expected: PASS with revised node-name expectations and payload checks.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/graph_builder.py testing/test_graph_builder.py
git commit -m "fix: simplify graph naming and enrich llm generation payloads"
```

### Task 7: Verify End-to-End Custom Hierarchy in a Real Planner-Worker Run

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/README_PLANNER_WORKER.md`
- Test: runtime validation via script + log inspection

**Step 1: Run focused test suite**

Run:
- `uv run python -m unittest testing.test_observability testing.test_graph_builder -v`
- `uv run python -m unittest testing.test_planner testing.test_worker_agent_unit testing.llm_agent_test -v`

Expected: PASS.

**Step 2: Run real planner-worker episode (20 or 50 horizon)**

Run: `uv run python scripts/run_llm_agent.py --agent-type planner-worker --horizon 20 --run-name custom-step-hierarchy-h20 --tags bench,custom-hierarchy`

Expected console:
- Episode completes without exceptions
- Local JSONL log file created for run

**Step 3: Validate local event shape**

Run: `rg -n '"event_type": "llm.generation"|"event_type": "tool.call"|"event_type": "planner.assignment"|"event_type": "action.commit"' logs/agent_runs/<run_id>.jsonl`

Expected:
- Events include step number and agent role
- No regression in event counts

**Step 4: Validate LangFuse trace structure manually**

Expected in UI:
- Top-level run trace
- Child spans: `step_0`, `step_1`, ...
- Under each step: `planner` (when replanned), `worker_0`, `worker_1`
- Under each role: LLM generation and tool spans only (no router noise)

**Step 5: Commit docs update**

```bash
git add src/overcooked_ai_py/agents/llm/README_PLANNER_WORKER.md
git commit -m "docs: document custom step-based LangFuse hierarchy"
```

## Risks and Guardrails

- LangFuse SDK API drift risk: isolate SDK calls inside `observability.py` and keep all failures non-fatal.
- Missing cost column risk: ensure generation objects include `usage` plus explicit `cost_details` when rates are known.
- Span lifecycle leaks risk: close role and step spans in `finally` blocks.
- Backward compatibility: keep local JSONL schema stable (`event_type`, `step`, `agent_role`, `payload`).

## Done Criteria

- No `dispatch`/graph-router clutter in LangFuse for planner-worker runs.
- Trace hierarchy is `step -> planner/worker -> llm/tool`.
- Costs remain visible in LangFuse total-cost column.
- All updated tests pass plus one real horizon-20 (or horizon-50) run is validated.
