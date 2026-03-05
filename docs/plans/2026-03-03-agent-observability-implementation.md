# Agent Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add robust observability for both `llm` and `planner-worker` modes via per-run JSONL logs and optional LangFuse tracing, without touching core framework behavior.

**Architecture:** Add a shared observability module in the LLM-agent package that owns run context, local file logging, and optional LangFuse callback plumbing. Thread observability through runner and LLM-agent classes so all graph invocations can emit consistent events and metadata. Keep failures non-fatal by treating observability as best-effort.

**Tech Stack:** Python 3.10, LangGraph, LangChain/LiteLLM, LangFuse Python SDK, `unittest`.

---

Skill references: `@test-driven-development`, `@verification-before-completion`.

### Task 1: Create Observability Core (RunContext + JSONL Logger)

**Files:**
- Create: `src/overcooked_ai_py/agents/llm/observability.py`
- Create: `testing/test_observability.py`
- Test: `testing/test_observability.py`

**Step 1: Write the failing tests**

```python
def test_file_logger_creates_run_file(self):
    ctx = RunContext(run_id="r1", run_name="bench-a", mode="llm", layout="cramped_room", model="gpt-4o")
    logger = FileRunLogger(base_dir=self.tmpdir, context=ctx)
    logger.emit("run.start", {"x": 1})
    self.assertTrue(logger.file_path.exists())

def test_event_contains_common_fields(self):
    ctx = RunContext(run_id="r2", run_name="bench-b", mode="planner-worker", layout="cramped_room", model="gpt-4o")
    logger = FileRunLogger(base_dir=self.tmpdir, context=ctx)
    logger.emit("action.commit", {"action": "move_up"}, step=4, agent_role="worker_0")
    row = json.loads(logger.file_path.read_text().splitlines()[0])
    self.assertEqual(row["run_id"], "r2")
    self.assertEqual(row["event_type"], "action.commit")
    self.assertEqual(row["mode"], "planner-worker")
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_observability -v`  
Expected: FAIL with `ModuleNotFoundError` or missing `RunContext/FileRunLogger`.

**Step 3: Write minimal implementation**

```python
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
    def emit(self, event_type: str, payload: dict, step: int | None = None, agent_role: str = "runner") -> None:
        ...
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_observability -v`  
Expected: PASS for core logger tests.

**Step 5: Commit**

```bash
git add testing/test_observability.py src/overcooked_ai_py/agents/llm/observability.py
git commit -m "feat: add observability core and per-run JSONL logger"
```

### Task 2: Add Tag Normalization and Mode-Tag Enforcement

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/observability.py`
- Modify: `testing/test_observability.py`
- Test: `testing/test_observability.py`

**Step 1: Write the failing tests**

```python
def test_required_mode_tag_added_for_llm(self):
    tags = normalize_tags(["exp:bench1"], mode="llm", layout="cramped_room")
    self.assertIn("mode:llm", tags)
    self.assertIn("layout:cramped_room", tags)

def test_required_mode_tag_added_for_planner_worker(self):
    tags = normalize_tags([], mode="planner-worker", layout="coordination_ring")
    self.assertIn("mode:planner-worker", tags)
    self.assertIn("layout:coordination_ring", tags)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_observability.TestObservabilityTags -v`  
Expected: FAIL with missing `normalize_tags`.

**Step 3: Write minimal implementation**

```python
def normalize_tags(user_tags: list[str], mode: str, layout: str) -> list[str]:
    tags = [t.strip() for t in user_tags if t and t.strip()]
    required = [f"mode:{mode}", f"layout:{layout}"]
    for tag in required:
        if tag not in tags:
            tags.append(tag)
    return tags
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_observability.TestObservabilityTags -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add testing/test_observability.py src/overcooked_ai_py/agents/llm/observability.py
git commit -m "feat: enforce mode and layout tags for observability"
```

### Task 3: Add Optional LangFuse Reporter and Invoke Config Builder

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/observability.py`
- Modify: `testing/test_observability.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Test: `testing/test_observability.py`

**Step 1: Write the failing tests**

```python
@patch("overcooked_ai_py.agents.llm.observability.CallbackHandler")
def test_build_invoke_config_includes_callback(self, mock_handler):
    reporter = LangFuseReporter(enabled=True, context=self.ctx)
    cfg = reporter.build_invoke_config({"recursion_limit": 15})
    self.assertIn("callbacks", cfg)
    self.assertEqual(cfg["recursion_limit"], 15)

def test_langfuse_reporter_disabled_is_noop(self):
    reporter = LangFuseReporter(enabled=False, context=self.ctx)
    cfg = reporter.build_invoke_config({})
    self.assertEqual(cfg, {})
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_observability.TestLangFuseReporter -v`  
Expected: FAIL with missing `LangFuseReporter`.

**Step 3: Write minimal implementation**

```python
class LangFuseReporter:
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
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_observability.TestLangFuseReporter -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add testing/test_observability.py src/overcooked_ai_py/agents/llm/observability.py pyproject.toml .env.example
git commit -m "feat: add optional LangFuse reporter and callback config plumbing"
```

### Task 4: Integrate Runner Metadata Flags and Run Lifecycle Events

**Files:**
- Modify: `scripts/run_llm_agent.py`
- Modify: `testing/test_observability.py`
- Test: `testing/test_observability.py`

**Step 1: Write the failing tests**

```python
def test_build_run_context_uses_defaults(self):
    args = Namespace(run_name=None, run_title="", tags="", experiment="default-exp", variant="baseline", notes="")
    ctx = build_run_context(args, mode="llm", layout="cramped_room", model="gpt-4o")
    self.assertEqual(ctx.mode, "llm")
    self.assertIn("mode:llm", ctx.tags)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_observability.TestRunContextFactory -v`  
Expected: FAIL with missing `build_run_context`.

**Step 3: Write minimal implementation**

```python
parser.add_argument("--run-name", default=None)
parser.add_argument("--run-title", default="")
parser.add_argument("--tags", default="")
parser.add_argument("--experiment", default="default-exp")
parser.add_argument("--variant", default="baseline")
parser.add_argument("--notes", default="")

ctx = build_run_context(args, mode=args.agent_type, layout=args.layout, model=model)
sink.emit("run.start", {"horizon": args.horizon})
...
sink.emit("run.end", {"total_reward": total_reward, "steps": step, "elapsed_s": elapsed})
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_observability.TestRunContextFactory -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/run_llm_agent.py testing/test_observability.py
git commit -m "feat: add run metadata flags and lifecycle observability events"
```

### Task 5: Instrument Shared Graph Builder for LLM/Tool/Error Events

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/graph_builder.py`
- Modify: `testing/test_graph_builder.py`
- Test: `testing/test_graph_builder.py`

**Step 1: Write the failing tests**

```python
@patch("overcooked_ai_py.agents.llm.graph_builder.ChatLiteLLM")
def test_observability_receives_llm_generation_event(self, mock_llm_class):
    from langchain_core.messages import AIMessage

    sink = MagicMock()
    mock_llm_instance = MagicMock()
    mock_llm_class.return_value = mock_llm_instance
    mock_llm_with_tools = MagicMock()
    mock_llm_instance.bind_tools.return_value = mock_llm_with_tools
    mock_llm_with_tools.invoke.return_value = AIMessage(content="Planner reasoning", tool_calls=[])

    graph = build_react_graph(
        model_name=self.model_name,
        system_prompt=self.system_prompt,
        observation_tools=self.observation_tools,
        action_tools=self.action_tools,
        action_tool_names=self.action_tool_names,
        get_chosen_fn=lambda: self.action_chosen,
        debug=False,
        observability=sink,
        role_name="planner",
    )
    graph.invoke({"messages": [("user", "x")]})
    sink.emit.assert_any_call("llm.generation", unittest.mock.ANY, step=None, agent_role="planner")

@patch("overcooked_ai_py.agents.llm.graph_builder.ChatLiteLLM")
def test_observability_receives_tool_call_event(self, mock_llm_class):
    from langchain_core.messages import AIMessage

    sink = MagicMock()
    mock_llm_instance = MagicMock()
    mock_llm_class.return_value = mock_llm_instance
    mock_llm_with_tools = MagicMock()
    mock_llm_instance.bind_tools.return_value = mock_llm_with_tools
    mock_llm_with_tools.invoke.return_value = AIMessage(
        content="Calling action tool",
        tool_calls=[{"name": "do_action", "args": {}, "id": "1"}],
    )

    graph = build_react_graph(
        model_name=self.model_name,
        system_prompt=self.system_prompt,
        observation_tools=self.observation_tools,
        action_tools=self.action_tools,
        action_tool_names=self.action_tool_names,
        get_chosen_fn=lambda: self.action_chosen,
        debug=False,
        observability=sink,
        role_name="planner",
    )
    graph.invoke({"messages": [("user", "x")]})
    sink.emit.assert_any_call("tool.call", unittest.mock.ANY, step=None, agent_role="planner")
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_graph_builder.TestGraphBuilder -v`  
Expected: FAIL with unexpected keyword `observability` in `build_react_graph`.

**Step 3: Write minimal implementation**

```python
def build_react_graph(
    model_name: str,
    system_prompt: str,
    observation_tools: list,
    action_tools: list,
    action_tool_names: set[str],
    get_chosen_fn: Callable,
    debug: bool = False,
    debug_prefix: str = "[LLM]",
    api_base: str = None,
    api_key: str = None,
    llm_timeout_seconds: float = 35.0,
    observability=None,
    role_name: str = "llm",
):
    # existing graph setup...
    if observability:
        observability.emit(
            "llm.generation",
            {"content_preview": (response.content or "")[:200]},
            step=None,
            agent_role=role_name,
        )

    for tc in last_message.tool_calls:
        if observability:
            observability.emit(
                "tool.call",
                {"tool_name": tc["name"], "args": tc.get("args", {})},
                step=None,
                agent_role=role_name,
            )
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_graph_builder.TestGraphBuilder -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/graph_builder.py testing/test_graph_builder.py
git commit -m "feat: emit llm and tool observability events from graph builder"
```

### Task 6: Wire Planner/Worker/LLMAgent/Legacy Graph to Observability + Callback Config

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner.py`
- Modify: `src/overcooked_ai_py/agents/llm/worker_agent.py`
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`
- Modify: `src/overcooked_ai_py/agents/llm/graph.py`
- Modify: `testing/test_planner.py`
- Modify: `testing/test_worker_agent_unit.py`
- Modify: `testing/llm_agent_test.py`
- Test: `testing/test_planner.py`
- Test: `testing/test_worker_agent_unit.py`
- Test: `testing/llm_agent_test.py`

**Step 1: Write the failing tests**

```python
def test_planner_emits_assignment_event(self):
    state = self.mdp.get_standard_start_state()
    planner = Planner(
        model_name="gpt-4o",
        observability=self.sink,
        invoke_config={"callbacks": ["dummy-callback"]},
    )
    planner._graph = Mock()
    planner._graph.invoke.return_value = {"messages": []}
    planner.maybe_replan(state)
    self.sink.emit.assert_any_call("planner.assignment", unittest.mock.ANY, step=state.timestep, agent_role="planner")

def test_worker_emits_action_commit(self):
    state = self.mdp.get_standard_start_state()
    worker = WorkerAgent(self.planner, "worker_0", model_name="gpt-4o")
    worker._graph = Mock()
    worker._tool_state.set_action(Action.STAY)
    worker.observability = self.sink
    worker.action(state)
    self.sink.emit.assert_any_call("action.commit", unittest.mock.ANY, step=state.timestep, agent_role="worker_0")
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_planner testing.test_worker_agent_unit testing.llm_agent_test -v`  
Expected: FAIL due to new constructor args/events not implemented.

**Step 3: Write minimal implementation**

```python
class Planner:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        replan_interval: int = 5,
        debug: bool = False,
        horizon: Optional[int] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        observability=None,
        invoke_config: Optional[dict] = None,
    ):
        self.observability = observability
        self.invoke_config = invoke_config or {}

self._graph.invoke({"messages": messages}, config={**self.invoke_config, "recursion_limit": 20})
```

```python
if self.observability:
    self.observability.emit("action.commit", {"action": action_name}, step=state.timestep, agent_role=self.worker_id)
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_planner testing.test_worker_agent_unit testing.llm_agent_test -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner.py src/overcooked_ai_py/agents/llm/worker_agent.py src/overcooked_ai_py/agents/llm/llm_agent.py src/overcooked_ai_py/agents/llm/graph.py testing/test_planner.py testing/test_worker_agent_unit.py testing/llm_agent_test.py
git commit -m "feat: wire observability and callback config through llm agents"
```

### Task 7: Verification, Docs, and Smoke Runs

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/README_PLANNER_WORKER.md`
- Modify: `README.md` (if runner flags are documented there)
- Test: `testing/test_observability.py`

**Step 1: Write failing doc/CLI test**

```python
def test_cli_help_includes_observability_flags(self):
    import subprocess

    proc = subprocess.run(
        ["uv", "run", "python", "scripts/run_llm_agent.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = proc.stdout + proc.stderr
    for flag in [
        "--run-name",
        "--run-title",
        "--tags",
        "--experiment",
        "--variant",
        "--notes",
    ]:
        self.assertIn(flag, help_text)
```

**Step 2: Run test to verify it fails/passes at the right phase**

Run: `uv run python -m unittest testing.test_observability.TestRunScriptCli -v`  
Expected:
- FAIL before Task 4 implementation is complete
- PASS after Task 4 implementation is complete

**Step 3: Run full validation**

Run:
- `uv run python -m unittest testing.test_observability testing.test_graph_builder testing.test_planner testing.test_worker_agent_unit testing.llm_agent_test -v`
- `uv run python scripts/run_llm_agent.py --agent-type llm --horizon 20 --run-name smoke-llm --tags bench,smoke`
- `uv run python scripts/run_llm_agent.py --agent-type planner-worker --horizon 20 --run-name smoke-pw --tags bench,smoke`

Expected:
- All tests PASS
- Two new files under `logs/agent_runs/`
- File events include `mode:llm` and `mode:planner-worker`
- Runs succeed with and without LangFuse keys present

**Step 4: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/README_PLANNER_WORKER.md README.md
git commit -m "docs: add observability usage and verification notes"
```

### Final Verification Gate

Before declaring completion:

1. Re-run the full unit test command.
2. Re-run both smoke scripts.
3. Confirm JSONL log schema is consistent and parseable.
4. Confirm no edits were made outside intended LLM-agent and runner surfaces.
