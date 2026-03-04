import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
from argparse import Namespace

from overcooked_ai_py.agents.llm.observability import (
    FileRunLogger,
    LangFuseReporter,
    ObservabilityHub,
    RunContext,
    _extract_model_from_llm_result,
    _extract_usage_from_llm_result,
    _normalize_usage,
    build_run_context,
    estimate_model_cost_usd,
    get_model_cost_rates,
    normalize_model_name,
    normalize_tags,
)


class TestObservabilityCore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_file_logger_creates_run_file(self):
        ctx = RunContext(
            run_id="r1",
            run_name="bench-a",
            mode="llm",
            layout="cramped_room",
            model="gpt-4o",
        )
        logger = FileRunLogger(base_dir=self.tmpdir, context=ctx)
        logger.emit("run.start", {"x": 1})
        self.assertTrue(logger.file_path.exists())

    def test_event_contains_common_fields(self):
        ctx = RunContext(
            run_id="r2",
            run_name="bench-b",
            mode="planner-worker",
            layout="cramped_room",
            model="gpt-4o",
        )
        logger = FileRunLogger(base_dir=self.tmpdir, context=ctx)
        logger.emit("action.commit", {"action": "move_up"}, step=4, agent_role="worker_0")
        row = json.loads(logger.file_path.read_text().splitlines()[0])
        self.assertEqual(row["run_id"], "r2")
        self.assertEqual(row["event_type"], "action.commit")
        self.assertEqual(row["mode"], "planner-worker")

    def test_hub_emits_to_file_and_langfuse(self):
        ctx = RunContext(
            run_id="r_hub",
            run_name="bench-hub",
            mode="planner-worker",
            layout="cramped_room",
            model="gpt-4o",
        )
        logger = FileRunLogger(base_dir=self.tmpdir, context=ctx)
        langfuse = Mock()
        hub = ObservabilityHub(file_logger=logger, langfuse=langfuse)

        hub.emit(
            "tool.call",
            {"tool_name": "check_path", "args": {"target": "pot"}},
            step=2,
            agent_role="worker_0",
        )

        row = json.loads(logger.file_path.read_text().splitlines()[0])
        self.assertEqual(row["event_type"], "tool.call")
        self.assertEqual(row["step"], 2)
        self.assertEqual(row["agent_role"], "worker_0")
        langfuse.emit_event.assert_called_once_with(
            "tool.call",
            {"tool_name": "check_path", "args": {"target": "pot"}},
            step=2,
            agent_role="worker_0",
        )

    def test_hub_uses_active_step_for_events_without_step(self):
        ctx = RunContext(
            run_id="r_hub2",
            run_name="bench-hub2",
            mode="planner-worker",
            layout="cramped_room",
            model="gpt-4o",
        )
        logger = FileRunLogger(base_dir=self.tmpdir, context=ctx)
        langfuse = Mock()
        hub = ObservabilityHub(file_logger=logger, langfuse=langfuse)

        hub.start_step(7)
        hub.emit("llm.generation", {"content": "x"}, agent_role="planner")
        hub.end_step()

        row = json.loads(logger.file_path.read_text().splitlines()[0])
        self.assertEqual(row["event_type"], "llm.generation")
        self.assertEqual(row["step"], 7)
        langfuse.emit_event.assert_called_once_with(
            "llm.generation",
            {"content": "x"},
            step=7,
            agent_role="planner",
        )

class TestObservabilityTags(unittest.TestCase):
    def test_required_mode_tag_added_for_llm(self):
        tags = normalize_tags(["exp:bench1"], mode="llm", layout="cramped_room")
        self.assertIn("mode:llm", tags)
        self.assertIn("layout:cramped_room", tags)

    def test_required_mode_tag_added_for_planner_worker(self):
        tags = normalize_tags([], mode="planner-worker", layout="coordination_ring")
        self.assertIn("mode:planner-worker", tags)
        self.assertIn("layout:coordination_ring", tags)


class TestLangFuseReporter(unittest.TestCase):
    def setUp(self):
        self.ctx = RunContext(
            run_id="r3",
            run_name="bench-c",
            mode="llm",
            layout="cramped_room",
            model="gpt-4o",
        )

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse", create=True)
    def test_build_invoke_config_disables_graph_callbacks(self, mock_langfuse):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        cfg = reporter.build_invoke_config({"recursion_limit": 15})
        self.assertEqual(cfg, {"recursion_limit": 15})
        self.assertIsNotNone(reporter._client)
        mock_langfuse.assert_called_once_with()

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse", create=True)
    def test_start_run_sets_trace_tags(self, mock_langfuse):
        ctx = RunContext(
            run_id="abcdefabcdefabcdefabcdefabcdefab",
            run_name="tag-test",
            mode="planner-worker",
            layout="cramped_room",
            model="openai/moonshotai.kimi-k2.5",
            tags=["bench", "custom-hierarchy", "mode:planner-worker"],
        )
        reporter = LangFuseReporter(enabled=True, context=ctx)
        reporter.start_run()

        mock_langfuse.return_value.start_span.assert_called_once_with(
            name="tag-test",
            metadata=unittest.mock.ANY,
        )
        trace = mock_langfuse.return_value.start_span.return_value
        self.assertEqual(reporter.get_trace_id(), trace.trace_id)
        trace.update_trace.assert_called_once_with(
            name="tag-test",
            tags=["bench", "custom-hierarchy", "mode:planner-worker"],
            session_id="abcdefabcdefabcdefabcdefabcdefab",
            metadata=unittest.mock.ANY,
        )

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse", create=True)
    def test_start_run_enqueues_ingestion_tags(self, mock_langfuse):
        ctx = RunContext(
            run_id="abcdefabcdefabcdefabcdefabcdefab",
            run_name="tag-test-ingestion",
            mode="planner-worker",
            layout="cramped_room",
            model="openai/moonshotai.kimi-k2.5",
            tags=["bench", "tags-fix"],
        )
        reporter = LangFuseReporter(enabled=True, context=ctx)
        reporter.start_run()

        mock_langfuse.return_value._create_trace_tags_via_ingestion.assert_called_once_with(
            trace_id=mock_langfuse.return_value.start_span.return_value.trace_id,
            tags=["bench", "tags-fix"],
        )

    def test_langfuse_reporter_disabled_is_noop(self):
        reporter = LangFuseReporter(enabled=False, context=self.ctx)
        cfg = reporter.build_invoke_config({})
        self.assertEqual(cfg, {})
        self.assertIsNone(reporter._client)

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse", create=True)
    def test_langfuse_client_init_failure_is_noop(self, mock_langfuse):
        mock_langfuse.side_effect = RuntimeError("init failed")
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        cfg = reporter.build_invoke_config({"recursion_limit": 15})
        self.assertEqual(cfg, {"recursion_limit": 15})
        self.assertIsNone(reporter._client)


class TestLangFuseHierarchyReporter(unittest.TestCase):
    def setUp(self):
        self.ctx = RunContext(
            run_id="abcdefabcdefabcdefabcdefabcdefab",
            run_name="bench-hierarchy",
            mode="planner-worker",
            layout="cramped_room",
            model="openai/moonshotai.kimi-k2.5",
        )

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse", create=True)
    def test_start_step_then_role_creates_nested_spans(self, mock_langfuse):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        reporter.start_run()
        reporter.start_step(3)
        reporter.start_role("worker_0")
        reporter.end_role()
        reporter.end_step()
        reporter.end_run({"steps": 5})

        client = mock_langfuse.return_value
        run_span = client.start_span.return_value
        step_span = run_span.start_span.return_value
        role_span = step_span.start_span.return_value

        client.start_span.assert_called_once()
        run_span.start_span.assert_any_call(name="step_3", metadata=unittest.mock.ANY)
        step_span.start_span.assert_called_once_with(
            name="worker_0",
            metadata=unittest.mock.ANY,
        )
        role_span.end.assert_called_once()
        step_span.end.assert_called_once()
        run_span.end.assert_called_once()

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse", create=True)
    def test_emit_llm_generation_creates_generation_with_cost(self, mock_langfuse):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        reporter.start_run()
        reporter.start_step(0)
        reporter.start_role("planner")
        reporter.emit_event(
            "llm.generation",
            {
                "content": "reasoning",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "estimated_cost_usd": 0.00001,
            },
            step=0,
            agent_role="planner",
        )

        role_span = (
            mock_langfuse.return_value.start_span.return_value.start_span.return_value.start_span.return_value
        )
        generation = role_span.start_generation.return_value
        call_kwargs = role_span.start_generation.call_args.kwargs
        self.assertEqual(call_kwargs["name"], "llm.generation")
        self.assertEqual(call_kwargs["model"], "moonshotai.kimi-k2.5")
        self.assertEqual(call_kwargs["usage_details"], {"input": 10, "output": 2, "total": 12})
        self.assertIn("total", call_kwargs["cost_details"])
        generation.end.assert_called_once()

    @patch("overcooked_ai_py.agents.llm.observability.Langfuse", create=True)
    def test_emit_tool_call_creates_tool_span(self, mock_langfuse):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        reporter.start_run()
        reporter.start_step(1)
        reporter.start_role("worker_1")
        reporter.emit_event(
            "tool.call",
            {"tool_name": "check_path", "args": {"target": "pot"}},
            step=1,
            agent_role="worker_1",
        )

        role_span = (
            mock_langfuse.return_value.start_span.return_value.start_span.return_value.start_span.return_value
        )
        tool_span = role_span.start_span.return_value
        role_span.start_span.assert_called_with(
            name="check_path",
            input={"target": "pot"},
            metadata=unittest.mock.ANY,
        )
        tool_span.end.assert_called_once()


class TestRunContextFactory(unittest.TestCase):
    def test_build_run_context_uses_defaults(self):
        args = Namespace(
            run_name=None,
            run_title="",
            tags="",
            experiment="default-exp",
            variant="baseline",
            notes="",
        )
        ctx = build_run_context(args, mode="llm", layout="cramped_room", model="gpt-4o")
        self.assertEqual(ctx.mode, "llm")
        self.assertIn("mode:llm", ctx.tags)


class TestObservabilityCosts(unittest.TestCase):
    def test_normalize_model_name(self):
        self.assertEqual(
            normalize_model_name("openai/moonshotai.kimi-k2.5"),
            "moonshotai.kimi-k2.5",
        )
        self.assertEqual(normalize_model_name("moonshotai.kimi-k2.5"), "moonshotai.kimi-k2.5")

    def test_estimate_model_cost(self):
        rates = get_model_cost_rates("openai/moonshotai.kimi-k2.5")
        self.assertEqual(rates, {"input": 0.6, "output": 3.03})
        cost = estimate_model_cost_usd(
            "openai/moonshotai.kimi-k2.5",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
        self.assertAlmostEqual(cost, 2.115, places=6)

    def test_normalize_usage_extracts_tokens(self):
        usage = _normalize_usage({"prompt_tokens": 120, "completion_tokens": 30})
        self.assertEqual(usage, {"input": 120, "output": 30, "total": 150})

    def test_extract_usage_from_llm_result(self):
        class Dummy:
            llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}}
            generations = []

        usage = _extract_usage_from_llm_result(Dummy())
        self.assertEqual(usage, {"input": 100, "output": 20, "total": 120})

    def test_extract_model_from_llm_result(self):
        class Dummy:
            llm_output = {"model_name": "openai/moonshotai.kimi-k2.5"}

        model = _extract_model_from_llm_result(Dummy())
        self.assertEqual(model, "moonshotai.kimi-k2.5")


class TestRunScriptCli(unittest.TestCase):
    def test_cli_help_includes_observability_flags(self):
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

    @patch("scripts.run_llm_agent.time.time", side_effect=[100.0, 101.0])
    @patch("scripts.run_llm_agent.ObservabilityHub")
    @patch("scripts.run_llm_agent.LangFuseReporter")
    @patch("scripts.run_llm_agent.FileRunLogger")
    @patch("scripts.run_llm_agent.AgentPair")
    @patch("scripts.run_llm_agent.make_greedy_partner")
    @patch("scripts.run_llm_agent.LLMAgent")
    @patch("scripts.run_llm_agent.OvercookedEnv.from_mdp")
    @patch("scripts.run_llm_agent.OvercookedGridworld.from_layout_name")
    @patch("scripts.run_llm_agent.argparse.ArgumentParser.parse_args")
    def test_runner_step_scope_order(
        self,
        mock_parse_args,
        mock_from_layout,
        mock_env_from_mdp,
        mock_llm_agent,
        mock_make_partner,
        mock_agent_pair,
        mock_file_logger,
        mock_langfuse_reporter,
        mock_hub_cls,
        _mock_time,
    ):
        import scripts.run_llm_agent as run_script

        mock_parse_args.return_value = Namespace(
            model="gpt-4o",
            layout="cramped_room",
            horizon=1,
            debug=False,
            visualize=False,
            fps=2,
            agent_type="llm",
            replan_interval=5,
            run_name="scope-order-test",
            run_title="",
            tags="",
            experiment="default-exp",
            variant="baseline",
            notes="",
        )

        mdp = Mock()
        mock_from_layout.return_value = mdp

        state = SimpleNamespace(timestep=0, players=[])
        env = Mock()
        env.state = state
        env.step.return_value = (state, 0, True, {})
        mock_env_from_mdp.return_value = env

        pair = Mock()
        pair.joint_action.return_value = ((Mock(), {}), (Mock(), {}))
        mock_agent_pair.return_value = pair

        mock_make_partner.return_value = Mock()
        mock_llm_agent.return_value = Mock()

        hub = Mock()
        hub.build_invoke_config.return_value = {}
        mock_hub_cls.return_value = hub

        with patch.dict(os.environ, {}, clear=True):
            run_script.main()

        expected_calls = [
            call.emit("run.start", {"horizon": 1}, step=None, agent_role="runner"),
            call.start_run(),
            call.start_step(0),
            call.end_step(),
            call.emit(
                "run.end",
                {"total_reward": 0, "steps": 1, "elapsed_s": 1.0},
                step=None,
                agent_role="runner",
            ),
            call.end_run({"total_reward": 0, "steps": 1, "elapsed_s": 1.0}),
        ]
        start_idx = hub.method_calls.index(expected_calls[0])
        self.assertEqual(hub.method_calls[start_idx:start_idx + 6], expected_calls)
        hub.start_run.assert_called_once_with()
        hub.start_step.assert_called_once_with(0)
        hub.end_step.assert_called_once_with()
        hub.end_run.assert_called_once_with({"total_reward": 0, "steps": 1, "elapsed_s": 1.0})
