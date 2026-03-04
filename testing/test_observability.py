import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

from overcooked_ai_py.agents.llm.observability import (
    FileRunLogger,
    LangFuseReporter,
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

    @patch("overcooked_ai_py.agents.llm.observability.CallbackHandler")
    def test_build_invoke_config_includes_callback(self, mock_handler):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        cfg = reporter.build_invoke_config({"recursion_limit": 15})
        self.assertIn("callbacks", cfg)
        self.assertEqual(cfg["recursion_limit"], 15)

    @patch("overcooked_ai_py.agents.llm.observability.CallbackHandler")
    def test_build_invoke_config_sets_run_name_and_metadata(self, mock_handler):
        ctx = RunContext(
            run_id="r3",
            run_name="bench-c",
            mode="llm",
            layout="cramped_room",
            model="openai/moonshotai.kimi-k2.5",
        )
        reporter = LangFuseReporter(enabled=True, context=ctx)
        cfg = reporter.build_invoke_config({"recursion_limit": 15})
        self.assertEqual(cfg["run_name"], "bench-c")
        self.assertEqual(cfg["metadata"]["langfuse_session_id"], "r3")
        self.assertEqual(cfg["metadata"]["langfuse_tags"], ctx.tags)
        self.assertEqual(cfg["metadata"]["ls_model_name"], "moonshotai.kimi-k2.5")
        self.assertEqual(cfg["metadata"]["model_cost_input_usd_per_million"], 0.6)
        self.assertEqual(cfg["metadata"]["model_cost_output_usd_per_million"], 3.03)

    @patch("overcooked_ai_py.agents.llm.observability.CallbackHandler")
    def test_prefers_current_callback_signature(self, mock_handler):
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        self.assertIsNotNone(reporter._callback)
        mock_handler.assert_called_once_with(
            update_trace=True,
            trace_context={"trace_id": "r3"},
        )

    def test_langfuse_reporter_disabled_is_noop(self):
        reporter = LangFuseReporter(enabled=False, context=self.ctx)
        cfg = reporter.build_invoke_config({})
        self.assertEqual(cfg, {})

    @patch("overcooked_ai_py.agents.llm.observability.CallbackHandler")
    def test_langfuse_handler_init_failure_is_noop(self, mock_handler):
        mock_handler.side_effect = RuntimeError("init failed")
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        cfg = reporter.build_invoke_config({"recursion_limit": 15})
        self.assertEqual(cfg, {"recursion_limit": 15})

    @patch("overcooked_ai_py.agents.llm.observability.CallbackHandler")
    def test_langfuse_handler_signature_fallback(self, mock_handler):
        callback_obj = object()
        calls = []

        def _factory(*args, **kwargs):
            calls.append((args, kwargs))
            if kwargs:
                raise TypeError("unexpected keyword")
            return callback_obj

        mock_handler.side_effect = _factory
        reporter = LangFuseReporter(enabled=True, context=self.ctx)
        cfg = reporter.build_invoke_config({"recursion_limit": 15})
        self.assertIs(reporter._callback, callback_obj)
        self.assertIn("callbacks", cfg)
        self.assertGreaterEqual(len(calls), 2)


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
