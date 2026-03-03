import json
import tempfile
import unittest
from unittest.mock import patch

from overcooked_ai_py.agents.llm.observability import (
    FileRunLogger,
    LangFuseReporter,
    RunContext,
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

    def test_langfuse_reporter_disabled_is_noop(self):
        reporter = LangFuseReporter(enabled=False, context=self.ctx)
        cfg = reporter.build_invoke_config({})
        self.assertEqual(cfg, {})
