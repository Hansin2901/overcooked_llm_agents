import json
import tempfile
import unittest

from overcooked_ai_py.agents.llm.observability import FileRunLogger, RunContext


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
