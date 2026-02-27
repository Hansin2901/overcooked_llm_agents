import unittest
from overcooked_ai_py.agents.llm.task import Task


class TestTask(unittest.TestCase):

    def test_creation_with_required_fields(self):
        task = Task(description="Pick up an onion", worker_id="worker_0", created_at=5)
        self.assertEqual(task.description, "Pick up an onion")
        self.assertEqual(task.worker_id, "worker_0")
        self.assertEqual(task.created_at, 5)

    def test_default_values(self):
        task = Task(description="Deliver soup", worker_id="worker_1", created_at=0)
        self.assertFalse(task.completed)
        self.assertEqual(task.steps_active, 0)

    def test_creation_with_all_fields(self):
        task = Task(
            description="Place onion in pot",
            worker_id="worker_0",
            created_at=10,
            completed=True,
            steps_active=7,
        )
        self.assertTrue(task.completed)
        self.assertEqual(task.steps_active, 7)

    def test_mutation(self):
        task = Task(description="Get dish", worker_id="worker_1", created_at=3)
        task.completed = True
        task.steps_active = 12
        self.assertTrue(task.completed)
        self.assertEqual(task.steps_active, 12)


if __name__ == "__main__":
    unittest.main()
