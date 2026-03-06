"""Tests for planner and worker system prompts in state_serializer.py"""

import unittest

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.agents.llm.state_serializer import (
    build_planner_system_prompt,
    build_worker_system_prompt,
)


class TestPlannerSystemPrompt(unittest.TestCase):
    """Test planner system prompt generation."""

    def setUp(self):
        """Set up test environment with cramped_room layout."""
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.worker_ids = ["worker_0", "worker_1"]

    def test_planner_prompt_includes_all_worker_ids(self):
        """Planner prompt should list all worker IDs."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids, horizon=200)

        for worker_id in self.worker_ids:
            self.assertIn(
                worker_id, prompt, f"Planner prompt should include {worker_id}"
            )

    def test_planner_prompt_mentions_task_assignment(self):
        """Planner prompt should emphasize task assignment role."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)

        # Check for key task assignment concepts
        task_keywords = [
            "assign",
            "task",
            "coordinate",
            "PLANNER",
        ]

        for keyword in task_keywords:
            self.assertIn(
                keyword.lower(),
                prompt.lower(),
                f"Planner prompt should mention '{keyword}'",
            )

    def test_planner_prompt_mentions_no_worker_communication(self):
        """Planner prompt should emphasize workers can't communicate."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)

        # Workers cannot communicate is a critical constraint
        self.assertIn(
            "CANNOT communicate",
            prompt,
            "Planner should know workers can't communicate",
        )
        self.assertIn(
            "independently",
            prompt.lower(),
            "Planner should understand workers work independently",
        )

    def test_planner_prompt_clarifies_cooking_start(self):
        """Planner prompt should clarify that full pots need cooking start action."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)
        self.assertIn("NOT ready soup", prompt)
        self.assertIn("INTERACT", prompt)

    def test_planner_prompt_includes_layout_info(self):
        """Planner prompt should include layout for cramped_room."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)

        # Check for layout section
        self.assertIn("LAYOUT:", prompt, "Prompt should have LAYOUT section")
        self.assertIn("Legend:", prompt, "Prompt should have legend")

        # Check for key location types
        self.assertIn("KEY LOCATIONS:", prompt, "Prompt should list key locations")

    def test_planner_prompt_includes_coordination_strategy(self):
        """Planner prompt should include strategy guidance."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)

        strategy_keywords = [
            "strategy",
            "coordination",
            "efficiency",
            "divide",
        ]

        for keyword in strategy_keywords:
            self.assertIn(
                keyword.lower(),
                prompt.lower(),
                f"Planner should have guidance about '{keyword}'",
            )

    def test_planner_prompt_includes_horizon(self):
        """Planner prompt should mention horizon if provided."""
        horizon = 400
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids, horizon=horizon)

        self.assertIn(str(horizon), prompt, "Prompt should include horizon value")
        self.assertIn("timesteps", prompt.lower(), "Prompt should mention timesteps")

    def test_planner_prompt_with_different_worker_counts(self):
        """Planner prompt should work with different numbers of workers."""
        # Test with 3 workers
        worker_ids_3 = ["worker_0", "worker_1", "worker_2"]
        prompt = build_planner_system_prompt(self.mdp, worker_ids_3)

        for worker_id in worker_ids_3:
            self.assertIn(worker_id, prompt)

        # Test with 1 worker (edge case)
        worker_ids_1 = ["worker_0"]
        prompt = build_planner_system_prompt(self.mdp, worker_ids_1)
        self.assertIn("worker_0", prompt)

    def test_planner_prompt_no_action_tools_mentioned(self):
        """Planner shouldn't mention low-level action tools."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)

        # Planner doesn't execute actions directly
        self.assertNotIn(
            "move_up",
            prompt.lower(),
            "Planner shouldn't mention specific movement actions",
        )
        self.assertNotIn(
            "action tool", prompt.lower(), "Planner doesn't use action tools directly"
        )

    def test_planner_prompt_forbids_completion_assumptions(self):
        """Planner prompt should forbid assuming task completion."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)
        self.assertIn("Do not assume a worker will complete", prompt)
        self.assertIn("current observed state", prompt)

    def test_planner_prompt_includes_state_grounded_constraints(self):
        """Planner prompt should include all state-grounded planning constraints."""
        prompt = build_planner_system_prompt(self.mdp, self.worker_ids)

        # Check for all 4 key constraints
        self.assertIn("Plan based ONLY on current observed state", prompt)
        self.assertIn("Re-evaluate worker positions and inventory", prompt)
        self.assertIn("Assign both workers every replan", prompt)
        self.assertIn("Workers may get blocked, distracted", prompt)


class TestWorkerSystemPrompt(unittest.TestCase):
    """Test worker system prompt generation."""

    def setUp(self):
        """Set up test environment with cramped_room layout."""
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.agent_index = 0
        self.worker_id = "worker_0"

    def test_worker_prompt_mentions_assigned_task_execution(self):
        """Worker prompt should emphasize executing assigned tasks."""
        prompt = build_worker_system_prompt(
            self.mdp, self.agent_index, self.worker_id, horizon=200
        )

        task_keywords = [
            "task assigned",
            "execute",
            "assigned to you",
            "current task",
        ]

        # At least some of these should be present
        found_keywords = [kw for kw in task_keywords if kw.lower() in prompt.lower()]
        self.assertGreater(
            len(found_keywords),
            0,
            f"Worker prompt should mention task execution. Found: {found_keywords}",
        )

    def test_worker_prompt_does_not_mention_other_workers(self):
        """Worker prompt should NOT mention other workers or coordination."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)

        # Worker shouldn't know about the multi-worker system
        self.assertNotIn("worker_1", prompt, "Worker shouldn't see other worker IDs")
        self.assertNotIn(
            "other workers", prompt.lower(), "Worker shouldn't know about other workers"
        )
        self.assertNotIn(
            "coordinate with", prompt.lower(), "Worker doesn't coordinate with others"
        )
        self.assertNotIn(
            "planner",
            prompt.lower(),
            "Worker shouldn't mention the planner role explicitly",
        )

    def test_worker_prompt_includes_layout_info(self):
        """Worker prompt should include layout for cramped_room."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)

        # Check for layout section
        self.assertIn("LAYOUT:", prompt, "Prompt should have LAYOUT section")
        self.assertIn("Legend:", prompt, "Prompt should have legend")

        # Check for key location types
        self.assertIn("KEY LOCATIONS:", prompt, "Prompt should list key locations")

    def test_worker_prompt_includes_worker_id(self):
        """Worker prompt should identify the worker by their ID."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)

        self.assertIn(self.worker_id, prompt, "Worker should know their own ID")

    def test_worker_prompt_includes_agent_index(self):
        """Worker prompt should mention which player they are."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)

        self.assertIn(
            f"Player {self.agent_index}",
            prompt,
            "Worker should know their player index",
        )

    def test_worker_prompt_includes_action_guide(self):
        """Worker prompt should include detailed action instructions."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)

        # Check for action guidance
        action_keywords = [
            "ACTION GUIDE",
            "pick up",
            "INTERACT",
            "stand adjacent",
            "face it",
        ]

        for keyword in action_keywords:
            self.assertIn(
                keyword, prompt, f"Worker should have guidance about '{keyword}'"
            )

    def test_worker_prompt_clarifies_cooking_start(self):
        """Worker prompt should clarify full pot vs ready soup."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)
        self.assertIn("FULL but idle", prompt)
        self.assertIn("INTERACT with empty hands", prompt)
        self.assertIn("NEVER try to pick up soup unless pot status says READY", prompt)

    def test_worker_prompt_mentions_partner_entity(self):
        """Worker should know about the @ entity but not that it's another worker."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)

        # Should mention the @ entity for navigation
        self.assertIn("@", prompt, "Worker should see @ in the environment")
        self.assertIn(
            "entity",
            prompt.lower(),
            "Worker should be aware of other entity for navigation",
        )

    def test_worker_prompt_includes_navigation_tips(self):
        """Worker prompt should include navigation guidance."""
        prompt = build_worker_system_prompt(self.mdp, self.agent_index, self.worker_id)

        nav_keywords = [
            "navigate",
            "blocked",
            "facing",
        ]

        for keyword in nav_keywords:
            self.assertIn(
                keyword.lower(),
                prompt.lower(),
                f"Worker should have navigation tip about '{keyword}'",
            )

    def test_worker_prompt_with_different_indices(self):
        """Worker prompt should work for both player 0 and player 1."""
        # Test player 0
        prompt_0 = build_worker_system_prompt(self.mdp, 0, "worker_0")
        self.assertIn("Player 0", prompt_0)
        self.assertIn("worker_0", prompt_0)

        # Test player 1
        prompt_1 = build_worker_system_prompt(self.mdp, 1, "worker_1")
        self.assertIn("Player 1", prompt_1)
        self.assertIn("worker_1", prompt_1)

    def test_worker_prompt_includes_horizon(self):
        """Worker prompt should mention horizon if provided."""
        horizon = 400
        prompt = build_worker_system_prompt(
            self.mdp, self.agent_index, self.worker_id, horizon=horizon
        )

        self.assertIn(str(horizon), prompt, "Prompt should include horizon value")
        self.assertIn("timesteps", prompt.lower(), "Prompt should mention timesteps")


class TestPromptConsistency(unittest.TestCase):
    """Test consistency between planner and worker prompts."""

    def setUp(self):
        """Set up test environment."""
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.worker_ids = ["worker_0", "worker_1"]
        self.horizon = 200

    def test_both_prompts_include_same_layout(self):
        """Planner and worker should see the same layout."""
        planner_prompt = build_planner_system_prompt(
            self.mdp, self.worker_ids, horizon=self.horizon
        )
        worker_prompt = build_worker_system_prompt(
            self.mdp, 0, "worker_0", horizon=self.horizon
        )

        # Both should have layout section
        self.assertIn("LAYOUT:", planner_prompt)
        self.assertIn("LAYOUT:", worker_prompt)

        # Extract layout grids (simple check - both should have same dimensions)
        # This is a basic check; full grid comparison would be more complex
        self.assertIn("Legend:", planner_prompt)
        self.assertIn("Legend:", worker_prompt)

    def test_both_prompts_include_game_rules(self):
        """Both planner and worker should understand basic game rules."""
        planner_prompt = build_planner_system_prompt(self.mdp, self.worker_ids)
        worker_prompt = build_worker_system_prompt(self.mdp, 0, "worker_0")

        # Common game rule keywords
        common_keywords = [
            "soup",
            "ingredient",
            "pot",
            "INTERACT",
            "dispenser",
            "serving",
        ]

        for keyword in common_keywords:
            self.assertIn(
                keyword, planner_prompt, f"Planner should understand '{keyword}'"
            )
            self.assertIn(
                keyword, worker_prompt, f"Worker should understand '{keyword}'"
            )

    def test_prompts_have_different_roles(self):
        """Planner and worker should have clearly different roles."""
        planner_prompt = build_planner_system_prompt(self.mdp, self.worker_ids)
        worker_prompt = build_worker_system_prompt(self.mdp, 0, "worker_0")

        # Planner-specific content
        self.assertIn("PLANNER", planner_prompt)
        self.assertNotIn("PLANNER", worker_prompt)

        # Worker-specific content (mentions being a specific player)
        self.assertIn("Player 0", worker_prompt)
        self.assertNotIn("Player 0", planner_prompt)
        self.assertNotIn("Player 1", planner_prompt)


if __name__ == "__main__":
    unittest.main()
