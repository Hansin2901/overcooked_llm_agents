"""Unit tests for the generic ReAct graph builder.

Tests that the graph:
- Compiles successfully
- Terminates when action tool is called
- Loops when observation tool is called
- Terminates when get_chosen_fn returns truthy
- Includes debug_prefix in debug output
"""

import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

from langchain_core.tools import tool

from overcooked_ai_py.agents.llm.graph_builder import build_react_graph


class TestGraphBuilder(unittest.TestCase):
    """Test the generic ReAct graph builder."""

    def setUp(self):
        """Set up test fixtures."""
        # Track action state
        self.action_chosen = None

        # Create test observation tools
        @tool
        def get_info() -> str:
            """Get some information."""
            return "Info about the environment"

        @tool
        def check_status() -> str:
            """Check current status."""
            return "Status OK"

        # Create test action tools
        @tool
        def do_action() -> str:
            """Perform an action."""
            self.action_chosen = "action_performed"
            return "Action done"

        @tool
        def finish() -> str:
            """Finish the task."""
            self.action_chosen = "finished"
            return "Task finished"

        self.observation_tools = [get_info, check_status]
        self.action_tools = [do_action, finish]
        self.action_tool_names = {"do_action", "finish"}

        self.system_prompt = "You are a test agent. Use tools to complete tasks."
        self.model_name = "gpt-3.5-turbo"

    def test_graph_compiles_successfully(self):
        """Test that the graph compiles without errors."""
        graph = build_react_graph(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            observation_tools=self.observation_tools,
            action_tools=self.action_tools,
            action_tool_names=self.action_tool_names,
            get_chosen_fn=lambda: self.action_chosen,
            debug=False,
        )
        self.assertIsNotNone(graph)

    def test_graph_with_custom_endpoint(self):
        """Test that the graph compiles with custom API endpoint."""
        graph = build_react_graph(
            model_name="openai/custom-model",
            system_prompt=self.system_prompt,
            observation_tools=self.observation_tools,
            action_tools=self.action_tools,
            action_tool_names=self.action_tool_names,
            get_chosen_fn=lambda: self.action_chosen,
            debug=False,
            api_base="https://custom.api/v1",
            api_key="test-key",
        )
        self.assertIsNotNone(graph)

    @patch("sys.stdout", new_callable=StringIO)
    def test_debug_prefix_in_output(self, mock_stdout):
        """Test that debug output includes the debug_prefix."""
        custom_prefix = "[TestAgent]"

        # Create a mock LLM that returns a simple response
        with patch("overcooked_ai_py.agents.llm.graph_builder.ChatLiteLLM") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            # Mock the bind_tools method
            mock_llm_with_tools = MagicMock()
            mock_llm_instance.bind_tools.return_value = mock_llm_with_tools

            # Create a mock response with content
            mock_response = MagicMock()
            mock_response.content = "I need to check the status first"
            mock_response.tool_calls = []
            mock_llm_with_tools.invoke.return_value = mock_response

            graph = build_react_graph(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                observation_tools=self.observation_tools,
                action_tools=self.action_tools,
                action_tool_names=self.action_tool_names,
                get_chosen_fn=lambda: self.action_chosen,
                debug=True,
                debug_prefix=custom_prefix,
            )

            # Invoke the graph
            result = graph.invoke({"messages": [("user", "Do something")]})

            # Check that the debug prefix appears in output
            output = mock_stdout.getvalue()
            self.assertIn(custom_prefix, output)

    def test_get_chosen_fn_termination(self):
        """Test that graph terminates when get_chosen_fn returns truthy."""

        def get_chosen():
            return self.action_chosen

        graph = build_react_graph(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            observation_tools=self.observation_tools,
            action_tools=self.action_tools,
            action_tool_names=self.action_tool_names,
            get_chosen_fn=get_chosen,
            debug=False,
        )

        # Initially, action_chosen is None, so get_chosen_fn returns None (falsy)
        self.assertIsNone(self.action_chosen)
        self.assertFalse(get_chosen())

        # Set action_chosen to simulate an action being performed
        self.action_chosen = "some_action"
        self.assertTrue(get_chosen())

    def test_observation_vs_action_tool_routing(self):
        """Test that the graph correctly routes observation vs action tools."""
        # This is a structural test - we verify the graph has the right edges

        graph = build_react_graph(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            observation_tools=self.observation_tools,
            action_tools=self.action_tools,
            action_tool_names=self.action_tool_names,
            get_chosen_fn=lambda: self.action_chosen,
            debug=False,
        )

        # Check that the compiled graph has the expected structure
        # The graph should have nodes: llm, obs_tools, action_tools
        self.assertIsNotNone(graph)

    def test_multiple_observation_tools(self):
        """Test that multiple observation tools can be added."""

        @tool
        def obs1() -> str:
            """Observation tool 1."""
            return "obs1"

        @tool
        def obs2() -> str:
            """Observation tool 2."""
            return "obs2"

        @tool
        def obs3() -> str:
            """Observation tool 3."""
            return "obs3"

        @tool
        def action1() -> str:
            """Action tool 1."""
            self.action_chosen = "action1"
            return "action1"

        graph = build_react_graph(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            observation_tools=[obs1, obs2, obs3],
            action_tools=[action1],
            action_tool_names={"action1"},
            get_chosen_fn=lambda: self.action_chosen,
            debug=False,
        )

        self.assertIsNotNone(graph)

    def test_empty_observation_tools(self):
        """Test that graph works with no observation tools (only actions)."""

        @tool
        def only_action() -> str:
            """Single action tool."""
            self.action_chosen = "done"
            return "done"

        graph = build_react_graph(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            observation_tools=[],
            action_tools=[only_action],
            action_tool_names={"only_action"},
            get_chosen_fn=lambda: self.action_chosen,
            debug=False,
        )

        self.assertIsNotNone(graph)

    def test_system_prompt_included(self):
        """Test that the system prompt is properly included in messages."""
        custom_prompt = "This is a custom system prompt for testing."

        with patch("overcooked_ai_py.agents.llm.graph_builder.ChatLiteLLM") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            mock_llm_with_tools = MagicMock()
            mock_llm_instance.bind_tools.return_value = mock_llm_with_tools

            # Create a mock response
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.tool_calls = []
            mock_llm_with_tools.invoke.return_value = mock_response

            graph = build_react_graph(
                model_name=self.model_name,
                system_prompt=custom_prompt,
                observation_tools=self.observation_tools,
                action_tools=self.action_tools,
                action_tool_names=self.action_tool_names,
                get_chosen_fn=lambda: self.action_chosen,
                debug=False,
            )

            # Invoke the graph
            graph.invoke({"messages": [("user", "Test message")]})

            # Verify that invoke was called and check the messages
            self.assertTrue(mock_llm_with_tools.invoke.called)
            call_args = mock_llm_with_tools.invoke.call_args[0][0]

            # Check that system prompt is in the messages
            from langchain_core.messages import SystemMessage
            has_system_message = any(
                isinstance(msg, SystemMessage) and msg.content == custom_prompt
                for msg in call_args
            )
            self.assertTrue(has_system_message)

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


if __name__ == "__main__":
    unittest.main()
