import agent_loop
from tool_manager import ToolResult


class FakeFunctionCall:
    """Small fake function call object matching Gemini fields used by run_agent."""

    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakeResponse:
    """Small fake Gemini response object for agent loop tests."""

    def __init__(self, text=None, function_calls=None):
        self.text = text
        self.function_calls = function_calls


def test_run_agent_direct_answer_trace(monkeypatch, tmp_path):
    """Direct answers use one LLM call and save trace."""
    monkeypatch.setattr(agent_loop, "ask_llm_with_tools", lambda contents: FakeResponse(text="Hi."))
    monkeypatch.setattr(agent_loop, "save_turn", lambda message, answer: None)
    monkeypatch.setattr(agent_loop, "save_trace", lambda trace: None)
    monkeypatch.setattr(agent_loop, "load_history", lambda: [])
    monkeypatch.setattr(agent_loop, "preferences_context", lambda: "")

    result = agent_loop.run_agent("hello")

    assert result.answer == "Hi."
    assert [event.name for event in result.trace].count("llm_request_started") == 1
    assert result.trace[-1].name == "memory_saved"


def test_run_agent_tool_path_uses_two_llm_calls(monkeypatch):
    """Tool answers use initial and final LLM calls."""
    responses = [
        FakeResponse(
            function_calls=[
                FakeFunctionCall("calculator", {"expression": "6 * 7"}),
            ]
        ),
        FakeResponse(text="6 times 7 is 42."),
    ]
    monkeypatch.setattr(agent_loop, "ask_llm_with_tools", lambda contents: responses.pop(0))
    monkeypatch.setattr(agent_loop, "save_turn", lambda message, answer: None)
    monkeypatch.setattr(agent_loop, "save_trace", lambda trace: None)
    monkeypatch.setattr(agent_loop, "load_history", lambda: [])
    monkeypatch.setattr(agent_loop, "preferences_context", lambda: "")

    result = agent_loop.run_agent("what is 6 times 7?")

    assert result.answer == "I used calculator.\n6 times 7 is 42."
    assert [event.name for event in result.trace].count("llm_request_started") == 2
    assert any(event.name == "tool_call_requested" for event in result.trace)


def test_run_agent_stops_when_final_llm_call_limit_reached(monkeypatch):
    """Agent stops before final LLM call if the configured LLM limit is reached."""
    monkeypatch.setattr(agent_loop, "MAX_LLM_CALLS", 1)
    monkeypatch.setattr(
        agent_loop,
        "ask_llm_with_tools",
        lambda contents: FakeResponse(
            function_calls=[
                FakeFunctionCall("calculator", {"expression": "6 * 7"}),
            ]
        ),
    )
    monkeypatch.setattr(agent_loop, "save_turn", lambda message, answer: None)
    monkeypatch.setattr(agent_loop, "save_trace", lambda trace: None)
    monkeypatch.setattr(agent_loop, "load_history", lambda: [])
    monkeypatch.setattr(agent_loop, "preferences_context", lambda: "")

    result = agent_loop.run_agent("what is 6 times 7?")

    assert "reached its LLM call limit" in result.answer
    assert any(event.name == "step_limit_reached" for event in result.trace)


def test_run_agent_multi_step_two_tool_path(monkeypatch):
    """Agent can run two tools across multiple LLM turns."""
    responses = [
        FakeResponse(
            function_calls=[
                FakeFunctionCall("search_notes", {"query": "memory"}),
            ]
        ),
        FakeResponse(
            function_calls=[
                FakeFunctionCall(
                    "save_note",
                    {
                        "title": "Memory Summary",
                        "content": "Agents can use memory.",
                    },
                ),
            ]
        ),
        FakeResponse(text="I searched your notes and saved a summary."),
    ]

    class FakeToolManager:
        """Fake tool manager for multi-step agent tests."""

        def run(self, tool_name, tool_args):
            return ToolResult(ok=True, output=f"{tool_name} output")

    monkeypatch.setattr(agent_loop, "ask_llm_with_tools", lambda contents: responses.pop(0))
    monkeypatch.setattr(agent_loop, "tool_manager", FakeToolManager())
    monkeypatch.setattr(agent_loop, "save_turn", lambda message, answer: None)
    monkeypatch.setattr(agent_loop, "save_trace", lambda trace: None)
    monkeypatch.setattr(agent_loop, "load_history", lambda: [])
    monkeypatch.setattr(agent_loop, "preferences_context", lambda: "")

    result = agent_loop.run_agent("search notes and save summary")

    assert result.answer == (
        "I used search_notes, save_note.\n"
        "I searched your notes and saved a summary."
    )
    assert [event.name for event in result.trace].count("llm_request_started") == 3
    assert [event.name for event in result.trace].count("tool_call_requested") == 2


def test_run_agent_handles_multiple_tool_calls_from_one_response(monkeypatch):
    """Agent can execute multiple tool calls returned in one LLM response."""
    responses = [
        FakeResponse(
            function_calls=[
                FakeFunctionCall("recall_preferences", {}),
                FakeFunctionCall("search_notes", {"query": "agents"}),
            ]
        ),
        FakeResponse(text="I recalled preferences and searched notes."),
    ]

    class FakeToolManager:
        """Fake tool manager for multi-call response tests."""

        def run(self, tool_name, tool_args):
            return ToolResult(ok=True, output=f"{tool_name} output")

    monkeypatch.setattr(agent_loop, "ask_llm_with_tools", lambda contents: responses.pop(0))
    monkeypatch.setattr(agent_loop, "tool_manager", FakeToolManager())
    monkeypatch.setattr(agent_loop, "save_turn", lambda message, answer: None)
    monkeypatch.setattr(agent_loop, "save_trace", lambda trace: None)
    monkeypatch.setattr(agent_loop, "load_history", lambda: [])
    monkeypatch.setattr(agent_loop, "preferences_context", lambda: "")

    result = agent_loop.run_agent("recall preferences and search notes")

    assert result.answer == (
        "I used recall_preferences, search_notes.\n"
        "I recalled preferences and searched notes."
    )
    assert [event.name for event in result.trace].count("llm_request_started") == 2
    assert [event.name for event in result.trace].count("tool_call_requested") == 2
