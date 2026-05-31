import agent_loop


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
