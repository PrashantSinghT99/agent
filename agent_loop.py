from llm_client import ask_llm_with_tools
from memory import load_history, preferences_context, save_turn, to_gemini_contents
from trace import AgentResult, TraceEvent, save_trace
from tool_manager import ToolManager


tool_manager = ToolManager()
MAX_LLM_CALLS = 2
MAX_TOOL_CALLS = 1


def _save_and_return(
    message: str,
    answer: str,
    trace: list[TraceEvent],
) -> AgentResult:
    """Save the final turn and return an AgentResult.

    Args:
        message: Current user message.
        answer: Final answer to show the user.
        trace: Debug trace events for this turn.

    Returns:
        AgentResult with answer and trace.
    """
    save_turn(message, answer)
    trace.append(TraceEvent("memory_saved", {"messages_added": 2}))
    save_trace(trace)
    return AgentResult(answer=answer, trace=trace)


def run_agent(user_message: str) -> AgentResult:
    """Run one agent turn for a user message.

    Args:
        user_message: Text entered by the user.

    Returns:
        AgentResult containing the final answer and debug trace.
    """
    message = user_message.strip()
    trace = [TraceEvent("user_message_received", {"message": message})]
    llm_calls = 0
    tool_calls = 0

    history = load_history()
    trace.append(TraceEvent("memory_loaded", {"messages": len(history)}))

    contents = to_gemini_contents(history)
    preference_context = preferences_context()
    if preference_context:
        contents.append(
            {
                "role": "user",
                "parts": [{"text": preference_context}],
            }
        )
        trace.append(TraceEvent("preferences_loaded", {"included": True}))

    contents.append(
        {
            "role": "user",
            "parts": [{"text": message}],
        }
    )
    if llm_calls >= MAX_LLM_CALLS:
        answer = "I stopped because the agent reached its LLM call limit."
        trace.append(
            TraceEvent(
                "step_limit_reached",
                {
                    "limit": "llm_calls",
                    "max": MAX_LLM_CALLS,
                },
            )
        )
        return _save_and_return(message, answer, trace)

    llm_calls += 1
    trace.append(
        TraceEvent(
            "llm_request_started",
            {
                "stage": "initial",
                "contents": len(contents),
                "llm_calls": llm_calls,
                "max_llm_calls": MAX_LLM_CALLS,
            },
        )
    )

    try:
        first_response = ask_llm_with_tools(contents)
    except Exception as e:
        answer = "I could not reach the language model. Please try again."
        trace.append(
            TraceEvent(
                "llm_request_failed",
                {
                    "stage": "initial",
                    "error": str(e),
                },
            )
        )
        return _save_and_return(message, answer, trace)

    function_calls = first_response.function_calls

    if not function_calls:
        answer = first_response.text
        trace.append(TraceEvent("llm_answer_received", {"used_tool": False}))
        return _save_and_return(message, answer, trace)

    function_call = function_calls[0]
    tool_name = function_call.name
    tool_args = dict(function_call.args)

    if tool_calls >= MAX_TOOL_CALLS:
        answer = "I stopped because the agent reached its tool call limit."
        trace.append(
            TraceEvent(
                "step_limit_reached",
                {
                    "limit": "tool_calls",
                    "max": MAX_TOOL_CALLS,
                },
            )
        )
        return _save_and_return(message, answer, trace)

    tool_calls += 1
    trace.append(
        TraceEvent(
            "tool_call_requested",
            {
                "tool": tool_name,
                "args": tool_args,
                "tool_calls": tool_calls,
                "max_tool_calls": MAX_TOOL_CALLS,
            },
        )
    )

    tool_result = tool_manager.run(tool_name, tool_args)
    trace.append(
        TraceEvent(
            "tool_executed",
            {
                "tool": tool_name,
                "ok": tool_result.ok,
                "result": tool_result.output,
                "error": tool_result.error,
            },
        )
    )

    if not tool_result.ok:
        answer = tool_result.output
        return _save_and_return(message, answer, trace)

    contents.append(
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "name": tool_name,
                        "args": tool_args,
                    }
                }
            ],
        }
    )
    contents.append(
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": tool_name,
                        "response": {
                            "result": tool_result.output,
                        },
                    }
                }
            ],
        }
    )
    trace.append(TraceEvent("tool_result_appended", {"contents": len(contents)}))

    if llm_calls >= MAX_LLM_CALLS:
        answer = f"I used {tool_name}, but stopped before the final model call because the agent reached its LLM call limit. Tool result: {tool_result.output}"
        trace.append(
            TraceEvent(
                "step_limit_reached",
                {
                    "limit": "llm_calls",
                    "max": MAX_LLM_CALLS,
                },
            )
        )
        return _save_and_return(message, answer, trace)

    llm_calls += 1
    trace.append(
        TraceEvent(
            "llm_request_started",
            {
                "stage": "final",
                "contents": len(contents),
                "llm_calls": llm_calls,
                "max_llm_calls": MAX_LLM_CALLS,
            },
        )
    )

    try:
        final_response = ask_llm_with_tools(contents)
    except Exception as e:
        answer = f"I used {tool_name}, but I could not ask the model to write the final answer. Tool result: {tool_result.output}"
        trace.append(
            TraceEvent(
                "llm_request_failed",
                {
                    "stage": "final",
                    "error": str(e),
                },
            )
        )
        return _save_and_return(message, answer, trace)

    answer = f"I used {tool_name}.\n{final_response.text}"
    trace.append(TraceEvent("llm_final_answer_received", {"tool": tool_name}))

    return _save_and_return(message, answer, trace)
