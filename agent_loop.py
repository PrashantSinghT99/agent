from llm_client import ask_llm_with_tools
from memory import load_history, preferences_context, save_turn, to_gemini_contents
from trace import AgentResult, TraceEvent, save_trace
from tool_manager import ToolManager


tool_manager = ToolManager()

# Guardrails prevent the model from repeatedly asking for tools forever.
MAX_LLM_CALLS = 5
MAX_TOOL_CALLS = 3
MAX_STEPS = 6


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
    used_tools = []

    # Memory and preferences are added before the current message so Gemini has context.
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

    # `contents` is the transcript sent to Gemini for this agent turn.
    contents.append(
        {
            "role": "user",
            "parts": [{"text": message}],
        }
    )

    # Each loop iteration gives Gemini a chance to either answer or request tools.
    for step in range(1, MAX_STEPS + 1):
        trace.append(
            TraceEvent(
                "agent_step_started",
                {
                    "step": step,
                    "max_steps": MAX_STEPS,
                },
            )
        )

        # Stop before the LLM call if the turn has already used its call budget.
        if llm_calls >= MAX_LLM_CALLS:
            answer = "I stopped because the agent reached its LLM call limit."
            trace.append(
                TraceEvent(
                    "step_limit_reached",
                    {
                        "limit": "llm_calls",
                        "max": MAX_LLM_CALLS,
                        "step": step,
                    },
                )
            )
            return _save_and_return(message, answer, trace)

        llm_calls += 1
        stage = "initial" if llm_calls == 1 else "after_tool"
        trace.append(
            TraceEvent(
                "llm_request_started",
                {
                    "stage": stage,
                    "contents": len(contents),
                    "llm_calls": llm_calls,
                    "max_llm_calls": MAX_LLM_CALLS,
                    "step": step,
                },
            )
        )

        try:
            response = ask_llm_with_tools(contents)
        except Exception as e:
            if used_tools:
                answer = "I used tools, but I could not reach the language model for the final answer."
            else:
                answer = "I could not reach the language model. Please try again."

            trace.append(
                TraceEvent(
                    "llm_request_failed",
                    {
                        "stage": stage,
                        "error": str(e),
                        "step": step,
                    },
                )
            )
            return _save_and_return(message, answer, trace)

        function_calls = response.function_calls

        # No function calls means Gemini is done and has produced the final answer.
        if not function_calls:
            answer = response.text
            if used_tools:
                answer = f"I used {', '.join(used_tools)}.\n{answer}"

            trace.append(
                TraceEvent(
                    "llm_answer_received",
                    {
                        "used_tool": bool(used_tools),
                        "step": step,
                    },
                )
            )
            return _save_and_return(message, answer, trace)

        # Gemini can request multiple tools in one response; count them before running any.
        if tool_calls + len(function_calls) > MAX_TOOL_CALLS:
            answer = "I stopped because the agent reached its tool call limit."
            trace.append(
                TraceEvent(
                    "step_limit_reached",
                    {
                        "limit": "tool_calls",
                        "current": tool_calls,
                        "requested": len(function_calls),
                        "max": MAX_TOOL_CALLS,
                        "step": step,
                    },
                )
            )
            return _save_and_return(message, answer, trace)

        model_parts = []
        response_parts = []

        # Run each requested tool in Python, never inside the LLM.
        for function_call in function_calls:
            tool_name = function_call.name
            tool_args = dict(function_call.args)
            tool_calls += 1
            used_tools.append(tool_name)

            trace.append(
                TraceEvent(
                    "tool_call_requested",
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "tool_calls": tool_calls,
                        "max_tool_calls": MAX_TOOL_CALLS,
                        "step": step,
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
                        "step": step,
                    },
                )
            )

            if not tool_result.ok:
                answer = tool_result.output
                return _save_and_return(message, answer, trace)

            # Echo Gemini's tool request back into the transcript.
            model_parts.append(
                {
                    "function_call": {
                        "name": tool_name,
                        "args": tool_args,
                    }
                }
            )
            # Add Python's tool result so Gemini can reason about the next step.
            response_parts.append(
                {
                    "function_response": {
                        "name": tool_name,
                        "response": {
                            "result": tool_result.output,
                        },
                    }
                }
            )

        # Append all tool calls/results together before asking Gemini what to do next.
        contents.append(
            {
                "role": "model",
                "parts": model_parts,
            }
        )
        contents.append(
            {
                "role": "user",
                "parts": response_parts,
            }
        )
        trace.append(
            TraceEvent(
                "tool_result_appended",
                {
                    "contents": len(contents),
                    "step": step,
                },
            )
        )

    answer = "I stopped because the agent reached its step limit."
    trace.append(
        TraceEvent(
            "step_limit_reached",
            {
                "limit": "steps",
                "max": MAX_STEPS,
            },
        )
    )

    return _save_and_return(message, answer, trace)
