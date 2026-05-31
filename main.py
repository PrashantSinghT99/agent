from agent_loop import run_agent
from rich.console import Console
from rich.pretty import Pretty
import sys


console = Console()


TRACE_STYLES = {
    "user_message_received": "bold white",
    "memory_loaded": "dim",
    "memory_saved": "dim",
    "preferences_loaded": "magenta",
    "llm_request_started": "cyan",
    "llm_answer_received": "green",
    "llm_final_answer_received": "green",
    "llm_request_failed": "bold red",
    "tool_call_requested": "yellow",
    "tool_executed": "green",
    "tool_result_appended": "blue",
    "step_limit_reached": "bold red",
}


def print_trace(trace):
    """Print agent debug events.

    Args:
        trace: Ordered trace events returned by run_agent.
    """
    console.print("Trace:", style="bold underline")
    for event in trace:
        style = TRACE_STYLES.get(event.name, "white")
        console.print(f"- {event.name}: ", style=style, end="")
        console.print(Pretty(event.data), style=style)


debug = "--debug" in sys.argv


def main():
    """Run the CLI chat loop."""
    while True:
        user_input = input("You: ")

        if user_input.lower() in {"exit", "quit"}:
            break

        result = run_agent(user_input)
        console.print("Agent:", style="bold green", end=" ")
        console.print(result.answer)

        if debug:
            print_trace(result.trace)


if __name__ == "__main__":
    main()
