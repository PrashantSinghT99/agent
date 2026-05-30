from agent_loop import run_agent
import sys


def print_trace(trace):
    """Print agent debug events.

    Args:
        trace: Ordered trace events returned by run_agent.
    """
    print("Trace:")
    for event in trace:
        print(f"- {event.name}: {event.data}")


debug = "--debug" in sys.argv

while True:
    user_input = input("You: ")

    if user_input.lower() in {"exit", "quit"}:
        break

    result = run_agent(user_input)
    print("Agent:", result.answer)

    if debug:
        print_trace(result.trace)
