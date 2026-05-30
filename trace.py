from dataclasses import dataclass
from typing import Any


@dataclass
class TraceEvent:
    """One debug event from inside the agent loop.

    Args:
        name: Short event name, such as "tool_call_requested".
        data: Structured details for the event.
    """

    name: str
    data: dict[str, Any]


@dataclass
class AgentResult:
    """Final agent answer plus debug trace events.

    Args:
        answer: Text shown to the user.
        trace: Ordered list of internal agent events.
    """

    answer: str
    trace: list[TraceEvent]
