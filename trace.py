from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


TRACE_LOG_PATH = Path("data/traces.jsonl")


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


def save_trace(trace: list[TraceEvent], path: Path = TRACE_LOG_PATH) -> None:
    """Append trace events to a JSONL log file.

    Args:
        trace: Ordered trace events from one agent turn.
        path: File path for JSONL trace records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as log_file:
        for event in trace:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event.name,
                "data": event.data,
            }
            log_file.write(json.dumps(record) + "\n")
