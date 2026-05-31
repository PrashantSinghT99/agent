import json

from trace import TraceEvent, save_trace


def test_save_trace_writes_jsonl(tmp_path):
    """Trace events are appended as JSONL records."""
    trace_path = tmp_path / "traces.jsonl"
    save_trace(
        [
            TraceEvent("user_message_received", {"message": "hi"}),
            TraceEvent("memory_saved", {"messages_added": 2}),
        ],
        path=trace_path,
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert [record["event"] for record in records] == [
        "user_message_received",
        "memory_saved",
    ]
    assert records[0]["data"] == {"message": "hi"}
