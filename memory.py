from pathlib import Path
import json


MEMORY_PATH = Path("data/memory.json")
MAX_TURNS = 10


def load_history() -> list[dict[str, str]]:
    """Load recent chat history from disk.

    Returns:
        A list of messages with role and content keys.
    """
    if not MEMORY_PATH.exists():
        return []

    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return data[-MAX_TURNS * 2 :]


def save_turn(user_message: str, assistant_message: str) -> None:
    """Save one user/assistant turn to disk.

    Args:
        user_message: Text the user sent.
        assistant_message: Final answer from the agent.
    """
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = load_history()
    history.append({"role": "user", "content": user_message})
    history.append({"role": "model", "content": assistant_message})

    MEMORY_PATH.write_text(
        json.dumps(history[-MAX_TURNS * 2 :], indent=2),
        encoding="utf-8",
    )


def to_gemini_contents(history: list[dict[str, str]]) -> list[dict]:
    """Convert internal memory messages to Gemini contents.

    Args:
        history: Messages with role and content keys.

    Returns:
        Gemini-formatted content dictionaries.
    """
    contents = []

    for message in history:
        role = message.get("role")
        content = message.get("content", "")

        if role not in {"user", "model"} or not content:
            continue

        contents.append(
            {
                "role": role,
                "parts": [{"text": content}],
            }
        )

    return contents
