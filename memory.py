from pathlib import Path
import json


MEMORY_PATH = Path("data/memory.json")
PREFERENCES_PATH = Path("data/preferences.json")
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


def load_preferences() -> dict[str, str]:
    """Load saved user preferences from disk.

    Returns:
        A dictionary of preference keys and values.
    """
    if not PREFERENCES_PATH.exists():
        return {}

    try:
        data = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {str(key): str(value) for key, value in data.items()}


def remember_preference(key: str, value: str) -> str:
    """Save or update one user preference.

    Args:
        key: Preference name, such as "explanation_style".
        value: Preference value, such as "simple examples".

    Returns:
        A message describing the saved preference.
    """
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_key = key.strip()
    clean_value = value.strip()
    preferences = load_preferences()
    preferences[clean_key] = clean_value
    PREFERENCES_PATH.write_text(
        json.dumps(preferences, indent=2),
        encoding="utf-8",
    )

    return f"Remembered preference: {clean_key} = {clean_value}"


def recall_preferences() -> str:
    """Return all saved user preferences.

    Returns:
        A newline-separated list of preferences, or a message if none exist.
    """
    preferences = load_preferences()

    if not preferences:
        return "No preferences saved."

    return "\n".join(f"{key}: {value}" for key, value in sorted(preferences.items()))


def preferences_context() -> str:
    """Build a short context string from saved preferences.

    Returns:
        Text that can be added to the model context, or an empty string.
    """
    preferences = load_preferences()

    if not preferences:
        return ""

    lines = ["User preferences:"]
    lines.extend(f"- {key}: {value}" for key, value in sorted(preferences.items()))
    return "\n".join(lines)
