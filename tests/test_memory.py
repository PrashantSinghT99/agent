import memory


def test_save_and_load_history(tmp_path, monkeypatch):
    """Memory saves and loads user/model turns from a temp file."""
    memory_path = tmp_path / "memory.json"
    monkeypatch.setattr(memory, "MEMORY_PATH", memory_path)

    memory.save_turn("hello", "hi")

    assert memory.load_history() == [
        {"role": "user", "content": "hello"},
        {"role": "model", "content": "hi"},
    ]


def test_to_gemini_contents_filters_invalid_messages():
    """Memory converts only valid user/model messages to Gemini format."""
    history = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "ignored"},
        {"role": "model", "content": "hi"},
        {"role": "user", "content": ""},
    ]

    assert memory.to_gemini_contents(history) == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
    ]


def test_load_history_returns_empty_for_bad_json(tmp_path, monkeypatch):
    """Memory handles corrupted JSON without crashing."""
    memory_path = tmp_path / "memory.json"
    memory_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(memory, "MEMORY_PATH", memory_path)

    assert memory.load_history() == []


def test_remember_and_recall_preferences(tmp_path, monkeypatch):
    """Preferences can be saved and recalled from a temp file."""
    preferences_path = tmp_path / "preferences.json"
    monkeypatch.setattr(memory, "PREFERENCES_PATH", preferences_path)

    result = memory.remember_preference("explanation_style", "simple examples")

    assert result == "Remembered preference: explanation_style = simple examples"
    assert memory.load_preferences() == {"explanation_style": "simple examples"}
    assert memory.recall_preferences() == "explanation_style: simple examples"


def test_preferences_context(tmp_path, monkeypatch):
    """Preferences context formats saved preferences for the model."""
    preferences_path = tmp_path / "preferences.json"
    monkeypatch.setattr(memory, "PREFERENCES_PATH", preferences_path)

    memory.remember_preference("level", "beginner")

    assert memory.preferences_context() == "User preferences:\n- level: beginner"
