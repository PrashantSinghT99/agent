from tools import notes


def test_save_list_and_read_note(tmp_path, monkeypatch):
    """Notes can be saved, listed, and read from a temp notes directory."""
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)

    save_result = notes.save_note("Agent Basics", "Agents can use tools.")

    assert save_result == f"Saved note: {tmp_path / 'agent-basics.md'}"
    assert notes.list_notes() == "agent-basics.md"
    assert notes.read_note("Agent Basics") == "Agents can use tools.\n"


def test_read_missing_note(tmp_path, monkeypatch):
    """Reading a missing note returns a clear message."""
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)

    assert notes.read_note("Missing Note") == "Note not found: missing-note.md"


def test_search_notes_returns_matching_snippet(tmp_path, monkeypatch):
    """Searching notes returns matching filenames and snippets."""
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    notes.save_note("Agent Loop", "An agent loop can call tools and use memory.")

    result = notes.search_notes("tools")

    assert "agent-loop.md:" in result
    assert "tools" in result


def test_search_notes_handles_no_matches(tmp_path, monkeypatch):
    """Searching notes returns a clear message when nothing matches."""
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    notes.save_note("Agent Loop", "An agent loop can call tools.")

    assert notes.search_notes("embeddings") == "No notes matched: embeddings"
