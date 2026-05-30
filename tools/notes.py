from pathlib import Path
import re


NOTES_DIR = Path("data/notes")


def _safe_filename(title: str) -> str:
    """Convert a note title into a safe markdown filename.

    Args:
        title: User-provided note title.

    Returns:
        A lowercase filename ending in .md.
    """
    filename = title.strip().lower()
    filename = re.sub(r"[^a-z0-9]+", "-", filename).strip("-")
    return f"{filename or 'untitled'}.md"


def save_note(title: str, content: str) -> str:
    """Save a note as a markdown file.

    Args:
        title: Short note title. This is used to create the filename.
        content: Text content to write into the note.

    Returns:
        A message describing where the note was saved.
    """
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    note_path = NOTES_DIR / _safe_filename(title)
    note_path.write_text(content.strip() + "\n", encoding="utf-8")
    return f"Saved note: {note_path}"


def list_notes() -> str:
    """List all saved markdown notes.

    Returns:
        A newline-separated list of note filenames, or a message if no notes exist.
    """
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    notes = sorted(NOTES_DIR.glob("*.md"))
    if not notes:
        return "No notes found."

    return "\n".join(note.name for note in notes)


def read_note(title: str) -> str:
    """Read a saved note by title.

    Args:
        title: Note title or filename-like title to read.

    Returns:
        The note content, or a message if the note does not exist.
    """
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    note_path = NOTES_DIR / _safe_filename(title)
    if not note_path.exists():
        return f"Note not found: {note_path.name}"

    return note_path.read_text(encoding="utf-8")
