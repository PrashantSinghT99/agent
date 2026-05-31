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


def search_notes(query: str) -> str:
    """Search saved notes by keyword.

    Args:
        query: Text to search for in note filenames and note content.

    Returns:
        Matching note filenames with short snippets, or a message if no match exists.
    """
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    normalized_query = query.strip().lower()
    if not normalized_query:
        return "Please provide a search query."

    matches = []
    for note_path in sorted(NOTES_DIR.glob("*.md")):
        content = note_path.read_text(encoding="utf-8")
        haystack = f"{note_path.stem}\n{content}".lower()

        if normalized_query not in haystack:
            continue

        snippet = _snippet_for_match(content, normalized_query)
        matches.append(f"{note_path.name}: {snippet}")

    if not matches:
        return f"No notes matched: {query}"

    return "\n".join(matches)


def _snippet_for_match(content: str, query: str, radius: int = 50) -> str:
    """Build a short content snippet around the first query match.

    Args:
        content: Full note content.
        query: Lowercase search query.
        radius: Number of characters to include around the match.

    Returns:
        A compact single-line snippet.
    """
    normalized_content = content.lower()
    match_index = normalized_content.find(query)

    if match_index == -1:
        return content.strip().splitlines()[0][:100] if content.strip() else "(empty note)"

    start = max(match_index - radius, 0)
    end = min(match_index + len(query) + radius, len(content))
    snippet = content[start:end].strip().replace("\n", " ")

    if start > 0:
        snippet = f"...{snippet}"
    if end < len(content):
        snippet = f"{snippet}..."

    return snippet
