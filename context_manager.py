"""Context budget manager for controlling prompt size.

Gemini has a context window limit. This module tracks approximate token usage
and truncates memory, preferences, and tool results before they blow up the prompt.

Token estimation rule-of-thumb used here:
    approx_tokens = len(text) // 4

This is not exact but is fast, requires no API call, and is good enough for budgeting.
"""

# ─── Budget constants ──────────────────────────────────────────────────────────

# Hard ceiling on total estimated prompt tokens per agent turn.
MAX_CONTEXT_TOKENS = 8_000

# Maximum characters allowed for a single tool result before truncation.
MAX_TOOL_RESULT_CHARS = 2_000

# Maximum characters allowed for the combined preferences context string.
MAX_PREFERENCES_CHARS = 500

# Maximum number of history messages (each message = 1 item) kept in context.
# 20 items = 10 user/model turns, mirroring memory.py's MAX_TURNS default.
MAX_HISTORY_MESSAGES = 20


# ─── Token counting ────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """Estimate the token count for a string.

    Uses the ``len // 4`` heuristic: fast, no API call, good enough for budgeting.

    Args:
        text: Any string to estimate.

    Returns:
        Approximate number of tokens.
    """
    return max(0, len(text) // 4)


def count_contents_tokens(contents: list[dict]) -> int:
    """Estimate the total token count for a Gemini contents list.

    Each content item is a dict with a ``role`` and a ``parts`` list.
    Each part may have a ``text`` key, ``function_call``, or ``function_response``.

    Args:
        contents: Gemini-formatted conversation contents.

    Returns:
        Approximate total token count across all parts.
    """
    total = 0
    for item in contents:
        for part in item.get("parts", []):
            if "text" in part:
                total += count_tokens(part["text"])
            elif "function_call" in part:
                fc = part["function_call"]
                total += count_tokens(fc.get("name", ""))
                total += count_tokens(str(fc.get("args", {})))
            elif "function_response" in part:
                fr = part["function_response"]
                total += count_tokens(fr.get("name", ""))
                total += count_tokens(str(fr.get("response", {})))
    return total


# ─── Truncation helpers ────────────────────────────────────────────────────────


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Trim a string to ``max_chars`` characters and signal whether it was cut.

    Args:
        text: The string to possibly trim.
        max_chars: Maximum allowed character length.

    Returns:
        A tuple of (possibly truncated string, was_truncated flag).
    """
    if len(text) <= max_chars:
        return text, False

    truncated = text[:max_chars].rstrip()
    return truncated + "\n[...truncated]", True


def truncate_tool_result(result: str) -> tuple[str, bool]:
    """Trim a tool result to ``MAX_TOOL_RESULT_CHARS``.

    Tool results from note reads or searches can be very long.
    This keeps them within a safe size before injecting into context.

    Args:
        result: Raw tool output string.

    Returns:
        A tuple of (possibly truncated string, was_truncated flag).
    """
    return truncate_text(result, MAX_TOOL_RESULT_CHARS)


def truncate_preferences(preferences_text: str) -> tuple[str, bool]:
    """Trim the preferences context block to ``MAX_PREFERENCES_CHARS``.

    Args:
        preferences_text: Formatted preferences string from ``preferences_context()``.

    Returns:
        A tuple of (possibly truncated string, was_truncated flag).
    """
    return truncate_text(preferences_text, MAX_PREFERENCES_CHARS)


# ─── History trimming ──────────────────────────────────────────────────────────


def trim_history(contents: list[dict]) -> tuple[list[dict], int]:
    """Drop the oldest history messages if the list exceeds ``MAX_HISTORY_MESSAGES``.

    Keeps the most recent messages so the model always sees fresh context.
    Dropped count is returned so the caller can emit a trace event.

    Args:
        contents: Gemini-formatted history contents (user/model turns only,
                  NOT including the current user message or preferences).

    Returns:
        A tuple of (trimmed contents list, number of messages dropped).
    """
    if len(contents) <= MAX_HISTORY_MESSAGES:
        return contents, 0

    dropped = len(contents) - MAX_HISTORY_MESSAGES
    return contents[dropped:], dropped


# ─── Budget check ──────────────────────────────────────────────────────────────


def check_context_budget(contents: list[dict]) -> dict:
    """Measure current estimated token usage against the budget.

    Args:
        contents: Full Gemini contents list for the current turn, after all
                  truncation and trimming has been applied.

    Returns:
        A dict with keys:
            - ``estimated_tokens``: int — current approximate token count
            - ``max_tokens``: int — the hard ceiling constant
            - ``within_budget``: bool — True if under budget
            - ``usage_pct``: float — fraction of budget consumed (0.0–1.0+)
    """
    estimated = count_contents_tokens(contents)
    within = estimated <= MAX_CONTEXT_TOKENS
    return {
        "estimated_tokens": estimated,
        "max_tokens": MAX_CONTEXT_TOKENS,
        "within_budget": within,
        "usage_pct": round(estimated / MAX_CONTEXT_TOKENS, 3),
    }
