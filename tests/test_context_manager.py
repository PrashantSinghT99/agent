"""Tests for context_manager.py.

All tests are deterministic — no LLM calls, no file I/O.
"""

import pytest

from context_manager import (
    MAX_CONTEXT_TOKENS,
    MAX_HISTORY_MESSAGES,
    MAX_PREFERENCES_CHARS,
    MAX_TOOL_RESULT_CHARS,
    check_context_budget,
    count_contents_tokens,
    count_tokens,
    trim_history,
    truncate_preferences,
    truncate_text,
    truncate_tool_result,
)


# ─── count_tokens ──────────────────────────────────────────────────────────────


class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_short_string(self):
        # 4 chars → 1 token
        assert count_tokens("abcd") == 1

    def test_long_string(self):
        # 400 chars → 100 tokens
        text = "a" * 400
        assert count_tokens(text) == 100

    def test_returns_int(self):
        assert isinstance(count_tokens("hello"), int)

    def test_never_negative(self):
        # Even 1-char strings should return 0, never negative.
        assert count_tokens("x") >= 0


# ─── count_contents_tokens ────────────────────────────────────────────────────


class TestCountContentsTokens:
    def _make_text_item(self, role: str, text: str) -> dict:
        return {"role": role, "parts": [{"text": text}]}

    def test_empty_contents(self):
        assert count_contents_tokens([]) == 0

    def test_single_text_part(self):
        # 40 chars → 10 tokens
        text = "a" * 40
        contents = [self._make_text_item("user", text)]
        assert count_contents_tokens(contents) == 10

    def test_multiple_text_parts(self):
        contents = [
            self._make_text_item("user", "a" * 80),   # 20 tokens
            self._make_text_item("model", "b" * 40),  # 10 tokens
        ]
        assert count_contents_tokens(contents) == 30

    def test_function_call_part(self):
        contents = [
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "calculator",
                            "args": {"expression": "2 + 2"},
                        }
                    }
                ],
            }
        ]
        # Just check it runs and returns a positive int.
        result = count_contents_tokens(contents)
        assert isinstance(result, int)
        assert result >= 0

    def test_function_response_part(self):
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "calculator",
                            "response": {"result": "4"},
                        }
                    }
                ],
            }
        ]
        result = count_contents_tokens(contents)
        assert isinstance(result, int)
        assert result >= 0

    def test_item_without_parts_does_not_crash(self):
        contents = [{"role": "user"}]
        assert count_contents_tokens(contents) == 0


# ─── truncate_text ────────────────────────────────────────────────────────────


class TestTruncateText:
    def test_short_text_not_truncated(self):
        text = "hello"
        result, was_truncated = truncate_text(text, max_chars=100)
        assert result == text
        assert was_truncated is False

    def test_exact_length_not_truncated(self):
        text = "a" * 100
        result, was_truncated = truncate_text(text, max_chars=100)
        assert result == text
        assert was_truncated is False

    def test_long_text_is_truncated(self):
        text = "a" * 500
        result, was_truncated = truncate_text(text, max_chars=100)
        assert was_truncated is True
        assert "[...truncated]" in result

    def test_truncated_result_not_longer_than_limit_plus_suffix(self):
        text = "a" * 500
        max_chars = 100
        result, _ = truncate_text(text, max_chars=max_chars)
        # result = first 100 chars (stripped) + "\n[...truncated]"
        assert len(result) <= max_chars + len("\n[...truncated]")

    def test_empty_string_not_truncated(self):
        result, was_truncated = truncate_text("", max_chars=10)
        assert result == ""
        assert was_truncated is False


# ─── truncate_tool_result ─────────────────────────────────────────────────────


class TestTruncateToolResult:
    def test_short_result_unchanged(self):
        short = "small result"
        result, truncated = truncate_tool_result(short)
        assert result == short
        assert truncated is False

    def test_long_result_is_truncated(self):
        long_result = "x" * (MAX_TOOL_RESULT_CHARS + 500)
        result, truncated = truncate_tool_result(long_result)
        assert truncated is True
        assert "[...truncated]" in result

    def test_truncated_result_respects_char_limit(self):
        long_result = "x" * (MAX_TOOL_RESULT_CHARS * 2)
        result, _ = truncate_tool_result(long_result)
        assert len(result) <= MAX_TOOL_RESULT_CHARS + len("\n[...truncated]")


# ─── truncate_preferences ────────────────────────────────────────────────────


class TestTruncatePreferences:
    def test_short_prefs_unchanged(self):
        prefs = "tone: formal"
        result, truncated = truncate_preferences(prefs)
        assert result == prefs
        assert truncated is False

    def test_long_prefs_are_truncated(self):
        long_prefs = "key: value\n" * 100  # very long
        result, truncated = truncate_preferences(long_prefs)
        assert truncated is True
        assert "[...truncated]" in result
        assert len(result) <= MAX_PREFERENCES_CHARS + len("\n[...truncated]")


# ─── trim_history ─────────────────────────────────────────────────────────────


def _make_history(n: int) -> list[dict]:
    """Build a fake Gemini contents list of length n."""
    return [{"role": "user", "parts": [{"text": f"msg {i}"}]} for i in range(n)]


class TestTrimHistory:
    def test_short_history_unchanged(self):
        history = _make_history(5)
        result, dropped = trim_history(history)
        assert dropped == 0
        assert result == history

    def test_exactly_max_not_trimmed(self):
        history = _make_history(MAX_HISTORY_MESSAGES)
        result, dropped = trim_history(history)
        assert dropped == 0
        assert len(result) == MAX_HISTORY_MESSAGES

    def test_long_history_is_trimmed(self):
        history = _make_history(MAX_HISTORY_MESSAGES + 5)
        result, dropped = trim_history(history)
        assert dropped == 5
        assert len(result) == MAX_HISTORY_MESSAGES

    def test_keeps_most_recent_messages(self):
        history = _make_history(MAX_HISTORY_MESSAGES + 3)
        result, _ = trim_history(history)
        # The last item in result should be the last item from the original list.
        assert result[-1] == history[-1]

    def test_empty_history_unchanged(self):
        result, dropped = trim_history([])
        assert result == []
        assert dropped == 0


# ─── check_context_budget ────────────────────────────────────────────────────


class TestCheckContextBudget:
    def _make_contents(self, total_chars: int) -> list[dict]:
        """Build a single-item contents list whose text has `total_chars` chars."""
        return [{"role": "user", "parts": [{"text": "a" * total_chars}]}]

    def test_returns_required_keys(self):
        result = check_context_budget(self._make_contents(100))
        assert "estimated_tokens" in result
        assert "max_tokens" in result
        assert "within_budget" in result
        assert "usage_pct" in result

    def test_within_budget_when_small(self):
        result = check_context_budget(self._make_contents(40))  # 10 tokens
        assert result["within_budget"] is True
        assert result["estimated_tokens"] == 10

    def test_over_budget_when_large(self):
        # MAX_CONTEXT_TOKENS * 4 chars → exactly MAX_CONTEXT_TOKENS tokens
        over_chars = (MAX_CONTEXT_TOKENS + 1) * 4
        result = check_context_budget(self._make_contents(over_chars))
        assert result["within_budget"] is False

    def test_max_tokens_matches_constant(self):
        result = check_context_budget([])
        assert result["max_tokens"] == MAX_CONTEXT_TOKENS

    def test_usage_pct_type(self):
        result = check_context_budget(self._make_contents(400))
        assert isinstance(result["usage_pct"], float)

    def test_empty_contents_zero_tokens(self):
        result = check_context_budget([])
        assert result["estimated_tokens"] == 0
        assert result["within_budget"] is True
        assert result["usage_pct"] == 0.0
