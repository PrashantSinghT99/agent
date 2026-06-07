"""Tests for tools/schemas.py.

Covers:
  - Pydantic model validation (happy path and failure cases)
  - build_gemini_declaration output shape and content
  - get_gemini_declarations round-trip through ToolManager registry
"""

import pytest
from pydantic import ValidationError

from tool_manager import get_gemini_declarations
from tools.schemas import (
    CalculatorArgs,
    ListNotesArgs,
    ReadNoteArgs,
    RecallPreferencesArgs,
    RememberPreferenceArgs,
    SaveNoteArgs,
    SearchNotesArgs,
    build_gemini_declaration,
)


# ─── CalculatorArgs ───────────────────────────────────────────────────────────


class TestCalculatorArgs:
    def test_valid_args(self):
        args = CalculatorArgs.model_validate({"expression": "2 + 2"})
        assert args.expression == "2 + 2"

    def test_missing_expression_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            CalculatorArgs.model_validate({})
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("expression",) for e in errors)

    def test_expression_must_be_string(self):
        # Pydantic coerces int to str in lax mode; pass a dict to force failure.
        with pytest.raises(ValidationError):
            CalculatorArgs.model_validate({"expression": {"nested": "dict"}})


# ─── SaveNoteArgs ─────────────────────────────────────────────────────────────


class TestSaveNoteArgs:
    def test_valid_args(self):
        args = SaveNoteArgs.model_validate({"title": "My Note", "content": "Hello"})
        assert args.title == "My Note"
        assert args.content == "Hello"

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            SaveNoteArgs.model_validate({"content": "Hello"})

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            SaveNoteArgs.model_validate({"title": "My Note"})

    def test_missing_both_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            SaveNoteArgs.model_validate({})
        errors = exc_info.value.errors()
        locs = [e["loc"] for e in errors]
        assert ("title",) in locs
        assert ("content",) in locs


# ─── ListNotesArgs / RecallPreferencesArgs (no fields) ───────────────────────


class TestNoFieldModels:
    def test_list_notes_accepts_empty_dict(self):
        args = ListNotesArgs.model_validate({})
        assert args.model_dump() == {}

    def test_recall_preferences_accepts_empty_dict(self):
        args = RecallPreferencesArgs.model_validate({})
        assert args.model_dump() == {}

    def test_list_notes_model_dump_empty(self):
        assert ListNotesArgs().model_dump() == {}


# ─── ReadNoteArgs ─────────────────────────────────────────────────────────────


class TestReadNoteArgs:
    def test_valid(self):
        args = ReadNoteArgs.model_validate({"title": "my-note"})
        assert args.title == "my-note"

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            ReadNoteArgs.model_validate({})


# ─── SearchNotesArgs ──────────────────────────────────────────────────────────


class TestSearchNotesArgs:
    def test_valid(self):
        args = SearchNotesArgs.model_validate({"query": "agent"})
        assert args.query == "agent"

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            SearchNotesArgs.model_validate({})


# ─── RememberPreferenceArgs ───────────────────────────────────────────────────


class TestRememberPreferenceArgs:
    def test_valid(self):
        args = RememberPreferenceArgs.model_validate({"key": "tone", "value": "formal"})
        assert args.key == "tone"
        assert args.value == "formal"

    def test_missing_key_raises(self):
        with pytest.raises(ValidationError):
            RememberPreferenceArgs.model_validate({"value": "formal"})

    def test_missing_value_raises(self):
        with pytest.raises(ValidationError):
            RememberPreferenceArgs.model_validate({"key": "tone"})


# ─── build_gemini_declaration ─────────────────────────────────────────────────


class TestBuildGeminiDeclaration:
    def test_output_has_required_keys(self):
        decl = build_gemini_declaration("calculator", "Evaluate math.", CalculatorArgs)
        assert "name" in decl
        assert "description" in decl
        assert "parameters" in decl

    def test_name_matches(self):
        decl = build_gemini_declaration("calculator", "Evaluate math.", CalculatorArgs)
        assert decl["name"] == "calculator"

    def test_description_matches(self):
        decl = build_gemini_declaration("calculator", "Evaluate math.", CalculatorArgs)
        assert decl["description"] == "Evaluate math."

    def test_parameters_type_is_object(self):
        decl = build_gemini_declaration("calculator", "Evaluate math.", CalculatorArgs)
        assert decl["parameters"]["type"] == "OBJECT"

    def test_required_fields_present(self):
        decl = build_gemini_declaration("calculator", "Evaluate math.", CalculatorArgs)
        assert "expression" in decl["parameters"].get("required", [])

    def test_field_type_is_string(self):
        decl = build_gemini_declaration("calculator", "Evaluate math.", CalculatorArgs)
        prop = decl["parameters"]["properties"]["expression"]
        assert prop["type"] == "STRING"

    def test_field_description_included(self):
        decl = build_gemini_declaration("calculator", "Evaluate math.", CalculatorArgs)
        prop = decl["parameters"]["properties"]["expression"]
        assert "description" in prop

    def test_no_fields_model_has_empty_properties(self):
        decl = build_gemini_declaration("list_notes", "List notes.", ListNotesArgs)
        assert decl["parameters"]["properties"] == {}
        # No required key for empty models.
        assert "required" not in decl["parameters"]

    def test_two_field_model(self):
        decl = build_gemini_declaration("save_note", "Save a note.", SaveNoteArgs)
        props = decl["parameters"]["properties"]
        assert "title" in props
        assert "content" in props
        assert set(decl["parameters"]["required"]) == {"title", "content"}


# ─── get_gemini_declarations (integration) ────────────────────────────────────


class TestGetGeminiDeclarations:
    """End-to-end: registry → Pydantic schemas → Gemini declarations."""

    EXPECTED_TOOL_NAMES = {
        "calculator",
        "save_note",
        "list_notes",
        "read_note",
        "search_notes",
        "remember_preference",
        "recall_preferences",
    }

    def test_returns_list(self):
        decls = get_gemini_declarations()
        assert isinstance(decls, list)

    def test_all_tools_present(self):
        decls = get_gemini_declarations()
        names = {d["name"] for d in decls}
        assert names == self.EXPECTED_TOOL_NAMES

    def test_every_declaration_has_required_keys(self):
        for decl in get_gemini_declarations():
            assert "name" in decl, f"Missing 'name' in {decl}"
            assert "description" in decl, f"Missing 'description' in {decl}"
            assert "parameters" in decl, f"Missing 'parameters' in {decl}"

    def test_every_declaration_has_object_parameters(self):
        for decl in get_gemini_declarations():
            assert decl["parameters"]["type"] == "OBJECT", decl["name"]
