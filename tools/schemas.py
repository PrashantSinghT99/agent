"""Pydantic argument models for every registered tool.

Each model is the single source of truth for:
  - argument names and types
  - field descriptions (used to build Gemini tool declarations)
  - validation rules (enforced before any Python tool function is called)

Adding a new tool = add one model here + wire it in tool_manager.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Per-tool argument models ──────────────────────────────────────────────────


class CalculatorArgs(BaseModel):
    """Arguments for the calculator tool."""

    expression: str = Field(
        description="The math expression to evaluate, for example: 45 * 12",
    )


class SaveNoteArgs(BaseModel):
    """Arguments for save_note."""

    title: str = Field(description="A short title for the note.")
    content: str = Field(description="The note content to save.")


class ListNotesArgs(BaseModel):
    """Arguments for list_notes (no fields required)."""


class ReadNoteArgs(BaseModel):
    """Arguments for read_note."""

    title: str = Field(description="The title of the note to read.")


class SearchNotesArgs(BaseModel):
    """Arguments for search_notes."""

    query: str = Field(description="The keyword or phrase to search for in saved notes.")


class RememberPreferenceArgs(BaseModel):
    """Arguments for remember_preference."""

    key: str = Field(description="Short preference key, for example: explanation_style.")
    value: str = Field(description="Preference value to remember.")


class RecallPreferencesArgs(BaseModel):
    """Arguments for recall_preferences (no fields required)."""


# ─── Gemini declaration builder ───────────────────────────────────────────────

# Pydantic type → Gemini JSON schema type string.
_PYDANTIC_TO_GEMINI_TYPE: dict[str, str] = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _pydantic_to_gemini_type(json_type: str) -> str:
    """Map a JSON Schema type string to a Gemini type string.

    Args:
        json_type: JSON Schema type such as ``"string"`` or ``"integer"``.

    Returns:
        Gemini type string such as ``"STRING"``.
    """
    return _PYDANTIC_TO_GEMINI_TYPE.get(json_type, "STRING")


def build_gemini_declaration(
    name: str,
    description: str,
    args_model: type[BaseModel],
) -> dict:
    """Build a Gemini function declaration dict from a Pydantic model.

    This replaces hand-written declaration blocks in ``llm_client.py``.
    The tool schema and field descriptions come entirely from the model.

    Args:
        name: Tool name the LLM will use when requesting this tool.
        description: Short sentence describing what the tool does.
        args_model: Pydantic ``BaseModel`` subclass whose fields define the parameters.

    Returns:
        A dict matching the Gemini ``function_declarations`` schema.
    """
    json_schema = args_model.model_json_schema()
    properties = json_schema.get("properties", {})
    required_fields = json_schema.get("required", [])

    gemini_properties: dict[str, dict] = {}
    for field_name, field_schema in properties.items():
        gemini_field: dict[str, str] = {
            "type": _pydantic_to_gemini_type(field_schema.get("type", "string")),
        }
        if "description" in field_schema:
            gemini_field["description"] = field_schema["description"]
        gemini_properties[field_name] = gemini_field

    parameters: dict = {"type": "OBJECT", "properties": gemini_properties}
    if required_fields:
        parameters["required"] = required_fields

    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }
