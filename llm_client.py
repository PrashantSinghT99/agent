from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = """
You are a helpful study agent for a beginner learning AI agents.
Keep answers clear and concise.
Use tools when they are useful instead of guessing.
Do not claim a note exists unless you used a note tool or saw it in context.
If a tool returns an error, explain the problem simply.
""".strip()


TOOL_DECLARATIONS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic math expression.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "expression": {
                    "type": "STRING",
                    "description": "The math expression to evaluate, for example: 45 * 12",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "save_note",
        "description": "Save a short note for the user.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "title": {
                    "type": "STRING",
                    "description": "A short title for the note.",
                },
                "content": {
                    "type": "STRING",
                    "description": "The note content to save.",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "list_notes",
        "description": "List all saved notes.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "read_note",
        "description": "Read the content of a saved note by title.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "title": {
                    "type": "STRING",
                    "description": "The title of the note to read.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "search_notes",
        "description": "Search saved notes by keyword and return matching snippets.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "The keyword or phrase to search for in saved notes.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember_preference",
        "description": "Save a user preference for future conversations.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "key": {
                    "type": "STRING",
                    "description": "Short preference key, for example: explanation_style.",
                },
                "value": {
                    "type": "STRING",
                    "description": "Preference value to remember.",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall_preferences",
        "description": "Recall saved user preferences.",
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
]


def ask_llm(prompt: str) -> str:
    """Send a plain prompt to Gemini without tool declarations.

    Args:
        prompt: User text to send to the model.

    Returns:
        Gemini response text.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text


def ask_llm_with_tools(contents):
    """Send conversation contents to Gemini with available tool declarations.

    Args:
        contents: Gemini-formatted conversation contents.

    Returns:
        Gemini SDK response object, which may contain text or function calls.
    """
    # Tool declarations tell Gemini what it may request; Python still executes tools.
    return client.models.generate_content(
        model=MODEL,
        contents=contents,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": [{"function_declarations": TOOL_DECLARATIONS}],
            "automatic_function_calling": {"disable": True},
        },
    )
