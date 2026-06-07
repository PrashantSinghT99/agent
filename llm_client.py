"""Gemini LLM client.

Tool declarations are generated automatically from the Pydantic schemas
defined in tools/schemas.py via tool_manager.get_gemini_declarations().
There is no hand-written tool schema in this file.
"""

from dotenv import load_dotenv
from google import genai
import os

from tool_manager import get_gemini_declarations

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

    Tool declarations are generated at call time from the Pydantic schemas
    so they always match the registered tools exactly.

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
            "tools": [{"function_declarations": get_gemini_declarations()}],
            "automatic_function_calling": {"disable": True},
        },
    )
