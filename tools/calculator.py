# tools/calculator.py
import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    """Evaluate a restricted Python AST node as a math expression.

    Args:
        node: AST node produced by ast.parse(..., mode="eval").

    Returns:
        Numeric result for the expression.

    Raises:
        ValueError: If the expression contains unsupported syntax.
    """
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        return OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        operand = _evaluate(node.operand)
        return OPERATORS[type(node.op)](operand)

    raise ValueError("Only basic math expressions are allowed.")


def calculate(expression: str) -> str:
    """Evaluate a basic math expression.

    Args:
        expression: Math expression such as "45 * 12" or "2 ** 5".

    Returns:
        The calculated result as text, or an error message.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        return str(_evaluate(parsed))
    except Exception as e:
        return f"Calculator error: {e}"
