"""Safely evaluate a recognised arithmetic expression without Python eval()."""

from __future__ import annotations

import ast
import operator

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_expression(expression: str) -> int | float:
    """Evaluate digits, parentheses and + - * / operators using a strict AST allowlist."""
    normalised = expression.replace("÷", "/").replace("×", "*").strip()
    if not normalised or len(normalised) > 100:
        raise ValueError("Expression must contain between 1 and 100 characters.")
    try:
        tree = ast.parse(normalised, mode="eval")
    except SyntaxError as exc:
        raise ValueError("The recognised symbols do not form a valid expression.") from exc

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand))
        raise ValueError("Only numbers, parentheses, +, -, *, and / are permitted.")

    try:
        result = visit(tree)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero is not allowed.") from exc
    if abs(result) > 1e12:
        raise ValueError("Result is outside the supported range.")
    return int(result) if isinstance(result, float) and result.is_integer() else result

