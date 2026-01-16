from __future__ import annotations

import ast
import re
from dataclasses import dataclass


class CalcError(ValueError):
    pass


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


@dataclass(frozen=True)
class CalcResult:
    value: float
    formatted: str


def _preprocess(expr: str) -> str:
    """
    Мини-поддержка процентов:
    '3.5% * 12000' -> '(3.5/100) * 12000'
    """
    s = expr.strip()
    if len(s) > 200:
        raise CalcError("Expression too long (max 200 chars).")

    # заменить запятые в десятичных на точки (частый ввод)
    s = s.replace(",", ".")

    # number% -> (number/100)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    return s


def _validate_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_ast(node.body)
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return
        raise CalcError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise CalcError("Only + - * / // ** operators are allowed.")
        _validate_ast(node.left)
        _validate_ast(node.right)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise CalcError("Only unary + and - are allowed.")
        _validate_ast(node.operand)
        return

    raise CalcError(f"Disallowed expression element: {type(node).__name__}")


def safe_calc(expr: str) -> CalcResult:
    s = _preprocess(expr)
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"Invalid expression: {e.msg}") from e

    _validate_ast(tree)

    value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}})
    try:
        v = float(value)
    except Exception as e:
        raise CalcError("Expression did not evaluate to a number.") from e

    if not (abs(v) < 1e308):
        raise CalcError("Result magnitude is too large.")

    # аккуратное форматирование
    formatted = f"{v:.6f}".rstrip("0").rstrip(".")
    return CalcResult(value=v, formatted=formatted)
