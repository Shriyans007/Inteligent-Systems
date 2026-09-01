import pytest

from extension.safe_expression import evaluate_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2+3*4", 14), ("(2+3)*4", 20), ("18÷3-1", 5), ("-4+10", 6)],
)
def test_supported_expressions(expression, expected):
    assert evaluate_expression(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "2**8", "abc", "1/0", ""])
def test_unsafe_or_invalid_expressions_are_rejected(expression):
    with pytest.raises(ValueError):
        evaluate_expression(expression)

