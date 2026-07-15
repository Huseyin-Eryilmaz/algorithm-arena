import numpy as np
import pytest

from algorithm_arena.benchmarks.custom import InvalidExpressionError, parse_expression


def test_parses_simple_sphere_expression():
    fn, expr = parse_expression("x**2 + y**2")
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    scores = fn(positions)

    np.testing.assert_allclose(scores, [0.0, 2.0, 4.0])


def test_parses_expression_with_trig_functions():
    fn, expr = parse_expression("sin(x) + cos(y)")
    positions = np.array([[0.0, 0.0]])
    scores = fn(positions)

    np.testing.assert_allclose(scores, [np.sin(0.0) + np.cos(0.0)])


def test_constant_expression_broadcasts_to_all_agents():
    """Even an expression without x/y, like '5', must broadcast correctly into a vector."""
    fn, expr = parse_expression("5")
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    scores = fn(positions)

    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, [5.0, 5.0, 5.0])


def test_rejects_undefined_variable():
    with pytest.raises(InvalidExpressionError, match="Only the variables 'x' and 'y'"):
        parse_expression("x**2 + z**2")


def test_rejects_malformed_expression():
    with pytest.raises(InvalidExpressionError):
        parse_expression("sin(x")  # unclosed parenthesis


def test_rejects_dangerous_input_gracefully():
    """
    A dangerous input (e.g. an import attempt) must already be rejected by
    sympy as mathematically meaningless and raise an error, never run
    silently.
    """
    with pytest.raises(InvalidExpressionError):
        parse_expression("__import__('os').system('echo test')")


def test_rejects_dunder_pattern_via_whitelist():
    """Any input containing an underscore must be filtered by the regex before reaching parse_expr."""
    with pytest.raises(InvalidExpressionError):
        parse_expression("__import__")


def test_caret_is_interpreted_as_power():
    """convert_xor is enabled, so 'x^2' should behave like 'x**2'."""
    fn, expr = parse_expression("x^2 + y^2")
    positions = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(fn(positions), [13.0])


def test_expression_with_only_one_variable():
    """An expression using only x must still broadcast over all agents."""
    fn, expr = parse_expression("sin(x)")
    positions = np.array([[0.0, 5.0], [np.pi / 2, -3.0]])
    np.testing.assert_allclose(fn(positions), [0.0, 1.0], atol=1e-12)


def test_rejects_quotes_via_whitelist():
    """Quotes never reach parse_expr — the regex whitelist blocks them first."""
    with pytest.raises(InvalidExpressionError, match="disallowed characters"):
        parse_expression("exp('x')")


def test_rejects_brackets_and_attribute_access():
    """Indexing/attribute-style payloads are blocked by the whitelist."""
    with pytest.raises(InvalidExpressionError):
        parse_expression("x[0]")
    with pytest.raises(InvalidExpressionError):
        parse_expression("x.__class__")


def test_rejects_disallowed_function_name():
    """A syntactically valid call to a non-whitelisted name must fail."""
    with pytest.raises(InvalidExpressionError):
        parse_expression("open(x)")


def test_rejects_empty_expression():
    with pytest.raises(InvalidExpressionError):
        parse_expression("")
