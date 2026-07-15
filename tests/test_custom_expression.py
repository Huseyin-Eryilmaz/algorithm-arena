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
    with pytest.raises(InvalidExpressionError, match="Sadece 'x' ve 'y'"):
        parse_expression("x**2 + z**2")


def test_rejects_malformed_expression():
    with pytest.raises(InvalidExpressionError):
        parse_expression("sin(x")  # kapanmamış parantez


def test_rejects_dangerous_input_gracefully():
    """
    Tehlikeli bir input (örn. import denemesi) sympy tarafından zaten
    matematiksel olarak anlamsız sayılıp hata fırlatmalı, sessizce
    çalışmamalı.
    """
    with pytest.raises(InvalidExpressionError):
        parse_expression("__import__('os').system('echo test')")


def test_rejects_dunder_pattern_via_whitelist():
    """Alt çizgi içeren hiçbir input, parse_expr'e ulaşmadan regex'te elenmeli."""
    with pytest.raises(InvalidExpressionError):
        parse_expression("__import__")
