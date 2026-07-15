import numpy as np
import pytest

from algorithm_arena.benchmarks.functions import BENCHMARKS

# Known global minimum locations for each benchmark (2D).
KNOWN_MINIMA = {
    "sphere": np.array([[0.0, 0.0]]),
    "rastrigin": np.array([[0.0, 0.0]]),
    "ackley": np.array([[0.0, 0.0]]),
    "rosenbrock": np.array([[1.0, 1.0]]),
}


@pytest.mark.parametrize("key", list(BENCHMARKS.keys()))
def test_benchmark_value_at_known_global_minimum(key):
    """Each benchmark must evaluate to its documented global minimum
    at the known minimizer (up to floating-point error)."""
    benchmark = BENCHMARKS[key]
    value = benchmark.fn(KNOWN_MINIMA[key])
    assert value.shape == (1,)
    assert abs(value[0] - benchmark.global_minimum) < 1e-8


@pytest.mark.parametrize("key", list(BENCHMARKS.keys()))
def test_benchmark_is_vectorized(key):
    """Benchmarks must accept a whole (n_agents, n_dims) batch at once
    and return one score per agent."""
    benchmark = BENCHMARKS[key]
    rng = np.random.default_rng(0)
    positions = rng.uniform(
        benchmark.default_bounds.lower,
        benchmark.default_bounds.upper,
        size=(50, benchmark.n_dims),
    )
    scores = benchmark.fn(positions)
    assert scores.shape == (50,)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("key", list(BENCHMARKS.keys()))
def test_benchmark_batch_matches_single_evaluations(key):
    """Evaluating a batch must give the same result as evaluating each
    point individually — guards against broken axis handling."""
    benchmark = BENCHMARKS[key]
    rng = np.random.default_rng(1)
    positions = rng.uniform(
        benchmark.default_bounds.lower,
        benchmark.default_bounds.upper,
        size=(10, benchmark.n_dims),
    )
    batch_scores = benchmark.fn(positions)
    single_scores = np.array([benchmark.fn(p[None, :])[0] for p in positions])
    np.testing.assert_allclose(batch_scores, single_scores)


@pytest.mark.parametrize("key", list(BENCHMARKS.keys()))
def test_global_minimum_is_a_lower_bound(key):
    """No sampled point may score below the documented global minimum."""
    benchmark = BENCHMARKS[key]
    rng = np.random.default_rng(2)
    positions = rng.uniform(
        benchmark.default_bounds.lower,
        benchmark.default_bounds.upper,
        size=(1000, benchmark.n_dims),
    )
    scores = benchmark.fn(positions)
    assert np.all(scores >= benchmark.global_minimum - 1e-9)
