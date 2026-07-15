import numpy as np

from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.evaluation.multi_run import (
    compare_two_algorithms,
    run_multiple_seeds,
)
from algorithm_arena.optimizers import PSO, CuckooSearch


def test_run_multiple_seeds_returns_correct_shape():
    benchmark = BENCHMARKS["sphere"]
    scores = run_multiple_seeds(
        PSO,
        benchmark.fn,
        benchmark.default_bounds,
        n_agents=10,
        max_iter=20,
        n_runs=5,
        base_seed=0,
    )
    assert scores.shape == (5,)
    assert np.all(scores >= 0)  # Sphere scores can never be negative


def test_run_multiple_seeds_is_deterministic_given_same_base_seed():
    """Running twice with the same base_seed must produce bit-identical results."""
    benchmark = BENCHMARKS["sphere"]
    scores_1 = run_multiple_seeds(
        PSO,
        benchmark.fn,
        benchmark.default_bounds,
        n_agents=10,
        max_iter=20,
        n_runs=5,
        base_seed=42,
    )
    scores_2 = run_multiple_seeds(
        PSO,
        benchmark.fn,
        benchmark.default_bounds,
        n_agents=10,
        max_iter=20,
        n_runs=5,
        base_seed=42,
    )
    np.testing.assert_array_equal(scores_1, scores_2)


def test_different_base_seeds_produce_different_results():
    """Different base_seeds must produce different results (and generally different score distributions)."""
    benchmark = BENCHMARKS["rastrigin"]
    scores_a = run_multiple_seeds(
        PSO,
        benchmark.fn,
        benchmark.default_bounds,
        n_agents=10,
        max_iter=20,
        n_runs=5,
        base_seed=0,
    )
    scores_b = run_multiple_seeds(
        PSO,
        benchmark.fn,
        benchmark.default_bounds,
        n_agents=10,
        max_iter=20,
        n_runs=5,
        base_seed=100,
    )
    assert not np.array_equal(scores_a, scores_b)


def test_compare_two_algorithms_identifies_better_one():
    """
    We run PSO on Rastrigin with few iterations and compare it against
    itself — no difference expected. Then we compare against a different
    algorithm (CuckooSearch, which typically converges more slowly at low
    iteration counts) and expect a meaningful difference.
    """
    benchmark = BENCHMARKS["rastrigin"]

    pso_scores = run_multiple_seeds(
        PSO,
        benchmark.fn,
        benchmark.default_bounds,
        n_agents=20,
        max_iter=15,
        n_runs=15,
        base_seed=0,
    )
    cuckoo_scores = run_multiple_seeds(
        CuckooSearch,
        benchmark.fn,
        benchmark.default_bounds,
        n_agents=20,
        max_iter=15,
        n_runs=15,
        base_seed=0,
    )

    result = compare_two_algorithms("PSO", pso_scores, "Cuckoo Search", cuckoo_scores)

    assert result.algorithm_a == "PSO"
    assert result.algorithm_b == "Cuckoo Search"
    assert 0.0 <= result.p_value <= 1.0
    assert result.better in (
        "PSO",
        "Cuckoo Search",
        "No difference (not statistically significant)",
    )


def test_compare_two_algorithms_handles_identical_scores():
    """If the two arrays are bit-identical (e.g. same algorithm, same seed), no error must be raised."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_two_algorithms("A", scores, "B", scores.copy())

    assert result.p_value == 1.0
    assert result.significant is False
