"""Property-style tests that every optimizer must satisfy, regardless of
its internal search strategy."""

import numpy as np
import pytest

from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.optimizers import (
    PSO,
    CuckooSearch,
    GeneticAlgorithm,
    GreyWolfOptimizer,
    HarrisHawksOptimization,
    SimulatedAnnealing,
)

OPTIMIZER_CLASSES = [
    PSO,
    GeneticAlgorithm,
    GreyWolfOptimizer,
    HarrisHawksOptimization,
    SimulatedAnnealing,
    CuckooSearch,
]


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_positions_stay_within_bounds(optimizer_cls):
    """Every reported position in every iteration must respect the search
    space bounds — agents are clipped, never allowed to escape."""
    benchmark = BENCHMARKS["ackley"]
    optimizer = optimizer_cls(n_agents=20, seed=3)

    for state in optimizer.optimize(
        objective_fn=benchmark.fn,
        bounds=benchmark.default_bounds,
        max_iter=30,
    ):
        assert np.all(state.positions >= benchmark.default_bounds.lower - 1e-12)
        assert np.all(state.positions <= benchmark.default_bounds.upper + 1e-12)


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_same_seed_reproduces_identical_run(optimizer_cls):
    """Two runs with the same seed must produce bit-identical trajectories —
    this is what makes paired statistical comparison valid."""
    benchmark = BENCHMARKS["rastrigin"]

    def run():
        optimizer = optimizer_cls(n_agents=15, seed=123)
        return list(
            optimizer.optimize(
                objective_fn=benchmark.fn,
                bounds=benchmark.default_bounds,
                max_iter=20,
            )
        )

    states_a = run()
    states_b = run()

    assert len(states_a) == len(states_b)
    for sa, sb in zip(states_a, states_b):
        np.testing.assert_array_equal(sa.positions, sb.positions)
        assert sa.best_score == sb.best_score


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_different_seeds_produce_different_runs(optimizer_cls):
    """Different seeds should (in practice) lead to different trajectories —
    guards against a silently ignored seed parameter."""
    benchmark = BENCHMARKS["rastrigin"]

    def final_positions(seed):
        optimizer = optimizer_cls(n_agents=15, seed=seed)
        final = optimizer.run_to_completion(
            objective_fn=benchmark.fn,
            bounds=benchmark.default_bounds,
            max_iter=20,
        )
        return final.positions

    assert not np.array_equal(final_positions(0), final_positions(999))


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_best_score_is_consistent_with_best_position(optimizer_cls):
    """The reported best_score must equal the objective evaluated at the
    reported best_position — they must never drift apart."""
    benchmark = BENCHMARKS["sphere"]
    optimizer = optimizer_cls(n_agents=20, seed=5)

    final = optimizer.run_to_completion(
        objective_fn=benchmark.fn,
        bounds=benchmark.default_bounds,
        max_iter=30,
    )

    recomputed = benchmark.fn(final.best_position[None, :])[0]
    assert abs(recomputed - final.best_score) < 1e-9


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_yields_exactly_max_iter_states(optimizer_cls):
    """The generator contract: exactly one state per iteration."""
    benchmark = BENCHMARKS["sphere"]
    optimizer = optimizer_cls(n_agents=10, seed=0)

    states = list(
        optimizer.optimize(
            objective_fn=benchmark.fn,
            bounds=benchmark.default_bounds,
            max_iter=7,
        )
    )
    assert [s.iteration for s in states] == list(range(7))
