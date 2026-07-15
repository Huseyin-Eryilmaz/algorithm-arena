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


CONVERGENCE_ITERS = {
    "PSO": 100,
    "GeneticAlgorithm": 100,
    "GreyWolfOptimizer": 100,
    "HarrisHawksOptimization": 100,
    "SimulatedAnnealing": 100,
    "CuckooSearch": 200,  # Lévy-flight based; converges more slowly but more steadily
}


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_converges_on_sphere(optimizer_cls):
    benchmark = BENCHMARKS["sphere"]
    optimizer = optimizer_cls(n_agents=30, seed=42)
    max_iter = CONVERGENCE_ITERS[optimizer_cls.__name__]

    final_state = optimizer.run_to_completion(
        objective_fn=benchmark.fn,
        bounds=benchmark.default_bounds,
        max_iter=max_iter,
    )

    assert final_state.best_score < 1e-2, (
        f"{optimizer.name} failed to converge on Sphere: {final_state.best_score}"
    )


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_state_shapes_are_consistent(optimizer_cls):
    """Are the array shapes inside OptimizationState consistent with expectations?"""
    benchmark = BENCHMARKS["sphere"]
    optimizer = optimizer_cls(n_agents=15, seed=1)

    states = list(
        optimizer.optimize(
            objective_fn=benchmark.fn,
            bounds=benchmark.default_bounds,
            max_iter=5,
        )
    )

    assert len(states) == 5
    for state in states:
        assert state.positions.shape == (15, 2)
        assert state.scores.shape == (15,)


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_best_score_never_increases(optimizer_cls):
    """The best score must never get worse across iterations (must be monotonically non-increasing)."""
    benchmark = BENCHMARKS["rastrigin"]
    optimizer = optimizer_cls(n_agents=30, seed=7)

    scores_over_time = [
        state.best_score
        for state in optimizer.optimize(
            objective_fn=benchmark.fn,
            bounds=benchmark.default_bounds,
            max_iter=50,
        )
    ]

    for earlier, later in zip(scores_over_time, scores_over_time[1:]):
        assert later <= earlier + 1e-9  # small float tolerance
