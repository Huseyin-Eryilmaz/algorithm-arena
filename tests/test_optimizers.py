import pytest

from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.optimizers import PSO, GeneticAlgorithm

OPTIMIZER_CLASSES = [PSO, GeneticAlgorithm]


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_converges_on_sphere(optimizer_cls):
    """
    Sphere en kolay fonksiyon (tek minimum, konveks) — her ciddi algoritma
    yeterli iterasyonda global minimuma (0,0 civarına) yakınsamalı.
    """
    benchmark = BENCHMARKS["sphere"]
    optimizer = optimizer_cls(n_agents=30, seed=42)

    final_state = optimizer.run_to_completion(
        objective_fn=benchmark.fn,
        bounds=benchmark.default_bounds,
        max_iter=100,
    )

    assert (
        final_state.best_score < 1e-2
    ), f"{optimizer.name} Sphere'de yakınsayamadı: {final_state.best_score}"


@pytest.mark.parametrize("optimizer_cls", OPTIMIZER_CLASSES)
def test_state_shapes_are_consistent(optimizer_cls):
    """OptimizationState içindeki array boyutları beklenenle tutarlı mı."""
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
    """En iyi skor iterasyonlar boyunca hiç kötüleşmemeli (monoton azalan olmalı)."""
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
        assert later <= earlier + 1e-9  # küçük float toleransı
