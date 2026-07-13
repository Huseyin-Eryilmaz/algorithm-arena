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
    assert np.all(scores >= 0)  # Sphere skorları hiç negatif olamaz


def test_run_multiple_seeds_is_deterministic_given_same_base_seed():
    """Aynı base_seed ile iki kere çalıştırınca birebir aynı sonuç gelmeli."""
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
    """Farklı base_seed'ler farklı (ve genelde farklı skor dağılımları) üretmeli."""
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
    Yapay olarak PSO'yu Sphere'de çok iyi (az iterasyon farkıyla) çalıştırıp
    kendisiyle karşılaştırıyoruz - farksız çıkmalı. Sonra farklı bir
    algoritmayla (CuckooSearch, tipik olarak daha yavaş yakınsar düşük
    iterasyonda) karşılaştırıp anlamlı bir fark bekliyoruz.
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
        "Fark yok (istatistiksel olarak anlamsız)",
    )


def test_compare_two_algorithms_handles_identical_scores():
    """İki dizi birebir aynıysa (örn. aynı algoritma, aynı seed) hata fırlatmamalı."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_two_algorithms("A", scores, "B", scores.copy())

    assert result.p_value == 1.0
    assert result.significant is False
