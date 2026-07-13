from __future__ import annotations

import numpy as np
import warnings
from algorithm_arena.optimizers.base import Bounds, ObjectiveFn, Optimizer
from typing import cast

from dataclasses import dataclass
from scipy import stats


def run_multiple_seeds(
    optimizer_cls: type[Optimizer],
    objective_fn: ObjectiveFn,
    bounds: Bounds,
    n_agents: int,
    max_iter: int,
    n_runs: int,
    base_seed: int = 0,
    **optimizer_kwargs,
) -> np.ndarray:
    """
    Aynı algoritmayı n_runs farklı seed ile çalıştırıp her run'ın nihai
    best_score'unu bir array olarak döner. Seed'ler base_seed, base_seed+1, ...
    şeklinde deterministik üretilir — böylece iki algoritma karşılaştırılırken
    aynı seed dizisiyle çalıştıklarından emin oluruz (paired comparison için şart).
    """
    scores = np.empty(n_runs)
    for i in range(n_runs):
        seed = base_seed + i
        optimizer = optimizer_cls(n_agents=n_agents, seed=seed, **optimizer_kwargs)
        final_state = optimizer.run_to_completion(objective_fn, bounds, max_iter)
        scores[i] = final_state.best_score
    return scores


@dataclass
class ComparisonResult:
    """İki algoritmanın N-seed sonuçlarının istatistiksel karşılaştırması."""

    algorithm_a: str
    algorithm_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float
    significant: bool  # p_value < alpha ise True
    better: str  # hangi algoritmanın ortalama olarak daha iyi olduğu


def compare_two_algorithms(
    name_a: str,
    scores_a: np.ndarray,
    name_b: str,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    İki algoritmanın eşleştirilmiş (aynı seed sırasıyla üretilmiş) sonuçlarını
    Wilcoxon signed-rank test ile karşılaştırır.

    Not: Wilcoxon testi, iki dizi arasındaki farkların TAMAMEN sıfır olduğu
    durumda (algoritmalar her seed'de birebir aynı sonucu verirse) hata
    fırlatır — bu genelde stokastik algoritmalarda pratikte olmaz ama
    yine de bu köşe durumunu ele alıyoruz.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in scalar divide",
                category=RuntimeWarning,
            )
            wilcoxon_result = stats.wilcoxon(scores_a, scores_b)
        p_value = float(cast(float, wilcoxon_result[1]))
    except ValueError:
        p_value = 1.0

    mean_a, mean_b = float(np.mean(scores_a)), float(np.mean(scores_b))
    significant = p_value < alpha

    if not significant:
        better = "Fark yok (istatistiksel olarak anlamsız)"
    else:
        better = name_a if mean_a < mean_b else name_b  # minimizasyon: düşük skor iyi

    return ComparisonResult(
        algorithm_a=name_a,
        algorithm_b=name_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=float(np.std(scores_a)),
        std_b=float(np.std(scores_b)),
        p_value=float(p_value),
        significant=significant,
        better=better,
    )
