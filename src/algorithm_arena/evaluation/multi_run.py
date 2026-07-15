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
    Runs the same algorithm with n_runs different seeds and returns each
    run's final best_score as an array. Seeds are generated deterministically
    as base_seed, base_seed+1, ... — so when two algorithms are compared we
    can be sure they ran with the same seed sequence (required for a paired
    comparison).
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
    """Statistical comparison of two algorithms' N-seed results."""

    algorithm_a: str
    algorithm_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float
    significant: bool  # True if p_value < alpha
    better: str  # which algorithm is better on average


def compare_two_algorithms(
    name_a: str,
    scores_a: np.ndarray,
    name_b: str,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Compares the paired results of two algorithms (produced with the same
    seed sequence) using the Wilcoxon signed-rank test.

    Note: the Wilcoxon test raises an error when the differences between the
    two arrays are ALL zero (i.e. the algorithms produce identical results on
    every seed) — this rarely happens in practice with stochastic algorithms,
    but we still handle the corner case.
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
        better = "No difference (not statistically significant)"
    else:
        better = (
            name_a if mean_a < mean_b else name_b
        )  # minimization: lower score is better

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
