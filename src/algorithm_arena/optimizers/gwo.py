from __future__ import annotations

from typing import Iterator

import numpy as np

from algorithm_arena.optimizers.base import (
    Bounds,
    ObjectiveFn,
    Optimizer,
    OptimizationState,
)


class GreyWolfOptimizer(Optimizer):
    """
    Grey Wolf Optimizer (GWO).
    Popülasyondaki en iyi 3 birey (alpha, beta, delta) lider kabul edilir,
    diğer tüm kurtlar bu üçünün "işaret ettiği" ortalama yöne hareket eder.
    `a` katsayısı 2 -> 0 lineer azalarak keşiften sömürüye geçişi sağlar.
    """

    @property
    def name(self) -> str:
        return "Grey Wolf Optimizer"

    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> Iterator[OptimizationState]:
        n_dims = bounds.n_dims

        positions = self.rng.uniform(
            bounds.lower, bounds.upper, size=(self.n_agents, n_dims)
        )
        scores = objective_fn(positions)

        def top_three(pos: np.ndarray, sc: np.ndarray):
            order = np.argsort(sc)[:3]
            return pos[order].copy(), sc[order].copy()

        leaders_pos, leaders_score = top_three(positions, scores)
        alpha, beta, delta = leaders_pos[0], leaders_pos[1], leaders_pos[2]
        alpha_score = leaders_score[0]

        global_best_position = alpha.copy()
        global_best_score = alpha_score

        for iteration in range(max_iter):
            a = 2 - iteration * (2 / max_iter)

            def move_towards(leader: np.ndarray) -> np.ndarray:
                r1 = self.rng.random((self.n_agents, n_dims))
                r2 = self.rng.random((self.n_agents, n_dims))
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - positions)
                return leader - A * D

            X1 = move_towards(alpha)
            X2 = move_towards(beta)
            X3 = move_towards(delta)

            positions = np.clip((X1 + X2 + X3) / 3.0, bounds.lower, bounds.upper)
            scores = objective_fn(positions)
            leaders_pos, leaders_score = top_three(positions, scores)
            alpha, beta, delta = leaders_pos[0], leaders_pos[1], leaders_pos[2]

            if leaders_score[0] < global_best_score:
                global_best_score = leaders_score[0]
                global_best_position = alpha.copy()

            yield OptimizationState(
                iteration=iteration,
                positions=positions.copy(),
                scores=scores.copy(),
                best_position=global_best_position.copy(),
                best_score=float(global_best_score),
                algorithm_name=self.name,
            )
