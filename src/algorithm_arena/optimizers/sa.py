from __future__ import annotations

from typing import Iterator

import numpy as np

from algorithm_arena.optimizers.base import (
    Bounds,
    ObjectiveFn,
    Optimizer,
    OptimizationState,
)


class SimulatedAnnealing(Optimizer):
    """
    Simulated Annealing (SA), n_agents bağımsız paralel zincir olarak.
    Her zincir kendi sıcaklığıyla soğur; kötü adımlar exp(-delta/T) olasılığıyla
    kabul edilir, bu da algoritmanın yerel minimuma saplanmasını engeller.
    """

    def __init__(
        self,
        n_agents: int = 30,
        seed: int | None = None,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.95,
        step_size_fraction: float = 0.1,  # sınır aralığının yüzdesi olarak komşu adım büyüklüğü
    ):
        super().__init__(n_agents=n_agents, seed=seed)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size_fraction = step_size_fraction

    @property
    def name(self) -> str:
        return "Simulated Annealing"

    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> Iterator[OptimizationState]:
        n_dims = bounds.n_dims
        step_size = self.step_size_fraction * (bounds.upper - bounds.lower)

        positions = self.rng.uniform(
            bounds.lower, bounds.upper, size=(self.n_agents, n_dims)
        )
        scores = objective_fn(positions)

        best_idx = np.argmin(scores)
        global_best_position = positions[best_idx].copy()
        global_best_score = scores[best_idx]

        temperature = self.initial_temp

        for iteration in range(max_iter):
            candidates = (
                positions
                + self.rng.normal(0, 1, size=(self.n_agents, n_dims)) * step_size
            )
            candidates = np.clip(candidates, bounds.lower, bounds.upper)
            candidate_scores = objective_fn(candidates)

            delta = candidate_scores - scores
            # delta <= 0 -> her zaman kabul; delta > 0 -> exp(-delta/T) olasılığıyla kabul
            acceptance_prob = np.exp(
                -np.clip(delta / max(temperature, 1e-9), 0, 50) * (delta > 0)
            )
            accept = (delta <= 0) | (self.rng.random(self.n_agents) < acceptance_prob)

            positions[accept] = candidates[accept]
            scores[accept] = candidate_scores[accept]

            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx].copy()

            temperature *= self.cooling_rate

            yield OptimizationState(
                iteration=iteration,
                positions=positions.copy(),
                scores=scores.copy(),
                best_position=global_best_position.copy(),
                best_score=float(global_best_score),
                algorithm_name=self.name,
            )
