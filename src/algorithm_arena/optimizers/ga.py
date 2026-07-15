from __future__ import annotations

from typing import Iterator

import numpy as np

from algorithm_arena.optimizers.base import (
    Bounds,
    ObjectiveFn,
    Optimizer,
    OptimizationState,
)


class GeneticAlgorithm(Optimizer):
    """
    Classic real-valued genetic algorithm: tournament selection, arithmetic
    crossover, Gaussian mutation, and elitism.
    """

    def __init__(
        self,
        n_agents: int = 30,
        seed: int | None = None,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,  # mutation magnitude as a fraction of the bound range
        crossover_rate: float = 0.8,
        elite_fraction: float = 0.1,  # fraction of top individuals carried over unchanged
        tournament_size: int = 3,
    ):
        super().__init__(n_agents=n_agents, seed=seed)
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size

    @property
    def name(self) -> str:
        return "Genetic Algorithm"

    def _tournament_select(
        self, positions: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Runs n_agents tournaments and selects that many parents (minimization: lowest score wins)."""
        n = len(positions)
        selected = np.empty_like(positions)
        for i in range(n):
            competitors = self.rng.choice(n, size=self.tournament_size, replace=False)
            winner = competitors[np.argmin(scores[competitors])]
            selected[i] = positions[winner]
        return selected

    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> Iterator[OptimizationState]:
        n_dims = bounds.n_dims
        n_elite = max(1, int(self.n_agents * self.elite_fraction))
        mutation_scale = self.mutation_strength * (bounds.upper - bounds.lower)

        positions = self.rng.uniform(
            bounds.lower, bounds.upper, size=(self.n_agents, n_dims)
        )
        scores = objective_fn(positions)

        best_idx = np.argmin(scores)
        global_best_position = positions[best_idx].copy()
        global_best_score = scores[best_idx]

        for iteration in range(max_iter):
            elite_idx = np.argsort(scores)[:n_elite]
            elites = positions[elite_idx].copy()

            parents = self._tournament_select(positions, scores)
            partner_order = self.rng.permutation(self.n_agents)
            partners = parents[partner_order]

            crossover_mask = self.rng.random(self.n_agents) < self.crossover_rate
            alpha = self.rng.random((self.n_agents, n_dims))
            offspring = np.where(
                crossover_mask[:, None],
                alpha * parents + (1 - alpha) * partners,
                parents,
            )

            mutation_mask = (
                self.rng.random((self.n_agents, n_dims)) < self.mutation_rate
            )
            noise = self.rng.normal(0, 1, size=(self.n_agents, n_dims)) * mutation_scale
            offspring = offspring + mutation_mask * noise
            offspring = np.clip(offspring, bounds.lower, bounds.upper)

            # Elitism: the best individuals pass directly into the new generation (so they are never lost)
            offspring[:n_elite] = elites

            positions = offspring
            scores = objective_fn(positions)

            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx].copy()

            yield OptimizationState(
                iteration=iteration,
                positions=positions.copy(),
                scores=scores.copy(),
                best_position=global_best_position.copy(),
                best_score=float(global_best_score),
                algorithm_name=self.name,
            )
