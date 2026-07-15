from __future__ import annotations

from typing import Iterator

import numpy as np

from algorithm_arena.optimizers.base import (
    Bounds,
    ObjectiveFn,
    Optimizer,
    OptimizationState,
)


class PSO(Optimizer):
    """
    Particle Swarm Optimization.
    Each particle is "pulled" towards the best position it has personally
    found (pbest) and the best position known by the swarm (gbest) —
    inspired by the flocking behavior of birds.
    """

    def __init__(
        self,
        n_agents: int = 30,
        seed: int | None = None,
        w: float = 0.7,  # inertia coefficient — how much of the previous velocity is kept
        c1: float = 1.5,  # cognitive learning coefficient (pull towards pbest)
        c2: float = 1.5,  # social learning coefficient (pull towards gbest)
    ):
        super().__init__(n_agents=n_agents, seed=seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    @property
    def name(self) -> str:
        return "Particle Swarm Optimization"

    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> Iterator[OptimizationState]:
        n_dims = bounds.n_dims

        # Initialization: particles are scattered randomly within the bounds
        positions = self.rng.uniform(
            bounds.lower, bounds.upper, size=(self.n_agents, n_dims)
        )
        velocities = np.zeros((self.n_agents, n_dims))

        scores = objective_fn(positions)

        personal_best_positions = positions.copy()
        personal_best_scores = scores.copy()

        best_idx = np.argmin(scores)
        global_best_position = positions[best_idx].copy()
        global_best_score = scores[best_idx]

        for iteration in range(max_iter):
            r1 = self.rng.random((self.n_agents, n_dims))
            r2 = self.rng.random((self.n_agents, n_dims))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (personal_best_positions - positions)
                + self.c2 * r2 * (global_best_position - positions)
            )
            positions = positions + velocities
            positions = np.clip(positions, bounds.lower, bounds.upper)

            scores = objective_fn(positions)

            improved = scores < personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]

            current_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_best_idx]
                global_best_position = personal_best_positions[current_best_idx].copy()

            yield OptimizationState(
                iteration=iteration,
                positions=positions.copy(),
                scores=scores.copy(),
                best_position=global_best_position.copy(),
                best_score=float(global_best_score),
                algorithm_name=self.name,
            )
