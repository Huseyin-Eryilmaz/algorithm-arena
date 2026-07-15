from __future__ import annotations

from typing import Iterator

import numpy as np

from algorithm_arena.optimizers.base import (
    Bounds,
    ObjectiveFn,
    Optimizer,
    OptimizationState,
)


class HarrisHawksOptimization(Optimizer):
    """
    Harris Hawks Optimization (HHO).
    Models hawks hunting a rabbit: the escape energy E fluctuates randomly
    while decaying from 2 towards 0 as iterations progress. |E| >= 1 ->
    exploration, |E| < 1 -> exploitation (different attack styles:
    soft/hard besiege).
    """

    @property
    def name(self) -> str:
        return "Harris Hawks Optimization"

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

        best_idx = np.argmin(scores)
        rabbit_position = positions[best_idx].copy()
        rabbit_score = scores[best_idx]

        for iteration in range(max_iter):
            E0 = 2 * self.rng.random(self.n_agents) - 1  # initial energy in [-1, 1)
            E = 2 * E0 * (1 - iteration / max_iter)  # decays over time
            J = 2 * (
                1 - self.rng.random(self.n_agents)
            )  # the rabbit's random escape jump

            new_positions = positions.copy()

            for i in range(self.n_agents):
                e = abs(E[i])

                if e >= 1:
                    # --- Exploration: search around a random escape point or the swarm mean
                    if self.rng.random() < 0.5:
                        rand_idx = self.rng.integers(0, self.n_agents)
                        new_positions[i] = positions[
                            rand_idx
                        ] - self.rng.random() * np.abs(
                            positions[rand_idx]
                            - 2 * self.rng.random(n_dims) * positions[i]
                        )
                    else:
                        mean_pos = positions.mean(axis=0)
                        new_positions[i] = (
                            rabbit_position - mean_pos
                        ) - self.rng.random() * (
                            bounds.lower
                            + self.rng.random() * (bounds.upper - bounds.lower)
                        )
                else:
                    # --- Exploitation: four attack styles, chosen based on e and r
                    r = self.rng.random()
                    if r >= 0.5 and e >= 0.5:
                        # Soft besiege
                        new_positions[i] = (
                            rabbit_position - positions[i]
                        ) - e * np.abs(J[i] * rabbit_position - positions[i])
                    elif r >= 0.5 and e < 0.5:
                        # Hard besiege
                        new_positions[i] = rabbit_position - e * np.abs(
                            rabbit_position - positions[i]
                        )
                    elif r < 0.5 and e >= 0.5:
                        # Soft besiege with progressive rapid dives (Levy flight approach simplified)
                        jump = rabbit_position - e * np.abs(
                            J[i] * rabbit_position - positions[i]
                        )
                        new_positions[i] = jump
                    else:
                        # Hard besiege with progressive rapid dives
                        mean_pos = positions.mean(axis=0)
                        jump = rabbit_position - e * np.abs(
                            J[i] * rabbit_position - mean_pos
                        )
                        new_positions[i] = jump

            positions = np.clip(new_positions, bounds.lower, bounds.upper)
            scores = objective_fn(positions)

            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < rabbit_score:
                rabbit_score = scores[current_best_idx]
                rabbit_position = positions[current_best_idx].copy()

            yield OptimizationState(
                iteration=iteration,
                positions=positions.copy(),
                scores=scores.copy(),
                best_position=rabbit_position.copy(),
                best_score=float(rabbit_score),
                algorithm_name=self.name,
            )
