from __future__ import annotations

from typing import Iterator

import numpy as np

from algorithm_arena.optimizers.base import (
    Bounds,
    ObjectiveFn,
    Optimizer,
    OptimizationState,
)


def _levy_flight(
    shape: tuple[int, ...], rng: np.random.Generator, beta: float = 1.5
) -> np.ndarray:
    """
    Mantegna algoritmasıyla Lévy-dağılımlı adım üretir.
    Çoğunlukla küçük adımlar, ara sıra çok büyük sıçramalar üretir — bu da
    Cuckoo Search'ün hem yerel arama hem global keşif yapabilmesini sağlar.
    """
    from scipy.special import gamma

    sigma_u = (
        gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = rng.normal(0, sigma_u, size=shape)
    v = rng.normal(0, 1, size=shape)
    return u / np.abs(v) ** (1 / beta)


class CuckooSearch(Optimizer):
    """
    Cuckoo Search (CS).
    Her iterasyonda yuvalar Lévy flight ile yeni pozisyonlara sıçrar (keşif),
    ayrıca pa oranındaki en kötü yuvalar terk edilip yeniden rastgele
    kurulur (çeşitliliği korumak için).
    """

    def __init__(
        self,
        n_agents: int = 30,
        seed: int | None = None,
        discovery_rate: float = 0.25,  # "pa" — ev sahibi kuş tarafından fark edilme olasılığı
        step_scale: float = 0.5,  # 0.01 -> 0.5
    ):
        super().__init__(n_agents=n_agents, seed=seed)
        self.discovery_rate = discovery_rate
        self.step_scale = step_scale

    @property
    def name(self) -> str:
        return "Cuckoo Search"

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
        global_best_position = positions[best_idx].copy()
        global_best_score = scores[best_idx]

        for iteration in range(max_iter):
            # --- Yeni çözümler: Lévy flight ile mevcut en iyiye doğru sıçrama
            step = _levy_flight((self.n_agents, n_dims), self.rng)
            new_positions = positions + self.step_scale * step * (
                positions - global_best_position
            )
            new_positions = np.clip(new_positions, bounds.lower, bounds.upper)
            new_scores = objective_fn(new_positions)

            improved = new_scores < scores
            positions[improved] = new_positions[improved]
            scores[improved] = new_scores[improved]

            # --- Kötü yuvaların bir kısmı terk edilip rastgele yeniden kuruluyor
            abandon_mask = self.rng.random(self.n_agents) < self.discovery_rate
            n_abandon = int(abandon_mask.sum())
            if n_abandon > 0:
                positions[abandon_mask] = self.rng.uniform(
                    bounds.lower, bounds.upper, size=(n_abandon, n_dims)
                )
                scores[abandon_mask] = objective_fn(positions[abandon_mask])

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
