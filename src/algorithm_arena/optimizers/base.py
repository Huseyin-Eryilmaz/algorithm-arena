from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

ObjectiveFn = Callable[[np.ndarray], np.ndarray]
"""
An objective function takes a position matrix of shape (n_agents, n_dims)
as input and returns a score vector of shape (n_agents,).
Being vectorized matters for performance — instead of looping over each
agent in Python, it lets numpy do the batch computation at C speed.
"""


@dataclass
class OptimizationState:
    """A snapshot of an optimization algorithm at a single iteration."""

    iteration: int
    positions: np.ndarray  # shape: (n_agents, n_dims) — current position of every agent
    scores: np.ndarray  # shape: (n_agents,) — current score of each agent
    best_position: np.ndarray  # best position found so far
    best_score: float  # best score found so far
    algorithm_name: str = ""  # so the dashboard knows which algorithm this is


@dataclass
class Bounds:
    """Lower/upper limits of the search space, per dimension."""

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def uniform(cls, low: float, high: float, n_dims: int) -> "Bounds":
        """Shortcut for when every dimension shares the same [low, high] range."""
        return cls(
            lower=np.full(n_dims, low, dtype=float),
            upper=np.full(n_dims, high, dtype=float),
        )

    @property
    def n_dims(self) -> int:
        return len(self.lower)


class Optimizer(ABC):
    """
    Common interface that every metaheuristic algorithm must follow.

    Design decision: optimize() is a generator (it uses yield, not return).
    This way:
      - The dashboard can render each iteration live (without waiting for the full result)
      - Headless benchmark mode (Phase 6) can consume only the final state
      - Early stopping can easily be applied from the outside
    """

    def __init__(self, n_agents: int = 30, seed: int | None = None):
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in the dashboard and reports, e.g. 'Particle Swarm Optimization'."""
        ...

    @abstractmethod
    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> Iterator[OptimizationState]:
        """Yields one OptimizationState at the end of every iteration."""
        ...

    def run_to_completion(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> OptimizationState:
        """
        Consumes the generator to the end and returns only the final state.
        Useful for fast test/benchmark scenarios, not for the dashboard.
        """
        final_state = None
        for final_state in self.optimize(objective_fn, bounds, max_iter):
            pass
        if final_state is None:
            raise RuntimeError(f"{self.name}: optimize() never yielded a state")
        return final_state


def collect_states(
    optimizer: "Optimizer",
    objective_fn: ObjectiveFn,
    bounds: Bounds,
    max_iter: int,
) -> list[OptimizationState]:
    """
    Consumes the optimize() generator and collects all states into a list.
    For scenarios that need the full history, like animation/visualization.
    Not for the dashboard's live stream (that consumes the generator directly).
    """
    return list(optimizer.optimize(objective_fn, bounds, max_iter))
