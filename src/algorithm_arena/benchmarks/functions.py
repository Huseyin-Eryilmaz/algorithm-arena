from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithm_arena.optimizers.base import Bounds, ObjectiveFn


@dataclass
class BenchmarkFunction:
    """Bir test fonksiyonunu, sınırlarını ve bilinen global minimumunu birlikte taşır."""

    name: str
    fn: ObjectiveFn
    default_bounds: Bounds
    global_minimum: float
    n_dims: int = 2


def sphere(x: np.ndarray) -> np.ndarray:
    """En basit fonksiyon: tek minimumlu, çanak şeklinde. x.shape = (n_agents, n_dims)."""
    return np.sum(x**2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    """Çok sayıda yerel minimumu olan, algoritmaları 'kandırmak' için klasik test fonksiyonu."""
    n_dims = x.shape[1]
    A = 10
    return A * n_dims + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)


def ackley(x: np.ndarray) -> np.ndarray:
    """Ortası düz, kenarlara doğru sert inişli — algoritmaların 'yerel minimuma takılma' testi."""
    n_dims = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n_dims))
    term2 = -np.exp(sum_cos / n_dims)
    return term1 + term2 + 20 + np.e


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """'Banana function' — dar, kavisli bir vadi, algoritmaların yakınsama hızını test eder."""
    return np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1
    )


BENCHMARKS: dict[str, BenchmarkFunction] = {
    "sphere": BenchmarkFunction(
        name="Sphere",
        fn=sphere,
        default_bounds=Bounds.uniform(-5.12, 5.12, n_dims=2),
        global_minimum=0.0,
    ),
    "rastrigin": BenchmarkFunction(
        name="Rastrigin",
        fn=rastrigin,
        default_bounds=Bounds.uniform(-5.12, 5.12, n_dims=2),
        global_minimum=0.0,
    ),
    "ackley": BenchmarkFunction(
        name="Ackley",
        fn=ackley,
        default_bounds=Bounds.uniform(-32.768, 32.768, n_dims=2),
        global_minimum=0.0,
    ),
    "rosenbrock": BenchmarkFunction(
        name="Rosenbrock",
        fn=rosenbrock,
        default_bounds=Bounds.uniform(-2.048, 2.048, n_dims=2),
        global_minimum=0.0,
    ),
}
