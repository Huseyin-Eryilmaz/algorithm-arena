from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

ObjectiveFn = Callable[[np.ndarray], np.ndarray]
"""
Bir objective fonksiyonu, (n_agents, n_dims) şeklinde bir pozisyon matrisini
girdi olarak alır, (n_agents,) şeklinde bir skor vektörü döner.
Vektörize olması performans için önemli — her ajan için tek tek Python
döngüsüne girmek yerine numpy'ın C-hızında toplu hesaplama yapmasını sağlar.
"""


@dataclass
class OptimizationState:
    """Bir optimizasyon algoritmasının tek bir iterasyondaki anlık görüntüsü."""

    iteration: int
    positions: np.ndarray  # şekil: (n_agents, n_dims) — tüm ajanların o anki konumu
    scores: np.ndarray  # şekil: (n_agents,) — her ajanın o anki skoru
    best_position: np.ndarray  # şu ana kadarki en iyi konum
    best_score: float  # şu ana kadarki en iyi skor
    algorithm_name: str = ""  # dashboard'da hangi algoritma olduğunu bilmek için


@dataclass
class Bounds:
    """Arama uzayının alt/üst sınırları, her boyut için."""

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def uniform(cls, low: float, high: float, n_dims: int) -> "Bounds":
        """Tüm boyutlar aynı [low, high] aralığındaysa kısayol."""
        return cls(
            lower=np.full(n_dims, low, dtype=float),
            upper=np.full(n_dims, high, dtype=float),
        )

    @property
    def n_dims(self) -> int:
        return len(self.lower)


class Optimizer(ABC):
    """
    Tüm metasezgisel algoritmaların uyması gereken ortak arayüz.

    Tasarım kararı: optimize() bir generator'dır (return değil yield kullanır).
    Bu sayede:
      - Dashboard her iterasyonu anlık olarak çizebilir (tam sonucu beklemeden)
      - Headless benchmark modu (Faz 6) sadece son state'i alıp devam edebilir
      - Erken durdurma (early stopping) dışarıdan kolayca uygulanabilir
    """

    def __init__(self, n_agents: int = 30, seed: int | None = None):
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Dashboard ve raporlarda gösterilecek okunabilir isim, örn. 'Particle Swarm Optimization'."""
        ...

    @abstractmethod
    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> Iterator[OptimizationState]:
        """Her iterasyon sonunda bir OptimizationState yield eder."""
        ...

    def run_to_completion(
        self,
        objective_fn: ObjectiveFn,
        bounds: Bounds,
        max_iter: int,
    ) -> OptimizationState:
        """
        Generator'ı sonuna kadar tüketip sadece son state'i döner.
        Dashboard değil, hızlı test/benchmark senaryoları için kullanışlı.
        """
        final_state = None
        for final_state in self.optimize(objective_fn, bounds, max_iter):
            pass
        if final_state is None:
            raise RuntimeError(f"{self.name}: optimize() hiç state yield etmedi")
        return final_state


def collect_states(
    optimizer: "Optimizer",
    objective_fn: ObjectiveFn,
    bounds: Bounds,
    max_iter: int,
) -> list[OptimizationState]:
    """
    optimize() generator'ını tüketip tüm state'leri listeye toplar.
    Animasyon/görselleştirme gibi tüm geçmişe ihtiyaç duyan senaryolar için.
    Dashboard'daki canlı akış için değil (o direkt generator'ı tüketecek).
    """
    return list(optimizer.optimize(objective_fn, bounds, max_iter))
