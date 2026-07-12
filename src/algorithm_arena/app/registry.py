"""
Dashboard'da kullanıcıya gösterilecek isimler ile gerçek sınıflar/objeler
arasındaki eşleme. UI kodunu (dashboard.py) optimizer/benchmark detaylarından
izole etmek için ayrı bir dosyada tutuyoruz.
"""

from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.optimizers import (
    PSO,
    CuckooSearch,
    GeneticAlgorithm,
    GreyWolfOptimizer,
    HarrisHawksOptimization,
    SimulatedAnnealing,
)
from algorithm_arena.optimizers.base import Optimizer

OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "Particle Swarm Optimization": PSO,
    "Genetic Algorithm": GeneticAlgorithm,
    "Grey Wolf Optimizer": GreyWolfOptimizer,
    "Harris Hawks Optimization": HarrisHawksOptimization,
    "Simulated Annealing": SimulatedAnnealing,
    "Cuckoo Search": CuckooSearch,
}

BENCHMARK_REGISTRY = {benchmark.name: key for key, benchmark in BENCHMARKS.items()}
# Örn: {"Sphere": "sphere", "Rastrigin": "rastrigin", ...}
