"""
Mapping between the names shown to the user in the dashboard and the actual
classes/objects. Kept in a separate file to isolate the UI code (dashboard.py)
from optimizer/benchmark details.
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
# E.g.: {"Sphere": "sphere", "Rastrigin": "rastrigin", ...}
