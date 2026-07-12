"""
Faz 3 görsellerini manuel olarak deneyip HTML dosyası olarak kaydetmek için.
Kullanım: uv run python scripts/demo_visualization.py
"""

from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.optimizers import PSO, GreyWolfOptimizer, HarrisHawksOptimization
from algorithm_arena.optimizers.base import collect_states
from algorithm_arena.visualization import (
    build_contour_animation,
    build_convergence_plot,
)

benchmark = BENCHMARKS["rastrigin"]

# --- Tek algoritmanın contour animasyonu
optimizer = PSO(n_agents=25, seed=0)
states = collect_states(optimizer, benchmark.fn, benchmark.default_bounds, max_iter=60)

fig1 = build_contour_animation(benchmark, states)
fig1.write_html("scripts/output_contour.html")
print("Contour animasyonu kaydedildi: scripts/output_contour.html")

# --- Üç algoritmanın convergence karşılaştırması
results = {}
for cls in [PSO, GreyWolfOptimizer, HarrisHawksOptimization]:
    opt = cls(n_agents=25, seed=0)
    results[opt.name] = collect_states(
        opt, benchmark.fn, benchmark.default_bounds, max_iter=60
    )

fig2 = build_convergence_plot(results)
fig2.write_html("scripts/output_convergence.html")
print("Convergence grafiği kaydedildi: scripts/output_convergence.html")
