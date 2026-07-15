import plotly.graph_objects as go

from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.optimizers import PSO
from algorithm_arena.optimizers.base import collect_states
from algorithm_arena.visualization import (
    build_contour_animation,
    build_convergence_plot,
)


def test_contour_animation_has_correct_frame_count():
    benchmark = BENCHMARKS["sphere"]
    optimizer = PSO(n_agents=10, seed=0)
    states = collect_states(
        optimizer, benchmark.fn, benchmark.default_bounds, max_iter=15
    )

    fig = build_contour_animation(benchmark, states)
    frames = list(fig.frames)

    assert isinstance(fig, go.Figure)
    assert len(frames) == 15


def test_contour_animation_rejects_non_2d_benchmark():
    """Only 2D functions are supported for now; must fail explicitly."""
    import pytest
    from algorithm_arena.optimizers.base import Bounds
    from algorithm_arena.benchmarks.functions import BenchmarkFunction, sphere

    fake_3d_benchmark = BenchmarkFunction(
        name="Sphere3D",
        fn=sphere,
        default_bounds=Bounds.uniform(-5, 5, n_dims=3),
        global_minimum=0.0,
        n_dims=3,
    )

    with pytest.raises(ValueError, match="2-dimensional"):
        build_contour_animation(fake_3d_benchmark, states=[])


def test_convergence_plot_has_one_trace_per_algorithm():
    benchmark = BENCHMARKS["sphere"]
    results = {}
    for seed in [0, 1]:
        optimizer = PSO(n_agents=10, seed=seed)
        results[f"PSO-seed{seed}"] = collect_states(
            optimizer, benchmark.fn, benchmark.default_bounds, max_iter=10
        )

    fig = build_convergence_plot(results)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_boxplot_has_one_box_per_algorithm():
    import numpy as np

    from algorithm_arena.visualization import build_boxplot

    results = {
        "PSO": np.array([0.1, 0.2, 0.3]),
        "GA": np.array([0.4, 0.5, 0.6]),
    }
    fig = build_boxplot(results)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert all(trace.type == "box" for trace in fig.data)
