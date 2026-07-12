from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from algorithm_arena.benchmarks.functions import BenchmarkFunction
from algorithm_arena.optimizers.base import OptimizationState


def _compute_contour_grid(
    benchmark: BenchmarkFunction, resolution: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Benchmark fonksiyonunu resolution x resolution'lık bir grid üzerinde
    değerlendirir. Sadece 2 boyutlu fonksiyonlar için anlamlı (contour plot
    doğası gereği 2D) — n_dims > 2 ise ValueError fırlatır.
    """
    if benchmark.n_dims != 2:
        raise ValueError(
            f"Contour plot sadece 2 boyutlu fonksiyonlar için desteklenir, "
            f"'{benchmark.name}' {benchmark.n_dims} boyutlu."
        )

    x = np.linspace(
        benchmark.default_bounds.lower[0], benchmark.default_bounds.upper[0], resolution
    )
    y = np.linspace(
        benchmark.default_bounds.lower[1], benchmark.default_bounds.upper[1], resolution
    )
    X, Y = np.meshgrid(x, y)

    # objective_fn (n_agents, n_dims) bekliyor, grid noktalarını düzleştirip
    # (resolution*resolution, 2) şekline getirip topluca değerlendiriyoruz
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    Z = benchmark.fn(grid_points).reshape(X.shape)

    return X, Y, Z


def build_contour_animation(
    benchmark: BenchmarkFunction,
    states: list[OptimizationState],
    resolution: int = 100,
) -> go.Figure:
    """
    Benchmark fonksiyonunun contour haritası üzerinde, optimizer'ın
    ajan pozisyonlarını iterasyon iterasyon gösteren animasyonlu figür üretir.
    """
    X, Y, Z = _compute_contour_grid(benchmark, resolution)

    contour_trace = go.Contour(
        x=X[0],
        y=Y[:, 0],
        z=Z,
        colorscale="Viridis",
        showscale=False,
        contours=dict(coloring="heatmap"),
        opacity=0.85,
    )

    # İlk kare: ilk iterasyondaki ajan pozisyonları
    first_state = states[0]
    scatter_trace = go.Scatter(
        x=first_state.positions[:, 0],
        y=first_state.positions[:, 1],
        mode="markers",
        marker=dict(size=9, color="white", line=dict(width=1.5, color="black")),
        name="Agents",
    )

    frames = [
        go.Frame(
            data=[
                contour_trace,  # her karede aynı, değişmiyor
                go.Scatter(
                    x=state.positions[:, 0],
                    y=state.positions[:, 1],
                    mode="markers",
                    marker=dict(
                        size=9, color="white", line=dict(width=1.5, color="black")
                    ),
                ),
            ],
            name=str(state.iteration),
            layout=go.Layout(
                title=f"{state.algorithm_name} — {benchmark.name} | "
                f"Iteration {state.iteration} | Best: {state.best_score:.6f}"
            ),
        )
        for state in states
    ]

    fig = go.Figure(
        data=[contour_trace, scatter_trace],
        frames=frames,
        layout=go.Layout(
            title=f"{states[0].algorithm_name} — {benchmark.name} | Iteration 0",
            xaxis=dict(title="x", range=[X.min(), X.max()]),
            yaxis=dict(title="y", range=[Y.min(), Y.max()]),
            width=700,
            height=600,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=100, redraw=True),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(state.iteration)],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                ),
                            ],
                            label=str(state.iteration),
                        )
                        for state in states
                    ],
                    active=0,
                )
            ],
        ),
    )

    return fig


def build_convergence_plot(
    results: dict[str, list[OptimizationState]],
    log_scale: bool = True,
) -> go.Figure:
    """
    Birden fazla algoritmanın best_score'unun iterasyona göre değişimini
    tek bir grafikte karşılaştırır — "kim daha hızlı yakınsıyor" sorusunun cevabı.
    """
    fig = go.Figure()

    for algorithm_name, states in results.items():
        iterations = [s.iteration for s in states]
        best_scores = [s.best_score for s in states]

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=best_scores,
                mode="lines",
                name=algorithm_name,
                line=dict(width=2.5),
            )
        )

    fig.update_layout(
        title="Convergence Comparison",
        xaxis_title="Iteration",
        yaxis_title="Best Score" + (" (log scale)" if log_scale else ""),
        yaxis_type="log" if log_scale else "linear",
        width=800,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig
