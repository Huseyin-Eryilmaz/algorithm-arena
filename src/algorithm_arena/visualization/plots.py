from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from algorithm_arena.benchmarks.functions import BenchmarkFunction
from algorithm_arena.optimizers.base import OptimizationState


def _compute_contour_grid(
    benchmark: BenchmarkFunction, resolution: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the benchmark function on a resolution x resolution grid.
    Only meaningful for 2-dimensional functions (contour plots are inherently
    2D) — raises ValueError if n_dims > 2.
    """
    if benchmark.n_dims != 2:
        raise ValueError(
            f"Contour plots are only supported for 2-dimensional functions, "
            f"'{benchmark.name}' has {benchmark.n_dims} dimensions."
        )

    x = np.linspace(
        benchmark.default_bounds.lower[0], benchmark.default_bounds.upper[0], resolution
    )
    y = np.linspace(
        benchmark.default_bounds.lower[1], benchmark.default_bounds.upper[1], resolution
    )
    X, Y = np.meshgrid(x, y)

    # objective_fn expects (n_agents, n_dims); we flatten the grid points into
    # a (resolution*resolution, 2) array and evaluate them all in one batch
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    Z = benchmark.fn(grid_points).reshape(X.shape)

    return X, Y, Z


def build_contour_animation(
    benchmark: BenchmarkFunction,
    states: list[OptimizationState],
    resolution: int = 100,
) -> go.Figure:
    """
    Builds an animated figure showing the optimizer's agent positions,
    iteration by iteration, on top of the benchmark function's contour map.
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

    # First frame: agent positions at the first iteration
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
                contour_trace,  # identical in every frame, never changes
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
    Compares how each algorithm's best_score evolves over iterations in a
    single chart — the answer to "who converges faster".
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


def build_boxplot(results: dict[str, np.ndarray]) -> go.Figure:
    """
    Compares the N-seed best_score distributions of multiple algorithms with
    boxplots. A boxplot shows the spread of the distribution (variance,
    outliers) alongside the mean, so it is more informative than comparing
    means alone — an algorithm can be good on average but inconsistent.
    """
    fig = go.Figure()

    for algorithm_name, scores in results.items():
        fig.add_trace(
            go.Box(
                y=scores,
                name=algorithm_name,
                boxpoints="all",  # also show every individual point, not just the box
                jitter=0.4,
                pointpos=-1.8,
            )
        )

    fig.update_layout(
        title="Best Score Distribution Across Runs",
        yaxis_title="Best Score",
        width=800,
        height=500,
        showlegend=False,
    )

    return fig
