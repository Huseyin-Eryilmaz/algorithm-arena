"""
Algorithm Arena — Streamlit Dashboard.
To run: uv run streamlit run src/algorithm_arena/app/dashboard.py
"""

import streamlit as st
import numpy as np

from algorithm_arena.app.registry import BENCHMARK_REGISTRY, OPTIMIZER_REGISTRY
from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.optimizers.base import collect_states
from algorithm_arena.visualization import (
    build_contour_animation,
    build_convergence_plot,
)
from algorithm_arena.benchmarks.custom import InvalidExpressionError, parse_expression
from algorithm_arena.benchmarks.functions import BenchmarkFunction
from algorithm_arena.optimizers.base import Bounds

from itertools import combinations

from algorithm_arena.evaluation.multi_run import (
    compare_two_algorithms,
    run_multiple_seeds,
)
from algorithm_arena.visualization import build_boxplot

st.set_page_config(page_title="Algorithm Arena", layout="wide")

st.title("🐺 Algorithm Arena")
st.caption("Metaheuristic optimization algorithms racing on benchmark functions")

tab_single, tab_race, tab_stats = st.tabs(
    ["🎯 Single Run", "🏁 Race Mode", "📊 Statistical Comparison"]
)

with tab_single:
    col_controls, col_display = st.columns([1, 3])

    with col_controls:
        st.subheader("Settings")

        algorithm_name: str = st.selectbox(
            "Algorithm", options=list(OPTIMIZER_REGISTRY.keys()), key="single_algo"
        )  # type: ignore[assignment]

        function_mode = st.radio(
            "Function Source",
            options=["Preset Benchmark", "Custom Expression"],
            key="single_mode",
        )

        # We give both branches default values, so that whichever mode is
        # entered, every variable is defined and has the correct type.
        benchmark_name: str | None = None
        custom_expr_str: str = "x**2 + y**2"
        custom_low: float = -5.0
        custom_high: float = 5.0

        if function_mode == "Preset Benchmark":
            benchmark_name = st.selectbox(
                "Benchmark Function",
                options=list(BENCHMARK_REGISTRY.keys()),
                key="single_bench",
            )
        else:
            custom_expr_str = st.text_input(
                "Expression (use x, y)",
                value=custom_expr_str,
                key="single_custom_expr",
                help="Example: x**2 + y**2 + 10*sin(x)  |  Supported functions: sin, cos, exp, sqrt, log...",
            )
            custom_low = st.number_input(
                "Lower Bound", value=custom_low, key="single_custom_low"
            )
            custom_high = st.number_input(
                "Upper Bound", value=custom_high, key="single_custom_high"
            )

        n_agents = st.slider(
            "Number of Agents",
            min_value=5,
            max_value=100,
            value=30,
            key="single_agents",
        )
        max_iter = st.slider(
            "Iterations", min_value=10, max_value=200, value=60, key="single_iter"
        )
        seed = st.number_input("Random Seed", min_value=0, value=42, key="single_seed")

        run_button = st.button(
            "▶ Run Optimization", key="single_run_btn", type="primary"
        )

    with col_display:
        if run_button:
            if function_mode == "Preset Benchmark":
                assert (
                    benchmark_name is not None
                )  # in this mode the selectbox always returns a value
                benchmark_key = BENCHMARK_REGISTRY[benchmark_name]
                benchmark = BENCHMARKS[benchmark_key]
                display_name = benchmark_name
            else:
                try:
                    custom_fn, parsed_expr = parse_expression(custom_expr_str)
                except InvalidExpressionError as e:
                    st.error(str(e))
                    st.stop()

                benchmark = BenchmarkFunction(
                    name=f"Custom: {custom_expr_str}",
                    fn=custom_fn,
                    default_bounds=Bounds.uniform(custom_low, custom_high, n_dims=2),
                    global_minimum=float("nan"),
                )
                display_name = f"Custom ({parsed_expr})"
                st.caption(f"Parsed as: `{parsed_expr}`")

            optimizer_cls = OPTIMIZER_REGISTRY[algorithm_name]
            optimizer = optimizer_cls(n_agents=n_agents, seed=int(seed))

            with st.spinner(f"Running {algorithm_name} on {display_name}..."):
                states = collect_states(
                    optimizer, benchmark.fn, benchmark.default_bounds, max_iter=max_iter
                )

            final_state = states[-1]
            st.metric("Best Score Found", f"{final_state.best_score:.6f}")

            fig_contour = build_contour_animation(benchmark, states)
            st.plotly_chart(fig_contour, width="stretch")

            fig_convergence = build_convergence_plot({algorithm_name: states})
            st.plotly_chart(fig_convergence, width="stretch")
        else:
            st.info("👈 Choose your settings and press 'Run Optimization'.")

with tab_race:
    col_controls, col_display = st.columns([1, 3])

    with col_controls:
        st.subheader("Settings")

        selected_algorithms = st.multiselect(
            "Algorithms to Compare",
            options=list(OPTIMIZER_REGISTRY.keys()),
            default=list(OPTIMIZER_REGISTRY.keys())[:3],
            key="race_algos",
        )
        benchmark_name_race = st.selectbox(
            "Benchmark Function",
            options=list(BENCHMARK_REGISTRY.keys()),
            key="race_bench",
        )

        n_agents_race = st.slider(
            "Number of Agents", min_value=5, max_value=100, value=30, key="race_agents"
        )
        max_iter_race = st.slider(
            "Iterations", min_value=10, max_value=200, value=60, key="race_iter"
        )
        seed_race = st.number_input(
            "Random Seed", min_value=0, value=42, key="race_seed"
        )

        race_button = st.button("🏁 Start Race", key="race_run_btn", type="primary")

    with col_display:
        if race_button:
            if not selected_algorithms:
                st.warning("Select at least one algorithm.")
            else:
                benchmark_key = BENCHMARK_REGISTRY[benchmark_name_race]
                benchmark = BENCHMARKS[benchmark_key]

                results = {}
                progress = st.progress(0.0, text="Running algorithms...")

                for i, algo_name in enumerate(selected_algorithms):
                    optimizer_cls = OPTIMIZER_REGISTRY[algo_name]
                    optimizer = optimizer_cls(
                        n_agents=n_agents_race, seed=int(seed_race)
                    )
                    results[algo_name] = collect_states(
                        optimizer,
                        benchmark.fn,
                        benchmark.default_bounds,
                        max_iter=max_iter_race,
                    )
                    progress.progress(
                        (i + 1) / len(selected_algorithms), text=f"{algo_name} done"
                    )

                progress.empty()

                # Leaderboard: table sorted by best score
                st.subheader("🏆 Leaderboard")
                leaderboard = sorted(
                    ((name, states[-1].best_score) for name, states in results.items()),
                    key=lambda pair: pair[1],
                )
                for rank, (name, score) in enumerate(leaderboard, start=1):
                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
                    st.write(f"{medal} **{name}** — `{score:.6f}`")

                fig_convergence = build_convergence_plot(results)
                st.plotly_chart(fig_convergence, width="stretch")
        else:
            st.info(
                "👈 Select the algorithms you want to compare and press 'Start Race'."
            )

with tab_stats:
    col_controls, col_display = st.columns([1, 3])

    with col_controls:
        st.subheader("Settings")

        stat_algorithms = st.multiselect(
            "Algorithms to Compare",
            options=list(OPTIMIZER_REGISTRY.keys()),
            default=list(OPTIMIZER_REGISTRY.keys())[:3],
            key="stat_algos",
        )
        stat_benchmark_name = st.selectbox(
            "Benchmark Function",
            options=list(BENCHMARK_REGISTRY.keys()),
            key="stat_bench",
        )
        stat_n_agents = st.slider(
            "Number of Agents", min_value=5, max_value=100, value=30, key="stat_agents"
        )
        stat_max_iter = st.slider(
            "Iterations per Run", min_value=10, max_value=200, value=60, key="stat_iter"
        )
        stat_n_runs = st.slider(
            "Number of Runs (seeds)",
            min_value=5,
            max_value=50,
            value=20,
            key="stat_runs",
            help="Each algorithm is run with this many different seeds, for the statistical comparison.",
        )
        stat_alpha = st.select_slider(
            "Significance Level (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            key="stat_alpha",
        )

        stat_button = st.button(
            "📊 Run Statistical Comparison", key="stat_run_btn", type="primary"
        )

    with col_display:
        if stat_button:
            if len(stat_algorithms) < 2:
                st.warning("Select at least 2 algorithms to compare.")
            else:
                benchmark_key = BENCHMARK_REGISTRY[stat_benchmark_name]
                benchmark = BENCHMARKS[benchmark_key]

                all_scores: dict[str, np.ndarray] = {}
                progress = st.progress(0.0, text="Running multi-seed evaluation...")

                for i, algo_name in enumerate(stat_algorithms):
                    optimizer_cls = OPTIMIZER_REGISTRY[algo_name]
                    all_scores[algo_name] = run_multiple_seeds(
                        optimizer_cls,
                        benchmark.fn,
                        benchmark.default_bounds,
                        n_agents=stat_n_agents,
                        max_iter=stat_max_iter,
                        n_runs=stat_n_runs,
                    )
                    progress.progress(
                        (i + 1) / len(stat_algorithms), text=f"{algo_name} done"
                    )

                progress.empty()

                st.subheader("📦 Score Distribution")
                fig_box = build_boxplot(all_scores)
                st.plotly_chart(fig_box, width="stretch")

                st.subheader("📈 Summary Statistics")
                summary_rows = [
                    {
                        "Algorithm": name,
                        "Mean": f"{np.mean(scores):.6f}",
                        "Std": f"{np.std(scores):.6f}",
                        "Best": f"{np.min(scores):.6f}",
                        "Worst": f"{np.max(scores):.6f}",
                    }
                    for name, scores in all_scores.items()
                ]
                st.table(summary_rows)

                st.subheader("🔬 Pairwise Wilcoxon Test Results")
                st.caption(
                    f"α = {stat_alpha} — if p < α, the difference is considered statistically significant."
                )
                for name_a, name_b in combinations(stat_algorithms, 2):
                    result = compare_two_algorithms(
                        name_a,
                        all_scores[name_a],
                        name_b,
                        all_scores[name_b],
                        alpha=stat_alpha,
                    )
                    icon = "✅" if result.significant else "⚪"
                    st.write(
                        f"{icon} **{name_a}** vs **{name_b}**: "
                        f"p = `{result.p_value:.4f}` — {result.better}"
                    )
        else:
            st.info(
                "👈 Select the algorithms you want to compare and press 'Run Statistical Comparison'."
            )
