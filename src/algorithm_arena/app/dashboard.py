"""
Algorithm Arena — Streamlit Dashboard.
Çalıştırmak için: uv run streamlit run src/algorithm_arena/app/dashboard.py
"""

import streamlit as st

from algorithm_arena.app.registry import BENCHMARK_REGISTRY, OPTIMIZER_REGISTRY
from algorithm_arena.benchmarks.functions import BENCHMARKS
from algorithm_arena.optimizers.base import collect_states
from algorithm_arena.visualization import (
    build_contour_animation,
    build_convergence_plot,
)

st.set_page_config(page_title="Algorithm Arena", layout="wide")

st.title("🐺 Algorithm Arena")
st.caption("Metaheuristic optimization algorithms racing on benchmark functions")

tab_single, tab_race = st.tabs(["🎯 Single Run", "🏁 Race Mode"])

with tab_single:
    col_controls, col_display = st.columns([1, 3])

    with col_controls:
        st.subheader("Settings")

        algorithm_name = st.selectbox(
            "Algorithm", options=list(OPTIMIZER_REGISTRY.keys()), key="single_algo"
        )
        benchmark_name = st.selectbox(
            "Benchmark Function",
            options=list(BENCHMARK_REGISTRY.keys()),
            key="single_bench",
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
            benchmark_key = BENCHMARK_REGISTRY[benchmark_name]
            benchmark = BENCHMARKS[benchmark_key]
            optimizer_cls = OPTIMIZER_REGISTRY[algorithm_name]
            optimizer = optimizer_cls(n_agents=n_agents, seed=int(seed))

            with st.spinner(f"Running {algorithm_name} on {benchmark_name}..."):
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
            st.info("👈 Ayarları seç ve 'Run Optimization' butonuna bas.")

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
                st.warning("En az bir algoritma seç.")
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

                # Leaderboard: en iyi skora göre sıralı tablo
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
                "👈 Karşılaştırmak istediğin algoritmaları seç ve 'Start Race' butonuna bas."
            )
