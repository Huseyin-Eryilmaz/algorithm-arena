"""Dashboard tests.

Uses Streamlit's official testing framework (streamlit.testing.v1.AppTest)
to run the app headlessly — no browser required.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = str(
    Path(__file__).parent.parent / "src" / "algorithm_arena" / "app" / "dashboard.py"
)


def test_registry_exposes_all_optimizers_and_benchmarks():
    """Catches basic problems like syntax errors or missing imports."""
    import importlib

    module = importlib.import_module("algorithm_arena.app.registry")
    assert hasattr(module, "OPTIMIZER_REGISTRY")
    assert hasattr(module, "BENCHMARK_REGISTRY")
    assert len(module.OPTIMIZER_REGISTRY) == 6


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
    at.run()
    return at


def test_dashboard_renders_without_exception(app):
    assert not app.exception


def test_dashboard_has_expected_controls(app):
    """The three tabs contribute selectboxes, sliders, and run buttons."""
    assert len(app.tabs) == 3
    assert len(app.button) >= 3  # one run button per tab
    assert len(app.selectbox) >= 2


def test_single_run_produces_a_result(app):
    """Clicking 'Run Optimization' with default settings must complete
    without an exception and report a best-score metric."""
    # Keep the run cheap: fewest agents/iterations the sliders allow.
    app.slider(key="single_agents").set_value(5)
    app.slider(key="single_iter").set_value(10)
    app.button(key="single_run_btn").click().run()

    assert not app.exception
    assert len(app.metric) == 1
    assert app.metric[0].label == "Best Score Found"


def test_single_run_with_invalid_custom_expression_shows_error(app):
    """An invalid custom expression must surface as st.error, not a crash."""
    app.radio(key="single_mode").set_value("Custom Expression")
    app.run()
    app.text_input(key="single_custom_expr").set_value("import os")
    app.slider(key="single_agents").set_value(5)
    app.slider(key="single_iter").set_value(10)
    app.button(key="single_run_btn").click().run()

    assert not app.exception
    assert len(app.error) == 1
