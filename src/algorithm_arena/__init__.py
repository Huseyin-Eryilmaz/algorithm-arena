def main() -> None:
    """Launches the Streamlit dashboard when invoked via the `algorithm-arena` command."""
    import subprocess
    import sys
    from pathlib import Path

    dashboard_path = Path(__file__).parent / "app" / "dashboard.py"
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    except KeyboardInterrupt:
        pass  # normal shutdown via Ctrl+C, no need to print a traceback
