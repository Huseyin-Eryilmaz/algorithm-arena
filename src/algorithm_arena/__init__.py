def main() -> None:
    """`algorithm-arena` komutuyla çağrıldığında Streamlit dashboard'unu başlatır."""
    import subprocess
    import sys
    from pathlib import Path

    dashboard_path = Path(__file__).parent / "app" / "dashboard.py"
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    except KeyboardInterrupt:
        pass  # Ctrl+C ile normal kapatma, traceback basmaya gerek yok
