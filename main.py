"""
Totoro Entrypoint
Run with:
  uv run main.py
or:
  streamlit run app.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app.py"
    print("🌿 Launching Totoro Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

if __name__ == "__main__":
    main()
