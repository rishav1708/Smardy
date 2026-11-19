import runpy
from pathlib import Path

# Streamlit Cloud entry point: runs the main app
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
