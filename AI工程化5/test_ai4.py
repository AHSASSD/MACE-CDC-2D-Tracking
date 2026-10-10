"""Run the unchanged AI4 baseline self-check from the AI5 workspace."""

from pathlib import Path
import runpy
import sys

AI4 = Path(__file__).resolve().parents[1] / "AI工程化4"
sys.path.insert(0, str(AI4))
runpy.run_path(str(AI4 / "test_ai4.py"), run_name="__main__")
