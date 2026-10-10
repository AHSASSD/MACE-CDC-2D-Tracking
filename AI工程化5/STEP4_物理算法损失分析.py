from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "AI工程化4" / "step4_hit_audit" / "analyze_physics_algorithm_loss.py"),
    run_name="__main__",
)
