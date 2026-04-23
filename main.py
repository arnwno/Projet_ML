"""
Entry point — Nuclear Waste Canister Temperature Prediction
CIVIL-226 Introduction to ML — EPFL 2026

Usage:
    python main.py --step 1      # EDA
    python main.py --step 2      # Preprocessing & outlier removal
    python main.py --step 3      # Baseline (IDW)
    python main.py --step 4      # Models (Ridge / RF / GBM)
    python main.py --step 5      # Predictions & submission
    python main.py --step all    # Run everything
"""
import argparse
import runpy
from pathlib import Path

STEPS = {
    1: ("EDA",                   "steps/01_eda.py"),
    2: ("Preprocessing",         "steps/02_preprocessing.py"),
    3: ("Baseline IDW",          "steps/03_baseline.py"),
    4: ("Model training",        "steps/04_models.py"),
    5: ("Submission",            "steps/05_submission.py"),
}


ROOT = Path(__file__).parent


def run_step(n: int):
    name, path = STEPS[n]
    print(f"\n{'='*60}")
    print(f"  STEP {n} - {name}")
    print(f"{'='*60}\n")
    runpy.run_path(str(ROOT / path), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="1", help="Step number or 'all'")
    args = parser.parse_args()

    if args.step == "all":
        for n in STEPS:
            run_step(n)
    else:
        run_step(int(args.step))
