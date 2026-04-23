"""
Entry point — Nuclear Waste Canister Temperature Prediction
CIVIL-226 Introduction to ML — EPFL 2026

Usage:
    python main.py --step 1      # EDA
    python main.py --step 2      # Preprocessing
    python main.py --step 3      # Baseline (IDW)
    python main.py --step 4      # Models (Ridge / RF / GBM)
    python main.py --step 5      # Predictions & submission
    python main.py --step all    # Run everything
"""
import argparse

STEPS = {
    1: ("EDA",                  "steps.01_eda"),
    2: ("Preprocessing",        "steps.02_preprocessing"),
    3: ("Baseline IDW",         "steps.03_baseline"),
    4: ("Model training",       "steps.04_models"),
    5: ("Submission",           "steps.05_submission"),
}


def run_step(n: int):
    name, module = STEPS[n]
    print(f"\n{'='*60}")
    print(f"  STEP {n} — {name}")
    print(f"{'='*60}\n")
    import importlib
    importlib.import_module(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="1", help="Step number or 'all'")
    args = parser.parse_args()

    if args.step == "all":
        for n in STEPS:
            run_step(n)
    else:
        run_step(int(args.step))
