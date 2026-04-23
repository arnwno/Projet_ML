from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data_parquet_2026"
FIG_DIR  = Path(__file__).parent.parent / "outputs" / "figures"

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and merge train/test/sensors parquet files. Returns (train, test, sensors)."""
    print("Loading data...")
    train   = pd.read_parquet(DATA_DIR / "train.parquet")
    test    = pd.read_parquet(DATA_DIR / "test.parquet")
    sensors = pd.read_parquet(DATA_DIR / "sensors.parquet")

    for df in [train, test, sensors]:
        df["sensor"] = df["sensor"].astype(str)

    train = train.merge(sensors, on="sensor", how="left")
    test  = test.merge(sensors,  on="sensor", how="left")

    train["time_yr"] = train["time"] / SECONDS_PER_YEAR
    test["time_yr"]  = test["time"]  / SECONDS_PER_YEAR

    n_train = train["sensor"].nunique()
    n_test  = test["sensor"].nunique()
    print(f"  Train : {len(train):>10,} rows | {n_train} sensors")
    print(f"  Test  : {len(test):>10,} rows | {n_test} sensors")
    print(f"  Sensors file: {len(sensors)} total positions")

    return train, test, sensors
