"""
Step 2 — Preprocessing & Outlier Removal
Produces cleaned train/test as parquet in outputs/
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_data, FIG_DIR, SECONDS_PER_YEAR

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = FIG_DIR.parent.parent / "outputs"

train, test, sensors = load_data()

# ── 1. Capteurs morts ─────────────────────────────────────────────────────────
# N927 : bloqué à 6039°C sur tous ses pas de temps
# N277 : 100% valeurs négatives
dead_sensors = ["N927", "N277"]
train = train[~train["sensor"].isin(dead_sensors)].copy()
print(f"Capteurs morts supprimés : {dead_sensors}")
print(f"  Train restant : {len(train):,} lignes, {train['sensor'].nunique()} capteurs")

# ── 2. Valeurs négatives → NaN (artefacts FEM) ────────────────────────────────
n_neg = (train["temperature"] < 0).sum()
train.loc[train["temperature"] < 0, "temperature"] = np.nan
print(f"\nValeurs négatives -> NaN : {n_neg:,} ({n_neg/len(train)*100:.2f}%)")

# ── 3. Bilan NaN final ────────────────────────────────────────────────────────
total_nan = train["temperature"].isna().sum()
print(f"NaN total après nettoyage : {total_nan:,} ({total_nan/len(train)*100:.2f}%)")

# ── 4. Sauvegarde ─────────────────────────────────────────────────────────────
train.to_parquet(OUT / "train_clean.parquet", index=False)
print(f"\nSauvegardé : outputs/train_clean.parquet")


# ── Fig — Avant / Après sur snapshot à 50 ans ────────────────────────────────
print("\n[Fig] Comparaison avant/après outlier removal...")

train_raw, _, _ = load_data()
available = train_raw["time"].unique()
t50 = available[np.argmin(np.abs(available - 50 * SECONDS_PER_YEAR))]

snap_raw   = train_raw[train_raw["time"] == t50]
snap_clean = train[train["time"] == t50]

vmax = snap_clean["temperature"].quantile(0.98)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, snap, title in zip(axes,
                            [snap_raw, snap_clean],
                            ["Brut (avec outliers)", "Nettoyé"]):
    sc = ax.scatter(snap["coor_x"], snap["coor_y"],
                    c=snap["temperature"].clip(upper=vmax),
                    cmap="inferno", s=20, vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="°C")
    ax.set_title(f"t = 50 ans — {title}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_outlier_removal.png", dpi=150)
print(f"Figure sauvegardée : outputs/figures/fig7_outlier_removal.png")
