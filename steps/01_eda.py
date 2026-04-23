"""
Step 1 — Exploratory Data Analysis
Produces 6 figures saved in outputs/figures/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_data, FIG_DIR, SECONDS_PER_YEAR

FIG_DIR.mkdir(parents=True, exist_ok=True)

train, test, sensors = load_data()

train_sensors = set(train["sensor"].unique())
test_sensors  = set(test["sensor"].unique())
sens_train    = sensors[sensors["sensor"].isin(train_sensors)]
sens_test     = sensors[sensors["sensor"].isin(test_sensors)]


# ── Fig 1 — Sensor map (train vs test) ────────────────────────────────────────
print("\n[Fig 1] Sensor map...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(sens_train["coor_x"], sens_train["coor_y"],
           s=40, c="steelblue", label=f"Train ({len(sens_train)})", alpha=0.8)
ax.scatter(sens_test["coor_x"], sens_test["coor_y"],
           s=40, c="tomato", label=f"Test ({len(sens_test)})", alpha=0.8, marker="^")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Sensor positions — train vs test")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_sensor_map.png", dpi=150)


# ── Fig 2 — Temperature time series for 8 train sensors ───────────────────────
print("[Fig 2] Temperature time series...")

# Exclude dead sensors
sample_sensors = (
    sens_train[~sens_train["sensor"].isin(["N927", "N277"])]
    .sort_values("coor_x")
    .iloc[np.linspace(0, len(sens_train) - 3, 8, dtype=int)]["sensor"]
    .tolist()
)

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.cm.plasma
for i, s in enumerate(sample_sensors):
    sub = train[train["sensor"] == s].sort_values("time_yr")
    xi  = sensors.loc[sensors["sensor"] == s, "coor_x"].values[0]
    # Clip outliers for display
    temp_display = sub["temperature"].clip(lower=0, upper=200)
    ax.plot(sub["time_yr"], temp_display,
            color=cmap(i / len(sample_sensors)),
            lw=1.0, label=f"{s} (x={xi:.1f}m)")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Temperature over time — selected train sensors")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_time_series.png", dpi=150)


# ── Fig 3 — Spatial temperature maps at 4 key snapshots ───────────────────────
print("[Fig 3] Spatial temperature maps...")

available    = train["time"].unique()
snapshots_yr = [1, 10, 50, 200]

snaps = {}
for yr in snapshots_yr:
    nearest    = available[np.argmin(np.abs(available - yr * SECONDS_PER_YEAR))]
    snaps[yr]  = train[train["time"] == nearest]

# Shared color scale on clean values only
all_temps   = np.concatenate([s["temperature"].clip(lower=0).dropna().values for s in snaps.values()])
vmax_global = np.percentile(all_temps, 98)

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

for ax, yr in zip(axes, snapshots_yr):
    snap = snaps[yr]
    temp_plot = snap["temperature"].clip(lower=0, upper=vmax_global)
    sc = ax.scatter(snap["coor_x"], snap["coor_y"],
                    c=temp_plot, cmap="inferno",
                    s=35, vmin=0, vmax=vmax_global)
    plt.colorbar(sc, ax=ax, label="°C", fraction=0.02, pad=0.01)
    ax.set_title(f"t = {yr} ans", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

plt.suptitle(f"Distribution spatiale des temperatures (echelle commune 0-{vmax_global:.0f} °C)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_spatial_maps.png", dpi=150)


# ── Fig 4 — Power decay over time ─────────────────────────────────────────────
print("[Fig 4] Power over time...")

ref_sensor = [s for s in train_sensors if s not in ["N927", "N277"]][0]
sub = train[train["sensor"] == ref_sensor].sort_values("time_yr")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sub["time_yr"], sub["power"], color="darkorange", lw=1.2)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Power (W)")
ax.set_title(f"Power decay over time — sensor {ref_sensor}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_power_decay.png", dpi=150)


# ── Fig 5 — NaN analysis ──────────────────────────────────────────────────────
print("[Fig 5] NaN analysis...")

# NaN analysis sur données BRUTES — les NaN originaux, pas ceux créés par le preprocessing
nan_per_sensor = (
    train.groupby("sensor", observed=True)["temperature"]
    .apply(lambda x: x.isna().sum())
    .reset_index(name="nan_count")
)
nan_per_sensor = nan_per_sensor.merge(sensors, on="sensor", how="left")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(nan_per_sensor["nan_count"], bins=30, color="steelblue", edgecolor="white")
axes[0].set_xlabel("NaN count par capteur")
axes[0].set_ylabel("Nombre de capteurs")
axes[0].set_title("Vraies valeurs manquantes par capteur (donnees brutes)")
axes[0].grid(True, alpha=0.3)

sc = axes[1].scatter(nan_per_sensor["coor_x"], nan_per_sensor["coor_y"],
                     c=nan_per_sensor["nan_count"], cmap="YlOrRd", s=50)
plt.colorbar(sc, ax=axes[1], label="NaN count")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
axes[1].set_title("Distribution spatiale des valeurs manquantes")
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_nan_analysis.png", dpi=150)


# ── Fig 6 — Temperature distribution & T vs distance ─────────────────────────
print("[Fig 6] Temperature distribution...")

# Utiliser les données nettoyées si disponibles, sinon filtrer à la volée
clean_path = FIG_DIR.parent.parent / "outputs" / "train_clean.parquet"
if clean_path.exists():
    import pandas as pd
    train_clean = pd.read_parquet(clean_path)
else:
    train_clean = train[~train["sensor"].isin(["N927", "N277"])].copy()
    train_clean.loc[train_clean["temperature"] < 0, "temperature"] = float("nan")

temp_clean = train_clean["temperature"].dropna()
t50_clean  = train_clean[train_clean["time"] == available[np.argmin(np.abs(available - 50 * SECONDS_PER_YEAR))]]
t50_clean  = t50_clean.dropna(subset=["temperature"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogramme avec densité normalisée
axes[0].hist(temp_clean, bins=100, color="mediumseagreen", edgecolor="none", density=True)
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Densite")
axes[0].set_title("Distribution des temperatures (donnees nettoyees)")
axes[0].set_xlim(0, temp_clean.quantile(0.999))
axes[0].grid(True, alpha=0.3)

# T vs x à 50 ans
axes[1].scatter(t50_clean["coor_x"], t50_clean["temperature"],
                s=10, alpha=0.6, color="darkorchid")
axes[1].set_xlabel("x (m) — distance du canister")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("Temperature vs position x  (t = 50 ans)")
axes[1].set_ylim(0, t50_clean["temperature"].quantile(0.99))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_temp_distribution.png", dpi=150)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — EDA")
print("=" * 60)
print(f"Time range       : 0 -> {train['time_yr'].max():.1f} years")
print(f"Time steps       : {train['time'].nunique()} steps / sensor")
print(f"Spatial domain   : x=[{sensors['coor_x'].min():.1f}, {sensors['coor_x'].max():.1f}] m"
      f"  y=[{sensors['coor_y'].min():.1f}, {sensors['coor_y'].max():.1f}] m")
clean_temp = train["temperature"].clip(lower=0)
print(f"Temp range clean : {clean_temp.min():.1f} -> {clean_temp.quantile(0.999):.1f} C (99.9th pct)")
print(f"Power range      : {train['power'].min():.1f} -> {train['power'].max():.1f} W")
print(f"NaN in train     : {train['temperature'].isna().sum():,} ({train['temperature'].isna().mean()*100:.2f}%)")
print(f"Sensors with NaN : {(nan_per_sensor['nan_count'] > 0).sum()} / {len(nan_per_sensor)}")
print(f"\nFigures saved in : {FIG_DIR.resolve()}")
