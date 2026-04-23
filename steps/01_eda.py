"""
Step 1 — Exploratory Data Analysis
Produces 6 figures saved in outputs/figures/
"""
import numpy as np
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

fig, ax = plt.subplots(figsize=(16, 4))
ax.scatter(sens_train["coor_x"], sens_train["coor_y"],
           s=30, c="steelblue", label=f"Train ({len(sens_train)})", alpha=0.8)
ax.scatter(sens_test["coor_x"],  sens_test["coor_y"],
           s=30, c="tomato",     label=f"Test ({len(sens_test)})",  alpha=0.8, marker="^")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Sensor positions — train vs test")
ax.legend()
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_sensor_map.png", dpi=150)
plt.show()


# ── Fig 2 — Temperature time series for 8 train sensors ───────────────────────
print("[Fig 2] Temperature time series...")

sample_sensors = (
    sens_train.sort_values("coor_x")
    .iloc[np.linspace(0, len(sens_train) - 1, 8, dtype=int)]["sensor"]
    .tolist()
)

fig, ax = plt.subplots(figsize=(14, 5))
cmap = plt.cm.plasma
for i, s in enumerate(sample_sensors):
    sub = train[train["sensor"] == s].sort_values("time_yr")
    xi  = sensors.loc[sensors["sensor"] == s, "coor_x"].values[0]
    ax.plot(sub["time_yr"], sub["temperature"],
            color=cmap(i / len(sample_sensors)),
            lw=0.8, label=f"{s} (x={xi:.1f}m)")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Temperature over time — selected train sensors")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_time_series.png", dpi=150)
plt.show()


# ── Fig 3 — Spatial temperature maps at 4 key snapshots ───────────────────────
print("[Fig 3] Spatial temperature maps...")

available    = train["time"].unique()
snapshots_yr = [1, 10, 50, 200]

# Collect all snapshots first to compute a shared color scale
snaps = {}
for yr in snapshots_yr:
    nearest       = available[np.argmin(np.abs(available - yr * SECONDS_PER_YEAR))]
    snaps[yr]     = train[train["time"] == nearest]

all_temps = np.concatenate([s["temperature"].dropna().values for s in snaps.values()])
vmin_global = 0
vmax_global = np.percentile(all_temps, 98)

fig, axes = plt.subplots(2, 2, figsize=(16, 7))
axes = axes.flatten()

for ax, yr in zip(axes, snapshots_yr):
    snap = snaps[yr]
    sc = ax.scatter(snap["coor_x"], snap["coor_y"],
                    c=snap["temperature"], cmap="inferno",
                    s=25, vmin=vmin_global, vmax=vmax_global)
    plt.colorbar(sc, ax=ax, label="°C")
    nearest_yr = available[np.argmin(np.abs(available - yr * SECONDS_PER_YEAR))] / SECONDS_PER_YEAR
    ax.set_title(f"t = {yr} yr")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

plt.suptitle(f"Spatial temperature distribution (shared scale 0-{vmax_global:.0f} °C)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_spatial_maps.png", dpi=150)
plt.show()


# ── Fig 4 — Power decay over time ─────────────────────────────────────────────
print("[Fig 4] Power over time...")

ref_sensor = list(train_sensors)[0]
sub = train[train["sensor"] == ref_sensor].sort_values("time_yr")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sub["time_yr"], sub["power"], color="darkorange", lw=0.8)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Power (W)")
ax.set_title(f"Power decay over time — sensor {ref_sensor}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_power_decay.png", dpi=150)
plt.show()


# ── Fig 5 — NaN analysis ──────────────────────────────────────────────────────
print("[Fig 5] NaN analysis...")

nan_per_sensor = (
    train.groupby("sensor")["temperature"]
    .apply(lambda x: x.isna().sum())
    .reset_index(name="nan_count")
)
nan_per_sensor = nan_per_sensor.merge(sensors, on="sensor", how="left")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(nan_per_sensor["nan_count"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_xlabel("NaN count per sensor")
axes[0].set_ylabel("Number of sensors")
axes[0].set_title("Distribution of missing temperature values per sensor")
axes[0].grid(True, alpha=0.3)

sc = axes[1].scatter(nan_per_sensor["coor_x"], nan_per_sensor["coor_y"],
                     c=nan_per_sensor["nan_count"], cmap="Reds", s=40)
plt.colorbar(sc, ax=axes[1], label="NaN count")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
axes[1].set_title("Spatial distribution of missing values")
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_nan_analysis.png", dpi=150)
plt.show()


# ── Fig 6 — Temperature distribution & T vs distance ─────────────────────────
print("[Fig 6] Temperature distribution...")

snap_50yr = train[train["time"] == available[np.argmin(np.abs(available - 50 * SECONDS_PER_YEAR))]]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

temp_clipped = train["temperature"].clip(upper=train["temperature"].quantile(0.995))
axes[0].hist(temp_clipped.dropna(), bins=80, color="mediumseagreen", edgecolor="white")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Count")
axes[0].set_title("Temperature distribution (train, clipped at 99.5th pct)")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(snap_50yr["coor_x"], snap_50yr["temperature"], s=5, alpha=0.4, color="darkorchid")
axes[1].set_xlabel("x (m) — distance from canister")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("Temperature vs x-position at t ≈ 50 yr")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_temp_distribution.png", dpi=150)
plt.show()


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — EDA")
print("=" * 60)
print(f"Time range       : 0 -> {train['time_yr'].max():.1f} years")
print(f"Time steps       : {train['time'].nunique()} steps / sensor")
print(f"Spatial domain   : x=[{sensors['coor_x'].min():.1f}, {sensors['coor_x'].max():.1f}] m"
      f"  y=[{sensors['coor_y'].min():.1f}, {sensors['coor_y'].max():.1f}] m")
print(f"Temp range       : {train['temperature'].min():.1f} -> {train['temperature'].max():.1f} C")
print(f"Power range      : {train['power'].min():.1f} -> {train['power'].max():.1f} W")
print(f"NaN in train     : {train['temperature'].isna().sum():,} ({train['temperature'].isna().mean()*100:.2f}%)")
print(f"Sensors with NaN : {(nan_per_sensor['nan_count'] > 0).sum()} / {len(nan_per_sensor)}")
print(f"\nFigures saved in : {FIG_DIR.resolve()}")
