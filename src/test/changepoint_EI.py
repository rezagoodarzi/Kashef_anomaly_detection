import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path(r"C:\Users\Axis\Documents\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv")
CP_PATH    = Path(r"C:\Users\Axis\Documents\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline_changepoint_nopelt\change_points.csv")
OUTPUT_DIR = Path(r"C:\Users\Axis\Documents\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline_changepoint_nopelt_time_start")

# ── User inputs ───────────────────────────────────────────────────────────────
center_lat = float("35.840455")
center_lon = float("50.981873")
radius_km  = float("3")
time_start = pd.Timestamp("2025-09-29 01:00:00")
time_end   = pd.Timestamp("2025-09-29 23:00:00")
avg_rssi   = -113



# ── Constants ─────────────────────────────────────────────────────────────────
RSSI_COL   = "RSSI_PUCCH(EUCell_Eric)(CRA)"
R_EARTH_KM = 6371.0

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH, usecols=["Date", "NE", RSSI_COL, "Latitude", "Longitude"])
df["Date"] = pd.to_datetime(df["Date"])

cp = pd.read_csv(CP_PATH)
cp.columns = cp.columns.str.strip()
cp["timestamp"] = pd.to_datetime(cp["timestamp"])

# ── Geo-fence: sectors within radius ─────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R_EARTH_KM * 2 * np.arcsin(np.sqrt(a))

sector_meta = df.groupby("NE").agg(Latitude=("Latitude", "first"), Longitude=("Longitude", "first"))
sector_meta["dist_km"] = haversine_km(
    center_lat, center_lon,
    sector_meta["Latitude"].values,
    sector_meta["Longitude"].values,
)
geo_sectors = sector_meta[sector_meta["dist_km"] <= radius_km].index.tolist()

if not geo_sectors:
    raise ValueError("No sectors found inside the given radius.")

print(f"Sectors within {radius_km} km: {len(geo_sectors)}")

# ── Filter data to time window ────────────────────────────────────────────────
df_window = df[
    df["NE"].isin(geo_sectors) &
    df["Date"].between(time_start, time_end)
].copy()

# ── Condition 1: sector has at least one change point in the time window ──────
cp_window = cp[
    cp["sector"].isin(geo_sectors) &
    cp["timestamp"].between(time_start, time_end)
]
sectors_with_cp = set(cp_window["sector"].unique())

# ── Condition 2: sector avg RSSI in the window is >= threshold ────────────────
sector_avg = df_window.groupby("NE")[RSSI_COL].mean()
sectors_above_avg = set(sector_avg[sector_avg >= avg_rssi].index)

# ── Union of both conditions ──────────────────────────────────────────────────
qualifying_sectors = list(sectors_with_cp | sectors_above_avg)

if not qualifying_sectors:
    raise ValueError("No sectors satisfy the change-point or avg-RSSI condition in the given time window.")

print(f"Qualifying sectors (CP in window OR avg RSSI >= {avg_rssi}): {len(qualifying_sectors)}")
print(f"  — with change point in window:    {len(sectors_with_cp & set(qualifying_sectors))}")
print(f"  — with avg RSSI above threshold:  {len(sectors_above_avg & set(qualifying_sectors))}")

# ── Plot ──────────────────────────────────────────────────────────────────────
df_plot      = df[df["NE"].isin(qualifying_sectors)].copy()
color_cycle  = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(18, 8))

for i, ne in enumerate(qualifying_sectors):
    color     = color_cycle[i % len(color_cycle)]
    sector_df = df_plot[df_plot["NE"] == ne].sort_values("Date")

    ax.plot(sector_df["Date"], sector_df[RSSI_COL],
            label=str(ne), color=color, alpha=0.65, linewidth=1)

    ne_cp = cp_window[cp_window["sector"] == ne]
    for _, row in ne_cp.iterrows():
        ax.annotate(
            f"{row['direction']} {row['magnitude']:.2f}",
            xy=(row["timestamp"], ax.get_ylim()[0]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=6,
            color=color,
            rotation=90,
        )

def highlight_time_window(ax):
    """
    Adds a vertical background span to the current axes over the interval [time_start, time_end].
    """
    ax.axvspan(time_start, time_end, color="yellow", alpha=0.5, label="Highlighted Window")

highlight_time_window(ax)
ax.axvspan(time_start, time_end, alpha=0.08, color="yellow", label="Query window")
ax.axhline(avg_rssi, color="red", linestyle=":", linewidth=1, label=f"avg_rssi threshold ({avg_rssi})")

ax.set_title(
    f"RSSI — qualifying sectors within {radius_km} km | window [{time_start} → {time_end}]\n"
    f"(n={len(qualifying_sectors)}, CP in window OR avg RSSI ≥ {avg_rssi})"
)
ax.set_xlabel("Date/Time")
ax.set_ylabel(RSSI_COL)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize="x-small", ncol=3, frameon=True)
plt.tight_layout()
plt.show()

# ── Export results ────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cp_export = (
    cp_window[cp_window["sector"].isin(qualifying_sectors)]
    .sort_values(["sector", "timestamp"])
)
cp_csv_path = OUTPUT_DIR / "geo_window_change_points.csv"
cp_export.to_csv(cp_csv_path, index=False)
print(f"Saved {len(cp_export)} change points to: {cp_csv_path}")

sectors_export = pd.DataFrame({"sector": sorted(set(qualifying_sectors))})
sectors_csv_path = OUTPUT_DIR / "geo_window_sectors.csv"
sectors_export.to_csv(sectors_csv_path, index=False)
print(f"Saved sector list to: {sectors_csv_path}")