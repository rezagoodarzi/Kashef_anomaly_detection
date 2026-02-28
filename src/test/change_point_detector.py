#!/usr/bin/env python3
"""
Change Point Detector
Detects change points in all sector RSSI data using PELT algorithm.
Outputs change_points.csv with value before/after change and aligned sector counts.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

# =============================================================================
# CONFIG – Tune these parameters
# =============================================================================

@dataclass
class Config:
    # Change point sensitivity (higher = fewer change points, lower = more)
    PELT_PENALTY: float = 1

    # Minimum data points between two consecutive change points
    MIN_SEGMENT_SIZE: int = 1

    # Sectors with fewer data points than this are skipped
    MIN_DATA_POINTS: int = 6

    # Window (number of points) used to compute value_before / value_after
    VALUE_WINDOW: int = 1

    # Time window (± hours) to count how many other sectors also changed
    ALIGNMENT_WINDOW_HOURS: float = 1.0

    # ── Magnitude threshold ──────────────────────────────────────────────
    # 'manual'   → keep CPs with magnitude > MANUAL_THRESHOLD
    # 'adaptive' → keep CPs with magnitude > ADAPTIVE_SENSITIVITY × sector_std
    # 'none'     → keep all CPs (no magnitude filter)
    THRESHOLD_MODE: str = 'adaptive'

    # Manual mode: fixed magnitude threshold (dB)
    MANUAL_THRESHOLD: float = 6.0

    # Adaptive mode: multiplier on each sector's own std deviation
    # Lower = more sensitive (keeps smaller changes), Higher = stricter
    ADAPTIVE_SENSITIVITY: float = 0.8

    # Column names
    RSSI_COL: str = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
    DATE_COL: str = 'Date'
    SECTOR_COL: str = 'NE'

    # Fill value for NaN that can't be forward/backward filled
    NAN_FILL: float = -110.0

    # Parallelism: 0 = use all CPU cores, 1 = single core, N = use N cores
    NUM_WORKERS: int = 0


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_data(path: str, cfg: Config) -> pd.DataFrame:
    """Load CSV, parse dates, fill NaN per sector (ffill → bfill → default)."""
    df = pd.read_csv(path, usecols=[cfg.DATE_COL, cfg.SECTOR_COL, cfg.RSSI_COL])
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL])
    df.sort_values([cfg.SECTOR_COL, cfg.DATE_COL], inplace=True)

    # Fill NaN within each sector, then global fallback
    df[cfg.RSSI_COL] = (
        df.groupby(cfg.SECTOR_COL)[cfg.RSSI_COL]
          .transform(lambda s: s.ffill().bfill())
          .fillna(cfg.NAN_FILL)
    )

    print(f"Loaded {len(df):,} records | {df[cfg.SECTOR_COL].nunique()} sectors")
    return df



def _process_one_sector(args):
    """
    Worker function for multiprocessing — processes a single sector.
    Must be at module level to be picklable on Windows.
    Returns: (results_list, filtered_count)
    """
    sector, signal, penalty, min_seg, min_dp, w, mode, manual_thresh, adaptive_sens = args

    if len(signal) < min_dp or np.std(signal) == 0:
        return [], 0

    normed = (signal - np.mean(signal)) / np.std(signal)
    cps = rpt.Pelt(model="rbf", min_size=min_seg).fit(normed).predict(pen=penalty)
    cps = cps[:-1]

    sector_std = float(np.std(signal))
    if mode == 'manual':
        mag_threshold = manual_thresh
    elif mode == 'adaptive':
        mag_threshold = adaptive_sens * sector_std
    else:
        mag_threshold = 0.0

    results = []
    filtered = 0
    for idx in cps:
        if idx >= len(signal):
            continue
        vb = round(float(np.mean(signal[max(0, idx - w): idx])), 2)
        va = round(float(np.mean(signal[idx: min(len(signal), idx + w)])), 2)
        mag = round(abs(va - vb), 2)

        if mag < mag_threshold:
            filtered += 1
            continue

        results.append((sector, idx, vb, va, mag, 'UP' if va > vb else 'DOWN'))

    return results, filtered


def find_aligned_sectors(cp_df: pd.DataFrame, window_hours: float):
    """
    For each change point, find which OTHER sectors also have
    a change point within ± window_hours.
    Uses sorted timestamps + binary search for speed.

    Returns:
        (counts, names) — int array and list of semicolon-separated sector strings
    """
    ts = cp_df['timestamp'].values.astype('int64')       # nanoseconds
    sectors = cp_df['sector'].values
    window_ns = int(pd.Timedelta(hours=window_hours).value)

    counts = np.zeros(len(cp_df), dtype=int)
    names = []

    for i in range(len(cp_df)):
        left  = np.searchsorted(ts, ts[i] - window_ns, side='left')
        right = np.searchsorted(ts, ts[i] + window_ns, side='right')
        # unique sectors in the window, excluding current sector
        nearby = sorted(set(sectors[left:right]) - {sectors[i]})
        counts[i] = len(nearby)
        names.append(';'.join(nearby))

    return counts, names


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def find_change_points(data_path: str, output_dir: str, cfg: Config = None) -> pd.DataFrame:
    """
    Detect change points for ALL sectors → save change_points.csv.

    CSV columns:
        sector, timestamp, value_before, value_after, magnitude, direction,
        num_aligned_sectors, aligned_sectors (semicolon-separated)
    """
    cfg = cfg or Config()

    df = load_data(data_path, cfg)

    pivot = (
        df.pivot_table(index=cfg.DATE_COL, columns=cfg.SECTOR_COL,
                       values=cfg.RSSI_COL, aggfunc='mean')
          .sort_index()
          .ffill().bfill()
          .fillna(cfg.NAN_FILL)
    )

    timestamps = pivot.index
    sectors = pivot.columns.tolist()
    w = cfg.VALUE_WINDOW

    n_sectors = len(sectors)
    n_workers = cfg.NUM_WORKERS if cfg.NUM_WORKERS > 0 else cpu_count()
    mode = cfg.THRESHOLD_MODE.lower()

    print(f"Running PELT on {n_sectors} sectors ({len(timestamps)} time steps)...")
    if mode == 'manual':
        print(f"  Threshold mode: MANUAL (magnitude > {cfg.MANUAL_THRESHOLD})")
    elif mode == 'adaptive':
        print(f"  Threshold mode: ADAPTIVE (magnitude > {cfg.ADAPTIVE_SENSITIVITY} × sector_std)")
    else:
        print(f"  Threshold mode: NONE (keep all change points)")
    print(f"  Workers: {n_workers} CPU cores")

    worker_args = [
        (sector, pivot[sector].values, cfg.PELT_PENALTY, cfg.MIN_SEGMENT_SIZE,
         cfg.MIN_DATA_POINTS, w, mode, cfg.MANUAL_THRESHOLD, cfg.ADAPTIVE_SENSITIVITY)
        for sector in sectors
    ]

    rows = []
    filtered_count = 0

    if n_workers == 1:
        # Single core — no overhead
        for i, args in enumerate(worker_args, 1):
            if i % 500 == 0 or i == n_sectors:
                print(f"  [{i}/{n_sectors}] sectors processed...")
            results, filt = _process_one_sector(args)
            rows.extend(results)
            filtered_count += filt
    else:
        # Multi core
        with Pool(processes=n_workers) as pool:
            done = 0
            for results, filt in pool.imap_unordered(_process_one_sector, worker_args, chunksize=64):
                rows.extend(results)
                filtered_count += filt
                done += 1
                if done % 500 == 0 or done == n_sectors:
                    print(f"  [{done}/{n_sectors}] sectors processed...")

    if filtered_count:
        print(f"  Filtered out {filtered_count:,} low-magnitude change points")

    if not rows:
        print("No change points detected.")
        return pd.DataFrame()

    # Worker returns indices → map back to actual timestamps
    cp_df = pd.DataFrame(rows, columns=[
        'sector', 'cp_idx', 'value_before', 'value_after',
        'magnitude', 'direction'
    ])
    cp_df['timestamp'] = cp_df['cp_idx'].map(lambda i: timestamps[i] if i < len(timestamps) else pd.NaT)
    cp_df.drop(columns='cp_idx', inplace=True)
    cp_df.sort_values('timestamp', inplace=True, ignore_index=True)

    print(f"Found {len(cp_df):,} change points")

    print("Finding aligned sectors...")
    counts, names = find_aligned_sectors(cp_df, cfg.ALIGNMENT_WINDOW_HOURS)
    cp_df['num_aligned_sectors'] = counts
    cp_df['aligned_sectors'] = names

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'change_points.csv'
    cp_df.to_csv(csv_path, index=False)

    with_align = (cp_df['num_aligned_sectors'] > 0).sum()
    print(f"\n--- Done ---")
    print(f"Total change points : {len(cp_df):,}")
    print(f"With aligned sectors: {with_align:,} ({100 * with_align / len(cp_df):.1f}%)")
    print(f"Direction split     : UP={(cp_df['direction']=='UP').sum():,}  DOWN={(cp_df['direction']=='DOWN').sum():,}")
    print(f"Saved to            : {csv_path}")

    return cp_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH  = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv'
    OUTPUT_DIR = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline_changepoint'

    # --- Customize here ---
    config = Config(
        PELT_PENALTY=1.0,          # PELT sensitivity (higher = fewer CPs)
        ALIGNMENT_WINDOW_HOURS=0, # ± hours to count aligned sectors
        VALUE_WINDOW=1,             # points for value_before / value_after

        # ── Threshold options ──
        # Option 1: 'manual'   → fixed magnitude cutoff
        # Option 2: 'adaptive' → auto threshold per sector (sensitivity × std)
        # Option 3: 'none'     → keep all change points
        THRESHOLD_MODE='adaptive',
        MANUAL_THRESHOLD=6.0,       # used when mode='manual' (dB)
        ADAPTIVE_SENSITIVITY=0.5,   # used when mode='adaptive' (lower=more sensitive)
    )

    find_change_points(DATA_PATH, OUTPUT_DIR, config)
