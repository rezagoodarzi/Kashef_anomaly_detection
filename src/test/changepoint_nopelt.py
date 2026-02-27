#!/usr/bin/env python3
"""
Change Point Detector
Detects change points in all sector RSSI data using a manual sliding-window
mean-difference algorithm. No external change-point libraries required.

Algorithm summary
-----------------
For each interior position i in a signal, a score is computed as:

    score[i] = | mean(signal[i : i + DETECTION_WINDOW])
                - mean(signal[i - DETECTION_WINDOW : i]) |

Change points are positions where score[i] exceeds the active magnitude
threshold AND score[i] is the local maximum within MIN_SEGMENT_SIZE points
(greedy left-to-right sweep).

value_before / value_after use VALUE_WINDOW for their own mean, independent
of DETECTION_WINDOW, consistent with the original design.

Outputs change_points.csv with the same columns as the original script.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # Half-width (in data points) of the comparison windows used to score
    # each candidate position. Larger = smoother, fewer spurious detections.
    # Replaces PELT_PENALTY from the original script.
    DETECTION_WINDOW: int = 3

    # Minimum number of data points that must separate two change points.
    MIN_SEGMENT_SIZE: int = 1

    # Sectors with fewer data points than this are skipped.
    MIN_DATA_POINTS: int = 6

    # Window (number of points) used to compute value_before / value_after.
    VALUE_WINDOW: int = 1

    # Time window (± hours) to count how many other sectors also changed.
    ALIGNMENT_WINDOW_HOURS: float = 1.0

    # ── Magnitude threshold ──────────────────────────────────────────────
    # 'manual'   -> keep CPs with magnitude > MANUAL_THRESHOLD
    # 'adaptive' -> keep CPs with magnitude > ADAPTIVE_SENSITIVITY x sector_std
    # 'none'     -> keep all CPs (no magnitude filter)
    THRESHOLD_MODE: str = 'adaptive'

    # Manual mode: fixed magnitude threshold (dB)
    MANUAL_THRESHOLD: float = 6.0

    # Adaptive mode: multiplier on each sector's own std deviation.
    # Lower = more sensitive (keeps smaller changes), Higher = stricter.
    ADAPTIVE_SENSITIVITY: float = 0.8

    # Column names
    RSSI_COL: str  = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
    DATE_COL: str  = 'Date'
    SECTOR_COL: str = 'NE'

    # Fill value for NaN that cannot be forward/backward filled
    NAN_FILL: float = -110.0

    # Parallelism: 0 = all CPU cores, 1 = single core, N = N cores
    NUM_WORKERS: int = 0


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_data(path: str, cfg: Config) -> pd.DataFrame:
    """Load CSV, parse dates, fill NaN per sector (ffill -> bfill -> default)."""
    df = pd.read_csv(path, usecols=[cfg.DATE_COL, cfg.SECTOR_COL, cfg.RSSI_COL])
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL])
    df.sort_values([cfg.SECTOR_COL, cfg.DATE_COL], inplace=True)

    df[cfg.RSSI_COL] = (
        df.groupby(cfg.SECTOR_COL)[cfg.RSSI_COL]
          .transform(lambda s: s.ffill().bfill())
          .fillna(cfg.NAN_FILL)
    )

    print(f"Loaded {len(df):,} records | {df[cfg.SECTOR_COL].nunique()} sectors")
    return df


def _sliding_window_scores(signal: np.ndarray, dw: int) -> np.ndarray:
    """
    Compute the absolute mean-difference score at every interior position.

    score[i] = | mean(signal[i : i+dw]) - mean(signal[i-dw : i]) |

    Positions within dw of either boundary receive score = 0.0.
    Uses cumulative sum for O(n) computation.
    """
    n = len(signal)
    scores = np.zeros(n)
    if n < 2 * dw + 1:
        return scores

    # cumulative sum for fast window means
    cs = np.concatenate(([0.0], np.cumsum(signal)))

    for i in range(dw, n - dw):
        left_mean  = (cs[i]      - cs[i - dw]) / dw
        right_mean = (cs[i + dw] - cs[i])       / dw
        scores[i]  = abs(right_mean - left_mean)

    return scores


def _select_change_points(scores: np.ndarray, threshold: float,
                           min_seg: int, dw: int) -> list:
    """
    Greedy left-to-right sweep to select change point indices.

    At each position that exceeds the threshold, the local maximum within
    the next min_seg positions is chosen, then the cursor jumps past it
    by min_seg to enforce minimum spacing.
    """
    n = len(scores)
    cps = []
    i = dw  # skip boundary region

    while i < n - dw:
        if scores[i] > threshold:
            # find the peak within [i, i + min_seg)
            end = min(i + max(min_seg, 1), n - dw)
            peak = i + int(np.argmax(scores[i:end]))
            cps.append(peak)
            i = peak + max(min_seg, 1)
        else:
            i += 1

    return cps


def _process_one_sector(args):
    """
    Worker function for processing a single sector.

    Returns: (results_list, filtered_count)
    """
    (sector, signal, dw, min_seg, min_dp,
     value_w, mode, manual_thresh, adaptive_sens) = args

    if len(signal) < min_dp or np.std(signal) == 0:
        return [], 0

    sector_std = float(np.std(signal))

    if mode == 'manual':
        threshold = manual_thresh
    elif mode == 'adaptive':
        threshold = adaptive_sens * sector_std
    else:
        threshold = 0.0

    scores = _sliding_window_scores(signal, dw)
    cp_indices = _select_change_points(scores, threshold, min_seg, dw)

    results = []
    for idx in cp_indices:
        vb = round(float(np.mean(signal[max(0, idx - value_w): idx])),        2)
        va = round(float(np.mean(signal[idx: min(len(signal), idx + value_w)])), 2)
        mag = round(abs(va - vb), 2)
        results.append((sector, idx, vb, va, mag, 'UP' if va > vb else 'DOWN'))

    # filtered_count is 0 here because threshold is applied inside
    # _select_change_points via the score threshold, not as a post-filter.
    return results, 0


def find_aligned_sectors(cp_df: pd.DataFrame, window_hours: float):
    """
    For each change point, find which OTHER sectors also have a change point
    within +/- window_hours. Uses sorted timestamps + binary search.

    Returns: (counts, names) as int array and list of semicolon-separated strings.
    """
    ts      = cp_df['timestamp'].values.astype('int64')
    sectors = cp_df['sector'].values
    window_ns = int(pd.Timedelta(hours=window_hours).value)

    n = len(cp_df)
    counts = np.zeros(n, dtype=int)
    names  = ["" for _ in range(n)]

    def _aligned_for_index(i: int):
        left  = np.searchsorted(ts, ts[i] - window_ns, side='left')
        right = np.searchsorted(ts, ts[i] + window_ns, side='right')
        nearby = sorted(set(sectors[left:right]) - {sectors[i]})
        return i, len(nearby), ';'.join(nearby)

    # Use threads to parallelize over change points. This is mostly NumPy/indexing
    # work and is safe to run concurrently for speed-ups on large datasets.
    max_workers = min(cpu_count(), n) if n > 0 else 1
    if max_workers <= 1:
        for i in range(n):
            idx, cnt, nm = _aligned_for_index(i)
            counts[idx] = cnt
            names[idx] = nm
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_aligned_for_index, i) for i in range(n)]
            for fut in as_completed(futures):
                idx, cnt, nm = fut.result()
                counts[idx] = cnt
                names[idx] = nm

    return counts, names


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def find_change_points(data_path: str, output_dir: str, cfg: Config = None) -> pd.DataFrame:
    """
    Detect change points for all sectors and save change_points.csv.

    Output columns:
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
    sectors    = pivot.columns.tolist()
    n_sectors  = len(sectors)
    n_workers  = cfg.NUM_WORKERS if cfg.NUM_WORKERS > 0 else cpu_count()
    mode       = cfg.THRESHOLD_MODE.lower()

    print(f"Running manual sliding-window detector on {n_sectors} sectors "
          f"({len(timestamps)} time steps)...")
    if mode == 'manual':
        print(f"  Threshold mode : MANUAL (magnitude > {cfg.MANUAL_THRESHOLD} dB)")
    elif mode == 'adaptive':
        print(f"  Threshold mode : ADAPTIVE (magnitude > {cfg.ADAPTIVE_SENSITIVITY} x sector_std)")
    else:
        print(f"  Threshold mode : NONE (keep all change points)")
    print(f"  Detection window: {cfg.DETECTION_WINDOW} points each side")
    print(f"  Workers         : {n_workers} CPU core(s)")

    worker_args = [
        (sector, pivot[sector].values, cfg.DETECTION_WINDOW,
         cfg.MIN_SEGMENT_SIZE, cfg.MIN_DATA_POINTS, cfg.VALUE_WINDOW,
         mode, cfg.MANUAL_THRESHOLD, cfg.ADAPTIVE_SENSITIVITY)
        for sector in sectors
    ]

    rows = []

    if n_workers == 1:
        for i, args in enumerate(worker_args, 1):
            if i % 500 == 0 or i == n_sectors:
                print(f"  [{i}/{n_sectors}] sectors processed...")
            results, _ = _process_one_sector(args)
            rows.extend(results)
    else:
        # Multithreaded execution over sectors. Each sector is independent,
        # so we can safely process them concurrently with a thread pool.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_process_one_sector, args) for args in worker_args]
            done = 0
            for fut in as_completed(futures):
                results, _ = fut.result()
                rows.extend(results)
                done += 1
                if done % 500 == 0 or done == n_sectors:
                    print(f"  [{done}/{n_sectors}] sectors processed...")

    if not rows:
        print("No change points detected.")
        return pd.DataFrame()

    cp_df = pd.DataFrame(rows, columns=[
        'sector', 'cp_idx', 'value_before', 'value_after', 'magnitude', 'direction'
    ])
    cp_df['timestamp'] = cp_df['cp_idx'].map(
        lambda i: timestamps[i] if i < len(timestamps) else pd.NaT
    )
    cp_df.drop(columns='cp_idx', inplace=True)
    cp_df.sort_values('timestamp', inplace=True, ignore_index=True)

    print(f"Found {len(cp_df):,} change points")
    print("Finding aligned sectors...")

    counts, names = find_aligned_sectors(cp_df, cfg.ALIGNMENT_WINDOW_HOURS)
    cp_df['num_aligned_sectors'] = counts
    cp_df['aligned_sectors']     = names

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'change_points.csv'
    cp_df.to_csv(csv_path, index=False)

    with_align = (cp_df['num_aligned_sectors'] > 0).sum()
    print(f"\n--- Done ---")
    print(f"Total change points : {len(cp_df):,}")
    print(f"With aligned sectors: {with_align:,} "
          f"({100 * with_align / len(cp_df):.1f}%)")
    print(f"Direction split     : "
          f"UP={(cp_df['direction']=='UP').sum():,}  "
          f"DOWN={(cp_df['direction']=='DOWN').sum():,}")
    print(f"Saved to            : {csv_path}")

    return cp_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH  = r'C:\Users\Axis\Documents\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv'
    OUTPUT_DIR = r'C:\Users\Axis\Documents\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline_changepoint_nopelt'

    config = Config(
        DETECTION_WINDOW=1,         # points each side for scoring (replaces PELT_PENALTY)
        ALIGNMENT_WINDOW_HOURS=0,   # +/- hours to count aligned sectors
        VALUE_WINDOW=1,             # points for value_before / value_after

        # Threshold options:
        # 'manual'   -> fixed magnitude cutoff
        # 'adaptive' -> auto threshold per sector (sensitivity x std)
        # 'none'     -> keep all change points
        THRESHOLD_MODE='manual',
        MANUAL_THRESHOLD=2.5,       # used when mode='manual' (dB)
        ADAPTIVE_SENSITIVITY=0.5,   # used when mode='adaptive' (lower = more sensitive)
    )

    find_change_points(DATA_PATH, OUTPUT_DIR, config)