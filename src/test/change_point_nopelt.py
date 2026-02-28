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
from multiprocessing import Pool, cpu_count

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # Half-width (in data points) of the comparison windows used to score
    # each candidate position. Larger = smoother, fewer spurious detections.
    DETECTION_WINDOW: int = 2

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
    ADAPTIVE_SENSITIVITY: float = 1

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
    Worker function for multiprocessing — processes a single sector.
    Must be at module level to be picklable on Windows.

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


def _process_alignment_chunk(args):
    """
    Worker function for parallel alignment processing.
    Processes a chunk of change point indices.

    Args:
        args: (indices, ts, sectors, window_ns) tuple

    Returns:
        List of (index, count, name_string) tuples
    """
    indices, ts, sectors, window_ns = args
    results = []

    for i in indices:
        left  = np.searchsorted(ts, ts[i] - window_ns, side='left')
        right = np.searchsorted(ts, ts[i] + window_ns, side='right')
        nearby = sorted(set(sectors[left:right]) - {sectors[i]})
        results.append((i, len(nearby), ';'.join(nearby)))

    return results


def _build_aligned_sector_groups(cp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build groups of aligned sectors at each timestamp.

    A group is defined as all sectors that have a change point at the same
    timestamp and are aligned with at least one other sector at that time.

    This is intended to be saved as a separate CSV so that the main
    change_points.csv stays compact (only counts of aligned sectors).
    """
    if 'timestamp' not in cp_df.columns or 'sector' not in cp_df.columns:
        return pd.DataFrame()

    groups = []
    group_id = 1

    # cp_df is already sorted by timestamp in find_change_points
    for ts, grp in cp_df.groupby('timestamp'):
        sectors_set = set(grp['sector'])
        # Only keep groups with at least 2 sectors
        if len(sectors_set) <= 1:
            continue

        groups.append({
            'group_id': group_id,
            'timestamp': ts,
            'num_sectors': len(sectors_set),
            'sectors': ';'.join(sorted(sectors_set)),
        })
        group_id += 1

    if not groups:
        return pd.DataFrame()

    return pd.DataFrame(groups)


def find_aligned_sectors(cp_df: pd.DataFrame, window_hours: float, n_workers: int = 1):
    """
    For each change point, find which OTHER sectors also have a change point
    within +/- window_hours. Uses sorted timestamps + binary search.

    Can use multiprocessing for speed when processing many change points.

    Args:
        cp_df: DataFrame with 'timestamp' and 'sector' columns
        window_hours: Time window in hours
        n_workers: Number of parallel workers (0 = all cores, 1 = sequential)

    Returns: (counts, names) as int array and list of semicolon-separated strings.
    """
    ts      = cp_df['timestamp'].values.astype('int64')
    sectors = cp_df['sector'].values
    window_ns = int(pd.Timedelta(hours=window_hours).value)
    n = len(cp_df)

    # For small datasets, sequential is faster (less overhead)
    if n < 1000 or n_workers == 1:
        counts = np.zeros(n, dtype=int)
        names  = []

        for i in range(n):
            left  = np.searchsorted(ts, ts[i] - window_ns, side='left')
            right = np.searchsorted(ts, ts[i] + window_ns, side='right')
            nearby = sorted(set(sectors[left:right]) - {sectors[i]})
            counts[i] = len(nearby)
            names.append(';'.join(nearby))

        return counts, names

    # Parallel processing for larger datasets
    n_workers = n_workers if n_workers > 0 else cpu_count()
    indices = list(range(n))
    chunk_size = max(1, n // (n_workers * 4))  # 4 chunks per worker
    chunks = [indices[i:i + chunk_size] for i in range(0, n, chunk_size)]

    worker_args = [(chunk, ts, sectors, window_ns) for chunk in chunks]

    counts = np.zeros(n, dtype=int)
    names = [''] * n

    with Pool(processes=n_workers) as pool:
        for chunk_results in pool.imap(_process_alignment_chunk, worker_args):
            for idx, count, name_str in chunk_results:
                counts[idx] = count
                names[idx] = name_str

    return counts, names


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def find_change_points(data_path: str, output_dir: str, cfg: Config = None) -> pd.DataFrame:
    """
    Detect change points for all sectors and save change_points.csv.

    Output columns (change_points.csv):
        sector, timestamp, value_before, value_after, magnitude, direction,
        num_aligned_sectors
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
        with Pool(processes=n_workers) as pool:
            done = 0
            for results, _ in pool.imap_unordered(_process_one_sector, worker_args, chunksize=64):
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

    counts, names = find_aligned_sectors(cp_df, cfg.ALIGNMENT_WINDOW_HOURS, n_workers)
    cp_df['num_aligned_sectors'] = counts

    # Build aligned sector groups (separate CSV) BEFORE dropping any columns
    aligned_groups_df = _build_aligned_sector_groups(cp_df.assign(aligned_sectors=names))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save main change points CSV WITHOUT aligned_sectors list
    csv_path = out / 'change_points.csv'
    cp_df.to_csv(csv_path, index=False)

    # Save aligned sector groups CSV (if any)
    if aligned_groups_df is not None and not aligned_groups_df.empty:
        groups_path = out / 'aligned_sector_groups.csv'
        aligned_groups_df.to_csv(groups_path, index=False)
    else:
        groups_path = None

    with_align = (cp_df['num_aligned_sectors'] > 0).sum()
    print(f"\n--- Done ---")
    print(f"Total change points : {len(cp_df):,}")
    print(f"With aligned sectors: {with_align:,} "
          f"({100 * with_align / len(cp_df):.1f}%)")
    if groups_path is not None:
        print(f"Aligned sector groups : {len(aligned_groups_df):,}")
        print(f"Aligned groups CSV    : {groups_path}")
    print(f"Direction split     : "
          f"UP={(cp_df['direction']=='UP').sum():,}  "
          f"DOWN={(cp_df['direction']=='DOWN').sum():,}")
    print(f"Saved to            : {csv_path}")

    return cp_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH  = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv'
    OUTPUT_DIR = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline_changepoint_nopelt'

    config = Config(
        DETECTION_WINDOW=1,         # points each side for scoring (replaces PELT_PENALTY)
        ALIGNMENT_WINDOW_HOURS=0,   # +/- hours to count aligned sectors
        VALUE_WINDOW=1,             # points for value_before / value_after

        # 'manual'   -> fixed magnitude cutoff
        # 'adaptive' -> auto threshold per sector (sensitivity x std)
        # 'none'     -> keep all change points
        THRESHOLD_MODE='adaptive',
        MANUAL_THRESHOLD=6.0,      
        ADAPTIVE_SENSITIVITY=1,   # used when mode='adaptive' (lower = more sensitive)
    )

    find_change_points(DATA_PATH, OUTPUT_DIR, config)