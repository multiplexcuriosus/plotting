#!/usr/bin/env python3
"""Shared utilities for DLAB intercept dataset extraction and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPISODE_START = 1
EPISODE_STOP = 2
EPISODE_CANCEL_CURRENT = 3
EPISODE_CANCEL_LAST = 4


@dataclass(frozen=True)
class EpisodeWindow:
    """Committed episode window with local contiguous episode_id."""

    episode_id: int
    start_timestamp: float
    end_timestamp: float

    @property
    def duration_sec(self) -> float:
        return self.end_timestamp - self.start_timestamp


@dataclass(frozen=True)
class EventLogEntry:
    timestamp: float
    value: int


def default_source_id_from_bag_path(path_to_target_bag: str) -> str:
    """Use bag directory name and strip one trailing _bag suffix."""
    name = Path(path_to_target_bag).name
    return name[:-4] if name.endswith("_bag") else name


def parse_episode_control_events(
    events: Sequence[EventLogEntry],
) -> Tuple[List[EpisodeWindow], List[str]]:
    """Parse /episode/control contract into final committed windows.

    Contract:
      1 = start current episode
      2 = stop and commit current episode
      3 = cancel current unfinished episode
      4 = cancel last committed episode
    """
    current_start: Optional[float] = None
    committed: List[Tuple[float, float]] = []
    warnings: List[str] = []

    for entry in events:
        t = float(entry.timestamp)
        v = int(entry.value)

        if v == EPISODE_START:
            if current_start is None:
                current_start = t
            else:
                warnings.append("duplicate start while already recording; ignored")

        elif v == EPISODE_STOP:
            if current_start is None:
                warnings.append("stop while not recording; ignored")
            else:
                committed.append((current_start, t))
                current_start = None

        elif v == EPISODE_CANCEL_CURRENT:
            if current_start is None:
                warnings.append("cancel_current while not recording; ignored")
            else:
                current_start = None

        elif v == EPISODE_CANCEL_LAST:
            if current_start is not None:
                warnings.append("cancel_last while recording; ignored")
            elif committed:
                committed.pop()
            else:
                warnings.append("cancel_last with no committed episode; ignored")

        else:
            warnings.append(f"unknown episode marker {v}; ignored")

    if current_start is not None:
        warnings.append("bag ended during unfinished episode; discarded")

    windows = [
        EpisodeWindow(episode_id=i, start_timestamp=s, end_timestamp=e)
        for i, (s, e) in enumerate(committed)
    ]
    return windows, warnings


def find_episode_index_for_timestamp(
    timestamp: float,
    windows: Sequence[EpisodeWindow],
    start_idx: int,
) -> Tuple[Optional[int], int]:
    """Return local episode index containing timestamp with inclusive boundaries."""
    idx = start_idx
    n = len(windows)
    while idx < n and timestamp > windows[idx].end_timestamp:
        idx += 1
    if idx >= n:
        return None, idx
    window = windows[idx]
    if window.start_timestamp <= timestamp <= window.end_timestamp:
        return idx, idx
    return None, idx


def assign_timestamps_to_episode_ids(
    timestamps: Sequence[float],
    windows: Sequence[EpisodeWindow],
) -> List[Optional[int]]:
    """Assign each timestamp to one episode id or None if outside windows."""
    out: List[Optional[int]] = []
    cursor = 0
    for t in timestamps:
        ep_idx, cursor = find_episode_index_for_timestamp(float(t), windows, cursor)
        out.append(ep_idx)
        if cursor >= len(windows):
            out.extend([None] * (len(timestamps) - len(out)))
            break
    return out


def resolve_dataset_csv_dir(input_path: str) -> Path:
    """Resolve dataset_csv from a dataset, bag, or single-bag recording parent."""
    p = Path(input_path).expanduser().resolve()
    candidate_a = p / "manifest.csv"
    if candidate_a.exists():
        return p

    candidate_b = p / "dataset_csv" / "manifest.csv"
    if candidate_b.exists():
        return p / "dataset_csv"

    nested = sorted(p.glob("*_bag/dataset_csv/manifest.csv")) if p.is_dir() else []
    if len(nested) == 1:
        return nested[0].parent
    if len(nested) > 1:
        raise RuntimeError(
            f"Found multiple nested dataset_csv sources under '{p}': "
            f"{[str(path.parent) for path in nested]}. Pass one explicitly."
        )

    raise RuntimeError(
        f"Could not find manifest.csv under '{p}'. "
        "Run extract_intercept_dataset_csv.py on this bag first."
    )


def require_columns(df: pd.DataFrame, cols: Sequence[str], csv_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{csv_name} is missing required columns: {missing}")


def table_x_to_s(x_table: np.ndarray, line_center_x: float, s_sign: float) -> np.ndarray:
    if s_sign == 0.0:
        raise ValueError("s_sign must be non-zero")
    return s_sign * (x_table - line_center_x)


def s_to_table_x(s_value: np.ndarray, line_center_x: float, s_sign: float) -> np.ndarray:
    if s_sign == 0.0:
        raise ValueError("s_sign must be non-zero")
    return line_center_x + s_value / s_sign


def pick_representative_classic_row(
    rows: pd.DataFrame,
    selection: str,
) -> Optional[pd.Series]:
    if rows.empty:
        return None
    ordered = rows.sort_values("timestamp", kind="mergesort")
    if selection == "first":
        return ordered.iloc[0]
    if selection == "last":
        return ordered.iloc[-1]
    raise ValueError(f"Unsupported classic selection: {selection}")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _strictly_increasing_mask(t: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.zeros((0,), dtype=bool)
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    return keep


def detect_motion_onset_time(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    speed_threshold_mps: float,
    min_consecutive: int,
    smoothing_window: int,
    min_displacement_m: float,
) -> Optional[float]:
    """Return onset timestamp or None.

    Uses smoothed copy for event detection only.
    """
    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive")
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")
    if timestamps.size < 2:
        return None

    keep = _strictly_increasing_mask(timestamps)
    t = timestamps[keep]
    xs = x[keep]
    ys = y[keep]
    if t.size < 2:
        return None

    xs = _moving_average(xs, smoothing_window)
    ys = _moving_average(ys, smoothing_window)

    dt = np.diff(t)
    dx = np.diff(xs)
    dy = np.diff(ys)
    speed = np.hypot(dx, dy) / dt

    above = speed >= speed_threshold_mps
    if not np.any(above):
        return None

    initial_xy = np.array([xs[0], ys[0]], dtype=float)
    run = 0
    run_start_idx = 0
    for i, is_above in enumerate(above):
        if is_above:
            if run == 0:
                run_start_idx = i
            run += 1
            if run >= min_consecutive:
                onset_idx = run_start_idx + 1
                displacement = np.hypot(
                    xs[onset_idx] - initial_xy[0],
                    ys[onset_idx] - initial_xy[1],
                )
                if displacement >= min_displacement_m:
                    return float(t[onset_idx])
        else:
            run = 0

    return None


def detect_max_y_intercept(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    motion_onset_time: Optional[float],
) -> Tuple[Optional[float], Optional[float], bool]:
    """Return (t, x, used_no_onset_fallback)."""
    if timestamps.size == 0:
        return None, None, False

    if motion_onset_time is None:
        idx = int(np.argmax(y))
        return float(timestamps[idx]), float(x[idx]), True

    mask = timestamps >= motion_onset_time
    if not np.any(mask):
        return None, None, False
    y_sub = y[mask]
    x_sub = x[mask]
    t_sub = timestamps[mask]
    idx = int(np.argmax(y_sub))
    return float(t_sub[idx]), float(x_sub[idx]), False


def detect_middle_line_crossing(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    middle_line_y: float,
    motion_onset_time: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Return first post-onset forward crossing time/x or (None, None)."""
    if timestamps.size < 2:
        return None, None

    start_idx = 0
    if motion_onset_time is not None:
        idx_candidates = np.where(timestamps >= motion_onset_time)[0]
        if idx_candidates.size == 0:
            return None, None
        start_idx = int(idx_candidates[0])

    for i in range(start_idx, timestamps.size - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if y1 < y0:
            continue
        if (y0 - middle_line_y) * (y1 - middle_line_y) > 0.0:
            continue
        if y1 == y0:
            continue
        alpha = (middle_line_y - y0) / (y1 - y0)
        if alpha < 0.0 or alpha > 1.0:
            continue
        t = float(timestamps[i] + alpha * (timestamps[i + 1] - timestamps[i]))
        x_interp = float(x[i] + alpha * (x[i + 1] - x[i]))
        return t, x_interp

    return None, None


def interpolate_position_at_time(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    event_time: float,
) -> Optional[Tuple[float, float]]:
    """Interpolate x/y at event_time without extrapolation."""
    if timestamps.size == 0:
        return None
    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    if event_time < t0 or event_time > t1:
        return None

    hit = np.where(np.isclose(timestamps, event_time))[0]
    if hit.size > 0:
        idx = int(hit[0])
        return float(x[idx]), float(y[idx])

    insert = int(np.searchsorted(timestamps, event_time, side="right"))
    if insert <= 0 or insert >= timestamps.size:
        return None

    i0 = insert - 1
    i1 = insert
    denom = float(timestamps[i1] - timestamps[i0])
    if denom <= 0.0:
        return None
    alpha = (event_time - float(timestamps[i0])) / denom
    x_interp = float(x[i0] + alpha * (x[i1] - x[i0]))
    y_interp = float(y[i0] + alpha * (y[i1] - y[i0]))
    return x_interp, y_interp


def build_output_rows_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        ["source_id", "episode_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_classic_counts(
    manifest_df: pd.DataFrame,
    classic_df: pd.DataFrame,
) -> Tuple[int, int, int]:
    """Return per-episode counts of classic messages as (zero, one, multiple)."""
    keys = manifest_df[["source_id", "episode_id"]].copy()
    grouped = (
        classic_df.groupby(["source_id", "episode_id"], as_index=False)
        .size()
        .rename(columns={"size": "classic_count"})
    )
    merged = keys.merge(grouped, on=["source_id", "episode_id"], how="left")
    merged["classic_count"] = merged["classic_count"].fillna(0).astype(int)
    zero = int((merged["classic_count"] == 0).sum())
    one = int((merged["classic_count"] == 1).sum())
    multiple = int((merged["classic_count"] > 1).sum())
    return zero, one, multiple


def assign_global_episode_id(
    manifests_in_input_order: Sequence[pd.DataFrame],
) -> List[pd.DataFrame]:
    """Assign deterministic contiguous global episode ids."""
    out: List[pd.DataFrame] = []
    next_gid = 0
    for manifest in manifests_in_input_order:
        ordered = manifest.sort_values("episode_id", kind="mergesort").copy()
        ordered["global_episode_id"] = np.arange(
            next_gid,
            next_gid + len(ordered),
            dtype=int,
        )
        next_gid += len(ordered)
        out.append(ordered)
    return out


def validate_no_duplicate_source_ids(source_ids: Iterable[str]) -> None:
    seen = set()
    duplicates = set()
    for source_id in source_ids:
        if source_id in seen:
            duplicates.add(source_id)
        seen.add(source_id)
    if duplicates:
        dup = sorted(duplicates)
        raise RuntimeError(
            "Duplicate source_id values detected across inputs: "
            f"{dup}. Re-run extraction with --source-id to disambiguate."
        )


def classify_lateral_intercept(s_intercept: float, epsilon_m: float) -> str:
    """Classify the reached side using the canonical signed-s convention."""
    if not np.isfinite(epsilon_m) or epsilon_m < 0.0:
        raise ValueError("epsilon_m must be finite and non-negative")
    if not np.isfinite(s_intercept):
        return "unavailable"
    if s_intercept > epsilon_m:
        return "L"
    if s_intercept < -epsilon_m:
        return "R"
    return "deadband"


def directional_count_bias(n_left: int, n_right: int) -> float:
    """Return (N_L - N_R) / (N_L + N_R), or NaN without directional data."""
    denominator = int(n_left) + int(n_right)
    if denominator == 0:
        return float("nan")
    return float((int(n_left) - int(n_right)) / denominator)


def sample_descriptive_statistics(values: Sequence[float]) -> Dict[str, float]:
    """Return count/location/spread statistics with sample standard deviation."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    count = int(arr.size)
    if count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "sample_std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
        }
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    return {
        "count": count,
        "mean": float(np.mean(arr)),
        "sample_std": float(np.std(arr, ddof=1)) if count >= 2 else float("nan"),
        "median": float(np.median(arr)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def select_monotonic_approach_points(
    y: Sequence[float],
    s: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Select a deterministic strictly increasing-y copy in input-time order.

    Exact duplicate and backward-y points are discarded. This preserves the
    first observation at each reached y frontier without smoothing raw data.
    """
    y_arr = np.asarray(y, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    if y_arr.shape != s_arr.shape:
        raise ValueError("y and s must have identical shapes")

    out_y: List[float] = []
    out_s: List[float] = []
    for y_value, s_value in zip(y_arr, s_arr):
        if not (np.isfinite(y_value) and np.isfinite(s_value)):
            continue
        if not out_y or float(y_value) > out_y[-1]:
            out_y.append(float(y_value))
            out_s.append(float(s_value))
    return np.asarray(out_y, dtype=float), np.asarray(out_s, dtype=float)


def interpolate_approach_on_y_grid(
    y: Sequence[float],
    s: Sequence[float],
    y_grid: Sequence[float],
) -> np.ndarray:
    """Interpolate one episode on y_grid without extrapolation."""
    y_arr = np.asarray(y, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    grid = np.asarray(y_grid, dtype=float)
    out = np.full(grid.shape, np.nan, dtype=float)
    if y_arr.size < 2 or y_arr.shape != s_arr.shape:
        return out
    if np.any(np.diff(y_arr) <= 0.0):
        raise ValueError("episode y values must be strictly increasing")
    covered = (grid >= y_arr[0]) & (grid <= y_arr[-1])
    out[covered] = np.interp(grid[covered], y_arr, s_arr)
    return out


def summarize_episode_grid(values_by_episode: np.ndarray) -> Dict[str, np.ndarray]:
    """Summarize an episode-by-grid matrix with equal episode weight."""
    values = np.asarray(values_by_episode, dtype=float)
    if values.ndim != 2:
        raise ValueError("values_by_episode must be a two-dimensional array")
    n_grid = values.shape[1]
    count = np.sum(np.isfinite(values), axis=0).astype(int)
    mean = np.full(n_grid, np.nan, dtype=float)
    sample_std = np.full(n_grid, np.nan, dtype=float)
    q25 = np.full(n_grid, np.nan, dtype=float)
    q75 = np.full(n_grid, np.nan, dtype=float)
    for idx in range(n_grid):
        col = values[:, idx]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        mean[idx] = float(np.mean(col))
        if col.size >= 2:
            sample_std[idx] = float(np.std(col, ddof=1))
        q25[idx], q75[idx] = np.quantile(col, [0.25, 0.75])
    return {
        "count": count,
        "mean": mean,
        "sample_std": sample_std,
        "q25": q25,
        "q75": q75,
    }
