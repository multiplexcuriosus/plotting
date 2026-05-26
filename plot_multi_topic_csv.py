#!/usr/bin/env python3
"""
Plot multiple per-topic CSVs from bag_to_csv.py on one shared time axis.

Default use case:
    python3 plot_multi_topic_csv.py /tmp/mt_csv out.svg

This expects a directory with files like:
    cartesian_cmd__twist.csv
    episode__control.csv
    teleop__gripper_state_cmd.csv

It plots:
    - /cartesian_cmd/twist linear_x, linear_y, linear_z as continuous lines
    - /teleop/gripper_state_cmd data as a step signal on a secondary y-axis
    - /episode/control data as vertical event markers
    - optional red zero-command regions where selected twist columns are all zero

It also keeps the important functionality of the older single-CSV plotter:
    - skip-percent cropping
    - relative time axis
    - paired PlotJuggler/RQT *_x / *_y column handling
    - zero-region duration labels
    - output to file or interactive display
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TWIST_CSV = "cartesian_cmd__twist.csv"
DEFAULT_EPISODE_CSV = "episode__control.csv"
DEFAULT_GRIPPER_CSV = "teleop__gripper_state_cmd.csv"

EPISODE_LABELS = {
    1: "start",
    2: "stop",
    3: "cancel current",
    4: "cancel last",
}


@dataclass
class LoadedCsv:
    name: str
    path: Path
    df: pd.DataFrame
    time_col: Optional[str]
    time_s_abs: Optional[pd.Series]


def contiguous_true_regions(mask: pd.Series) -> List[Tuple[int, int]]:
    """
    Return index ranges where mask is continuously True.
    Each region is returned as (start_index, end_index), inclusive.
    """
    values = mask.to_numpy()
    regions: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for i, is_true in enumerate(values):
        if bool(is_true) and start is None:
            start = i
        elif not bool(is_true) and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(values) - 1))

    return regions


def format_duration(duration_s: float) -> str:
    """Format a duration measured in seconds for display."""
    seconds = abs(float(duration_s))

    if seconds >= 60:
        minutes = int(seconds // 60)
        rem_seconds = seconds % 60
        return f"{minutes}m {rem_seconds:.1f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds * 1e6:.1f}us"


def convert_time_to_seconds(values: pd.Series, input_unit: str) -> pd.Series:
    """Convert a numeric time series to seconds."""
    out = pd.to_numeric(values, errors="coerce")
    if input_unit == "s":
        return out
    if input_unit == "ms":
        return out / 1e3
    if input_unit == "us":
        return out / 1e6
    if input_unit == "ns":
        return out / 1e9
    raise ValueError(f"Unsupported time unit: {input_unit}")


def find_time_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    """
    Pick a usable time column.

    bag_to_csv.py writes t_abs and t_rel. For multi-topic overlays, t_abs is
    preferred because every per-topic CSV can then be aligned to one global
    relative axis.
    """
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Requested time column '{preferred}' not found. Available: {list(df.columns)}")
        return preferred

    candidates = [
        "t_abs",
        "timestamp",
        "time",
        "header_stamp",
        "t_rel",
        "t_episode",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {c.lower(): c for c in df.columns}
    for key in ["t_abs", "timestamp", "time", "header_stamp", "t_rel", "t_episode"]:
        if key in lower_map:
            return lower_map[key]

    return None


def load_csv(path: Path, name: str, time_column: Optional[str], time_unit: str) -> LoadedCsv:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    t_col = find_time_column(df, preferred=time_column)
    if t_col is None:
        t_abs = None
    else:
        t_abs = convert_time_to_seconds(df[t_col], time_unit)
        if t_abs.isna().all():
            raise ValueError(f"Time column '{t_col}' in {path} could not be parsed as numeric")

    return LoadedCsv(name=name, path=path, df=df, time_col=t_col, time_s_abs=t_abs)


def resolve_csv_path(input_path: Path, explicit: Optional[str], default_name: str) -> Optional[Path]:
    """Resolve a CSV path from either a directory input or explicit argument."""
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_absolute() and input_path.is_dir():
            p = input_path / p
        return p

    if input_path.is_dir():
        p = input_path / default_name
        return p if p.exists() else None

    # If input_path is a single file, only use it as the twist CSV by default.
    if input_path.is_file() and default_name == DEFAULT_TWIST_CSV:
        return input_path

    return None


def list_columns(paths: Sequence[Tuple[str, Optional[Path]]]) -> None:
    for name, path in paths:
        if path is None:
            print(f"\n[{name}] not found / not configured")
            continue
        if not path.exists():
            print(f"\n[{name}] missing: {path}")
            continue
        df = pd.read_csv(path, nrows=5)
        print(f"\n[{name}] {path}")
        for col in df.columns:
            print(f"  {col}")


def detect_paired_plotjuggler_columns(df: pd.DataFrame) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Detect old RQT/PlotJuggler-style columns such as:
        /cartesian_cmd/twist/twist/linear/x_x -> x-axis/time values
        /cartesian_cmd/twist/twist/linear/x_y -> actual signal values
    """
    pair_regexes = [
        re.compile(r"^.*/linear/([xyz])_([xy])$"),
        re.compile(r"^linear_([xyz])_([xy])$"),
    ]

    paired: Dict[str, Dict[str, str]] = {}
    for col_name in df.columns:
        for rgx in pair_regexes:
            match = rgx.match(col_name)
            if match:
                axis, component = match.groups()
                paired.setdefault(axis, {})[component] = col_name
                break

    has_xyz = all(
        axis in paired and "x" in paired[axis] and "y" in paired[axis]
        for axis in ["x", "y", "z"]
    )
    return paired if has_xyz else None


def auto_twist_columns(df: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested twist columns missing: {missing}. Available: {list(df.columns)}")
        return requested

    preferred_sets = [
        ["linear_x", "linear_y", "linear_z"],
        ["twist_linear_x", "twist_linear_y", "twist_linear_z"],
        ["x", "y", "z"],
    ]
    for cols in preferred_sets:
        if all(c in df.columns for c in cols):
            return cols

    # Fallback for older flat column names that include linear and axis.
    found: List[str] = []
    for axis in ["x", "y", "z"]:
        candidates = [
            c for c in df.columns
            if "linear" in c.lower() and c.lower().endswith(axis)
        ]
        if candidates:
            found.append(candidates[0])
    if len(found) == 3:
        return found

    raise ValueError(
        "Could not auto-detect twist columns. Use --twist-columns. "
        f"Available columns: {list(df.columns)}"
    )


def auto_value_column(df: pd.DataFrame, requested: Optional[str], purpose: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested {purpose} column '{requested}' not found. Available: {list(df.columns)}")
        return requested

    candidates = ["data", "value", purpose, f"{purpose}_data"]
    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = [
        c for c in df.columns
        if c not in {"t_abs", "t_rel", "t_episode", "episode_idx", "header_stamp"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Could not auto-detect {purpose} value column. Use --{purpose}-column. "
        f"Available columns: {list(df.columns)}"
    )


def crop_by_skip_percent(items: Sequence[LoadedCsv], skip_percent: float) -> None:
    """
    Crop the first part of the shared time range.

    The old script skipped rows of one CSV. For multiple topics with different
    rates, row-based skipping is misleading, so this crops by global time span.
    """
    if skip_percent < 0 or skip_percent >= 100:
        raise ValueError("skip_percent must be in the range [0, 100).")
    if skip_percent == 0:
        return

    all_times = []
    for item in items:
        if item.time_s_abs is not None:
            all_times.append(item.time_s_abs.dropna())
    if not all_times:
        return

    concatenated = pd.concat(all_times, ignore_index=True)
    t_min = float(concatenated.min())
    t_max = float(concatenated.max())
    cutoff = t_min + (t_max - t_min) * (skip_percent / 100.0)

    for item in items:
        if item.time_s_abs is None:
            continue
        keep = item.time_s_abs >= cutoff
        item.df = item.df.loc[keep].reset_index(drop=True)
        item.time_s_abs = item.time_s_abs.loc[keep].reset_index(drop=True)

        if item.df.empty:
            raise ValueError(f"No data left in {item.path} after applying skip_percent={skip_percent}")


def global_time_zero(items: Sequence[LoadedCsv]) -> float:
    all_times = []
    for item in items:
        if item.time_s_abs is not None and not item.time_s_abs.empty:
            all_times.append(item.time_s_abs.dropna())
    if not all_times:
        return 0.0
    return float(pd.concat(all_times, ignore_index=True).min())


def relative_time(item: LoadedCsv, t0: float) -> pd.Series:
    if item.time_s_abs is None:
        return pd.Series(item.df.index, index=item.df.index, dtype=float)
    return item.time_s_abs.reset_index(drop=True) - t0


def plot_zero_regions(
    ax,
    x_values: pd.Series,
    zero_mask: pd.Series,
    min_label_duration: float = 0.0,
    label: str = "linear x=y=z=0",
) -> None:
    """
    Draw continuous red horizontal line segments at y=0 for regions where zero_mask is True.
    Also labels each red segment with its duration.
    """
    zero_regions = contiguous_true_regions(zero_mask)
    if not zero_regions:
        return

    x_values = x_values.reset_index(drop=True)
    label_added = False

    y_min, y_max = ax.get_ylim()
    y_offset = 0.01 * (y_max - y_min) if y_max > y_min else 0.01
    y_text = 0.0 + y_offset

    for start_i, end_i in zero_regions:
        if start_i > 0:
            x_start = 0.5 * (x_values.iloc[start_i - 1] + x_values.iloc[start_i])
        else:
            x_start = x_values.iloc[start_i]

        if end_i < len(x_values) - 1:
            x_end = 0.5 * (x_values.iloc[end_i] + x_values.iloc[end_i + 1])
        else:
            x_end = x_values.iloc[end_i]

        duration = abs(float(x_end - x_start))

        ax.plot(
            [x_start, x_end],
            [0, 0],
            color="red",
            linewidth=5,
            solid_capstyle="butt",
            label=label if not label_added else None,
            zorder=10,
        )
        label_added = True

        if duration >= min_label_duration:
            x_mid = 0.5 * (x_start + x_end)
            ax.text(
                x_mid,
                y_text,
                format_duration(duration),
                color="red",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=11,
                clip_on=True,
            )


def plot_episode_markers(
    ax,
    episode: LoadedCsv,
    t0: float,
    value_col: str,
    annotate: bool,
) -> None:
    x = relative_time(episode, t0)
    values = pd.to_numeric(episode.df[value_col], errors="coerce")

    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.04 * (y_max - y_min) if y_max > y_min else y_max
    used_labels = set()

    for xi, value in zip(x, values):
        if pd.isna(value):
            continue
        int_value = int(value)
        event_name = EPISODE_LABELS.get(int_value, f"episode={int_value}")
        label = f"episode {event_name}"
        draw_label = label if label not in used_labels else None
        used_labels.add(label)

        # Keep styling simple but distinct enough for debugging.
        linestyle = "--" if int_value in (1, 2) else ":"
        linewidth = 1.3 if int_value in (1, 2) else 1.0
        ax.axvline(float(xi), linestyle=linestyle, linewidth=linewidth, alpha=0.75, label=draw_label)

        if annotate:
            ax.text(
                float(xi),
                y_text,
                event_name,
                rotation=90,
                fontsize=8,
                ha="right",
                va="top",
                alpha=0.85,
                clip_on=True,
            )


def combine_legends(ax, ax2=None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    # Preserve order while removing duplicate labels.
    seen = set()
    final_handles = []
    final_labels = []
    for h, lab in zip(handles, labels):
        if not lab or lab in seen:
            continue
        seen.add(lab)
        final_handles.append(h)
        final_labels.append(lab)

    if final_handles:
        ax.legend(final_handles, final_labels, loc="best", fontsize=9)


def plot_overlay(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser()

    twist_path = resolve_csv_path(input_path, args.twist_csv, DEFAULT_TWIST_CSV)
    gripper_path = resolve_csv_path(input_path, args.gripper_csv, DEFAULT_GRIPPER_CSV)
    episode_path = resolve_csv_path(input_path, args.episode_csv, DEFAULT_EPISODE_CSV)

    configured_paths = [
        ("twist", twist_path),
        ("gripper", gripper_path),
        ("episode", episode_path),
    ]

    if args.list_columns:
        list_columns(configured_paths)
        return

    if twist_path is None:
        raise RuntimeError(
            "No twist CSV found. Pass a directory containing cartesian_cmd__twist.csv "
            "or specify --twist-csv."
        )

    loaded: List[LoadedCsv] = []
    twist = load_csv(twist_path, "twist", args.time_column, args.time_unit)
    loaded.append(twist)

    gripper: Optional[LoadedCsv] = None
    if not args.no_gripper and gripper_path is not None and gripper_path.exists():
        gripper = load_csv(gripper_path, "gripper", args.time_column, args.time_unit)
        loaded.append(gripper)

    episode: Optional[LoadedCsv] = None
    if not args.no_episode and episode_path is not None and episode_path.exists():
        episode = load_csv(episode_path, "episode", args.time_column, args.time_unit)
        loaded.append(episode)

    crop_by_skip_percent(loaded, args.skip_percent)
    t0 = global_time_zero(loaded)

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    paired = detect_paired_plotjuggler_columns(twist.df)
    zero_x: Optional[pd.Series] = None
    zero_mask: Optional[pd.Series] = None

    if paired is not None and args.twist_columns is None:
        # Legacy single CSV / PlotJuggler style.
        value_cols = {axis: paired[axis]["y"] for axis in ["x", "y", "z"]}
        for axis in ["x", "y", "z"]:
            x_col = paired[axis]["x"]
            y_col = paired[axis]["y"]
            x_plot_abs = convert_time_to_seconds(twist.df[x_col], args.paired_time_unit)
            x_plot = x_plot_abs - float(x_plot_abs.dropna().iloc[0])
            y_plot = pd.to_numeric(twist.df[y_col], errors="coerce")
            if args.negate_twist:
                y_plot = -y_plot
            ax.plot(x_plot, y_plot, label=f"linear {axis}", linewidth=args.linewidth)

        zero_x = convert_time_to_seconds(twist.df[paired["x"]["x"]], args.paired_time_unit)
        zero_x = zero_x - float(zero_x.dropna().iloc[0])
        zero_mask = (
            twist.df[[value_cols["x"], value_cols["y"], value_cols["z"]]]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .le(args.zero_eps)
            .all(axis=1)
        )

    else:
        x_twist = relative_time(twist, t0)
        twist_cols = auto_twist_columns(twist.df, args.twist_columns)

        for col in twist_cols:
            y_plot = pd.to_numeric(twist.df[col], errors="coerce")
            if args.negate_twist:
                y_plot = -y_plot
            label = args.label_prefix + col if args.label_prefix else col
            ax.plot(x_twist, y_plot, label=label, linewidth=args.linewidth)

        zero_columns = args.zero_columns if args.zero_columns else twist_cols[:3]
        missing_zero_cols = [c for c in zero_columns if c not in twist.df.columns]
        if missing_zero_cols:
            raise ValueError(f"Zero-region columns missing: {missing_zero_cols}")

        zero_x = x_twist
        zero_mask = (
            twist.df[zero_columns]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .le(args.zero_eps)
            .all(axis=1)
        )

    ax.set_xlabel("Time [s]", fontsize=12, fontweight="bold")
    ax.set_ylabel(args.ylabel, fontsize=12, fontweight="bold")
    ax.set_title(args.title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Draw once after line plotting so y-limits are meaningful.
    if not args.no_zero_regions and zero_x is not None and zero_mask is not None:
        print(f"Rows where selected twist columns are zero: {int(zero_mask.sum())} / {len(zero_mask)}")
        plot_zero_regions(
            ax,
            zero_x,
            zero_mask,
            min_label_duration=args.min_label_duration,
            label=args.zero_label,
        )

    ax2 = None
    if gripper is not None:
        gripper_col = auto_value_column(gripper.df, args.gripper_column, "gripper")
        x_gripper = relative_time(gripper, t0)
        y_gripper = pd.to_numeric(gripper.df[gripper_col], errors="coerce")

        if args.gripper_on_secondary_axis:
            ax2 = ax.twinx()
            target_ax = ax2
            target_ax.set_ylabel(args.gripper_ylabel, fontsize=11, fontweight="bold")
            target_ax.set_ylim(args.gripper_ymin, args.gripper_ymax)
        else:
            target_ax = ax

        target_ax.step(
            x_gripper,
            y_gripper,
            where="post",
            label=args.gripper_label,
            linewidth=args.gripper_linewidth,
            alpha=0.85,
        )
        target_ax.scatter(
            x_gripper,
            y_gripper,
            s=args.gripper_marker_size,
            alpha=0.85,
        )

    if episode is not None:
        episode_col = auto_value_column(episode.df, args.episode_column, "episode")
        # Plot after zero regions; markers should span the final primary axis.
        plot_episode_markers(
            ax,
            episode,
            t0,
            value_col=episode_col,
            annotate=not args.no_episode_text,
        )

    combine_legends(ax, ax2)

    if args.xlim is not None:
        ax.set_xlim(args.xlim[0], args.xlim[1])

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay multiple per-topic bag_to_csv.py CSV files on one shared time axis."
    )
    parser.add_argument(
        "input",
        help=(
            "CSV output directory from bag_to_csv.py, or a single twist CSV. "
            "If this is a directory, default filenames are auto-used."
        ),
    )
    parser.add_argument("output", nargs="?", default=None, help="Optional output plot path: .svg, .pdf, .eps, .png, ...")

    parser.add_argument("--twist-csv", default=None, help="Path/name of twist CSV. Default: cartesian_cmd__twist.csv in input dir.")
    parser.add_argument("--gripper-csv", default=None, help="Path/name of gripper CSV. Default: teleop__gripper_state_cmd.csv in input dir.")
    parser.add_argument("--episode-csv", default=None, help="Path/name of episode CSV. Default: episode__control.csv in input dir.")

    parser.add_argument("--list-columns", action="store_true", help="Print columns of configured CSVs and exit.")
    parser.add_argument("--time-column", default=None, help="Time column to use in all CSVs. Default: auto, preferring t_abs.")
    parser.add_argument("--time-unit", choices=["s", "ms", "us", "ns"], default="s", help="Unit of --time-column. bag_to_csv.py t_abs is seconds.")
    parser.add_argument("--paired-time-unit", choices=["s", "ms", "us", "ns"], default="s", help="Unit for legacy paired *_x time columns.")

    parser.add_argument("--twist-columns", nargs="+", default=None, help="Twist/value columns to plot. Default: auto-detect linear_x/y/z.")
    parser.add_argument("--zero-columns", nargs="+", default=None, help="Columns used for zero-region detection. Default: plotted twist columns, first 3.")
    parser.add_argument("--zero-eps", type=float, default=1e-12, help="Absolute threshold for zero-region detection.")
    parser.add_argument("--no-zero-regions", action="store_true", help="Disable red x=y=z=0 duration bars.")
    parser.add_argument("--zero-label", default="linear x=y=z=0", help="Legend label for zero regions.")
    parser.add_argument("--min-label-duration", type=float, default=0.0, help="Only label zero regions with duration >= this many seconds.")

    parser.add_argument("--gripper-column", default=None, help="Column to use from gripper CSV. Default: auto-detect data.")
    parser.add_argument("--episode-column", default=None, help="Column to use from episode CSV. Default: auto-detect data.")
    parser.add_argument("--no-gripper", action="store_true", help="Do not plot gripper CSV even if present.")
    parser.add_argument("--no-episode", action="store_true", help="Do not plot episode CSV even if present.")
    parser.add_argument("--no-episode-text", action="store_true", help="Draw episode vertical lines without text labels.")

    parser.add_argument("--skip-percent", type=float, default=0.0, help="Crop the first N percent of the shared time span.")
    parser.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("START", "END"), help="Visible x-axis range in relative seconds.")
    parser.add_argument("--negate-twist", action="store_true", help="Multiply plotted twist columns by -1.")

    parser.add_argument("--title", default="Cartesian Twist + Episode / Gripper Events", help="Plot title.")
    parser.add_argument("--ylabel", default="Cartesian twist command [m/s]", help="Primary y-axis label.")
    parser.add_argument("--label-prefix", default="", help="Optional prefix for twist legend labels.")
    parser.add_argument("--gripper-label", default="gripper state", help="Legend label for gripper state.")
    parser.add_argument("--gripper-ylabel", default="Gripper state", help="Secondary y-axis label.")
    parser.add_argument("--gripper-on-secondary-axis", action=argparse.BooleanOptionalAction, default=True, help="Plot gripper on secondary y-axis. Default: true.")
    parser.add_argument("--gripper-ymin", type=float, default=-0.1, help="Secondary gripper axis min.")
    parser.add_argument("--gripper-ymax", type=float, default=1.1, help="Secondary gripper axis max.")

    parser.add_argument("--linewidth", type=float, default=2.0, help="Twist line width.")
    parser.add_argument("--gripper-linewidth", type=float, default=1.8, help="Gripper step line width.")
    parser.add_argument("--gripper-marker-size", type=float, default=25.0, help="Gripper marker size.")
    parser.add_argument("--width", type=float, default=12.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=6.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for raster formats.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_overlay(args)


if __name__ == "__main__":
    main()
