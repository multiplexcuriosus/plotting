#!/usr/bin/env python3
"""
Generic CSV plotting helper for ROS2 bag exports and PlotJuggler/RQT-style CSVs.

The default behavior tries to reproduce the old cartesian twist plotter:
- auto-detect /cartesian_cmd/twist/twist/linear/{x,y,z} columns
- auto-detect old paired PlotJuggler columns like .../linear/x_x and .../linear/x_y
- mark continuous regions where x=y=z=0 with red segments and duration labels

It also works with the bag_to_csv.py output generated earlier, where /cartesian_cmd/twist
is exported as columns like:
    t_abs,t_rel,header_stamp,frame_id,linear_x,linear_y,linear_z,angular_x,...

Examples:
    # Auto-detect cartesian twist linear columns from bag_to_csv output
    python3 plot_csv_topics.py cartesian_cmd__twist.csv twist.svg

    # Explicit columns from a joint_states CSV
    python3 plot_csv_topics.py joint_states.csv qpos.svg \
        --columns qpos_0 qpos_1 qpos_2 qpos_3 qpos_4 qpos_5 qpos_6 gripper_width \
        --ylabel "Joint position / gripper width"

    # Plot only z velocity and invert sign
    python3 plot_csv_topics.py cartesian_cmd__twist.csv z_vel.svg \
        --columns linear_z --negate

    # Old PlotJuggler/RQT-style CSV with paired *_x/*_y columns in microseconds
    python3 plot_csv_topics.py plotjuggler_export.csv twist.svg --time-unit us

    # Inspect columns
    python3 plot_csv_topics.py cartesian_cmd__twist.csv --list-columns
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Import pyplot after argparse setup is okay; we still use a normal import so plt.show() works.
import matplotlib.pyplot as plt


OLD_CARTESIAN_LINEAR_PREFIX = "/cartesian_cmd/twist/twist/linear/"
BAG_TO_CSV_LINEAR_COLS = ["linear_x", "linear_y", "linear_z"]
OLD_DIRECT_LINEAR_COLS = [
    f"{OLD_CARTESIAN_LINEAR_PREFIX}x",
    f"{OLD_CARTESIAN_LINEAR_PREFIX}y",
    f"{OLD_CARTESIAN_LINEAR_PREFIX}z",
]
TIME_CANDIDATES = [
    "t_episode",
    "t_rel",
    "timestamp",
    "time",
    "t_abs",
    "header_stamp",
]


@dataclass
class SeriesSpec:
    label: str
    y_col: str
    x_col: Optional[str] = None
    negate: bool = False


@dataclass
class TimeData:
    values: pd.Series
    label: str
    unit: str


def log(msg: str) -> None:
    print(msg, flush=True)


def contiguous_true_regions(mask: pd.Series) -> List[Tuple[int, int]]:
    """
    Return index ranges where mask is continuously True.
    Each region is returned as (start_index, end_index), inclusive.
    """
    values = mask.fillna(False).to_numpy(dtype=bool)
    regions: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for i, is_true in enumerate(values):
        if is_true and start is None:
            start = i
        elif not is_true and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(values) - 1))

    return regions


def format_duration(duration: float, unit: str = "s") -> str:
    """
    Format duration for display.

    duration is assumed to be measured in the same unit as the x-axis.
    Supported display units: samples, s.
    """
    duration = abs(float(duration))

    if unit == "samples":
        return f"{duration:.0f} samples"

    if unit != "s":
        raise ValueError(f"Unsupported duration display unit: {unit}")

    seconds = duration
    if seconds >= 60:
        minutes = int(seconds // 60)
        rem_seconds = seconds % 60
        return f"{minutes}m {rem_seconds:.1f}s"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds * 1e6:.1f}us"


def unit_scale_to_seconds(unit: str) -> float:
    if unit == "s":
        return 1.0
    if unit == "ms":
        return 1e-3
    if unit == "us":
        return 1e-6
    if unit == "ns":
        return 1e-9
    raise ValueError(f"Unsupported time unit: {unit}")


def infer_time_unit(values: pd.Series, column_name: str) -> str:
    """
    Infer raw time unit. Heuristic only.

    - bag_to_csv.py writes t_abs/t_rel/t_episode/header_stamp in seconds.
    - ROS bag timestamps from raw exports are often nanoseconds.
    - Some PlotJuggler/RQT exports use microseconds.
    """
    name = column_name.lower()
    if name in {"t_abs", "t_rel", "t_episode", "header_stamp"}:
        return "s"

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return "s"

    abs_max = float(numeric.abs().max())
    diffs = numeric.diff().abs()
    diffs = diffs[diffs > 0].dropna()
    median_diff = float(diffs.median()) if len(diffs) else 0.0

    # Typical absolute Unix/ROS nanoseconds: ~1e18.
    if abs_max > 1e14 or median_diff > 1e6:
        return "ns"

    # Typical microsecond timestamps: ~1e12, or 30 Hz spacing ~33333 us.
    if abs_max > 1e10 or median_diff > 1000:
        return "us"

    # Typical millisecond timestamps: 30 Hz spacing ~33 ms.
    if median_diff > 1:
        return "ms"

    return "s"


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    return pd.to_numeric(df[column], errors="coerce")


def to_plot_time(
    values: pd.Series,
    column_name: str,
    raw_unit: str = "auto",
    relative: bool = True,
) -> TimeData:
    """Convert raw time values to seconds for plotting."""
    x = pd.to_numeric(values, errors="coerce").reset_index(drop=True)

    if raw_unit == "auto":
        actual_unit = infer_time_unit(x, column_name)
    else:
        actual_unit = raw_unit

    if relative:
        first_valid = x.dropna().iloc[0] if not x.dropna().empty else 0.0
        x = x - first_valid
        label = "Relative time [s]"
    else:
        label = "Time [s]"

    x = x * unit_scale_to_seconds(actual_unit)
    return TimeData(values=x, label=label, unit="s")


def sample_index_time(n: int) -> TimeData:
    return TimeData(
        values=pd.Series(range(n), dtype=float),
        label="Sample index",
        unit="samples",
    )


def find_time_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested --time-column not found: {requested}")
        return requested

    lower_to_real = {col.lower(): col for col in df.columns}
    for candidate in TIME_CANDIDATES:
        if candidate.lower() in lower_to_real:
            return lower_to_real[candidate.lower()]

    # Fallback: first column containing "time".
    for col in df.columns:
        if "time" in col.lower():
            return col

    return None


def detect_old_cartesian_paired(df: pd.DataFrame) -> Optional[List[SeriesSpec]]:
    """
    Detect PlotJuggler/RQT-style paired columns:
        /cartesian_cmd/twist/twist/linear/x_x = x-axis/time values
        /cartesian_cmd/twist/twist/linear/x_y = actual x velocity values
    """
    pair_regex = re.compile(
        r"^/cartesian_cmd/twist/twist/linear/([xyz])_([xy])$"
    )
    paired_cols: Dict[str, Dict[str, str]] = {}

    for col_name in df.columns:
        match = pair_regex.match(col_name)
        if not match:
            continue
        axis, component = match.groups()
        paired_cols.setdefault(axis, {})[component] = col_name

    has_xyz = all(
        axis in paired_cols and "x" in paired_cols[axis] and "y" in paired_cols[axis]
        for axis in ["x", "y", "z"]
    )
    if not has_xyz:
        return None

    return [
        SeriesSpec(
            label=f"Linear {axis}",
            x_col=paired_cols[axis]["x"],
            y_col=paired_cols[axis]["y"],
            # Preserve old script behavior: paired y values were negated.
            negate=True,
        )
        for axis in ["x", "y", "z"]
    ]


def detect_auto_cartesian(df: pd.DataFrame) -> List[SeriesSpec]:
    """Auto-detect common cartesian linear velocity columns."""
    paired = detect_old_cartesian_paired(df)
    if paired is not None:
        return paired

    if all(col in df.columns for col in BAG_TO_CSV_LINEAR_COLS):
        return [
            SeriesSpec(label=f"Linear {axis}", y_col=f"linear_{axis}")
            for axis in ["x", "y", "z"]
        ]

    if all(col in df.columns for col in OLD_DIRECT_LINEAR_COLS):
        return [
            SeriesSpec(label=f"Linear {axis}", y_col=f"{OLD_CARTESIAN_LINEAR_PREFIX}{axis}")
            for axis in ["x", "y", "z"]
        ]

    raise ValueError(
        "Could not auto-detect cartesian linear columns. Use --list-columns, then pass "
        "--columns explicitly. Expected either linear_x/linear_y/linear_z or old "
        f"columns under {OLD_CARTESIAN_LINEAR_PREFIX!r}."
    )


def column_matches_component(col: str, component: str) -> bool:
    return (
        col == component
        or col.endswith(f"_{component}")
        or col.endswith(f"/{component}")
        or col.endswith(f".{component}")
    )


def select_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    prefix: Optional[str],
    components: Optional[Sequence[str]],
) -> List[SeriesSpec]:
    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                "Requested column(s) not found: "
                + ", ".join(missing)
                + "\nUse --list-columns to inspect available columns."
            )
        return [SeriesSpec(label=col, y_col=col) for col in columns]

    if prefix:
        selected = [col for col in df.columns if col.startswith(prefix)]
        if components:
            selected = [
                col
                for col in selected
                if any(column_matches_component(col, comp) for comp in components)
            ]
        if not selected:
            raise ValueError(
                f"No columns matched --prefix {prefix!r}"
                + (f" and --components {list(components)!r}" if components else "")
            )
        return [SeriesSpec(label=col, y_col=col) for col in selected]

    return detect_auto_cartesian(df)


def apply_skip_percent(df: pd.DataFrame, skip_percent: float) -> pd.DataFrame:
    if skip_percent < 0 or skip_percent >= 100:
        raise ValueError("--skip-percent must be in the range [0, 100).")

    skip_rows = int(len(df) * (skip_percent / 100.0))
    if skip_rows > 0:
        df = df.iloc[skip_rows:].reset_index(drop=True)

    if df.empty:
        raise ValueError("No data left after applying --skip-percent.")

    return df


def build_common_time(
    df: pd.DataFrame,
    requested_time_col: Optional[str],
    time_unit: str,
    relative_time: bool,
) -> TimeData:
    time_col = find_time_column(df, requested_time_col)
    if time_col is None:
        return sample_index_time(len(df))
    return to_plot_time(df[time_col], time_col, raw_unit=time_unit, relative=relative_time)


def plot_zero_regions(
    ax,
    x_values: pd.Series,
    zero_mask: pd.Series,
    duration_unit: str = "s",
    min_label_duration: float = 0.0,
    label: str = "zero command",
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

    # Need the current y-axis range after signal plotting.
    y_min, y_max = ax.get_ylim()
    if not math.isfinite(y_min) or not math.isfinite(y_max) or y_min == y_max:
        y_min, y_max = -1.0, 1.0
    y_offset = 0.01 * (y_max - y_min)
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

        if pd.isna(x_start) or pd.isna(x_end):
            continue

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
            duration_text = format_duration(duration, duration_unit)
            ax.text(
                x_mid,
                y_text,
                duration_text,
                color="red",
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=11,
                clip_on=True,
            )


def infer_zero_columns(series_specs: Sequence[SeriesSpec]) -> List[str]:
    """Infer x/y/z value columns for zero-region detection."""
    by_axis: Dict[str, str] = {}
    for spec in series_specs:
        col = spec.y_col
        label = spec.label.lower()
        for axis in ["x", "y", "z"]:
            if (
                col == f"linear_{axis}"
                or col == f"{OLD_CARTESIAN_LINEAR_PREFIX}{axis}"
                or col.endswith(f"/linear/{axis}_y")
                or label == f"linear {axis}"
            ):
                by_axis[axis] = col

    if all(axis in by_axis for axis in ["x", "y", "z"]):
        return [by_axis[axis] for axis in ["x", "y", "z"]]
    return []


def build_zero_mask(
    df: pd.DataFrame,
    zero_columns: Sequence[str],
    zero_eps: float,
) -> pd.Series:
    missing = [col for col in zero_columns if col not in df.columns]
    if missing:
        raise ValueError(f"--zero-columns not found: {missing}")
    values = pd.DataFrame({col: numeric_series(df, col).abs() for col in zero_columns})
    return values.le(zero_eps).all(axis=1)


def maybe_filter_x_range(
    df: pd.DataFrame,
    x_values: pd.Series,
    start_x: Optional[float],
    end_x: Optional[float],
) -> pd.DataFrame:
    if start_x is None and end_x is None:
        return df
    mask = pd.Series(True, index=df.index)
    if start_x is not None:
        mask &= x_values >= start_x
    if end_x is not None:
        mask &= x_values <= end_x
    filtered = df.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No data left after applying --start-x/--end-x.")
    return filtered


def plot_csv(args: argparse.Namespace) -> None:
    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError("CSV file is empty.")

    if args.list_columns:
        log("Columns:")
        for col in df.columns:
            log(f"  {col}")
        return

    df = apply_skip_percent(df, args.skip_percent)
    series_specs = select_columns(df, args.columns, args.prefix, args.components)

    # Explicit --negate overrides/extends auto legacy paired negation.
    if args.negate:
        for spec in series_specs:
            spec.negate = True
    elif args.no_legacy_paired_negate:
        for spec in series_specs:
            spec.negate = False

    # Build common time first. For paired old-style columns, each signal can use its own x column.
    common_time = build_common_time(
        df,
        requested_time_col=args.time_column,
        time_unit=args.time_unit,
        relative_time=not args.absolute_time,
    )

    # Optional x range filtering currently only uses common time. This is intentional;
    # for paired data, use --skip-percent or export a cleaner CSV if you need strict clipping.
    if args.start_x is not None or args.end_x is not None:
        if any(spec.x_col is not None for spec in series_specs):
            log("[WARNING] --start-x/--end-x with paired x/y columns filters using the common/auto time column only.")
        df = maybe_filter_x_range(df, common_time.values, args.start_x, args.end_x)
        common_time = build_common_time(
            df,
            requested_time_col=args.time_column,
            time_unit=args.time_unit,
            relative_time=not args.absolute_time,
        )

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    zero_x_values: Optional[pd.Series] = None
    zero_x_unit = common_time.unit

    for i, spec in enumerate(series_specs):
        y = numeric_series(df, spec.y_col).reset_index(drop=True)
        if spec.negate:
            y = -y

        if spec.x_col:
            x_time = to_plot_time(
                df[spec.x_col],
                spec.x_col,
                raw_unit=args.time_unit,
                relative=not args.absolute_time,
            )
            x = x_time.values
            if i == 0:
                zero_x_values = x
                zero_x_unit = x_time.unit
            x_label = x_time.label
        else:
            x = common_time.values
            if i == 0:
                zero_x_values = x
                zero_x_unit = common_time.unit
            x_label = common_time.label

        ax.plot(x, y, label=spec.label, linewidth=args.linewidth)

    # Zero-region detection.
    should_mark_zero = not args.no_zero_regions
    zero_columns = list(args.zero_columns or [])
    if should_mark_zero and not zero_columns:
        zero_columns = infer_zero_columns(series_specs)

    if should_mark_zero and zero_columns:
        zero_mask = build_zero_mask(df, zero_columns, args.zero_eps)
        log(f"Rows where selected zero columns are all zero: {int(zero_mask.sum())} / {len(df)}")
        if zero_x_values is None:
            zero_x_values = common_time.values
            zero_x_unit = common_time.unit
        plot_zero_regions(
            ax,
            zero_x_values,
            zero_mask,
            duration_unit=zero_x_unit,
            min_label_duration=args.min_label_duration,
            label=args.zero_label,
        )
    elif should_mark_zero:
        log("[INFO] zero-region annotation skipped; could not infer x/y/z zero columns. Use --zero-columns if needed.")

    title = args.title if args.title else csv_file.stem.replace("__", "/")
    ylabel = args.ylabel if args.ylabel else "Value"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(args.xlabel if args.xlabel else x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    if args.grid:
        ax.grid(True, alpha=0.3)
    if not args.no_legend:
        ax.legend(loc=args.legend_loc, fontsize=10)

    fig.tight_layout()

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        log(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected columns from a CSV file exported from ROS2 bag data. "
            "If no columns are given, cartesian twist linear x/y/z columns are auto-detected."
        )
    )
    parser.add_argument("csv_file", help="Path to input CSV file.")
    parser.add_argument("output_file", nargs="?", default=None, help="Optional output plot path, e.g. .svg, .pdf, .eps, .png.")

    selection = parser.add_argument_group("column selection")
    selection.add_argument("--columns", nargs="+", default=None, help="Exact CSV columns to plot as y-values.")
    selection.add_argument("--prefix", default=None, help="Plot all columns whose names start with this prefix.")
    selection.add_argument("--components", nargs="+", default=None, help="Filter --prefix selection by suffix/component, e.g. x y z.")
    selection.add_argument("--list-columns", action="store_true", help="Print CSV columns and exit.")

    time_group = parser.add_argument_group("time axis")
    time_group.add_argument("--time-column", default=None, help="Column to use as common x-axis. Default: auto-detect t_episode, t_rel, timestamp, time, t_abs, header_stamp.")
    time_group.add_argument("--time-unit", choices=["auto", "s", "ms", "us", "ns"], default="auto", help="Raw unit of time columns before conversion to seconds. Default: auto.")
    time_group.add_argument("--absolute-time", action="store_true", help="Do not subtract the first timestamp. Default: plot relative time from zero.")
    time_group.add_argument("--start-x", type=float, default=None, help="Optional lower x-axis bound after conversion to plotted units.")
    time_group.add_argument("--end-x", type=float, default=None, help="Optional upper x-axis bound after conversion to plotted units.")

    transform = parser.add_argument_group("data transforms")
    transform.add_argument("--skip-percent", type=float, default=0.0, help="Percentage of initial rows to skip before plotting, in [0, 100).")
    transform.add_argument("--negate", action="store_true", help="Negate all plotted y-values.")
    transform.add_argument("--no-legacy-paired-negate", action="store_true", help="Disable the old-script behavior that negates PlotJuggler paired cartesian columns.")

    zero = parser.add_argument_group("zero-region annotation")
    zero.add_argument("--zero-columns", nargs="+", default=None, help="Columns used for zero-region detection. If omitted, x/y/z linear columns are inferred when possible.")
    zero.add_argument("--zero-eps", type=float, default=1e-12, help="Absolute tolerance for treating values as zero. Default: 1e-12.")
    zero.add_argument("--no-zero-regions", action="store_true", help="Disable red zero-region segments and duration labels.")
    zero.add_argument("--min-label-duration", type=float, default=0.0, help="Only label red zero regions with duration >= this value in plotted x-axis units, usually seconds.")
    zero.add_argument("--zero-label", default="x=y=z=0", help="Legend label for zero-region segments.")

    style = parser.add_argument_group("plot style")
    style.add_argument("--title", default=None, help="Plot title. Default: derived from CSV filename.")
    style.add_argument("--xlabel", default=None, help="X-axis label override.")
    style.add_argument("--ylabel", default=None, help="Y-axis label override.")
    style.add_argument("--fig-width", type=float, default=12.0, help="Figure width in inches. Default: 12.")
    style.add_argument("--fig-height", type=float, default=6.0, help="Figure height in inches. Default: 6.")
    style.add_argument("--linewidth", type=float, default=2.0, help="Line width. Default: 2.")
    style.add_argument("--dpi", type=int, default=300, help="Output DPI. Default: 300.")
    style.add_argument("--grid", action=argparse.BooleanOptionalAction, default=True, help="Show grid. Default: true.")
    style.add_argument("--no-legend", action="store_true", help="Hide legend.")
    style.add_argument("--legend-loc", default="best", help="Matplotlib legend location. Default: best.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_x is not None and args.end_x is not None and args.end_x < args.start_x:
        raise ValueError("--end-x must be >= --start-x")
    if args.zero_eps < 0:
        raise ValueError("--zero-eps must be >= 0")
    plot_csv(args)


if __name__ == "__main__":
    main()
