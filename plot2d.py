import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse


def contiguous_true_regions(mask):
    """
    Return index ranges where mask is continuously True.
    Each region is returned as (start_index, end_index), inclusive.
    """
    values = mask.to_numpy()
    regions = []
    start = None

    for i, is_true in enumerate(values):
        if is_true and start is None:
            start = i
        elif not is_true and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(values) - 1))

    return regions


def format_duration(duration, unit="s"):
    """
    Format duration for display.

    duration is assumed to be measured in the same unit as the x-axis.
    """
    duration = abs(float(duration))

    if unit == "samples":
        return f"{duration:.0f} samples"

    if unit == "us":
        seconds = duration / 1e6
    elif unit == "ms":
        seconds = duration / 1e3
    elif unit == "s":
        seconds = duration
    else:
        raise ValueError(f"Unsupported duration unit: {unit}")

    if seconds >= 60:
        minutes = int(seconds // 60)
        rem_seconds = seconds % 60
        return f"{minutes}m {rem_seconds:.1f}s"
    elif seconds >= 1:
        return f"{seconds:.2f}s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.1f}ms"
    else:
        return f"{seconds * 1e6:.1f}us"
    
def to_relative_seconds(x_values, input_unit="s"):
    """
    Convert raw x-axis values to positive relative seconds starting at 0.
    """
    x_values = x_values.reset_index(drop=True)
    x_rel = x_values - x_values.iloc[0]

    if input_unit == "us":
        x_rel = x_rel / 1e6
    elif input_unit == "ms":
        x_rel = x_rel / 1e3
    elif input_unit == "s":
        pass
    else:
        raise ValueError(f"Unsupported input_unit: {input_unit}")

    return x_rel

def plot_zero_regions(
    ax,
    x_values,
    zero_mask,
    duration_unit="s",
    min_label_duration=0.0,
    label="x=y=z=0",
):
    """
    Draw continuous red horizontal line segments at y=0 for regions where zero_mask is True.
    Also labels each red segment with its duration.
    """
    zero_regions = contiguous_true_regions(zero_mask)

    if not zero_regions:
        return

    x_values = x_values.reset_index(drop=True)
    label_added = False

    # Text offset above y=0, based on current y-axis range.
    y_min, y_max = ax.get_ylim()
    y_offset = 0.01 * (y_max - y_min)
    y_text = 0.0 + y_offset

    for start_i, end_i in zero_regions:
        # Extend region boundaries halfway to neighboring samples.
        if start_i > 0:
            x_start = 0.5 * (x_values.iloc[start_i - 1] + x_values.iloc[start_i])
        else:
            x_start = x_values.iloc[start_i]

        if end_i < len(x_values) - 1:
            x_end = 0.5 * (x_values.iloc[end_i] + x_values.iloc[end_i + 1])
        else:
            x_end = x_values.iloc[end_i]

        duration = abs(x_end - x_start)

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


def plot_csv(
    csv_file,
    output_file=None,
    skip_percent=0.0,
    duration_unit="s",
    min_label_duration=0.0,
):
    """
    Read a CSV file and plot cartesian command twist linear components.

    Red horizontal line segments mark time regions where the actual x, y,
    and z linear command values are all zero. Each segment is labelled with
    its duration.
    """
    df = pd.read_csv(csv_file)

    if skip_percent < 0 or skip_percent >= 100:
        raise ValueError("skip_percent must be in the range [0, 100).")

    skip_rows = int(len(df) * (skip_percent / 100.0))
    if skip_rows > 0:
        df = df.iloc[skip_rows:].reset_index(drop=True)

    if df.empty:
        raise ValueError("No data left after applying skip_percent.")

    linear_prefix = "/cartesian_cmd/twist/twist/linear/"
    linear_cols = [col for col in df.columns if col.startswith(linear_prefix)]

    if not linear_cols:
        raise ValueError(
            f"No linear twist columns found. Expected columns starting with '{linear_prefix}'."
        )

    # Detect paired RQT/PlotJuggler-style columns:
    #   /linear/x_x = x-axis/time values for the x signal
    #   /linear/x_y = actual x velocity values
    pair_regex = re.compile(r"^/cartesian_cmd/twist/twist/linear/([xyz])_([xy])$")
    paired_cols = {}

    for col_name in linear_cols:
        match = pair_regex.match(col_name)
        if match:
            axis, component = match.groups()
            paired_cols.setdefault(axis, {})[component] = col_name

    has_paired_xyz = all(
        axis in paired_cols and "x" in paired_cols[axis] and "y" in paired_cols[axis]
        for axis in ["x", "y", "z"]
    )

    # Determine value columns for zero detection.
    # Important: *_x columns are plotting x-axis values, not command values.
    if has_paired_xyz:
        value_cols = {
            axis: paired_cols[axis]["y"]
            for axis in ["x", "y", "z"]
        }
    else:
        value_cols = {
            axis: f"{linear_prefix}{axis}"
            for axis in ["x", "y", "z"]
            if f"{linear_prefix}{axis}" in df.columns
        }

    missing_axes = [axis for axis in ["x", "y", "z"] if axis not in value_cols]
    if missing_axes:
        raise ValueError(
            f"Could not find actual value columns for axes: {missing_axes}. "
            f"Detected value columns: {value_cols}"
        )

    zero_eps = 1e-12
    zero_mask = (
        df[[value_cols["x"], value_cols["y"], value_cols["z"]]]
        .abs()
        .le(zero_eps)
        .all(axis=1)
    )

    print(f"Rows where x=y=z=0: {zero_mask.sum()} / {len(df)}")

    fig, ax = plt.subplots(figsize=(12, 6))

    if has_paired_xyz:
        for axis in ["x", "y", "z"]:
            x_col = paired_cols[axis]["x"] 
            y_col = paired_cols[axis]["y"] 

            x_plot = to_relative_seconds(df[x_col], input_unit=duration_unit)
            y_plot = -df[y_col]  # preserved from your original code

            ax.plot(
                x_plot, 
                y_plot,
                label=f"Linear {axis}",
                linewidth=2,
            )

        # Draw continuous red regions once, using the x-axis/time of the x signal.
        zero_time = to_relative_seconds(df[paired_cols["x"]["x"]], input_unit=duration_unit)
        plot_zero_regions(
            ax,
            zero_time,
            zero_mask,
            duration_unit=duration_unit,
            min_label_duration=min_label_duration,
        )

        ax.set_xlabel(f"Time [{duration_unit}]", fontsize=12, fontweight="bold")
        ax.set_ylabel("Linear Twist Command [m/s]", fontsize=12, fontweight="bold")
        ax.set_title("Cartesian Twist Commands", fontsize=14, fontweight="bold")

    else:
        time_col = [
            col for col in df.columns
            if "time" in col.lower() or col.lower() == "timestamp"
        ]

        if time_col:
            time = df[time_col[0]] / 1e6
            x_label = "Time [s]"
            duration_unit_for_plot = "s"
        else:
            time = pd.Series(df.index, index=df.index)
            x_label = "Sample Index"
            duration_unit_for_plot = "samples"

        for axis in ["x", "y", "z"]:
            col_name = value_cols[axis]

            ax.plot(
                time,
                df[col_name],
                label=f"Linear {axis}",
                linewidth=2,
            )

        plot_zero_regions(
            ax,
            time,
            zero_mask,
            duration_unit=duration_unit_for_plot,
            min_label_duration=min_label_duration,
        )

        ax.set_xlabel("Time [s]", fontsize=12, fontweight="bold")
        ax.set_ylabel("Linear Velocity (m/s)", fontsize=12, fontweight="bold")
        ax.set_title("Cartesian Command - Linear Velocity", fontsize=14, fontweight="bold")

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot cartesian twist linear command data from CSV."
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("output_file", nargs="?", default=None, help="Optional output plot path")
    parser.add_argument(
        "--skip-percent",
        type=float,
        default=0.0,
        help="Percentage of initial data rows to skip before plotting (0 to <100).",
    )
    parser.add_argument(
        "--duration-unit",
        choices=["s", "ms", "us"],
        default="s",
        help="Unit of the paired *_x time columns. Default: s.",
    )
    parser.add_argument(
        "--min-label-duration",
        type=float,
        default=0.0,
        help=(
            "Only label red regions with duration >= this value, measured in the "
            "same unit as --duration-unit. Useful to avoid clutter."
        ),
    )

    args = parser.parse_args()
    plot_csv(
        args.csv_file,
        args.output_file,
        args.skip_percent,
        args.duration_unit,
        args.min_label_duration,
    )