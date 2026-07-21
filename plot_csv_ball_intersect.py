#!/usr/bin/env python3
"""
Plot ball-interception episodes exported by bag_to_csv.py.

Designed for the current MVP bag topics:
  /scene_localizer/top_cam/ball_3d_table
  /scene/ball_trajectory_table
  /trajectory_executor/executed_goto_s
  /trajectory_executor/executed_goto_s_target_base
  /joint_states

Default output:
  1. interception_x_overview.png
       One subplot per episode, showing ball x in table coordinates and the
       time at which executed_goto_s was published.

  2. episode_XX_detail.png
       Detailed x/y plots plus execution information for every episode.

XY table-overlay output:
  Select --plot-mode xy to draw a top-down view of the table frame with the
  ball/TCP XY paths, path directions, and executed interception target
  overlaid. Select --plot-mode both to create the standard and XY plots:
    interception_xy_overview.png
    episode_XX_xy.png
  The table-frame origin is the bottom-left corner; the table extends along
  positive x and positive y. By default, the XY view draws the table only up
  to y=0.80 m even when the physical table is longer.

Optional TCP support:
  Supply --urdf and --tcp-frame to calculate TCP position from /joint_states
  with Pinocchio. Supply --table-pose-base TX TY TZ QX QY QZ QW to transform
  TCP and executed target points from robot-base coordinates into table
  coordinates and overlay them with the ball.

Typical use:
  python3 plot_csv_ball_intersect_mvp.py \
      --csv-dir /home/jau/data/bags/recording_20260716_161544/csv \
      --out-dir /home/jau/data/bags/recording_20260716_161544/plots

With FK and table pose:
  python3 plot_csv_ball_intersect_mvp.py \
      --csv-dir /path/to/csv \
      --out-dir /path/to/plots \
      --urdf /path/to/fr3.urdf \
      --tcp-frame right_fr3_hand_tcp \
      --table-pose-base TX TY TZ QX QY QZ QW

The seven --table-pose-base values describe the table frame pose in robot base:
  [translation xyz, quaternion xyzw] = T_base_table
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


TOPICS = {
    "ball": "/scene_localizer/top_cam/ball_3d_table",
    "trajectory": "/scene/ball_trajectory_table",
    "goto_s": "/trajectory_executor/executed_goto_s",
    "goto_target_base": "/trajectory_executor/executed_goto_s_target_base",
    "joints": "/joint_states",
}

ARM_JOINTS = [f"right_fr3_joint{i}" for i in range(1, 8)]

TIMING_SUMMARY_COLUMNS = (
    "start_to_goto_event_s",
    "goto_event_to_ball_max_y_s",
    "ball_max_y_to_episode_end_s",
)

AGGREGATE_SUMMARY_COLUMNS = (
    *TIMING_SUMMARY_COLUMNS,
    "executed_s_from_center_m",
    "duration_s",
)

# Default T_base_table = [tx, ty, tz, qx, qy, qz, qw]
DEFAULT_TABLE_POSE_BASE = (
    0.8040513709120621,
    0.7348098382481278,
    0.05733261031148451,
    0.0002519007405217038,
    0.005504935586029486,
    0.9999326183938553,
    0.01021718660979085,
)


@dataclass(frozen=True)
class TablePoseBase:
    """Pose T_base_table: table-frame origin/orientation expressed in base."""

    translation: np.ndarray
    rotation: np.ndarray

    def base_to_table(self, points_base: np.ndarray) -> np.ndarray:
        points = np.asarray(points_base, dtype=float)
        return (self.rotation.T @ (points - self.translation).T).T


@dataclass
class EpisodeData:
    idx: int
    start_abs: float
    end_abs: float
    ball: pd.DataFrame
    trajectory: pd.DataFrame
    goto_s: pd.DataFrame
    goto_target_base: pd.DataFrame
    joints: pd.DataFrame
    tcp_base: Optional[pd.DataFrame] = None
    tcp_table: Optional[pd.DataFrame] = None
    goto_target_table: Optional[pd.DataFrame] = None


def log(message: str) -> None:
    print(message, flush=True)


def topic_to_filename(topic: str) -> str:
    return topic.strip("/").replace("/", "__") + ".csv"


def read_optional_csv(csv_dir: Path, topic: str) -> pd.DataFrame:
    path = csv_dir / topic_to_filename(topic)
    if not path.exists():
        log(f"[WARNING] missing optional CSV: {path.name}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        log(f"[WARNING] empty CSV: {path.name}")
        return pd.DataFrame()

    for col in ["t_abs", "t_rel", "t_episode", "episode_idx", "header_stamp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log(f"[INFO] loaded {path.name}: {len(df)} rows")
    return df


def require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} CSV is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def find_fuzzy_column(
    df: pd.DataFrame,
    *,
    required_tokens: Sequence[str],
    suffixes: Sequence[str] = (),
    forbidden_tokens: Sequence[str] = (),
) -> Optional[str]:
    """Find a useful flattened ROS-message field without assuming exact schema."""
    matches: List[str] = []
    for col in df.columns:
        lower = col.lower()
        if not all(token.lower() in lower for token in required_tokens):
            continue
        if any(token.lower() in lower for token in forbidden_tokens):
            continue
        if suffixes and not any(lower.endswith(suffix.lower()) for suffix in suffixes):
            continue
        matches.append(col)

    if not matches:
        return None

    # Prefer shorter, less deeply nested names.
    return sorted(matches, key=lambda value: (len(value), value))[0]


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def quaternion_xyzw_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("table-pose quaternion has zero norm")
    x, y, z, w = q / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def parse_table_pose(values: Optional[Sequence[float]]) -> Optional[TablePoseBase]:
    if values is None:
        tx, ty, tz, qx, qy, qz, qw = DEFAULT_TABLE_POSE_BASE
        return TablePoseBase(
            translation=np.asarray([tx, ty, tz], dtype=float),
            rotation=quaternion_xyzw_to_rotation(qx, qy, qz, qw),
        )
    if len(values) != 7:
        raise ValueError("--table-pose-base requires exactly 7 values: tx ty tz qx qy qz qw")

    tx, ty, tz, qx, qy, qz, qw = map(float, values)
    return TablePoseBase(
        translation=np.asarray([tx, ty, tz], dtype=float),
        rotation=quaternion_xyzw_to_rotation(qx, qy, qz, qw),
    )


def episode_indices(frames: Iterable[pd.DataFrame]) -> List[int]:
    indices: set[int] = set()
    for df in frames:
        if df.empty or "episode_idx" not in df.columns:
            continue
        values = pd.to_numeric(df["episode_idx"], errors="coerce").dropna()
        indices.update(int(value) for value in values)
    return sorted(indices)


def select_episode(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    if df.empty or "episode_idx" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    values = pd.to_numeric(df["episode_idx"], errors="coerce")
    result = df.loc[values == idx].copy()
    if "t_abs" in result.columns:
        result = result.sort_values("t_abs")
    return result.reset_index(drop=True)


def infer_episode_bounds(parts: Sequence[pd.DataFrame]) -> Tuple[float, float]:
    starts: List[float] = []
    ends: List[float] = []

    for df in parts:
        if df.empty or "t_abs" not in df.columns:
            continue
        t_abs = numeric(df, "t_abs").dropna()
        if t_abs.empty:
            continue
        starts.append(float(t_abs.min()))
        ends.append(float(t_abs.max()))

        if "t_episode" in df.columns:
            t_episode = numeric(df, "t_episode")
            valid = pd.DataFrame({"t_abs": numeric(df, "t_abs"), "t_episode": t_episode}).dropna()
            if not valid.empty:
                starts.append(float((valid["t_abs"] - valid["t_episode"]).median()))

    if not starts or not ends:
        raise RuntimeError("could not infer episode time bounds")
    return min(starts), max(ends)


def relative_time(df: pd.DataFrame, episode_start_abs: float) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if "t_abs" in df.columns:
        return numeric(df, "t_abs") - episode_start_abs
    if "t_episode" in df.columns:
        return numeric(df, "t_episode")
    raise RuntimeError("CSV has neither t_abs nor t_episode")


def load_episodes(csv_dir: Path, requested: Optional[Sequence[int]]) -> List[EpisodeData]:
    frames: Dict[str, pd.DataFrame] = {
        key: read_optional_csv(csv_dir, topic) for key, topic in TOPICS.items()
    }

    ball = frames["ball"]
    if ball.empty:
        raise RuntimeError(
            f"Required ball CSV not found: {topic_to_filename(TOPICS['ball'])}. "
            "Export it with bag_to_csv.py first."
        )
    require_columns(ball, ["point_x", "point_y"], "ball")

    available = episode_indices(frames.values())
    if not available:
        raise RuntimeError(
            "No episode_idx columns found. Re-run bag_to_csv.py with --use_episode_windows."
        )

    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"requested episodes not found: {missing}; available={available}")
        selected = list(requested)
    else:
        selected = available

    episodes: List[EpisodeData] = []
    for idx in selected:
        episode_parts = {key: select_episode(df, idx) for key, df in frames.items()}
        start_abs, end_abs = infer_episode_bounds(list(episode_parts.values()))
        episodes.append(
            EpisodeData(
                idx=idx,
                start_abs=start_abs,
                end_abs=end_abs,
                ball=episode_parts["ball"],
                trajectory=episode_parts["trajectory"],
                goto_s=episode_parts["goto_s"],
                goto_target_base=episode_parts["goto_target_base"],
                joints=episode_parts["joints"],
            )
        )

    log(f"[INFO] plotting episodes: {[episode.idx for episode in episodes]}")
    return episodes


def compute_tcp_base_with_pinocchio(
    joints_df: pd.DataFrame,
    urdf: Path,
    tcp_frame: str,
) -> pd.DataFrame:
    if joints_df.empty:
        return pd.DataFrame()

    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError(
            "Pinocchio is required for TCP FK. Install/source it, or omit --urdf."
        ) from exc

    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()

    if not model.existFrame(tcp_frame):
        available_frames = [frame.name for frame in model.frames]
        close = [name for name in available_frames if "tcp" in name.lower() or "hand" in name.lower()]
        raise RuntimeError(
            f"TCP frame {tcp_frame!r} not found in URDF. Candidate frames: {close[:30]}"
        )
    frame_id = model.getFrameId(tcp_frame)

    column_by_joint: Dict[str, str] = {}
    for joint_name in ARM_JOINTS:
        candidates = [f"pos_{joint_name}", joint_name]
        column = first_existing_column(joints_df, candidates)
        if column is None:
            raise RuntimeError(
                f"joint-state CSV is missing {joint_name!r}; expected one of {candidates}"
            )
        if not model.existJointName(joint_name):
            raise RuntimeError(f"joint {joint_name!r} not found in URDF model")
        column_by_joint[joint_name] = column

    output_rows: List[Dict[str, float]] = []
    q_neutral = pin.neutral(model)

    for _, row in joints_df.iterrows():
        q = q_neutral.copy()
        valid = True
        for joint_name, column in column_by_joint.items():
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                valid = False
                break
            joint_id = model.getJointId(joint_name)
            joint_model = model.joints[joint_id]
            if joint_model.nq != 1:
                raise RuntimeError(
                    f"expected scalar joint {joint_name}, but URDF model nq={joint_model.nq}"
                )
            q[joint_model.idx_q] = float(value)

        if not valid:
            continue

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p = np.asarray(data.oMf[frame_id].translation, dtype=float)
        output_rows.append(
            {
                "t_abs": float(row["t_abs"]),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
            }
        )

    return pd.DataFrame(output_rows)


def transform_xyz_dataframe(
    df: pd.DataFrame,
    pose: TablePoseBase,
    xyz_columns: Tuple[str, str, str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x_col, y_col, z_col = xyz_columns
    require_columns(df, [x_col, y_col, z_col], "point")

    points = np.column_stack(
        [numeric(df, x_col), numeric(df, y_col), numeric(df, z_col)]
    )
    valid = np.isfinite(points).all(axis=1)
    transformed = np.full_like(points, np.nan, dtype=float)
    transformed[valid] = pose.base_to_table(points[valid])

    out = df.copy()
    out["x_table"] = transformed[:, 0]
    out["y_table"] = transformed[:, 1]
    out["z_table"] = transformed[:, 2]
    return out


def add_tcp_and_transforms(
    episodes: Sequence[EpisodeData],
    urdf: Optional[Path],
    tcp_frame: str,
    table_pose: Optional[TablePoseBase],
) -> None:
    for episode in episodes:
        if urdf is not None:
            log(f"[INFO] episode {episode.idx}: computing TCP FK")
            episode.tcp_base = compute_tcp_base_with_pinocchio(
                episode.joints, urdf=urdf, tcp_frame=tcp_frame
            )

        if table_pose is not None:
            if episode.tcp_base is not None and not episode.tcp_base.empty:
                episode.tcp_table = transform_xyz_dataframe(
                    episode.tcp_base, table_pose, ("x", "y", "z")
                )
            if not episode.goto_target_base.empty:
                episode.goto_target_table = transform_xyz_dataframe(
                    episode.goto_target_base,
                    table_pose,
                    ("point_x", "point_y", "point_z"),
                )


def trajectory_intercept_x_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    exact = first_existing_column(
        df,
        [
            "intersection_point_x",
            "intercept_point_x",
            "predicted_intersection_x",
            "intersection_x",
            "intercept_x",
        ],
    )
    if exact:
        return exact

    for tokens in [("intersection",), ("intercept",)]:
        found = find_fuzzy_column(
            df,
            required_tokens=tokens,
            suffixes=("_x", "/x", ".x"),
            forbidden_tokens=("velocity", "direction"),
        )
        if found:
            return found
    return None


def goto_s_value_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    exact = first_existing_column(df, ["data", "s", "goto_s"])
    if exact:
        return exact
    numeric_candidates = [
        col
        for col in df.columns
        if col not in {"t_abs", "t_rel", "t_episode", "episode_idx", "header_stamp"}
        and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    return numeric_candidates[0] if len(numeric_candidates) == 1 else None


def draw_execution_markers(ax, episode: EpisodeData, annotate: bool = True) -> None:
    if episode.goto_s.empty:
        return

    s_col = goto_s_value_column(episode.goto_s)
    times = relative_time(episode.goto_s, episode.start_abs)
    values = numeric(episode.goto_s, s_col) if s_col else pd.Series(np.nan, index=times.index)

    for event_idx, (time_value, s_value) in enumerate(zip(times, values)):
        if not math.isfinite(float(time_value)):
            continue
        label = "executed_goto_s event" if event_idx == 0 else None
        ax.axvline(float(time_value), linestyle="--", linewidth=1.4, label=label)

        if annotate:
            text = "executed_goto_s"
            if math.isfinite(float(s_value)):
                text += f"\ns={100.0 * float(s_value):.1f} cm"
            ax.annotate(
                text,
                xy=(float(time_value), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
            )


def plot_ball_axis(
    ax,
    episode: EpisodeData,
    component: str,
    include_tcp: bool,
    include_target: bool,
    show_trajectory_estimate: bool,
) -> None:
    point_col = f"point_{component}"
    ball_t = relative_time(episode.ball, episode.start_abs)
    ball_value_cm = 100.0 * numeric(episode.ball, point_col)
    ax.plot(ball_t, ball_value_cm, linewidth=1.8, label=f"Ball {component}_table")

    if component == "x" and show_trajectory_estimate:
        intersection_col = trajectory_intercept_x_column(episode.trajectory)
        if intersection_col:
            traj_t = relative_time(episode.trajectory, episode.start_abs)
            intercept_cm = 100.0 * numeric(episode.trajectory, intersection_col)
            ax.plot(
                traj_t,
                intercept_cm,
                linestyle=":",
                linewidth=1.5,
                label=f"Predicted intercept x ({intersection_col})",
            )

    if include_tcp and episode.tcp_table is not None and not episode.tcp_table.empty:
        tcp_t = relative_time(episode.tcp_table, episode.start_abs)
        tcp_cm = 100.0 * numeric(episode.tcp_table, f"{component}_table")
        ax.plot(tcp_t, tcp_cm, linewidth=1.6, label=f"TCP {component}_table")

    if include_target and episode.goto_target_table is not None and not episode.goto_target_table.empty:
        target_t = relative_time(episode.goto_target_table, episode.start_abs)
        target_cm = 100.0 * numeric(episode.goto_target_table, f"{component}_table")
        ax.scatter(
            target_t,
            target_cm,
            marker="x",
            s=55,
            linewidths=2,
            label=f"Executed target {component}_table",
            zorder=5,
        )

    draw_execution_markers(ax, episode, annotate=(component == "x"))
    ax.set_ylabel(f"{component} [cm]")
    ax.grid(True, alpha=0.3)


def compact_episode_annotation(episode: EpisodeData) -> str:
    parts: List[str] = []

    s_col = goto_s_value_column(episode.goto_s)
    if s_col and not episode.goto_s.empty:
        values = numeric(episode.goto_s, s_col).dropna()
        if not values.empty:
            parts.append("s=" + ", ".join(f"{100 * value:.1f} cm" for value in values))

    if not episode.goto_target_base.empty and "point_x" in episode.goto_target_base.columns:
        values = numeric(episode.goto_target_base, "point_x").dropna()
        if not values.empty:
            parts.append("target base x=" + ", ".join(f"{100 * value:.1f} cm" for value in values))

    return "\n".join(parts)


def finite_xy_time(
    df: pd.DataFrame,
    episode_start_abs: float,
    x_column: str,
    y_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite x/y/time samples, ordered by episode time."""
    if df.empty:
        return np.array([]), np.array([]), np.array([])

    require_columns(df, [x_column, y_column], "XY trajectory")
    samples = pd.DataFrame(
        {
            "x": numeric(df, x_column),
            "y": numeric(df, y_column),
            "t": relative_time(df, episode_start_abs),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    samples = samples.sort_values("t")
    return (
        samples["x"].to_numpy(dtype=float),
        samples["y"].to_numpy(dtype=float),
        samples["t"].to_numpy(dtype=float),
    )


def draw_table_frame_overlay(
    ax,
    table_size_m: Tuple[float, float],
    interception_y_cm: Optional[float],
) -> Tuple[float, float]:
    """Draw a top-down table whose frame origin is its bottom-left corner."""
    table_x_cm = 100.0 * float(table_size_m[0])
    table_y_cm = 100.0 * float(table_size_m[1])

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            table_x_cm,
            table_y_cm,
            facecolor="#eef5e9",
            edgecolor="#496a43",
            linewidth=2.0,
            label=f"Displayed table area (y≤{table_y_cm:.0f} cm)",
            zorder=0,
        )
    )

    if interception_y_cm is not None and math.isfinite(interception_y_cm):
        ax.plot(
            [0.0, table_x_cm],
            [interception_y_cm, interception_y_cm],
            color="#d65f8d",
            linestyle="--",
            linewidth=1.6,
            label=f"Interception line (y={interception_y_cm:.1f} cm)",
            zorder=1,
        )

    arrow_x = 0.22 * table_x_cm
    arrow_y = 0.16 * table_y_cm
    arrow_style = {"arrowstyle": "-|>", "color": "#303030", "lw": 1.4}
    ax.annotate("", xy=(arrow_x, 0.0), xytext=(0.0, 0.0), arrowprops=arrow_style, zorder=2)
    ax.annotate("", xy=(0.0, arrow_y), xytext=(0.0, 0.0), arrowprops=arrow_style, zorder=2)
    ax.text(arrow_x, 0.0, " +x", ha="left", va="center", fontsize=8, color="#303030")
    ax.text(0.0, arrow_y, " +y", ha="center", va="bottom", fontsize=8, color="#303030")
    ax.scatter([0.0], [0.0], marker="+", s=45, color="#303030", zorder=3)
    ax.annotate(
        "origin",
        xy=(0.0, 0.0),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        color="#303030",
    )
    return table_x_cm, table_y_cm


def infer_interception_y_cm(episode: EpisodeData) -> Optional[float]:
    """Infer the robot's horizontal interception line in table coordinates."""
    candidates = [
        (episode.goto_target_table, "y_table"),
        (episode.tcp_table, "y_table"),
    ]
    for df, column in candidates:
        if df is None or df.empty or column not in df.columns:
            continue
        values = numeric(df, column).replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            return 100.0 * float(values.median())
    return None


def first_goto_event_time_s(episode: EpisodeData) -> Optional[float]:
    """Return the first executed_goto_s topic timestamp relative to episode start."""
    if episode.goto_s.empty:
        return None
    event_times = pd.to_numeric(
        relative_time(episode.goto_s, episode.start_abs),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if event_times.empty:
        return None
    return float(event_times.iloc[0])


def timing_delta_text(start_s: Optional[float], end_s: Optional[float]) -> str:
    if start_s is None or end_s is None:
        return "n/a"
    delta = end_s - start_s
    if delta < 0.0:
        return f"{delta:.3f} s (reversed)"
    return f"{delta:.3f} s"


def episode_timing_metrics(episode: EpisodeData) -> Dict[str, Optional[float]]:
    goto_event_t = first_goto_event_time_s(episode)
    _, ball_y, ball_t = finite_xy_time(
        episode.ball,
        episode.start_abs,
        "point_x",
        "point_y",
    )
    farthest_y_t = float(ball_t[int(np.argmax(ball_y))]) if ball_y.size else None
    episode_duration = max(0.0, episode.end_abs - episode.start_abs)

    def delta(start_s: Optional[float], end_s: Optional[float]) -> Optional[float]:
        if start_s is None or end_s is None:
            return None
        return float(end_s - start_s)

    return {
        "start_to_goto_event_s": delta(0.0, goto_event_t),
        "goto_event_to_ball_max_y_s": delta(goto_event_t, farthest_y_t),
        "ball_max_y_to_episode_end_s": delta(farthest_y_t, episode_duration),
    }


def draw_xy_timing_overlay(
    ax,
    episode: EpisodeData,
    goto_event_t: Optional[float],
    farthest_y_t: Optional[float],
    compact: bool,
) -> None:
    episode_duration = max(0.0, episode.end_abs - episode.start_abs)
    lines = [
        "Timing from topic stamps",
        f"start → executed_goto_s: {timing_delta_text(0.0, goto_event_t)}",
        (
            "executed_goto_s → ball max-y: "
            f"{timing_delta_text(goto_event_t, farthest_y_t)}"
        ),
        f"ball max-y → episode end: {timing_delta_text(farthest_y_t, episode_duration)}",
    ]
    ax.text(
        0.98,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8 if compact else 8.5,
        linespacing=1.3,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.88,
        },
        zorder=20,
    )


def draw_xy_distance_overlay(
    ax,
    distance_cm: Optional[float],
    compact: bool,
) -> None:
    distance_text = "n/a" if distance_cm is None else f"{distance_cm:.2f} cm"
    ax.text(
        0.02,
        0.02,
        "Planar distance\nball max-y → final TCP\n" f"d_xy = {distance_text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8 if compact else 8.5,
        linespacing=1.3,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.88,
        },
        zorder=20,
    )


def draw_path_direction_arrows(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    count: int = 3,
) -> None:
    """Add a few arrowheads without covering a dense measured path."""
    if len(x) < 2:
        return

    arrow_count = min(count, len(x) - 1)
    centers = np.unique(
        np.linspace(1, len(x) - 1, arrow_count + 2, dtype=int)[1:-1]
    )
    half_span = max(1, len(x) // max(6 * arrow_count, 1))
    for center in centers:
        start = max(0, int(center) - half_span)
        end = min(len(x) - 1, int(center) + half_span)
        if start == end or math.hypot(float(x[end] - x[start]), float(y[end] - y[start])) < 1e-9:
            continue
        ax.annotate(
            "",
            xy=(float(x[end]), float(y[end])),
            xytext=(float(x[start]), float(y[start])),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.3},
            zorder=5,
        )


def plot_episode_xy_axis(
    ax,
    episode: EpisodeData,
    table_size_m: Tuple[float, float],
    annotate_execution: bool,
) -> None:
    """Plot one episode as a top-down table-frame XY path overlay."""
    table_x_cm, table_y_cm = draw_table_frame_overlay(
        ax,
        table_size_m,
        interception_y_cm=infer_interception_y_cm(episode),
    )

    ball_x, ball_y, ball_t = finite_xy_time(
        episode.ball,
        episode.start_abs,
        "point_x",
        "point_y",
    )
    ball_x_cm = 100.0 * ball_x
    ball_y_cm = 100.0 * ball_y
    ball_color = "#1976d2"
    ax.plot(
        ball_x_cm,
        ball_y_cm,
        color=ball_color,
        linewidth=2.2,
        label="Ball trajectory",
        zorder=4,
    )
    farthest_y_idx: Optional[int] = None
    farthest_y_t: Optional[float] = None
    if ball_x_cm.size:
        ax.scatter(
            [ball_x_cm[0]],
            [ball_y_cm[0]],
            marker="o",
            s=38,
            facecolors="white",
            edgecolors=ball_color,
            linewidths=1.6,
            label="Ball start/end",
            zorder=6,
        )
        ax.scatter(
            [ball_x_cm[-1]],
            [ball_y_cm[-1]],
            marker="s",
            s=34,
            color=ball_color,
            zorder=6,
        )
        farthest_y_idx = int(np.argmax(ball_y_cm))
        farthest_y_t = float(ball_t[farthest_y_idx])
        ax.scatter(
            [ball_x_cm[farthest_y_idx]],
            [ball_y_cm[farthest_y_idx]],
            marker="D",
            s=58,
            facecolors="#ffeb3b",
            edgecolors="#5d4037",
            linewidths=1.4,
            label="Ball farthest-y point",
            zorder=8,
        )

    all_x = [ball_x_cm]
    all_y = [ball_y_cm]
    max_y_to_final_tcp_distance_cm: Optional[float] = None

    if episode.tcp_table is not None and not episode.tcp_table.empty:
        tcp_x, tcp_y, _ = finite_xy_time(
            episode.tcp_table,
            episode.start_abs,
            "x_table",
            "y_table",
        )
        tcp_x_cm = 100.0 * tcp_x
        tcp_y_cm = 100.0 * tcp_y
        tcp_color = "#ef6c00"
        ax.plot(
            tcp_x_cm,
            tcp_y_cm,
            color=tcp_color,
            linewidth=2.0,
            label="TCP trajectory",
            zorder=4,
        )
        draw_path_direction_arrows(ax, tcp_x_cm, tcp_y_cm, tcp_color)
        if farthest_y_idx is not None and tcp_x_cm.size:
            max_y_to_final_tcp_distance_cm = math.hypot(
                float(ball_x_cm[farthest_y_idx] - tcp_x_cm[-1]),
                float(ball_y_cm[farthest_y_idx] - tcp_y_cm[-1]),
            )
        all_x.append(tcp_x_cm)
        all_y.append(tcp_y_cm)

    target_x_cm = np.array([])
    target_y_cm = np.array([])
    target_t = np.array([])
    if episode.goto_target_table is not None and not episode.goto_target_table.empty:
        target_x, target_y, target_t = finite_xy_time(
            episode.goto_target_table,
            episode.start_abs,
            "x_table",
            "y_table",
        )
        target_x_cm = 100.0 * target_x
        target_y_cm = 100.0 * target_y
        ax.scatter(
            target_x_cm,
            target_y_cm,
            marker="x",
            s=75,
            linewidths=2.4,
            color="#7b1fa2",
            label="Executed target",
            zorder=7,
        )
        all_x.append(target_x_cm)
        all_y.append(target_y_cm)

    # The topic timestamp does not itself identify goal-acceptance versus motion
    # completion semantics. Use the neutral term "event" and mark the nearest
    # measured ball position without inventing an interpolated spatial sample.
    goto_event_t = first_goto_event_time_s(episode)
    if goto_event_t is not None and ball_t.size:
        event_ball_idx = int(np.argmin(np.abs(ball_t - goto_event_t)))
        event_ball_x = float(ball_x_cm[event_ball_idx])
        event_ball_y = float(ball_y_cm[event_ball_idx])
        ax.scatter(
            [event_ball_x],
            [event_ball_y],
            marker="o",
            s=72,
            facecolors="white",
            edgecolors="#c62828",
            linewidths=2.0,
            label="Ball at executed_goto_s event",
            zorder=9,
        )

        if target_x_cm.size:
            target_idx = (
                int(np.argmin(np.abs(target_t - goto_event_t)))
                if target_t.size
                else 0
            )
            ax.plot(
                [event_ball_x, float(target_x_cm[target_idx])],
                [event_ball_y, float(target_y_cm[target_idx])],
                color="#c62828",
                linestyle="--",
                linewidth=1.8,
                label="Ball-at-event → executed target",
                zorder=7,
            )

        if annotate_execution:
            ax.annotate(
                f"executed_goto_s event\nt={goto_event_t:.2f} s",
                xy=(event_ball_x, event_ball_y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
            )

    draw_xy_timing_overlay(
        ax,
        episode,
        goto_event_t=goto_event_t,
        farthest_y_t=farthest_y_t,
        compact=not annotate_execution,
    )
    draw_xy_distance_overlay(
        ax,
        distance_cm=max_y_to_final_tcp_distance_cm,
        compact=not annotate_execution,
    )

    ax.set_xlabel("x_table [cm]")
    ax.set_ylabel("y_table [cm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    finite_x = np.concatenate([values[np.isfinite(values)] for values in all_x])
    finite_y = np.concatenate([values[np.isfinite(values)] for values in all_y])
    x_min = min(0.0, float(finite_x.min())) if finite_x.size else 0.0
    x_max = max(table_x_cm, float(finite_x.max())) if finite_x.size else table_x_cm
    y_min = min(0.0, float(finite_y.min())) if finite_y.size else 0.0
    y_max = max(table_y_cm, float(finite_y.max())) if finite_y.size else table_y_cm
    margin = 0.04 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)


def plot_xy_overview(
    episodes: Sequence[EpisodeData],
    output: Path,
    columns: int,
    table_size_m: Tuple[float, float],
) -> None:
    n = len(episodes)
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.5 * rows),
        squeeze=False,
    )

    for ax, episode in zip(axes.flat, episodes):
        plot_episode_xy_axis(
            ax,
            episode,
            table_size_m=table_size_m,
            annotate_execution=False,
        )
        duration = max(0.0, episode.end_abs - episode.start_abs)
        ax.set_title(
            f"Episode {episode.idx}  ({duration:.2f} s)",
            fontsize=11,
            fontweight="bold",
        )

    for ax in axes.flat[n:]:
        ax.axis("off")

    handles: List[object] = []
    labels: List[str] = []
    for ax in axes.flat[:n]:
        subplot_handles, subplot_labels = ax.get_legend_handles_labels()
        for handle, label in zip(subplot_handles, subplot_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(4, len(labels)),
            fontsize=9,
        )

    fig.suptitle(
        "Ball interception MVP — top-down table-frame XY trajectories",
        fontsize=15,
        fontweight="bold",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] wrote {output}")


def plot_episode_xy(
    episode: EpisodeData,
    output: Path,
    table_size_m: Tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    plot_episode_xy_axis(
        ax,
        episode,
        table_size_m=table_size_m,
        annotate_execution=True,
    )
    ax.set_title(
        f"Episode {episode.idx}: top-down table-frame XY trajectories",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] wrote {output}")


def plot_x_overview(
    episodes: Sequence[EpisodeData],
    output: Path,
    columns: int,
    show_trajectory_estimate: bool,
) -> None:
    n = len(episodes)
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.1 * columns, 3.3 * rows),
        squeeze=False,
        sharey=True,
    )

    for ax, episode in zip(axes.flat, episodes):
        plot_ball_axis(
            ax,
            episode,
            component="x",
            include_tcp=True,
            include_target=True,
            show_trajectory_estimate=show_trajectory_estimate,
        )
        duration = max(0.0, episode.end_abs - episode.start_abs)
        ax.set_xlim(left=0.0, right=max(duration, 0.1))
        ax.set_title(f"Episode {episode.idx}  ({duration:.2f} s)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode time [s]")

        annotation = compact_episode_annotation(episode)
        if annotation:
            ax.text(
                0.02,
                0.03,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round", "alpha": 0.75},
            )

    for ax in axes.flat[n:]:
        ax.axis("off")

    handles: List[object] = []
    labels: List[str] = []
    for ax in axes.flat[:n]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.966),
            ncol=min(4, len(labels)),
            fontsize=9,
        )

    fig.suptitle(
        "Ball interception MVP — x trajectory and executed GOTO_S timing",
        fontsize=15,
        fontweight="bold",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] wrote {output}")


def plot_episode_detail(
    episode: EpisodeData,
    output: Path,
    show_trajectory_estimate: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    plot_ball_axis(
        axes[0],
        episode,
        component="x",
        include_tcp=True,
        include_target=True,
        show_trajectory_estimate=show_trajectory_estimate,
    )
    axes[0].set_title(
        f"Episode {episode.idx}: lateral interception coordinate",
        fontsize=13,
        fontweight="bold",
    )

    plot_ball_axis(
        axes[1],
        episode,
        component="y",
        include_tcp=True,
        include_target=True,
        show_trajectory_estimate=False,
    )
    axes[1].set_title("Ball approach coordinate", fontsize=11)

    ax = axes[2]
    draw_execution_markers(ax, episode, annotate=False)

    s_col = goto_s_value_column(episode.goto_s)
    if s_col and not episode.goto_s.empty:
        event_t = relative_time(episode.goto_s, episode.start_abs)
        s_cm = 100.0 * numeric(episode.goto_s, s_col)
        ax.scatter(event_t, s_cm, marker="o", s=55, label="Executed GOTO_S s")
        for time_value, s_value in zip(event_t, s_cm):
            if math.isfinite(float(time_value)) and math.isfinite(float(s_value)):
                ax.annotate(
                    f"{s_value:.1f} cm",
                    (float(time_value), float(s_value)),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    if not episode.goto_target_base.empty and "point_x" in episode.goto_target_base.columns:
        target_t = relative_time(episode.goto_target_base, episode.start_abs)
        target_base_x_cm = 100.0 * numeric(episode.goto_target_base, "point_x")
        ax.scatter(
            target_t,
            target_base_x_cm,
            marker="x",
            s=65,
            linewidths=2,
            label="Executed target x_base",
        )

    ax.set_title("Execution values (different coordinate origins unless target is transformed)", fontsize=11)
    ax.set_xlabel("Episode time [s]")
    ax.set_ylabel("Value [cm]")
    ax.grid(True, alpha=0.3)

    duration = max(0.0, episode.end_abs - episode.start_abs)
    axes[-1].set_xlim(left=0.0, right=max(duration, 0.1))

    for subplot in axes:
        handles, labels = subplot.get_legend_handles_labels()
        if handles:
            subplot.legend(loc="best", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] wrote {output}")


def write_episode_summary(episodes: Sequence[EpisodeData], output: Path) -> None:
    rows: List[Dict[str, object]] = []
    for episode in episodes:
        row: Dict[str, object] = {
            "row_type": "episode",
            "statistic": "",
            "episode_idx": episode.idx,
            "duration_s": episode.end_abs - episode.start_abs,
            "ball_samples": len(episode.ball),
            "trajectory_samples": len(episode.trajectory),
            "goto_s_count": len(episode.goto_s),
            "target_count": len(episode.goto_target_base),
        }

        s_col = goto_s_value_column(episode.goto_s)
        if s_col and not episode.goto_s.empty:
            values = (
                numeric(episode.goto_s, s_col)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            times = relative_time(episode.goto_s, episode.start_abs).dropna()
            row["executed_s_m"] = ";".join(f"{value:.6f}" for value in values)
            row["executed_s_time_s"] = ";".join(f"{value:.6f}" for value in times)
            row["executed_s_from_center_m"] = (
                float(values.iloc[0]) if not values.empty else None
            )
        else:
            row["executed_s_m"] = ""
            row["executed_s_time_s"] = ""
            row["executed_s_from_center_m"] = None

        row.update(episode_timing_metrics(episode))
        rows.append(row)

    episode_rows = list(rows)
    reducers = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
    }
    for statistic, reducer in reducers.items():
        aggregate: Dict[str, object] = {
            "row_type": "summary_aggregate",
            "statistic": statistic,
            "episode_idx": "",
        }
        for column in AGGREGATE_SUMMARY_COLUMNS:
            values = [
                float(row[column])
                for row in episode_rows
                if row.get(column) is not None
                and math.isfinite(float(row[column]))
                and (
                    column not in TIMING_SUMMARY_COLUMNS
                    or float(row[column]) >= 0.0
                )
            ]
            aggregate[column] = float(reducer(values)) if values else None
        rows.append(aggregate)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    log(f"[INFO] wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create episode-aware ball-interception plots from bag_to_csv.py output."
    )
    parser.add_argument("--csv-dir", type=Path, required=True, help="Directory containing per-topic CSV files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for PNG plots and summary CSV.")
    parser.add_argument(
        "--plot-mode",
        choices=("standard", "xy", "both"),
        default="standard",
        help=(
            "Plot family to generate: standard x/y-versus-time plots, top-down "
            "table-frame XY overlays, or both. Default: standard."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional episode indices. Default: all committed episodes.",
    )
    parser.add_argument(
        "--overview-columns",
        type=int,
        default=3,
        help="Number of subplot columns in overview plots. Default: 3.",
    )
    parser.add_argument(
        "--no-detail-plots",
        action="store_true",
        help="Only create the selected overview plot(s) and summary CSV.",
    )
    parser.add_argument(
        "--no-trajectory-estimate",
        action="store_true",
        help="Do not attempt to plot predicted intercept x from BallTrajectory CSV.",
    )
    parser.add_argument(
        "--table-size",
        type=float,
        nargs=2,
        metavar=("X_METERS", "Y_METERS"),
        default=(0.60, 1.20),
        help=(
            "Physical table extent along table-frame x and y, in metres, used by "
            "the XY overlay. The frame origin is the bottom-left table corner. "
            "Default: 0.60 1.20."
        ),
    )
    parser.add_argument(
        "--xy-table-y-max",
        type=float,
        default=0.80,
        help=(
            "Upper table-frame y coordinate drawn by the XY overlay, in metres. "
            "Trajectory points beyond it remain visible outside the table patch. "
            "Default: 0.80."
        ),
    )

    fk = parser.add_argument_group("optional TCP forward kinematics")
    fk.add_argument("--urdf", type=Path, default=None, help="FR3 URDF used for Pinocchio FK.")
    fk.add_argument(
        "--tcp-frame",
        default="right_fr3_hand_tcp",
        help="URDF frame whose position is plotted. Default: right_fr3_hand_tcp.",
    )
    fk.add_argument(
        "--table-pose-base",
        type=float,
        nargs=7,
        metavar=("TX", "TY", "TZ", "QX", "QY", "QZ", "QW"),
        default=None,
        help=(
            "Pose T_base_table. If omitted, a built-in calibrated default pose is used. "
            "Enables TCP/target conversion into table coordinates."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {args.csv_dir}")
    if args.overview_columns <= 0:
        raise ValueError("--overview-columns must be positive")
    if any(value <= 0.0 for value in args.table_size):
        raise ValueError("both --table-size values must be positive")
    if args.xy_table_y_max <= 0.0:
        raise ValueError("--xy-table-y-max must be positive")
    if args.urdf is not None and not args.urdf.exists():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")

    table_pose = parse_table_pose(args.table_pose_base)
    episodes = load_episodes(args.csv_dir, requested=args.episodes)
    add_tcp_and_transforms(
        episodes,
        urdf=args.urdf,
        tcp_frame=args.tcp_frame,
        table_pose=table_pose,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_mode in {"standard", "both"}:
        plot_x_overview(
            episodes,
            args.out_dir / "interception_x_overview.png",
            columns=args.overview_columns,
            show_trajectory_estimate=not args.no_trajectory_estimate,
        )

        if not args.no_detail_plots:
            for episode in episodes:
                plot_episode_detail(
                    episode,
                    args.out_dir / f"episode_{episode.idx:02d}_detail.png",
                    show_trajectory_estimate=not args.no_trajectory_estimate,
                )

    if args.plot_mode in {"xy", "both"}:
        table_size_m = (
            float(args.table_size[0]),
            min(float(args.table_size[1]), float(args.xy_table_y_max)),
        )
        plot_xy_overview(
            episodes,
            args.out_dir / "interception_xy_overview.png",
            columns=args.overview_columns,
            table_size_m=table_size_m,
        )

        if not args.no_detail_plots:
            for episode in episodes:
                plot_episode_xy(
                    episode,
                    args.out_dir / f"episode_{episode.idx:02d}_xy.png",
                    table_size_m=table_size_m,
                )

    write_episode_summary(episodes, args.out_dir / "episode_summary.csv")

    log("")
    log("[INFO] done")
    if args.urdf is None:
        log("[INFO] TCP was not plotted. Add --urdf to enable FK from joint_states.")
    elif table_pose is None:
        log("[INFO] TCP FK was computed in base, but not overlaid with ball table coordinates.")
        log("       Add --table-pose-base TX TY TZ QX QY QZ QW to enable the overlay.")


if __name__ == "__main__":
    main()
