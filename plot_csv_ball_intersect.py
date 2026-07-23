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
    positive x and positive y. The displayed XY rectangle is configurable in
    centimetres using --table-x-min/--table_x_min, --table-x-max/--table_x_max,
    --table-y-min/--table_y_min, and --table-y-max/--table_y_max. Defaults:
    x in [0, 60] cm, y in [0, 80] cm.

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

Example XY crop in table-frame centimetres:
    python3 plot_csv_ball_intersect_mvp.py \
            --csv-dir /path/to/csv \
            --out-dir /path/to/plots \
            --plot-mode xy \
            --table-x-min 10 --table-x-max 50 \
            --table-y-min 10 --table-y-max 70

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


SCENE_PRIMARY = "#c62828"       # red
SCENE_SECONDARY = "#ef6c00"     # orange
SCENE_LIGHT = "#ff7043"
ROLLOUT_PRIMARY = "#00796b"     # turquoise
ROLLOUT_SECONDARY = "#1565c0"   # blue
EXECUTION_COLOR = "#7b1fa2"     # purple
NEUTRAL_COLOR = "#555555"


TOPICS = {
    "ball": "/scene_localizer/top_cam/ball_3d_table",
    "trajectory": "/scene/ball_trajectory_table",
    "goto_s": "/trajectory_executor/executed_goto_s",
    "goto_target_base": "/trajectory_executor/executed_goto_s_target_base",
    "joints": "/joint_states",
    "act_prediction": "/act/intercept_prediction",
    "sic_selected_s": "/interception_controller/selected_goto_s",
    "ric_selected_s": "/rollout_interception_controller/selected_goto_s",
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
    "prediction_count",
    "prediction_s_min_m",
    "prediction_s_mean_m",
    "prediction_s_max_m",
    "prediction_probability_min",
    "prediction_probability_mean",
    "prediction_probability_max",
    "sic_selected_s_m",
    "sic_selected_time_s",
    "ric_selected_s_m",
    "ric_selected_time_s",
    "ric_probability_at_selection",
    "executed_minus_sic_s_m",
    "executed_minus_ric_s_m",
    "ric_minus_sic_s_m",
    "abs_executed_minus_rollout_m",
    "abs_executed_from_middle_m",
    "start_to_scene_s",
    "start_to_rollout_s",
    "scene_to_rollout_s",
    "episode_length_s",
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
    act_prediction: pd.DataFrame
    sic_selected_s: pd.DataFrame
    ric_selected_s: pd.DataFrame
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
                act_prediction=episode_parts["act_prediction"],
                sic_selected_s=episode_parts["sic_selected_s"],
                ric_selected_s=episode_parts["ric_selected_s"],
            )
        )

    log(f"[INFO] plotting episodes: {[episode.idx for episode in episodes]}")
    return episodes


def resolve_prediction_columns_for_episodes(
    episodes: Sequence[EpisodeData],
    override_s_column: Optional[str],
    override_probability_column: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    non_empty_predictions = [ep.act_prediction for ep in episodes if not ep.act_prediction.empty]
    if not non_empty_predictions:
        return None, None

    union_columns: List[str] = sorted(
        {column for df in non_empty_predictions for column in df.columns}
    )
    probe = pd.DataFrame(columns=union_columns)

    s_col = prediction_s_column(probe, override=override_s_column)
    p_col = prediction_probability_column(probe, override=override_probability_column)

    # If lookup on union dataframe failed due no numeric data, retry on first data frame.
    if s_col is None:
        s_col = prediction_s_column(non_empty_predictions[0], override=override_s_column)
    if p_col is None:
        p_col = prediction_probability_column(non_empty_predictions[0], override=override_probability_column)

    if s_col is None or p_col is None:
        raise RuntimeError(
            "Unable to interpret non-empty ACT prediction CSV. "
            f"Resolved s column={s_col}, probability column={p_col}. "
            f"Available columns: {union_columns}. "
            "Use --prediction-s-column and --prediction-probability-column to override."
        )
    log(f"[INFO] rollout prediction columns: s={s_col}, probability={p_col}")
    return s_col, p_col


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


def prediction_s_column(
    df: pd.DataFrame,
    override: Optional[str] = None,
) -> Optional[str]:
    if df.empty:
        return None
    if override is not None:
        if override not in df.columns:
            raise RuntimeError(
                f"--prediction-s-column={override!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return override

    preferred = ["data_0", "predicted_s", "target_s", "s"]
    col = first_existing_column(df, preferred)
    if col is not None:
        return col

    fuzzy = find_fuzzy_column(
        df,
        required_tokens=("pred",),
        forbidden_tokens=("prob", "conf"),
    )
    if fuzzy is not None:
        return fuzzy

    numeric_candidates = [
        column
        for column in df.columns
        if column not in {"t_abs", "t_rel", "t_episode", "episode_idx", "header_stamp"}
        and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    return None


def prediction_probability_column(
    df: pd.DataFrame,
    override: Optional[str] = None,
) -> Optional[str]:
    if df.empty:
        return None
    if override is not None:
        if override not in df.columns:
            raise RuntimeError(
                f"--prediction-probability-column={override!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return override

    preferred = ["data_1", "probability", "execute_probability", "confidence"]
    col = first_existing_column(df, preferred)
    if col is not None:
        return col

    fuzzy = find_fuzzy_column(
        df,
        required_tokens=("prob",),
    )
    if fuzzy is not None:
        return fuzzy
    fuzzy = find_fuzzy_column(df, required_tokens=("conf",))
    if fuzzy is not None:
        return fuzzy
    return None


def selected_s_event(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    if df.empty:
        return None
    s_col = goto_s_value_column(df)
    if s_col is None:
        return None

    values = numeric(df, s_col)
    times = relative_time(df, float(numeric(df, "t_abs").dropna().min()))
    work = pd.DataFrame({"t": times, "s": values}).replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return None
    first = work.iloc[0]
    return float(first["t"]), float(first["s"])


def selected_s_event_with_episode_time(
    df: pd.DataFrame,
    episode_start_abs: float,
) -> Optional[Tuple[float, float]]:
    if df.empty:
        return None
    s_col = goto_s_value_column(df)
    if s_col is None:
        return None

    times = relative_time(df, episode_start_abs)
    values = numeric(df, s_col)
    work = pd.DataFrame({"t": times, "s": values}).replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return None
    first = work.iloc[0]
    return float(first["t"]), float(first["s"])


def interpolate_1d(
    t: np.ndarray,
    values: np.ndarray,
    query_t: float,
    max_gap: float,
) -> Optional[float]:
    if len(t) == 0:
        return None
    if query_t < float(t[0]) or query_t > float(t[-1]):
        return None

    idx = int(np.searchsorted(t, query_t))
    if idx == 0:
        return float(values[0]) if abs(float(t[0]) - query_t) <= max_gap else None
    if idx >= len(t):
        return float(values[-1]) if abs(float(t[-1]) - query_t) <= max_gap else None

    t0, t1 = float(t[idx - 1]), float(t[idx])
    v0, v1 = float(values[idx - 1]), float(values[idx])
    if t1 <= t0:
        return None
    if (t1 - t0) > max_gap:
        return None
    alpha = (query_t - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


def interpolate_xy_at_time(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    query_t: float,
    max_gap: float,
) -> Optional[Tuple[float, float]]:
    x_val = interpolate_1d(t, x, query_t, max_gap)
    y_val = interpolate_1d(t, y, query_t, max_gap)
    if x_val is None or y_val is None:
        return None
    return x_val, y_val


def map_s_to_table_x_m(
    s_m: np.ndarray,
    s_zero_x_m: float,
    s_sign: int,
) -> np.ndarray:
    return float(s_zero_x_m) + float(s_sign) * np.asarray(s_m, dtype=float)


def extract_prediction_series(
    episode: EpisodeData,
    s_column: Optional[str],
    probability_column: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        episode.act_prediction.empty
        or s_column is None
        or probability_column is None
        or s_column not in episode.act_prediction.columns
        or probability_column not in episode.act_prediction.columns
    ):
        return np.array([]), np.array([]), np.array([])

    samples = pd.DataFrame(
        {
            "t": relative_time(episode.act_prediction, episode.start_abs),
            "s": numeric(episode.act_prediction, s_column),
            "p": numeric(episode.act_prediction, probability_column),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values("t")
    if samples.empty:
        return np.array([]), np.array([]), np.array([])
    return (
        samples["t"].to_numpy(dtype=float),
        samples["s"].to_numpy(dtype=float),
        samples["p"].to_numpy(dtype=float),
    )


def first_executed_s_event(episode: EpisodeData) -> Optional[Tuple[float, float]]:
    if episode.goto_s.empty:
        return None
    s_col = goto_s_value_column(episode.goto_s)
    if s_col is None:
        return None
    work = pd.DataFrame(
        {
            "t": relative_time(episode.goto_s, episode.start_abs),
            "s": numeric(episode.goto_s, s_col),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return None
    first = work.iloc[0]
    return float(first["t"]), float(first["s"])


def executed_target_x_table_m(episode: EpisodeData) -> Optional[float]:
    if episode.goto_target_table is None or episode.goto_target_table.empty:
        return None
    if "x_table" not in episode.goto_target_table.columns:
        return None
    values = numeric(episode.goto_target_table, "x_table").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def select_target_xy_at_event(
    episode: EpisodeData,
    event_t: Optional[float],
) -> Optional[Tuple[float, float]]:
    if episode.goto_target_table is None or episode.goto_target_table.empty or event_t is None:
        return None
    x, y, t = finite_xy_time(
        episode.goto_target_table,
        episode.start_abs,
        "x_table",
        "y_table",
    )
    if len(t) == 0:
        return None
    idx = int(np.argmin(np.abs(t - event_t)))
    return float(x[idx]), float(y[idx])


def derive_rollout_metrics(
    episode: EpisodeData,
    prediction_s_column_name: Optional[str],
    prediction_probability_column_name: Optional[str],
    event_match_max_gap_sec: float,
    s_zero_x_m: float,
    s_sign: int,
) -> Dict[str, Optional[float]]:
    pred_t, pred_s, pred_p = extract_prediction_series(
        episode,
        prediction_s_column_name,
        prediction_probability_column_name,
    )
    sic_event = selected_s_event_with_episode_time(episode.sic_selected_s, episode.start_abs)
    ric_event = selected_s_event_with_episode_time(episode.ric_selected_s, episode.start_abs)
    exec_event = first_executed_s_event(episode)

    sic_s = None if sic_event is None else float(sic_event[1])
    sic_t = None if sic_event is None else float(sic_event[0])
    ric_s = None if ric_event is None else float(ric_event[1])
    ric_t = None if ric_event is None else float(ric_event[0])
    exec_s = None if exec_event is None else float(exec_event[1])

    ric_prob = None
    if ric_t is not None and len(pred_t) > 0:
        ric_prob = interpolate_1d(pred_t, pred_p, ric_t, event_match_max_gap_sec)

    mapped_exec_x = None if exec_s is None else float(map_s_to_table_x_m(np.asarray([exec_s]), s_zero_x_m, s_sign)[0])
    observed_exec_x = executed_target_x_table_m(episode)
    if mapped_exec_x is not None and observed_exec_x is not None:
        if abs(mapped_exec_x - observed_exec_x) > 0.03:
            log(
                f"[WARNING] episode {episode.idx}: s->x mapping and executed target differ by "
                f"{100.0 * abs(mapped_exec_x - observed_exec_x):.1f} cm; "
                "check --s-zero-x-m / --s-sign."
            )

    return {
        "prediction_count": float(len(pred_s)),
        "prediction_s_min_m": float(np.min(pred_s)) if len(pred_s) else None,
        "prediction_s_mean_m": float(np.mean(pred_s)) if len(pred_s) else None,
        "prediction_s_max_m": float(np.max(pred_s)) if len(pred_s) else None,
        "prediction_probability_min": float(np.min(pred_p)) if len(pred_p) else None,
        "prediction_probability_mean": float(np.mean(pred_p)) if len(pred_p) else None,
        "prediction_probability_max": float(np.max(pred_p)) if len(pred_p) else None,
        "sic_selected_s_m": sic_s,
        "sic_selected_time_s": sic_t,
        "ric_selected_s_m": ric_s,
        "ric_selected_time_s": ric_t,
        "ric_probability_at_selection": ric_prob,
        "executed_minus_sic_s_m": (None if exec_s is None or sic_s is None else float(exec_s - sic_s)),
        "executed_minus_ric_s_m": (None if exec_s is None or ric_s is None else float(exec_s - ric_s)),
        "ric_minus_sic_s_m": (None if ric_s is None or sic_s is None else float(ric_s - sic_s)),
    }


def rollout_episode_scalar_metrics(episode: EpisodeData) -> Dict[str, float]:
    """Per-episode first-event scalar metrics for rollout overview statistics."""
    nan = float("nan")
    scene_event = selected_s_event_with_episode_time(episode.sic_selected_s, episode.start_abs)
    rollout_event = selected_s_event_with_episode_time(episode.ric_selected_s, episode.start_abs)
    executed_event = first_executed_s_event(episode)

    scene_t = nan if scene_event is None else float(scene_event[0])
    scene_s = nan if scene_event is None else float(scene_event[1])
    rollout_t = nan if rollout_event is None else float(rollout_event[0])
    rollout_s = nan if rollout_event is None else float(rollout_event[1])
    executed_t = nan if executed_event is None else float(executed_event[0])
    executed_s = nan if executed_event is None else float(executed_event[1])

    if not math.isfinite(scene_t):
        scene_t = nan
    if not math.isfinite(scene_s):
        scene_s = nan
    if not math.isfinite(rollout_t):
        rollout_t = nan
    if not math.isfinite(rollout_s):
        rollout_s = nan
    if not math.isfinite(executed_t):
        executed_t = nan
    if not math.isfinite(executed_s):
        executed_s = nan

    abs_executed_minus_rollout_m = (
        abs(executed_s - rollout_s)
        if math.isfinite(executed_s) and math.isfinite(rollout_s)
        else nan
    )
    abs_executed_from_middle_m = abs(executed_s) if math.isfinite(executed_s) else nan
    start_to_scene_s = scene_t if math.isfinite(scene_t) else nan
    start_to_rollout_s = rollout_t if math.isfinite(rollout_t) else nan
    scene_to_rollout_s = (
        rollout_t - scene_t
        if math.isfinite(rollout_t) and math.isfinite(scene_t)
        else nan
    )
    episode_length_s = (
        float(episode.end_abs - episode.start_abs)
        if math.isfinite(float(episode.end_abs)) and math.isfinite(float(episode.start_abs))
        else nan
    )

    return {
        "abs_executed_minus_rollout_m": abs_executed_minus_rollout_m,
        "abs_executed_from_middle_m": abs_executed_from_middle_m,
        "start_to_scene_s": start_to_scene_s,
        "start_to_rollout_s": start_to_rollout_s,
        "scene_to_rollout_s": scene_to_rollout_s,
        "episode_length_s": episode_length_s,
        "scene_time_s": scene_t,
        "scene_s_m": scene_s,
        "rollout_time_s": rollout_t,
        "rollout_selected_s_m": rollout_s,
        "executed_time_s": executed_t,
        "executed_s_m": executed_s,
    }


def finite_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def rollout_aggregate_statistics(
    episodes: Sequence[EpisodeData],
) -> Dict[str, Dict[str, float]]:
    metrics = [rollout_episode_scalar_metrics(episode) for episode in episodes]

    def values_for(key: str) -> List[float]:
        return [float(row.get(key, float("nan"))) for row in metrics]

    keys = (
        "abs_executed_minus_rollout_m",
        "abs_executed_from_middle_m",
        "start_to_scene_s",
        "start_to_rollout_s",
        "scene_to_rollout_s",
        "episode_length_s",
    )
    return {key: finite_stats(values_for(key)) for key in keys}


def rollout_overview_statistics_text(
    aggregate: Dict[str, Dict[str, float]],
) -> str:
    def fmt_stats_row(
        label: str,
        stats: Dict[str, float],
        scale: float,
        decimals: int,
    ) -> str:
        count = int(stats["count"])
        if count == 0:
            value_text = "n/a / n/a / n/a"
        else:
            min_v = scale * float(stats["min"])
            mean_v = scale * float(stats["mean"])
            max_v = scale * float(stats["max"])
            value_text = f"{min_v:.{decimals}f} / {mean_v:.{decimals}f} / {max_v:.{decimals}f}"
        return f"{label:<28} {value_text:<21} (n={count})"

    lines = [
        "Aggregate statistics (min / mean / max)",
        "",
        fmt_stats_row("|Executed - Rollout| [cm]", aggregate["abs_executed_minus_rollout_m"], 100.0, 1),
        fmt_stats_row("|Executed s| [cm]", aggregate["abs_executed_from_middle_m"], 100.0, 1),
        fmt_stats_row("Start -> Scene [s]", aggregate["start_to_scene_s"], 1.0, 2),
        fmt_stats_row("Start -> Rollout [s]", aggregate["start_to_rollout_s"], 1.0, 2),
        fmt_stats_row("Scene -> Rollout [s]", aggregate["scene_to_rollout_s"], 1.0, 2),
        fmt_stats_row("Episode length [s]", aggregate["episode_length_s"], 1.0, 2),
    ]
    return "\n".join(lines)


def draw_rollout_xy_panel(
    ax,
    episode: EpisodeData,
    table_bounds_cm: Tuple[float, float, float, float],
    prediction_s_column_name: Optional[str],
    prediction_probability_column_name: Optional[str],
    s_zero_x_m: float,
    s_sign: int,
    interception_y_m: Optional[float],
    event_match_max_gap_sec: float,
    prediction_y_offset_cm: float,
    eef_y_offset_cm: float,
    annotate: bool,
) -> Dict[str, object]:
    ric_info_text: Optional[str] = None
    exec_info_text: Optional[str] = None

    inferred_interception_cm = infer_interception_y_cm(episode)
    interception_y_cm = (
        100.0 * float(interception_y_m)
        if interception_y_m is not None
        else inferred_interception_cm
    )
    if interception_y_cm is None:
        table_x_min_cm, table_x_max_cm, table_y_min_cm, table_y_max_cm = table_bounds_cm
        interception_y_cm = 0.5 * (table_y_min_cm + table_y_max_cm)

    draw_table_frame_overlay(ax, table_bounds_cm, interception_y_cm=interception_y_cm, line_color=SCENE_SECONDARY)

    ball_x, ball_y, ball_t = finite_xy_time(episode.ball, episode.start_abs, "point_x", "point_y")
    ball_x_cm = 100.0 * ball_x
    ball_y_cm = 100.0 * ball_y
    if len(ball_x_cm) > 0:
        ax.plot(ball_x_cm, ball_y_cm, color=SCENE_PRIMARY, linewidth=2.0, label="Ball trajectory", zorder=4)

    if episode.tcp_table is not None and not episode.tcp_table.empty:
        tcp_x, tcp_y, _ = finite_xy_time(
            episode.tcp_table,
            episode.start_abs,
            "x_table",
            "y_table",
        )
        if len(tcp_x) > 0:
            eef_y_cm = np.full_like(tcp_x, interception_y_cm + float(eef_y_offset_cm))
            eef_x_cm = 100.0 * tcp_x
            ax.plot(
                eef_x_cm,
                eef_y_cm,
                color="#ec407a",
                linewidth=1.9,
                label="EEF trajectory",
                zorder=4,
            )
            eef_start_x = float(eef_x_cm[0])
            eef_end_x = float(eef_x_cm[-1])
            eef_y_value = float(eef_y_cm[0])
            ax.plot([eef_start_x, eef_start_x], [interception_y_cm, eef_y_value], color="#ec407a", linewidth=1.0, zorder=5)
            ax.plot([eef_end_x, eef_end_x], [interception_y_cm, eef_y_value], color="#ec407a", linewidth=1.0, zorder=5)
            if len(eef_x_cm) > 1:
                arrow_start = max(0, len(eef_x_cm) // 3)
                arrow_end = min(len(eef_x_cm) - 1, arrow_start + max(1, len(eef_x_cm) // 5))
                if arrow_end > arrow_start:
                    ax.annotate(
                        "",
                        xy=(float(eef_x_cm[arrow_end]), eef_y_value),
                        xytext=(float(eef_x_cm[arrow_start]), eef_y_value),
                        arrowprops={"arrowstyle": "-|>", "color": "#ec407a", "lw": 1.1},
                        zorder=6,
                    )

    pred_t, pred_s, pred_p = extract_prediction_series(
        episode,
        prediction_s_column_name,
        prediction_probability_column_name,
    )
    if len(pred_s) > 0:
        pred_x_cm = 100.0 * map_s_to_table_x_m(pred_s, s_zero_x_m, s_sign)
        pred_y_cm = np.full_like(pred_x_cm, interception_y_cm + float(prediction_y_offset_cm))
        ax.plot(
            pred_x_cm,
            pred_y_cm,
            color=ROLLOUT_PRIMARY,
            linewidth=1.5,
            marker="o",
            markersize=4,
            markerfacecolor=ROLLOUT_PRIMARY,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.9,
            label="Rollout prediction evolution",
            zorder=6,
        )
        pred_start_x = float(pred_x_cm[0])
        pred_end_x = float(pred_x_cm[-1])
        pred_y_value = float(pred_y_cm[0])
        ax.plot([pred_start_x, pred_start_x], [interception_y_cm, pred_y_value], color=ROLLOUT_PRIMARY, linewidth=1.0, zorder=5)
        ax.plot([pred_end_x, pred_end_x], [interception_y_cm, pred_y_value], color=ROLLOUT_PRIMARY, linewidth=1.0, zorder=5)
        if len(pred_x_cm) > 1:
            arrow_start = max(0, len(pred_x_cm) // 3)
            arrow_end = min(len(pred_x_cm) - 1, arrow_start + max(1, len(pred_x_cm) // 5))
            if arrow_end > arrow_start:
                ax.annotate(
                    "",
                    xy=(float(pred_x_cm[arrow_end]), pred_y_value),
                    xytext=(float(pred_x_cm[arrow_start]), pred_y_value),
                    arrowprops={"arrowstyle": "-|>", "color": ROLLOUT_PRIMARY, "lw": 1.1},
                    zorder=7,
                )

    sic_event = selected_s_event_with_episode_time(episode.sic_selected_s, episode.start_abs)
    ric_event = selected_s_event_with_episode_time(episode.ric_selected_s, episode.start_abs)
    exec_event = first_executed_s_event(episode)

    if sic_event is not None:
        sic_t, sic_s = sic_event
        sic_x_cm = 100.0 * float(map_s_to_table_x_m(np.asarray([sic_s]), s_zero_x_m, s_sign)[0])
        ax.scatter([sic_x_cm], [interception_y_cm], marker="^", s=88, color=SCENE_SECONDARY, label="Scene selected s", zorder=8)

    ric_probability = None
    if ric_event is not None:
        ric_t, ric_s = ric_event
        ric_x_cm = 100.0 * float(map_s_to_table_x_m(np.asarray([ric_s]), s_zero_x_m, s_sign)[0])
        ax.scatter([ric_x_cm], [interception_y_cm], marker="v", s=88, color=ROLLOUT_SECONDARY, label="Rollout selected s", zorder=9)
        if len(pred_t) > 0:
            ric_probability = interpolate_1d(pred_t, pred_p, ric_t, event_match_max_gap_sec)

        ric_ball_xy = interpolate_xy_at_time(ball_x, ball_y, ball_t, ric_t, event_match_max_gap_sec)
        if ric_ball_xy is None:
            log(
                f"[WARNING] episode {episode.idx}: cannot interpolate ball pose at RIC selection "
                f"t={ric_t:.3f}s with max gap {event_match_max_gap_sec:.3f}s"
            )
        else:
            ric_ball_x_cm, ric_ball_y_cm = 100.0 * ric_ball_xy[0], 100.0 * ric_ball_xy[1]
            ax.scatter(
                [ric_ball_x_cm],
                [ric_ball_y_cm],
                marker="o",
                s=74,
                facecolors="white",
                edgecolors=ROLLOUT_SECONDARY,
                linewidths=1.8,
                label="Ball at Rollout selection",
                zorder=10,
            )
            ax.plot(
                [ric_ball_x_cm, ric_x_cm],
                [ric_ball_y_cm, interception_y_cm],
                color=ROLLOUT_SECONDARY,
                linestyle="--",
                linewidth=1.7,
                label="Ball@Rollout -> Rollout target",
                zorder=7,
            )
        if annotate:
            prob_text = "n/a" if ric_probability is None else f"{ric_probability:.2f}"
            ric_info_text = f"Rollout\nt={ric_t:.2f}s\ns={100 * ric_s:.1f}cm\np={prob_text}"

    if exec_event is not None:
        exec_t, exec_s = exec_event
        exec_x_cm = 100.0 * float(map_s_to_table_x_m(np.asarray([exec_s]), s_zero_x_m, s_sign)[0])
        ax.scatter([exec_x_cm], [interception_y_cm], marker="X", s=96, color=EXECUTION_COLOR, label="Actual executed s", zorder=9)

        if len(ball_t) > 0:
            exec_ball_xy = interpolate_xy_at_time(ball_x, ball_y, ball_t, exec_t, event_match_max_gap_sec)
            if exec_ball_xy is not None:
                exec_ball_x_cm, exec_ball_y_cm = 100.0 * exec_ball_xy[0], 100.0 * exec_ball_xy[1]
                ax.scatter(
                    [exec_ball_x_cm],
                    [exec_ball_y_cm],
                    marker="o",
                    s=70,
                    facecolors="white",
                    edgecolors=EXECUTION_COLOR,
                    linewidths=1.8,
                    label="Actual execution event",
                    zorder=10,
                )
                ax.plot(
                    [exec_ball_x_cm, exec_x_cm],
                    [exec_ball_y_cm, interception_y_cm],
                    color=EXECUTION_COLOR,
                    linestyle=":",
                    linewidth=1.7,
                    zorder=7,
                )
        if annotate:
            exec_info_text = (
                "Actual Execution\n"
                f"t={exec_t:.2f}s\n"
                f"s={100.0 * exec_s:.1f}cm"
            )

    ax.set_xlabel("x_table [cm]")
    ax.set_ylabel("y_table [cm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.axvline(100.0 * 0.30, color="black", linestyle=":", linewidth=0.8, zorder=2)
    table_x_min_cm, table_x_max_cm, table_y_min_cm, table_y_max_cm = table_bounds_cm
    ax.set_xlim(table_x_min_cm, table_x_max_cm)
    ax.set_ylim(table_y_min_cm, table_y_max_cm)

    if annotate:
        base_text_box = {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
        }
        if ric_info_text is not None:
            rollout_box = dict(base_text_box)
            rollout_box["edgecolor"] = ROLLOUT_SECONDARY
            ax.text(
                0.02,
                0.82,
                ric_info_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=ROLLOUT_SECONDARY,
                bbox=rollout_box,
                zorder=20,
            )
        if exec_info_text is not None:
            execution_box = dict(base_text_box)
            execution_box["edgecolor"] = EXECUTION_COLOR
            ax.text(
                0.02,
                0.62,
                exec_info_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=EXECUTION_COLOR,
                bbox=execution_box,
                zorder=20,
            )

    return {
        "ric_probability_at_selection": ric_probability,
    }


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
    table_bounds_cm: Tuple[float, float, float, float],
    interception_y_cm: Optional[float],
    line_color: str = "#d65f8d",
) -> None:
    """Draw a top-down table whose frame origin is its bottom-left corner."""
    table_x_min_cm, table_x_max_cm, table_y_min_cm, table_y_max_cm = table_bounds_cm
    table_width_cm = table_x_max_cm - table_x_min_cm
    table_height_cm = table_y_max_cm - table_y_min_cm

    ax.add_patch(
        Rectangle(
            (table_x_min_cm, table_y_min_cm),
            table_width_cm,
            table_height_cm,
            facecolor="#eef5e9",
            edgecolor="#496a43",
            linewidth=2.0,
            # label=(
            #     "Displayed table area "
            #     f"(x=[{table_x_min_cm:.1f}, {table_x_max_cm:.1f}] cm, "
            #     f"y=[{table_y_min_cm:.1f}, {table_y_max_cm:.1f}] cm)"
            # ),
            zorder=0,
        )
    )

    if interception_y_cm is not None and math.isfinite(interception_y_cm):
        ax.plot(
            [table_x_min_cm, table_x_max_cm],
            [interception_y_cm, interception_y_cm],
            color=line_color,
            linestyle="--",
            linewidth=1.6,
            label=f"Interception line (y={interception_y_cm:.1f} cm)",
            zorder=1,
        )

    origin_in_bounds = (
        table_x_min_cm <= 0.0 <= table_x_max_cm
        and table_y_min_cm <= 0.0 <= table_y_max_cm
    )
    if origin_in_bounds:
        arrow_dx = min(0.22 * table_width_cm, table_x_max_cm)
        arrow_dy = min(0.16 * table_height_cm, table_y_max_cm)
        arrow_style = {"arrowstyle": "-|>", "color": "#303030", "lw": 1.4}
        ax.annotate(
            "",
            xy=(arrow_dx, 0.0),
            xytext=(0.0, 0.0),
            arrowprops=arrow_style,
            zorder=2,
        )
        ax.annotate(
            "",
            xy=(0.0, arrow_dy),
            xytext=(0.0, 0.0),
            arrowprops=arrow_style,
            zorder=2,
        )
        ax.text(arrow_dx, 0.0, " +x", ha="left", va="center", fontsize=8, color="#303030")
        ax.text(0.0, arrow_dy, " +y", ha="center", va="bottom", fontsize=8, color="#303030")
        ax.scatter([0.0], [0.0], marker="+", s=45, color="#303030", zorder=3)
        ax.annotate(
            "origin",
            xy=(0.0, 0.0),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#303030",
        )


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
        f"start → exec_goto_s: {timing_delta_text(0.0, goto_event_t)}",
        (
            "exec_goto_s → ball max-y: "
            f"{timing_delta_text(goto_event_t, farthest_y_t)}"
        ),
        f"ball max-y → eps end: {timing_delta_text(farthest_y_t, episode_duration)}",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="baseline",
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
        0.5,
        "Planar distance\nball max-y → final TCP\n" f"d_xy = {distance_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
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
    table_bounds_cm: Tuple[float, float, float, float],
    annotate_execution: bool,
) -> None:
    """Plot one episode as a top-down table-frame XY path overlay."""
    draw_table_frame_overlay(
        ax,
        table_bounds_cm,
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
            #label="Ball start/end",
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
            #label="Ball farthest-y point",
            zorder=8,
        )

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
    table_x_min_cm, table_x_max_cm, table_y_min_cm, table_y_max_cm = table_bounds_cm
    ax.set_xlim(table_x_min_cm, table_x_max_cm)
    ax.set_ylim(table_y_min_cm, table_y_max_cm)


def plot_xy_overview(
    episodes: Sequence[EpisodeData],
    output: Path,
    columns: int,
    table_bounds_cm: Tuple[float, float, float, float],
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
            table_bounds_cm=table_bounds_cm,
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
            loc="center right",
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
    table_bounds_cm: Tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    plot_episode_xy_axis(
        ax,
        episode,
        table_bounds_cm=table_bounds_cm,
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
        ax.legend(loc="lower left",bbox_to_anchor=(0.02, 0.02),  fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] wrote {output}")


def plot_rollout_overview(
    episodes: Sequence[EpisodeData],
    output: Path,
    columns: int,
    table_bounds_cm: Tuple[float, float, float, float],
    prediction_s_column_name: Optional[str],
    prediction_probability_column_name: Optional[str],
    s_zero_x_m: float,
    s_sign: int,
    interception_y_m: Optional[float],
    event_match_max_gap_sec: float,
    prediction_y_offset_cm: float,
    eef_y_offset_cm: float,
) -> None:
    n = len(episodes)
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.6 * columns, 4.7 * rows),
        squeeze=False,
    )

    for ax, episode in zip(axes.flat, episodes):
        draw_rollout_xy_panel(
            ax,
            episode,
            table_bounds_cm=table_bounds_cm,
            prediction_s_column_name=prediction_s_column_name,
            prediction_probability_column_name=prediction_probability_column_name,
            s_zero_x_m=s_zero_x_m,
            s_sign=s_sign,
            interception_y_m=interception_y_m,
            event_match_max_gap_sec=event_match_max_gap_sec,
            prediction_y_offset_cm=prediction_y_offset_cm,
            eef_y_offset_cm=eef_y_offset_cm,
            annotate=False,
        )
        duration = max(0.0, episode.end_abs - episode.start_abs)
        ax.set_title(f"Episode {episode.idx} ({duration:.2f} s)", fontsize=11, fontweight="bold")

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
            bbox_to_anchor=(0.5, 0.97),
            ncol=min(4, len(labels)),
            fontsize=8,
        )

    aggregate_stats = rollout_aggregate_statistics(episodes)
    stats_text = rollout_overview_statistics_text(aggregate_stats)
    fig.text(
        0.81,
        0.90,
        stats_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 0.94,
        },
        zorder=30,
    )

    log("[INFO] rollout aggregate statistics")
    log(f"       |Executed - Rollout| [cm] min/mean/max (n={int(aggregate_stats['abs_executed_minus_rollout_m']['count'])}): "
        f"{100.0 * aggregate_stats['abs_executed_minus_rollout_m']['min']:.1f} / "
        f"{100.0 * aggregate_stats['abs_executed_minus_rollout_m']['mean']:.1f} / "
        f"{100.0 * aggregate_stats['abs_executed_minus_rollout_m']['max']:.1f}")
    log(f"       |Executed s| [cm] min/mean/max (n={int(aggregate_stats['abs_executed_from_middle_m']['count'])}): "
        f"{100.0 * aggregate_stats['abs_executed_from_middle_m']['min']:.1f} / "
        f"{100.0 * aggregate_stats['abs_executed_from_middle_m']['mean']:.1f} / "
        f"{100.0 * aggregate_stats['abs_executed_from_middle_m']['max']:.1f}")
    log(f"       Start->Scene [s] min/mean/max (n={int(aggregate_stats['start_to_scene_s']['count'])}): "
        f"{aggregate_stats['start_to_scene_s']['min']:.2f} / "
        f"{aggregate_stats['start_to_scene_s']['mean']:.2f} / "
        f"{aggregate_stats['start_to_scene_s']['max']:.2f}")
    log(f"       Start->Rollout [s] min/mean/max (n={int(aggregate_stats['start_to_rollout_s']['count'])}): "
        f"{aggregate_stats['start_to_rollout_s']['min']:.2f} / "
        f"{aggregate_stats['start_to_rollout_s']['mean']:.2f} / "
        f"{aggregate_stats['start_to_rollout_s']['max']:.2f}")
    log(f"       Scene->Rollout [s] min/mean/max (n={int(aggregate_stats['scene_to_rollout_s']['count'])}): "
        f"{aggregate_stats['scene_to_rollout_s']['min']:.2f} / "
        f"{aggregate_stats['scene_to_rollout_s']['mean']:.2f} / "
        f"{aggregate_stats['scene_to_rollout_s']['max']:.2f}")
    log(f"       Episode length [s] min/mean/max (n={int(aggregate_stats['episode_length_s']['count'])}): "
        f"{aggregate_stats['episode_length_s']['min']:.2f} / "
        f"{aggregate_stats['episode_length_s']['mean']:.2f} / "
        f"{aggregate_stats['episode_length_s']['max']:.2f}")

    fig.suptitle(
        "Scene and Rollout interception analysis",
        fontsize=15,
        fontweight="bold",
        y=0.996,
    )
    fig.tight_layout(rect=(0, 0.02, 0.80, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] wrote {output}")


def plot_episode_rollout(
    episode: EpisodeData,
    output: Path,
    table_bounds_cm: Tuple[float, float, float, float],
    prediction_s_column_name: Optional[str],
    prediction_probability_column_name: Optional[str],
    s_zero_x_m: float,
    s_sign: int,
    interception_y_m: Optional[float],
    event_match_max_gap_sec: float,
    main_x_scale: float,
    prediction_y_offset_cm: float,
    eef_y_offset_cm: float,
) -> None:
    try:
        fig = plt.figure(
            figsize=(11.8, 7.8),
            layout="compressed",
        )
    except TypeError:
        fig = plt.figure(
            figsize=(11.8, 7.8),
            constrained_layout=True,
        )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.28, 0.72],
        height_ratios=[1.0, 1.0],
        wspace=0.08,
        hspace=0.16,
    )
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1], sharex=ax_s)

    draw_rollout_xy_panel(
        ax_xy,
        episode,
        table_bounds_cm=table_bounds_cm,
        prediction_s_column_name=prediction_s_column_name,
        prediction_probability_column_name=prediction_probability_column_name,
        s_zero_x_m=s_zero_x_m,
        s_sign=s_sign,
        interception_y_m=interception_y_m,
        event_match_max_gap_sec=event_match_max_gap_sec,
        prediction_y_offset_cm=prediction_y_offset_cm,
        eef_y_offset_cm=eef_y_offset_cm,
        annotate=True,
    )
    ax_xy.set_aspect(1.0 / float(main_x_scale), adjustable="box")

    pred_t, pred_s, pred_p = extract_prediction_series(
        episode,
        prediction_s_column_name,
        prediction_probability_column_name,
    )
    sic_event = selected_s_event_with_episode_time(episode.sic_selected_s, episode.start_abs)
    ric_event = selected_s_event_with_episode_time(episode.ric_selected_s, episode.start_abs)
    exec_event = first_executed_s_event(episode)

    if len(pred_t) > 0:
        ax_s.plot(pred_t, 100.0 * pred_s, color=ROLLOUT_PRIMARY, linewidth=1.8, label="ACT pred. s")
        ax_p.plot(pred_t, pred_p, color=ROLLOUT_SECONDARY, linewidth=1.8, label="ACT exec. prob")

    if sic_event is not None:
        sic_t, sic_s = sic_event
        ax_s.scatter([sic_t], [100.0 * sic_s], marker="^", s=70, color=SCENE_SECONDARY, label="Classic s")
        ax_s.axvline(sic_t, color=SCENE_SECONDARY, linestyle="--", linewidth=1.1)
        ax_p.axvline(sic_t, color=SCENE_SECONDARY, linestyle="--", linewidth=1.1)

    ric_prob = None
    if ric_event is not None:
        ric_t, ric_s = ric_event
        ax_s.scatter([ric_t], [100.0 * ric_s], marker="v", s=70, color=ROLLOUT_SECONDARY, label="ACT sel s")
        ax_s.axvline(ric_t, color=ROLLOUT_SECONDARY, linestyle="--", linewidth=1.1)
        ax_p.axvline(ric_t, color=ROLLOUT_SECONDARY, linestyle="--", linewidth=1.1)
        if len(pred_t) > 0:
            ric_prob = interpolate_1d(pred_t, pred_p, ric_t, event_match_max_gap_sec)
            if ric_prob is not None:
                ax_p.scatter(
                    [ric_t],
                    [ric_prob],
                    marker="o",
                    s=58,
                    facecolors="white",
                    edgecolors=ROLLOUT_SECONDARY,
                    linewidths=1.8,
                    label="ACT exec event",
                    zorder=8,
                )

    if exec_event is not None:
        exec_t, exec_s = exec_event
        ax_s.scatter([exec_t], [100.0 * exec_s], marker="X", s=80, color=EXECUTION_COLOR, label="Actual exec. s")
        ax_s.axvline(exec_t, color=EXECUTION_COLOR, linestyle=":", linewidth=1.2)
        ax_p.axvline(exec_t, color=EXECUTION_COLOR, linestyle=":", linewidth=1.2, label="Actual . exec event")

    ax_s.set_ylabel("s [cm]")
    ax_s.set_title("Interception coordinate s(t)", fontsize=11)
    ax_s.grid(True, alpha=0.3)

    ax_p.set_xlabel("Episode time [s]")
    ax_p.set_ylabel("Probability")
    ax_p.set_title("Execute probability p(t)", fontsize=11)
    ax_p.grid(True, alpha=0.3)
    ax_p.set_ylim(-0.05, 1.05)

    if len(pred_t) > 0:
        ax_s.set_xlim(float(pred_t[0]), float(pred_t[-1]))

    xy_handles, xy_labels = ax_xy.get_legend_handles_labels()
    s_handles, s_labels = ax_s.get_legend_handles_labels()
    p_handles, p_labels = ax_p.get_legend_handles_labels()

    xy_requested = [
        "Interception line",
        "Ball trajectory",
        "EEF trajectory",
        "Rollout prediction evolution",
        "Scene selected s",
        "Rollout selected s",
        "Actual executed s",
        "Actual execution event",
    ]
    xy_legend_handles: List[object] = []
    xy_legend_labels: List[str] = []
    for requested in xy_requested:
        for handle, label in zip(xy_handles, xy_labels):
            matches = label.startswith("Interception line") if requested == "Interception line" else label == requested
            if matches and requested not in xy_legend_labels:
                xy_legend_handles.append(handle)
                xy_legend_labels.append(requested)
                break
    if xy_legend_handles:
        ax_xy.legend(
            xy_legend_handles,
            xy_legend_labels,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            bbox_transform=ax_xy.transAxes,
            ncol=2,
            fontsize=8,
            framealpha=0.9,
            facecolor="white",
            edgecolor="0.75",
        )

    s_requested = [
        "ACT pred. s",
        "Classic s",
        "ACT sel s",
        "Actual exec. s",
    ]
    s_legend_handles: List[object] = []
    s_legend_labels: List[str] = []
    for requested in s_requested:
        for handle, label in zip(s_handles, s_labels):
            if label == requested and requested not in s_legend_labels:
                s_legend_handles.append(handle)
                s_legend_labels.append(requested)
                break
    if s_legend_handles:
        ax_s.legend(s_legend_handles, s_legend_labels, loc="upper left", fontsize=8)

    p_requested = [
        "ACT exec. prob",
        "ACT exec event",
        "Actual . exec event",
    ]
    p_legend_handles: List[object] = []
    p_legend_labels: List[str] = []
    for requested in p_requested:
        for handle, label in zip(p_handles, p_labels):
            if label == requested and requested not in p_legend_labels:
                p_legend_handles.append(handle)
                p_legend_labels.append(requested)
                break
    if p_legend_handles:
        ax_p.legend(p_legend_handles, p_legend_labels, loc="upper left", fontsize=8)

    duration = max(0.0, episode.end_abs - episode.start_abs)
    fig.suptitle(
        f"Episode {episode.idx} rollout analysis ({duration:.2f} s)",
        fontsize=14,
        fontweight="bold",
    )
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


def write_episode_summary(
    episodes: Sequence[EpisodeData],
    output: Path,
    prediction_s_column_name: Optional[str],
    prediction_probability_column_name: Optional[str],
    event_match_max_gap_sec: float,
    s_zero_x_m: float,
    s_sign: int,
) -> None:
    def coerce_float_or_nan(value: object) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    rows: List[Dict[str, object]] = []
    for episode in episodes:
        row: Dict[str, object] = {
            "row_type": "episode",
            "statistic": "",
            "episode_idx": episode.idx,
            "duration_s": episode.end_abs - episode.start_abs,
            "episode_length_s": episode.end_abs - episode.start_abs,
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
            times = (
                relative_time(episode.goto_s, episode.start_abs)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
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
        row.update(
            derive_rollout_metrics(
                episode,
                prediction_s_column_name=prediction_s_column_name,
                prediction_probability_column_name=prediction_probability_column_name,
                event_match_max_gap_sec=event_match_max_gap_sec,
                s_zero_x_m=s_zero_x_m,
                s_sign=s_sign,
            )
        )
        row.update(rollout_episode_scalar_metrics(episode))
        rows.append(row)

    episode_rows = list(rows)
    aggregate_stats = {
        column: finite_stats([
            coerce_float_or_nan(row.get(column, float("nan"))) for row in episode_rows
        ])
        for column in AGGREGATE_SUMMARY_COLUMNS
    }
    for statistic in ("min", "mean", "max"):
        aggregate: Dict[str, object] = {
            "row_type": "summary_aggregate",
            "statistic": statistic,
            "episode_idx": "",
        }
        for column in AGGREGATE_SUMMARY_COLUMNS:
            stats = aggregate_stats[column]
            aggregate[column] = float(stats[statistic]) if stats["count"] > 0 else float("nan")
            aggregate[f"{column}_count"] = int(stats["count"])
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
        choices=("standard", "xy", "both", "rollout", "all"),
        default="standard",
        help=(
            "Plot family to generate: standard x/y-versus-time plots, top-down "
            "table-frame XY overlays, rollout analysis, legacy both (standard+xy), "
            "or all (standard+xy+rollout). Default: standard."
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
            "Physical table extent along table-frame x and y, in metres. "
            "This does not define the XY displayed crop; use --table-x-min/max "
            "and --table-y-min/max for display bounds. Default: 0.60 1.20."
        ),
    )
    parser.add_argument(
        "--table_x_min",
        "--table-x-min",
        type=float,
        default=0.0,
        help=(
            "Displayed XY lower x bound in centimetres. Default: 0.0. "
            "Alias: --table-x-min."
        ),
    )
    parser.add_argument(
        "--table_x_max",
        "--table-x-max",
        type=float,
        default=50.0,
        help=(
            "Displayed XY upper x bound in centimetres. Default: 60.0. "
            "Alias: --table-x-max."
        ),
    )
    parser.add_argument(
        "--table_y_min",
        "--table-y-min",
        type=float,
        default=10.0,
        help=(
            "Displayed XY lower y bound in centimetres. Default: 0.0. "
            "Alias: --table-y-min."
        ),
    )
    parser.add_argument(
        "--table_y_max",
        "--table-y-max",
        type=float,
        default=80.0,
        help=(
            "Displayed XY upper y bound in centimetres. Default: 80.0. "
            "Alias: --table-y-max."
        ),
    )
    parser.add_argument(
        "--prediction-s-column",
        type=str,
        default=None,
        help="Override ACT prediction s column name in /act/intercept_prediction CSV.",
    )
    parser.add_argument(
        "--prediction-probability-column",
        type=str,
        default=None,
        help="Override ACT prediction probability column name in /act/intercept_prediction CSV.",
    )
    parser.add_argument(
        "--event-match-max-gap-sec",
        type=float,
        default=0.10,
        help="Maximum interpolation bracket width for event-time matching. Default: 0.10 s.",
    )
    parser.add_argument(
        "--s-zero-x-m",
        type=float,
        default=0.30,
        help="Table x position mapped from s=0, in metres. Default: 0.30.",
    )
    parser.add_argument(
        "--s-sign",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="Sign for s-to-table-x mapping: x_table = s_zero_x_m + s_sign * s. Default: -1.",
    )
    parser.add_argument(
        "--interception-y-m",
        type=float,
        default=None,
        help="Override interception y level in table frame (metres) for rollout overlays.",
    )
    parser.add_argument(
        "--rollout-main-x-scale",
        type=float,
        default=1.0,
        help=(
            "Visual x-direction scale factor for the main XY panel in per-episode "
            "rollout plots only. Example: 1.5 stretches x by 1.5. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--rollout-prediction-y-offset-cm",
        type=float,
        default=2.0,
        help="Vertical offset above interception line for rollout prediction evolution, in cm. Default: 1.0.",
    )
    parser.add_argument(
        "--rollout-eef-y-offset-cm",
        type=float,
        default=3.0,
        help="Vertical offset above interception line for EEF trajectory in rollout plots, in cm. Default: 2.0.",
    )

    fk = parser.add_argument_group("optional TCP forward kinematics")
    fk.add_argument("--urdf", type=Path, default="/home/jau/dyros/src/plotting/runtime_fr3.urdf", help="FR3 URDF used for Pinocchio FK.")
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
    if args.table_x_min >= args.table_x_max:
        raise ValueError(
            "invalid XY display bounds: --table-x-min/--table_x_min must be "
            "less than --table-x-max/--table_x_max"
        )
    if args.table_y_min >= args.table_y_max:
        raise ValueError(
            "invalid XY display bounds: --table-y-min/--table_y_min must be "
            "less than --table-y-max/--table_y_max"
        )
    if args.urdf is not None and not args.urdf.exists():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")
    if args.event_match_max_gap_sec <= 0.0:
        raise ValueError("--event-match-max-gap-sec must be positive")
    if args.rollout_main_x_scale <= 0.0:
        raise ValueError("--rollout-main-x-scale must be positive")

    mapping_op = "+" if int(args.s_sign) >= 0 else "-"
    log(f"[INFO] s-to-table mapping: x_table = {float(args.s_zero_x_m):.3f} {mapping_op} s")

    table_pose = parse_table_pose(args.table_pose_base)
    episodes = load_episodes(args.csv_dir, requested=args.episodes)
    prediction_s_column_name, prediction_probability_column_name = resolve_prediction_columns_for_episodes(
        episodes,
        override_s_column=args.prediction_s_column,
        override_probability_column=args.prediction_probability_column,
    )
    add_tcp_and_transforms(
        episodes,
        urdf=args.urdf,
        tcp_frame=args.tcp_frame,
        table_pose=table_pose,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    do_standard = args.plot_mode in {"standard", "both", "all"}
    do_xy = args.plot_mode in {"xy", "both", "all"}
    do_rollout = args.plot_mode in {"rollout", "all"}

    if do_standard:
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

    table_bounds_cm = (
        float(args.table_x_min),
        float(args.table_x_max),
        float(args.table_y_min),
        float(args.table_y_max),
    )
    if do_xy:
        plot_xy_overview(
            episodes,
            args.out_dir / "interception_xy_overview.png",
            columns=args.overview_columns,
            table_bounds_cm=table_bounds_cm,
        )

        if not args.no_detail_plots:
            for episode in episodes:
                plot_episode_xy(
                    episode,
                    args.out_dir / f"episode_{episode.idx:02d}_xy.png",
                    table_bounds_cm=table_bounds_cm,
                )

    if do_rollout:
        plot_rollout_overview(
            episodes,
            args.out_dir / "interception_rollout_overview.png",
            columns=args.overview_columns,
            table_bounds_cm=table_bounds_cm,
            prediction_s_column_name=prediction_s_column_name,
            prediction_probability_column_name=prediction_probability_column_name,
            s_zero_x_m=float(args.s_zero_x_m),
            s_sign=int(args.s_sign),
            interception_y_m=args.interception_y_m,
            event_match_max_gap_sec=float(args.event_match_max_gap_sec),
            prediction_y_offset_cm=float(args.rollout_prediction_y_offset_cm),
            eef_y_offset_cm=float(args.rollout_eef_y_offset_cm),
        )

        if not args.no_detail_plots:
            for episode in episodes:
                plot_episode_rollout(
                    episode,
                    args.out_dir / f"episode_{episode.idx:02d}_rollout.png",
                    table_bounds_cm=table_bounds_cm,
                    prediction_s_column_name=prediction_s_column_name,
                    prediction_probability_column_name=prediction_probability_column_name,
                    s_zero_x_m=float(args.s_zero_x_m),
                    s_sign=int(args.s_sign),
                    interception_y_m=args.interception_y_m,
                    event_match_max_gap_sec=float(args.event_match_max_gap_sec),
                    main_x_scale=float(args.rollout_main_x_scale),
                    prediction_y_offset_cm=float(args.rollout_prediction_y_offset_cm),
                    eef_y_offset_cm=float(args.rollout_eef_y_offset_cm),
                )

    write_episode_summary(
        episodes,
        args.out_dir / "episode_summary.csv",
        prediction_s_column_name=prediction_s_column_name,
        prediction_probability_column_name=prediction_probability_column_name,
        event_match_max_gap_sec=float(args.event_match_max_gap_sec),
        s_zero_x_m=float(args.s_zero_x_m),
        s_sign=int(args.s_sign),
    )

    log("")
    log("[INFO] done")
    if args.urdf is None:
        log("[INFO] TCP was not plotted. Add --urdf to enable FK from joint_states.")
    elif table_pose is None:
        log("[INFO] TCP FK was computed in base, but not overlaid with ball table coordinates.")
        log("       Add --table-pose-base TX TY TZ QX QY QZ QW to enable the overlay.")


if __name__ == "__main__":
    main()
