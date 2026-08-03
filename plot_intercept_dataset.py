#!/usr/bin/env python3
"""Plot combined intercept dataset CSVs from one or more extracted sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from intercept_dataset_common import (
    assign_global_episode_id,
    classify_lateral_intercept,
    detect_max_y_intercept,
    detect_middle_line_crossing,
    detect_motion_onset_time,
    directional_count_bias,
    interpolate_approach_on_y_grid,
    interpolate_position_at_time,
    pick_representative_classic_row,
    require_columns,
    resolve_dataset_csv_dir,
    sample_descriptive_statistics,
    select_monotonic_approach_points,
    summarize_episode_grid,
    summarize_classic_counts,
    table_x_to_s,
    validate_no_duplicate_source_ids,
)


ALL_EVENT_NAMES = [
    "goto-selected",
    "motion-onset",
    "episode-start",
    "episode-end",
    "vision-intercept",
]

TRAJECTORY_DISTRIBUTION_GRID_POINTS = 200
LR_COLORS = {"L": "tab:blue", "R": "tab:orange", "deadband": "0.45"}


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class SourceDataset:
    source_id: str
    source_bag: str
    dataset_dir: Path
    manifest: pd.DataFrame
    ball: pd.DataFrame
    classic: pd.DataFrame


@dataclass
class CombinedDataset:
    manifest: pd.DataFrame
    ball: pd.DataFrame
    classic: pd.DataFrame
    source_summaries: List[Dict[str, Any]]
    dataset_dirs: List[Path]
    source_bags: Dict[str, str]


@dataclass
class TrajectoryDistributionAnalysis:
    episodes: pd.DataFrame
    summary: pd.DataFrame
    mean_grid: pd.DataFrame
    raw_segments: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]


def parse_bool_col(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "yes", "y"])


def _empty_csv(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([], columns=list(columns))


def load_one_source(input_path: str) -> SourceDataset:
    dataset_dir = resolve_dataset_csv_dir(input_path)

    manifest_path = dataset_dir / "manifest.csv"
    ball_path = dataset_dir / "ball_position_table.csv"
    classic_path = dataset_dir / "classic_selected_goto_s.csv"

    manifest = pd.read_csv(manifest_path)
    require_columns(
        manifest,
        [
            "source_id",
            "episode_id",
            "start_timestamp",
            "end_timestamp",
            "duration_sec",
            "ball_sample_count",
            "classic_goto_s_count",
            "ball_frame_id",
            "valid",
            "invalid_reason",
        ],
        "manifest.csv",
    )

    if ball_path.exists():
        ball = pd.read_csv(ball_path)
    else:
        ball = _empty_csv(
            [
                "source_id",
                "episode_id",
                "timestamp",
                "t_rel_sec",
                "header_timestamp",
                "x",
                "y",
                "z",
                "frame_id",
            ]
        )
    require_columns(
        ball,
        [
            "source_id",
            "episode_id",
            "timestamp",
            "t_rel_sec",
            "header_timestamp",
            "x",
            "y",
            "z",
            "frame_id",
        ],
        "ball_position_table.csv",
    )

    if classic_path.exists():
        classic = pd.read_csv(classic_path)
    else:
        classic = _empty_csv(
            ["source_id", "episode_id", "timestamp", "t_rel_sec", "s"]
        )
    require_columns(
        classic,
        ["source_id", "episode_id", "timestamp", "t_rel_sec", "s"],
        "classic_selected_goto_s.csv",
    )

    source_ids = manifest["source_id"].astype(str).dropna().unique().tolist()
    if len(source_ids) != 1:
        raise RuntimeError(
            f"{manifest_path} must contain exactly one source_id, got {source_ids}"
        )
    source_id = source_ids[0]
    source_bag = ""
    metadata_path = dataset_dir / "extraction_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_source_id = str(metadata.get("source_id", "")).strip()
            if metadata_source_id and metadata_source_id != source_id:
                raise RuntimeError(
                    f"{metadata_path} source_id '{metadata_source_id}' does not "
                    f"match manifest source_id '{source_id}'"
                )
            source_bag = str(metadata.get("source_bag_path", "")).strip()
        except (OSError, ValueError, TypeError) as exc:
            log(f"[WARNING] Could not read source bag metadata from {metadata_path}: {exc}")
    else:
        log(f"[WARNING] Missing {metadata_path}; per-episode source_bag will be blank")

    manifest = manifest.copy()
    manifest["episode_id"] = manifest["episode_id"].astype(int)
    manifest["valid"] = parse_bool_col(manifest["valid"])

    for col in ["start_timestamp", "end_timestamp", "duration_sec"]:
        manifest[col] = pd.to_numeric(manifest[col], errors="coerce")
    manifest["ball_sample_count"] = pd.to_numeric(
        manifest["ball_sample_count"], errors="coerce"
    ).fillna(0).astype(int)
    manifest["classic_goto_s_count"] = pd.to_numeric(
        manifest["classic_goto_s_count"], errors="coerce"
    ).fillna(0).astype(int)

    if not ball.empty:
        for col in ["episode_id", "timestamp", "t_rel_sec", "header_timestamp", "x", "y", "z"]:
            ball[col] = pd.to_numeric(ball[col], errors="coerce")
        ball["episode_id"] = ball["episode_id"].astype(int)

    if not classic.empty:
        for col in ["episode_id", "timestamp", "t_rel_sec", "s"]:
            classic[col] = pd.to_numeric(classic[col], errors="coerce")
        classic["episode_id"] = classic["episode_id"].astype(int)

    return SourceDataset(
        source_id=source_id,
        source_bag=source_bag,
        dataset_dir=dataset_dir,
        manifest=manifest,
        ball=ball,
        classic=classic,
    )


def combine_sources(inputs: Sequence[str]) -> CombinedDataset:
    sources = [load_one_source(x) for x in inputs]
    validate_no_duplicate_source_ids([s.source_id for s in sources])

    assigned = assign_global_episode_id([s.manifest for s in sources])

    out_sources: List[SourceDataset] = []
    for source, manifest_with_gid in zip(sources, assigned):
        keys = manifest_with_gid[["source_id", "episode_id", "global_episode_id"]]
        ball = source.ball.merge(
            keys,
            on=["source_id", "episode_id"],
            how="inner",
        )
        classic = source.classic.merge(
            keys,
            on=["source_id", "episode_id"],
            how="inner",
        )
        out_sources.append(
            SourceDataset(
                source_id=source.source_id,
                source_bag=source.source_bag,
                dataset_dir=source.dataset_dir,
                manifest=manifest_with_gid,
                ball=ball,
                classic=classic,
            )
        )

    summary_rows: List[Dict[str, Any]] = []
    for src in out_sources:
        committed = int(len(src.manifest))
        valid = int(src.manifest["valid"].sum()) if committed else 0
        invalid = committed - valid
        ball_samples = int(len(src.ball))
        zero, one, multi = summarize_classic_counts(src.manifest, src.classic)
        summary_rows.append(
            {
                "source_id": src.source_id,
                "committed_episodes": committed,
                "valid_trajectories": valid,
                "invalid_trajectories": invalid,
                "ball_sample_count": ball_samples,
                "classic_zero": zero,
                "classic_one": one,
                "classic_multiple": multi,
            }
        )

    manifest = pd.concat([s.manifest for s in out_sources], ignore_index=True)
    ball = pd.concat([s.ball for s in out_sources], ignore_index=True)
    classic = pd.concat([s.classic for s in out_sources], ignore_index=True)

    return CombinedDataset(
        manifest=manifest,
        ball=ball,
        classic=classic,
        source_summaries=summary_rows,
        dataset_dirs=[s.dataset_dir for s in out_sources],
        source_bags={s.source_id: s.source_bag for s in out_sources},
    )


def _resolve_enabled_events(mark_events: Sequence[str]) -> List[str]:
    values = list(mark_events)
    if "none" in values:
        return []
    if "all" in values:
        return list(ALL_EVENT_NAMES)
    out = []
    for event in values:
        if event not in ALL_EVENT_NAMES:
            raise RuntimeError(f"Unsupported event name: {event}")
        out.append(event)
    return out


def _event_styles() -> Dict[str, Dict[str, Any]]:
    return {
        "episode-start": {"marker": "o", "fillstyle": "full", "edgecolor": "k", "size": 20},
        "episode-end": {"marker": "o", "fillstyle": "none", "edgecolor": "k", "size": 20},
        "goto-selected": {"marker": "o", "fillstyle": "full", "edgecolor": "none", "size": 16},
        "motion-onset": {"marker": "o", "fillstyle": "full", "edgecolor": "none", "size": 16},
        "vision-intercept": {"marker": "o", "fillstyle": "full", "edgecolor": "none", "size": 16},
    }


def _episode_color(
    episode_gid: int,
    color_mode: str,
    cmap: Any,
    norm: Any,
) -> Any:
    if color_mode == "single":
        return "tab:blue"
    return cmap(norm(episode_gid))


def build_episode_color_normalization(total_episode_count: int) -> Any:
    return matplotlib.colors.Normalize(vmin=0, vmax=max(total_episode_count - 1, 1))


def _build_episode_classic_map(
    classic_df: pd.DataFrame,
    selection: str,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    if classic_df.empty:
        return out

    grouped = classic_df.groupby(["source_id", "episode_id"], sort=False)
    for (source_id, episode_id), rows in grouped:
        picked = pick_representative_classic_row(rows, selection)
        if picked is None:
            continue
        out[(str(source_id), int(episode_id))] = {
            "timestamp": float(picked["timestamp"]),
            "t_rel_sec": float(picked["t_rel_sec"]),
            "s": float(picked["s"]),
        }
    return out


def _compute_s_time_draw_end_time(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    motion_onset_time: Optional[float],
    middle_line_y: float,
) -> Tuple[Optional[float], str]:
    """Choose trajectory cutoff time for S-time plot.

    Priority:
    1) max-y time (post motion-onset when available)
    2) middle-line crossing fallback when max-y is unavailable
    3) final available sample when neither can be computed
    """
    max_y_t, _max_y_x, _used_fallback = detect_max_y_intercept(
        timestamps=timestamps,
        x=x,
        y=y,
        motion_onset_time=motion_onset_time,
    )
    if max_y_t is not None:
        return float(max_y_t), "max-y"

    crossing_t, _crossing_x = detect_middle_line_crossing(
        timestamps=timestamps,
        x=x,
        y=y,
        middle_line_y=middle_line_y,
        motion_onset_time=None,
    )
    if crossing_t is not None:
        return float(crossing_t), "middle-line-fallback"

    if timestamps.size == 0:
        return None, "unavailable"
    return float(timestamps[-1]), "final-sample-fallback"


def _collect_episode_events(
    combined: CombinedDataset,
    classic_map: Dict[Tuple[str, int], Dict[str, float]],
    args: argparse.Namespace,
    enabled_events: Sequence[str],
) -> Tuple[Dict[Tuple[str, int], Dict[str, Optional[float]]], Dict[str, int]]:
    manifest = combined.manifest
    ball = combined.ball

    event_info: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}
    stats = {
        "motion_onset_available": 0,
        "representative_goto_available": 0,
        "vision_intercept_available": 0,
        "goto_xy_outside_ball_range": 0,
    }

    valid_manifest = manifest[manifest["valid"]].copy()
    for row in valid_manifest.itertuples(index=False):
        key = (str(row.source_id), int(row.episode_id))
        epi_ball = ball[
            (ball["source_id"] == key[0]) & (ball["episode_id"] == key[1])
        ].sort_values("timestamp", kind="mergesort")
        if epi_ball.empty:
            continue

        t = epi_ball["timestamp"].to_numpy(dtype=float)
        x = epi_ball["x"].to_numpy(dtype=float)
        y = epi_ball["y"].to_numpy(dtype=float)

        data: Dict[str, Optional[float]] = {
            "episode_start_time": float(row.start_timestamp),
            "episode_end_time": float(row.end_timestamp),
            "episode_duration_sec": float(row.duration_sec),
            "motion_onset_time": None,
            "vision_intercept_time": None,
            "vision_intercept_x": None,
            "goto_selected_time": None,
            "goto_selected_t_rel": None,
            "goto_selected_s": None,
            "s_time_draw_end_time": None,
        }

        onset = detect_motion_onset_time(
            timestamps=t,
            x=x,
            y=y,
            speed_threshold_mps=args.motion_speed_threshold_mps,
            min_consecutive=args.motion_min_consecutive,
            smoothing_window=args.motion_smoothing_window,
            min_displacement_m=args.motion_min_displacement_m,
        )
        data["motion_onset_time"] = onset
        if onset is not None:
            stats["motion_onset_available"] += 1

        classic = classic_map.get(key)
        if classic is not None:
            data["goto_selected_time"] = classic["timestamp"]
            data["goto_selected_t_rel"] = classic["t_rel_sec"]
            data["goto_selected_s"] = classic["s"]
            stats["representative_goto_available"] += 1

        if args.vision_intercept_mode == "max-y":
            v_t, v_x, _used_fallback = detect_max_y_intercept(
                timestamps=t,
                x=x,
                y=y,
                motion_onset_time=onset,
            )
            data["vision_intercept_time"] = v_t
            data["vision_intercept_x"] = v_x
        else:
            v_t, v_x = detect_middle_line_crossing(
                timestamps=t,
                x=x,
                y=y,
                middle_line_y=args.middle_line_y,
                motion_onset_time=onset,
            )
            data["vision_intercept_time"] = v_t
            data["vision_intercept_x"] = v_x

        if data["vision_intercept_time"] is not None:
            stats["vision_intercept_available"] += 1

        draw_end_t, _draw_end_mode = _compute_s_time_draw_end_time(
            timestamps=t,
            x=x,
            y=y,
            motion_onset_time=onset,
            middle_line_y=args.middle_line_y,
        )
        data["s_time_draw_end_time"] = draw_end_t

        if "goto-selected" in enabled_events and data["goto_selected_time"] is not None:
            got = interpolate_position_at_time(
                timestamps=t,
                x=x,
                y=y,
                event_time=float(data["goto_selected_time"]),
            )
            if got is None:
                stats["goto_xy_outside_ball_range"] += 1

        event_info[key] = data

    return event_info, stats


def _analysis_exclusion_row(
    row: Any,
    source_bag: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "source_bag": source_bag,
        "source_id": str(row.source_id),
        "episode_id": int(row.episode_id),
        "global_episode_id": int(row.global_episode_id),
        "s_onset": np.nan,
        "s_int": np.nan,
        "delta_s": np.nan,
        "classification": "unavailable",
        "included": False,
        "exclusion_reason": reason,
    }


def _build_approach_segment(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    onset_time: float,
    crossing_time: float,
    crossing_x: float,
    line_center_x: float,
    s_sign: float,
    middle_line_y: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return raw and monotonic (s, y) onset-to-crossing segments."""
    onset_position = interpolate_position_at_time(
        timestamps, x, y, event_time=onset_time
    )
    if onset_position is None:
        return None

    interior = (timestamps > onset_time) & (timestamps < crossing_time)
    raw_x = np.concatenate(
        [
            np.asarray([onset_position[0]], dtype=float),
            x[interior],
            np.asarray([crossing_x], dtype=float),
        ]
    )
    raw_y = np.concatenate(
        [
            np.asarray([onset_position[1]], dtype=float),
            y[interior],
            np.asarray([middle_line_y], dtype=float),
        ]
    )
    raw_s = table_x_to_s(raw_x, line_center_x=line_center_x, s_sign=s_sign)
    monotonic_y, monotonic_s = select_monotonic_approach_points(raw_y, raw_s)

    if monotonic_y.size > 0 and np.isclose(monotonic_y[-1], middle_line_y):
        monotonic_y[-1] = float(middle_line_y)
        monotonic_s[-1] = float(raw_s[-1])
    elif monotonic_y.size > 0 and middle_line_y > monotonic_y[-1]:
        monotonic_y = np.append(monotonic_y, float(middle_line_y))
        monotonic_s = np.append(monotonic_s, float(raw_s[-1]))

    if monotonic_y.size < 2 or np.any(np.diff(monotonic_y) <= 0.0):
        return None
    return raw_s, raw_y, monotonic_s, monotonic_y


def _summary_row_for_scope(
    scope: str,
    source_id: str,
    source_bag: str,
    scope_episodes: pd.DataFrame,
) -> Dict[str, Any]:
    usable = scope_episodes[scope_episodes["included"]]
    n_usable = int(len(usable))
    counts = {
        group: int((usable["classification"] == group).sum())
        for group in ["L", "R", "deadband"]
    }
    result: Dict[str, Any] = {
        "scope": scope,
        "source_id": source_id,
        "source_bag": source_bag,
        "total_episodes": int(len(scope_episodes)),
        "usable_episodes": n_usable,
        "excluded_episodes": int(len(scope_episodes) - n_usable),
        "N_L": counts["L"],
        "N_R": counts["R"],
        "N_deadband": counts["deadband"],
        "percentage_denominator": "usable_including_deadband",
        "pct_L": 100.0 * counts["L"] / n_usable if n_usable else np.nan,
        "pct_R": 100.0 * counts["R"] / n_usable if n_usable else np.nan,
        "pct_deadband": 100.0 * counts["deadband"] / n_usable if n_usable else np.nan,
        "directional_count_bias_B": directional_count_bias(counts["L"], counts["R"]),
    }
    for group in ["L", "R"]:
        stats = sample_descriptive_statistics(
            usable.loc[usable["classification"] == group, "s_int"].to_numpy(dtype=float)
        )
        for name, value in stats.items():
            result[f"{group}_s_int_{name}"] = value
    return result


def analyze_trajectory_distribution(
    combined: CombinedDataset,
    args: argparse.Namespace,
) -> TrajectoryDistributionAnalysis:
    episode_rows: List[Dict[str, Any]] = []
    raw_segments: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    monotonic_segments: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}

    for row in combined.manifest.sort_values("global_episode_id").itertuples(index=False):
        source_id = str(row.source_id)
        key = (source_id, int(row.episode_id))
        source_bag = combined.source_bags.get(source_id, "")
        invalid_reason = "" if pd.isna(row.invalid_reason) else str(row.invalid_reason).strip()
        if not bool(row.valid):
            reason = invalid_reason or "manifest_invalid"
            episode_rows.append(_analysis_exclusion_row(row, source_bag, reason))
            continue

        epi_ball = combined.ball[
            (combined.ball["source_id"] == source_id)
            & (combined.ball["episode_id"] == int(row.episode_id))
        ].sort_values("timestamp", kind="mergesort")
        finite = np.isfinite(
            epi_ball[["timestamp", "x", "y"]].to_numpy(dtype=float)
        ).all(axis=1)
        epi_ball = epi_ball.loc[finite]
        if len(epi_ball) < 2:
            episode_rows.append(
                _analysis_exclusion_row(row, source_bag, "insufficient_finite_ball_samples")
            )
            continue

        t = epi_ball["timestamp"].to_numpy(dtype=float)
        x = epi_ball["x"].to_numpy(dtype=float)
        y = epi_ball["y"].to_numpy(dtype=float)
        onset = detect_motion_onset_time(
            timestamps=t,
            x=x,
            y=y,
            speed_threshold_mps=args.motion_speed_threshold_mps,
            min_consecutive=args.motion_min_consecutive,
            smoothing_window=args.motion_smoothing_window,
            min_displacement_m=args.motion_min_displacement_m,
        )
        if onset is None:
            episode_rows.append(
                _analysis_exclusion_row(row, source_bag, "motion_onset_unavailable")
            )
            continue

        crossing_time, crossing_x = detect_middle_line_crossing(
            timestamps=t,
            x=x,
            y=y,
            middle_line_y=args.middle_line_y,
            motion_onset_time=onset,
        )
        if crossing_time is None or crossing_x is None:
            episode_rows.append(
                _analysis_exclusion_row(row, source_bag, "middle_line_crossing_unavailable")
            )
            continue

        segment = _build_approach_segment(
            timestamps=t,
            x=x,
            y=y,
            onset_time=float(onset),
            crossing_time=float(crossing_time),
            crossing_x=float(crossing_x),
            line_center_x=args.line_center_x,
            s_sign=args.s_sign,
            middle_line_y=args.middle_line_y,
        )
        if segment is None:
            episode_rows.append(
                _analysis_exclusion_row(row, source_bag, "invalid_monotonic_approach_segment")
            )
            continue

        raw_s, raw_y, monotonic_s, monotonic_y = segment
        s_onset = float(raw_s[0])
        s_int = float(raw_s[-1])
        classification = classify_lateral_intercept(
            s_int, args.lr_deadband_epsilon_m
        )
        episode_rows.append(
            {
                "source_bag": source_bag,
                "source_id": source_id,
                "episode_id": int(row.episode_id),
                "global_episode_id": int(row.global_episode_id),
                "s_onset": s_onset,
                "s_int": s_int,
                "delta_s": float(s_int - s_onset),
                "classification": classification,
                "included": True,
                "exclusion_reason": "",
            }
        )
        raw_segments[key] = (raw_s, raw_y)
        monotonic_segments[key] = (monotonic_y, monotonic_s)

    episodes = pd.DataFrame(episode_rows)
    episodes["included"] = episodes["included"].astype(bool)

    scopes: List[Tuple[str, str, str, pd.DataFrame]] = [
        ("combined", "", "", episodes)
    ]
    for source_id in combined.manifest["source_id"].astype(str).drop_duplicates():
        scopes.append(
            (
                "source",
                source_id,
                combined.source_bags.get(source_id, ""),
                episodes[episodes["source_id"] == source_id],
            )
        )

    summary = pd.DataFrame(
        [
            _summary_row_for_scope(scope, source_id, source_bag, scope_episodes)
            for scope, source_id, source_bag, scope_episodes in scopes
        ]
    )

    grid_rows: List[Dict[str, Any]] = []
    for scope, source_id, _source_bag, scope_episodes in scopes:
        included = scope_episodes[scope_episodes["included"]]
        scope_keys = [
            (str(row.source_id), int(row.episode_id))
            for row in included.itertuples(index=False)
        ]
        if not scope_keys:
            continue
        y_min = min(float(monotonic_segments[key][0][0]) for key in scope_keys)
        y_grid = np.linspace(
            y_min, float(args.middle_line_y), TRAJECTORY_DISTRIBUTION_GRID_POINTS
        )
        for group in ["L", "R"]:
            group_rows = included[included["classification"] == group]
            matrices = []
            for row in group_rows.itertuples(index=False):
                key = (str(row.source_id), int(row.episode_id))
                epi_y, epi_s = monotonic_segments[key]
                matrices.append(interpolate_approach_on_y_grid(epi_y, epi_s, y_grid))
            values = (
                np.vstack(matrices)
                if matrices
                else np.empty((0, TRAJECTORY_DISTRIBUTION_GRID_POINTS), dtype=float)
            )
            stats = summarize_episode_grid(values)
            for idx, y_value in enumerate(y_grid):
                grid_rows.append(
                    {
                        "scope": scope,
                        "source_id": source_id,
                        "classification": group,
                        "y": float(y_value),
                        "contributing_episode_count": int(stats["count"][idx]),
                        "mean_s": float(stats["mean"][idx]),
                        "sample_std_s": float(stats["sample_std"][idx]),
                        "q25_s": float(stats["q25"][idx]),
                        "q75_s": float(stats["q75"][idx]),
                    }
                )

    return TrajectoryDistributionAnalysis(
        episodes=episodes,
        summary=summary,
        mean_grid=pd.DataFrame(grid_rows),
        raw_segments=raw_segments,
    )


def _plot_xy_overlay(
    combined: CombinedDataset,
    event_info: Dict[Tuple[str, int], Dict[str, Optional[float]]],
    classic_map: Dict[Tuple[str, int], Dict[str, float]],
    enabled_events: Sequence[str],
    args: argparse.Namespace,
    out_paths: Sequence[Path],
) -> None:
    manifest = combined.manifest
    ball = combined.ball

    valid_manifest = manifest[manifest["valid"]].copy()
    valid_keys = valid_manifest[["source_id", "episode_id", "global_episode_id"]]
    valid_ball = ball.merge(valid_keys, on=["source_id", "episode_id", "global_episode_id"], how="inner")

    fig, ax = plt.subplots(figsize=(9, 8))

    table_xlim = tuple(args.table_x_limits)
    table_ylim = tuple(args.table_y_limits)
    s_limits = table_x_to_s(
        np.array([table_xlim[0], table_xlim[1]], dtype=float),
        line_center_x=args.line_center_x,
        s_sign=args.s_sign,
    )
    s_min = float(min(s_limits))
    s_max = float(max(s_limits))
    # Right-to-left display means larger table x appears on the left.
    ax.set_xlim(s_min, s_max)
    ax.invert_xaxis()
    ax.set_ylim(table_ylim)
    ax.set_aspect("equal", adjustable="box")

    # Subtle table boundary and middle line.
    rect_s = [s_min, s_max, s_max, s_min, s_min]
    rect_y = [table_ylim[0], table_ylim[0], table_ylim[1], table_ylim[1], table_ylim[0]]
    ax.plot(rect_s, rect_y, color="0.55", lw=0.8, alpha=0.5)
    ax.axhline(args.middle_line_y, color="0.45", lw=0.8, alpha=0.4)

    if s_min <= 0.0 <= s_max:
        ax.axvline(0.0, color="0.5", lw=0.8, ls=":", alpha=0.6)
    if table_ylim[0] <= 0.0 <= table_ylim[1]:
        ax.axhline(0.0, color="0.5", lw=0.8, ls=":", alpha=0.6)

    cmap = plt.get_cmap("turbo")
    total_eps = len(valid_manifest)
    norm = build_episode_color_normalization(total_eps)

    styles = _event_styles()

    grouped = valid_ball.groupby(["source_id", "episode_id", "global_episode_id"], sort=True)
    for (source_id, episode_id, gid), rows in grouped:
        rows = rows.sort_values("timestamp", kind="mergesort")
        x_table = rows["x"].to_numpy(dtype=float)
        s_ball = table_x_to_s(
            x_table,
            line_center_x=args.line_center_x,
            s_sign=args.s_sign,
        )
        y = rows["y"].to_numpy(dtype=float)
        t = rows["timestamp"].to_numpy(dtype=float)
        color = _episode_color(int(gid), args.color_mode, cmap, norm)

        ax.plot(s_ball, y, color=color, lw=1.0, alpha=0.65)

        key = (str(source_id), int(episode_id))
        e = event_info.get(key, {})

        if "episode-start" in enabled_events and len(s_ball) > 0:
            st = styles["episode-start"]
            ax.scatter(
            [s_ball[0]],
                [y[0]],
                s=st["size"],
                marker=st["marker"],
                facecolors=color,
                edgecolors=st["edgecolor"],
                linewidths=0.3,
                alpha=0.95,
                zorder=4,
            )

        if "episode-end" in enabled_events and len(s_ball) > 0:
            st = styles["episode-end"]
            ax.scatter(
            [s_ball[-1]],
                [y[-1]],
                s=st["size"],
                marker=st["marker"],
                facecolors="none",
                edgecolors=st["edgecolor"],
                linewidths=0.6,
                alpha=0.95,
                zorder=4,
            )

        for ev_name, ev_key in [
            ("motion-onset", "motion_onset_time"),
            ("goto-selected", "goto_selected_time"),
            ("vision-intercept", "vision_intercept_time"),
        ]:
            if ev_name not in enabled_events:
                continue
            ev_time = e.get(ev_key)
            if ev_time is None:
                continue
            pos = interpolate_position_at_time(t, s_ball, y, float(ev_time))
            if pos is None:
                continue
            st = styles[ev_name]
            ax.scatter(
                [pos[0]],
                [pos[1]],
                s=st["size"],
                marker=st["marker"],
                facecolors=color,
                edgecolors=st["edgecolor"],
                linewidths=0.3,
                alpha=0.9,
                zorder=4,
            )

        if args.show_classic_goto:
            classic = classic_map.get(key)
            if classic is not None:
                s_target = float(classic["s"])
                ax.scatter(
                    [s_target],
                    [args.middle_line_y],
                    marker="x",
                    color=color,
                    s=24,
                    linewidths=0.8,
                    alpha=0.85,
                    zorder=5,
                )

    ax.set_xlabel("Ball Lateral s [m]")
    ax.set_ylabel("Table Y [m]")
    ax.set_title(f"Ball Trajectory Distribution - {total_eps} Episodes")

    if args.color_mode == "episode":
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Combined episode")

    ax.grid(False)
    fig.tight_layout()
    for path in out_paths:
        fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def _plot_s_time_overlay(
    combined: CombinedDataset,
    event_info: Dict[Tuple[str, int], Dict[str, Optional[float]]],
    classic_map: Dict[Tuple[str, int], Dict[str, float]],
    enabled_events: Sequence[str],
    args: argparse.Namespace,
    out_paths: Sequence[Path],
) -> None:
    manifest = combined.manifest
    ball = combined.ball

    valid_manifest = manifest[manifest["valid"]].copy()
    valid_keys = valid_manifest[["source_id", "episode_id", "global_episode_id", "start_timestamp", "end_timestamp"]]
    valid_ball = ball.merge(valid_keys, on=["source_id", "episode_id", "global_episode_id"], how="inner")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.get_cmap("turbo")
    total_eps = len(valid_manifest)
    norm = build_episode_color_normalization(total_eps)

    if not valid_ball.empty:
        valid_ball = valid_ball.sort_values(["global_episode_id", "timestamp"], kind="mergesort")
        grouped = valid_ball.groupby(["source_id", "episode_id", "global_episode_id"], sort=True)
        for (source_id, episode_id, gid), rows in grouped:
            key = (str(source_id), int(episode_id))
            e = event_info.get(key, {})

            t_abs = rows["timestamp"].to_numpy(dtype=float)
            x = rows["x"].to_numpy(dtype=float)
            draw_end_abs = e.get("s_time_draw_end_time")
            if draw_end_abs is not None:
                keep = t_abs <= (float(draw_end_abs) + 1e-12)
                if np.any(keep):
                    t_abs = t_abs[keep]
                    x = x[keep]

            if t_abs.size == 0:
                continue

            t_rel = t_abs - float(rows["start_timestamp"].iloc[0])
            s_ball = table_x_to_s(x, line_center_x=args.line_center_x, s_sign=args.s_sign)
            color = _episode_color(int(gid), args.color_mode, cmap, norm)

            ax.plot(t_rel, s_ball, color=color, lw=1.0, alpha=0.65)

            if args.show_classic_goto:
                classic = classic_map.get(key)
                if classic is not None:
                    t_sel = float(classic["t_rel_sec"])
                    s_sel = float(classic["s"])
                    ax.scatter([t_sel], [s_sel], marker="x", color=color, s=20, linewidths=0.8, alpha=0.85)
                    t_end = float(t_rel[-1])
                    if t_end >= t_sel:
                        ax.plot([t_sel, t_end], [s_sel, s_sel], color=color, lw=0.6, alpha=0.25)

            for ev_name, ev_key in [
                ("goto-selected", "goto_selected_t_rel"),
                ("motion-onset", "motion_onset_time"),
                ("episode-start", "episode_start_time"),
                ("episode-end", "episode_end_time"),
                ("vision-intercept", "vision_intercept_time"),
            ]:
                if ev_name not in enabled_events:
                    continue

                if ev_name == "episode-start":
                    t_line = 0.0
                elif ev_name == "episode-end":
                    t_line = float(e.get("episode_duration_sec")) if e.get("episode_duration_sec") is not None else None
                elif ev_name == "goto-selected":
                    t_line = e.get("goto_selected_t_rel")
                else:
                    ev_abs = e.get(ev_key)
                    if ev_abs is None:
                        t_line = None
                    else:
                        t_line = float(ev_abs - e["episode_start_time"])

                if t_line is None:
                    continue
                ax.axvline(t_line, color="0.2", lw=0.5, alpha=0.06)

    ax.axvline(0.0, color="0.45", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0.0, color="0.45", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Time from Episode Start [s]")
    ax.set_ylabel("Ball Lateral s [m]")
    ax.set_title(f"Ball Lateral Position vs Time - {len(valid_manifest)} Episodes")

    proxy_handles = []
    proxy_labels = []
    for event_name in enabled_events:
        proxy_handles.append(Line2D([0], [0], color="0.2", lw=0.8, alpha=0.25))
        proxy_labels.append(event_name)
    if proxy_handles:
        ax.legend(proxy_handles, proxy_labels, loc="best", framealpha=0.9)

    fig.tight_layout()
    for path in out_paths:
        fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def _plot_trajectory_distribution(
    analysis: TrajectoryDistributionAnalysis,
    args: argparse.Namespace,
    out_paths: Sequence[Path],
) -> None:
    included = analysis.episodes[analysis.episodes["included"]]
    combined_grid = analysis.mean_grid[analysis.mean_grid["scope"] == "combined"]
    summary = analysis.summary[analysis.summary["scope"] == "combined"].iloc[0]

    fig = plt.figure(figsize=(15, 8.5))
    grid_spec = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], hspace=0.34, wspace=0.28)
    ax_spatial = fig.add_subplot(grid_spec[:, 0])
    ax_distribution = fig.add_subplot(grid_spec[0, 1])
    ax_counts = fig.add_subplot(grid_spec[1, 1])

    table_xlim = tuple(args.table_x_limits)
    table_ylim = tuple(args.table_y_limits)
    s_limits = table_x_to_s(
        np.asarray(table_xlim, dtype=float),
        line_center_x=args.line_center_x,
        s_sign=args.s_sign,
    )
    s_min, s_max = float(min(s_limits)), float(max(s_limits))
    ax_spatial.set_xlim(s_min, s_max)
    ax_spatial.invert_xaxis()
    ax_spatial.set_ylim(table_ylim)
    ax_spatial.set_aspect("equal", adjustable="box")
    ax_spatial.plot(
        [s_min, s_max, s_max, s_min, s_min],
        [table_ylim[0], table_ylim[0], table_ylim[1], table_ylim[1], table_ylim[0]],
        color="0.55",
        lw=0.8,
        alpha=0.5,
    )
    ax_spatial.axhline(args.middle_line_y, color="0.35", lw=1.0, alpha=0.65)
    ax_spatial.axvline(0.0, color="0.5", lw=0.8, ls=":", alpha=0.65)

    for row in included.itertuples(index=False):
        key = (str(row.source_id), int(row.episode_id))
        segment = analysis.raw_segments.get(key)
        if segment is None:
            continue
        raw_s, raw_y = segment
        color = LR_COLORS[str(row.classification)]
        alpha = 0.17 if row.classification in ["L", "R"] else 0.35
        ax_spatial.plot(raw_s, raw_y, color=color, lw=0.8, alpha=alpha, zorder=1)

    for group in ["L", "R"]:
        group_grid = combined_grid[combined_grid["classification"] == group]
        if group_grid.empty:
            continue
        y_grid = group_grid["y"].to_numpy(dtype=float)
        mean_s = group_grid["mean_s"].to_numpy(dtype=float)
        q25 = group_grid["q25_s"].to_numpy(dtype=float)
        q75 = group_grid["q75_s"].to_numpy(dtype=float)
        color = LR_COLORS[group]
        ax_spatial.fill_betweenx(
            y_grid, q25, q75, color=color, alpha=0.20, linewidth=0.0, zorder=2
        )
        ax_spatial.plot(
            mean_s,
            y_grid,
            color=color,
            lw=3.0,
            label=f"{group} mean (N={int(summary[f'N_{group}'])})",
            zorder=3,
        )

    for group in ["L", "deadband", "R"]:
        rows = included[included["classification"] == group]
        if rows.empty:
            continue
        ax_spatial.scatter(
            rows["s_int"],
            np.full(len(rows), args.middle_line_y),
            s=22 if group != "deadband" else 28,
            color=LR_COLORS[group],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
            zorder=4,
        )
    ax_spatial.set_xlabel("Ball Lateral s [m] (+s = robot-base +X)")
    ax_spatial.set_ylabel("Table Y [m]")
    ax_spatial.set_title("Motion Onset to Interpolated Middle-Line Crossing")
    ax_spatial.legend(loc="best", framealpha=0.9)

    finite_s = included["s_int"].to_numpy(dtype=float)
    finite_s = finite_s[np.isfinite(finite_s)]
    if finite_s.size:
        bins = np.linspace(float(np.min(finite_s)), float(np.max(finite_s)), 25)
        if np.isclose(bins[0], bins[-1]):
            bins = np.linspace(bins[0] - 0.01, bins[-1] + 0.01, 25)
        for group in ["L", "R"]:
            values = included.loc[
                included["classification"] == group, "s_int"
            ].to_numpy(dtype=float)
            if values.size:
                ax_distribution.hist(
                    values,
                    bins=bins,
                    color=LR_COLORS[group],
                    alpha=0.48,
                    label=group,
                )
                mean = float(np.mean(values))
                median = float(np.median(values))
                ax_distribution.axvline(mean, color=LR_COLORS[group], lw=2.0)
                ax_distribution.axvline(
                    median, color=LR_COLORS[group], lw=1.6, ls="--"
                )
    eps = float(args.lr_deadband_epsilon_m)
    ax_distribution.axvspan(-eps, eps, color=LR_COLORS["deadband"], alpha=0.16)
    ax_distribution.axvline(0.0, color="0.2", lw=0.9, ls=":")
    ax_distribution.invert_xaxis()
    ax_distribution.set_xlabel("s at interception [m]")
    ax_distribution.set_ylabel("Episodes")
    ax_distribution.set_title(f"Interception Distribution (deadband ±{eps:.3f} m)")
    ax_distribution.legend(title="Solid mean; dashed median", fontsize=9, title_fontsize=8)

    groups = ["L", "deadband", "R"]
    counts = np.asarray(
        [summary["N_L"], summary["N_deadband"], summary["N_R"]], dtype=int
    )
    denominator = int(summary["usable_episodes"])
    percentages = 100.0 * counts / denominator if denominator else np.zeros(3)
    bars = ax_counts.bar(
        groups, counts, color=[LR_COLORS[group] for group in groups], alpha=0.78
    )
    for bar, group, count, percentage in zip(bars, groups, counts, percentages):
        label_group = "Deadband" if group == "deadband" else group
        ax_counts.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{label_group}: {count} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    bias = float(summary["directional_count_bias_B"])
    bias_text = "undefined" if not np.isfinite(bias) else f"{bias:+.4f}"
    ax_counts.text(
        0.5,
        0.94,
        f"B = (N_L - N_R) / (N_L + N_R) = {bias_text}",
        transform=ax_counts.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    ax_counts.text(
        0.5,
        0.86,
        "Percent of usable episodes (deadband included)",
        transform=ax_counts.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.3",
    )
    ax_counts.set_ylabel("Episodes")
    ax_counts.set_title(
        f"Counts and Directional Bias — usable {denominator}, excluded {int(summary['excluded_episodes'])}"
    )
    ax_counts.set_ylim(0.0, max(float(np.max(counts)) * 1.55, 1.0))
    ax_counts.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Combined L/R Trajectory-Distribution Analysis", fontsize=16)
    fig.subplots_adjust(top=0.91, bottom=0.09, left=0.07, right=0.98)
    for path in out_paths:
        fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def write_trajectory_distribution_outputs(
    out_dir: Path,
    analysis: TrajectoryDistributionAnalysis,
) -> List[str]:
    filenames = [
        "ball_trajectory_distribution_lr_summary.csv",
        "ball_trajectory_distribution_lr_episodes.csv",
        "ball_trajectory_distribution_lr_mean_grid.csv",
    ]
    analysis.summary.to_csv(out_dir / filenames[0], index=False)
    analysis.episodes.to_csv(out_dir / filenames[1], index=False)
    analysis.mean_grid.to_csv(out_dir / filenames[2], index=False)
    return filenames


def _resolve_output_dir(combined: CombinedDataset, args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser().resolve()
    if len(combined.dataset_dirs) == 1:
        return combined.dataset_dirs[0] / "plots"
    return Path("combined_dataset_plots").resolve()


def _ensure_formats(formats: Sequence[str]) -> List[str]:
    allowed = {"png", "svg", "pdf"}
    out = []
    for fmt in formats:
        f = fmt.strip().lower()
        if f not in allowed:
            raise RuntimeError(f"Unsupported format '{fmt}'. Allowed: {sorted(allowed)}")
        out.append(f)
    return out


def print_source_summary(combined: CombinedDataset) -> None:
    log("Per-source summary:")
    for row in combined.source_summaries:
        log(
            "  "
            f"source_id={row['source_id']} "
            f"committed={row['committed_episodes']} "
            f"valid={row['valid_trajectories']} "
            f"invalid={row['invalid_trajectories']} "
            f"ball_samples={row['ball_sample_count']} "
            f"classic_zero={row['classic_zero']} "
            f"classic_one={row['classic_one']} "
            f"classic_multiple={row['classic_multiple']}"
        )

    manifest = combined.manifest
    ball = combined.ball
    classic = combined.classic
    total_committed = int(len(manifest))
    total_valid = int(manifest["valid"].sum()) if total_committed else 0
    total_invalid = total_committed - total_valid
    zero, one, multiple = summarize_classic_counts(manifest, classic)
    log("Combined totals:")
    log(
        "  "
        f"committed={total_committed} "
        f"valid={total_valid} "
        f"invalid={total_invalid} "
        f"ball_samples={len(ball)} "
        f"classic_samples={len(classic)} "
        f"classic_zero={zero} "
        f"classic_one={one} "
        f"classic_multiple={multiple}"
    )


def write_plot_metadata(
    out_dir: Path,
    combined: CombinedDataset,
    args: argparse.Namespace,
    generated_files: Sequence[str],
) -> None:
    payload = {
        "input_source_ids": combined.manifest["source_id"].drop_duplicates().tolist(),
        "input_dataset_csv_paths": [str(p) for p in combined.dataset_dirs],
        "combined_episode_count": int(len(combined.manifest)),
        "combined_valid_trajectory_count": int(combined.manifest["valid"].sum()),
        "geometry": {
            "line_center_x": args.line_center_x,
            "middle_line_y": args.middle_line_y,
            "s_sign": args.s_sign,
            "table_x_limits": list(args.table_x_limits),
            "table_y_limits": list(args.table_y_limits),
        },
        "motion_onset": {
            "speed_threshold_mps": args.motion_speed_threshold_mps,
            "min_consecutive": args.motion_min_consecutive,
            "smoothing_window": args.motion_smoothing_window,
            "min_displacement_m": args.motion_min_displacement_m,
        },
        "vision_intercept_mode": args.vision_intercept_mode,
        "classic_selection": args.classic_selection,
        "generated_filenames": list(generated_files),
    }
    if args.trajectory_distribution:
        payload["trajectory_distribution"] = {
            "enabled": True,
            "interception_definition": "first_post_onset_forward_middle_line_crossing",
            "deadband_epsilon_m": args.lr_deadband_epsilon_m,
            "percentage_denominator": "usable_including_deadband",
            "mean_trajectory_grid_points": TRAJECTORY_DISTRIBUTION_GRID_POINTS,
            "mean_trajectory_independent_axis": "table_y",
            "monotonicization": "chronological_strictly_increasing_y_frontier",
            "extrapolation": False,
        }
    (out_dir / "plot_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one combined intercept-dataset view from one or more extracted dataset_csv sources.",
    )
    parser.add_argument("inputs", nargs="+", help="One or more dataset_csv directories or parent bag directories")
    parser.add_argument("--out-dir", default=None, help="Output directory for plots")
    parser.add_argument("--formats", nargs="+", default=["png"], help="Output formats: png svg pdf")
    parser.add_argument("--dpi", type=int, default=160, help="Raster DPI for saved figures")
    parser.add_argument("--color-mode", choices=["single", "episode"], default="episode")

    parser.add_argument("--show-classic-goto", action="store_true")
    parser.add_argument("--classic-selection", choices=["first", "last"], default="last")

    parser.add_argument(
        "--mark-events",
        nargs="+",
        default=["episode-start", "episode-end"],
        choices=ALL_EVENT_NAMES + ["all", "none"],
        help="Event markers to show: all none goto-selected motion-onset episode-start episode-end vision-intercept",
    )

    parser.add_argument("--motion-speed-threshold-mps", type=float, default=0.10)
    parser.add_argument("--motion-min-consecutive", type=int, default=3)
    parser.add_argument("--motion-smoothing-window", type=int, default=5)
    parser.add_argument("--motion-min-displacement-m", type=float, default=0.01)
    parser.add_argument(
        "--vision-intercept-mode",
        choices=["max-y", "middle-line-crossing"],
        default="max-y",
    )

    parser.add_argument("--line-center-x", type=float, default=0.3)
    parser.add_argument("--middle-line-y", type=float, default=0.6)
    parser.add_argument("--s-sign", type=float, default=-1.0)
    parser.add_argument("--table-x-limits", nargs=2, type=float, default=[0.0, 0.6])
    parser.add_argument("--table-y-limits", nargs=2, type=float, default=[0.0, 1.2])

    parser.add_argument(
        "--trajectory-distribution",
        action="store_true",
        help=(
            "Generate opt-in L/R trajectory-distribution figure and CSV reports "
            "using interpolated post-onset middle-line crossings"
        ),
    )
    parser.add_argument(
        "--lr-deadband-epsilon-m",
        type=float,
        default=0.005,
        help="L/R classification deadband half-width in metres (default: 0.005)",
    )

    parser.add_argument("--interactive", action="store_true", help="Show figures interactively after saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.motion_min_consecutive <= 0:
        raise RuntimeError("--motion-min-consecutive must be positive")
    if args.motion_smoothing_window <= 0:
        raise RuntimeError("--motion-smoothing-window must be positive")
    if args.s_sign == 0.0:
        raise RuntimeError("--s-sign must be non-zero")
    if not np.isfinite(args.lr_deadband_epsilon_m) or args.lr_deadband_epsilon_m < 0.0:
        raise RuntimeError("--lr-deadband-epsilon-m must be finite and non-negative")

    formats = _ensure_formats(args.formats)
    enabled_events = _resolve_enabled_events(args.mark_events)

    combined = combine_sources(args.inputs)
    print_source_summary(combined)

    classic_map = _build_episode_classic_map(combined.classic, args.classic_selection)
    event_info, event_stats = _collect_episode_events(
        combined,
        classic_map,
        args,
        enabled_events,
    )

    log("Event availability summary:")
    log(f"  motion onset available: {event_stats['motion_onset_available']}")
    log(f"  representative GOTO_S available: {event_stats['representative_goto_available']}")
    log(f"  vision intercept available: {event_stats['vision_intercept_available']}")
    log(
        "  event timestamps outside measured ball range: "
        f"{event_stats['goto_xy_outside_ball_range']}"
    )

    out_dir = _resolve_output_dir(combined, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Resolved output directory: {out_dir}")

    xy_paths = [out_dir / f"ball_trajectory_xy_overlay.{fmt}" for fmt in formats]
    st_paths = [out_dir / f"ball_lateral_s_vs_time.{fmt}" for fmt in formats]

    _plot_xy_overlay(combined, event_info, classic_map, enabled_events, args, xy_paths)
    _plot_s_time_overlay(combined, event_info, classic_map, enabled_events, args, st_paths)

    generated_files = [p.name for p in xy_paths + st_paths]
    if args.trajectory_distribution:
        analysis = analyze_trajectory_distribution(combined, args)
        distribution_formats = list(dict.fromkeys(["png"] + formats))
        distribution_paths = [
            out_dir / f"ball_trajectory_distribution_lr.{fmt}"
            for fmt in distribution_formats
        ]
        _plot_trajectory_distribution(analysis, args, distribution_paths)
        generated_files.extend(path.name for path in distribution_paths)
        generated_files.extend(
            write_trajectory_distribution_outputs(out_dir, analysis)
        )
        log("Trajectory-distribution summary:")
        for row in analysis.summary.itertuples(index=False):
            label = "combined" if row.scope == "combined" else row.source_id
            log(
                f"  {label}: usable={row.usable_episodes} excluded={row.excluded_episodes} "
                f"L={row.N_L} ({row.pct_L:.1f}%) "
                f"R={row.N_R} ({row.pct_R:.1f}%) "
                f"deadband={row.N_deadband} ({row.pct_deadband:.1f}%) "
                f"B={row.directional_count_bias_B:+.4f}"
            )
        excluded = analysis.episodes[~analysis.episodes["included"]]
        if not excluded.empty:
            log("Trajectory-distribution exclusions:")
            for reason, count in excluded["exclusion_reason"].value_counts().items():
                log(f"  {reason}: {count}")
    write_plot_metadata(out_dir, combined, args, generated_files)

    if args.interactive:
        plt.show()


if __name__ == "__main__":
    main()
