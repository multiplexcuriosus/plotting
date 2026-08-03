from __future__ import annotations

import inspect
import types
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from intercept_dataset_common import (
    detect_max_y_intercept,
    detect_middle_line_crossing,
    detect_motion_onset_time,
    interpolate_position_at_time,
    pick_representative_classic_row,
    s_to_table_x,
    table_x_to_s,
)
import plot_intercept_dataset as pid


matplotlib.use("Agg", force=True)


def _write_dataset_csv(base_dir: Path, source_id: str, with_classic: bool = True) -> Path:
    d = base_dir / source_id / "dataset_csv"
    d.mkdir(parents=True, exist_ok=True)

    manifest = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "episode_id": 0,
                "start_timestamp": 10.0,
                "end_timestamp": 12.0,
                "duration_sec": 2.0,
                "ball_sample_count": 3,
                "classic_goto_s_count": 2 if with_classic else 0,
                "ball_frame_id": "table",
                "valid": True,
                "invalid_reason": "",
            }
        ]
    )
    manifest.to_csv(d / "manifest.csv", index=False)

    ball = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "episode_id": 0,
                "timestamp": 10.0,
                "t_rel_sec": 0.0,
                "header_timestamp": 10.0,
                "x": 0.30,
                "y": 0.40,
                "z": 0.0,
                "frame_id": "table",
            },
            {
                "source_id": source_id,
                "episode_id": 0,
                "timestamp": 11.0,
                "t_rel_sec": 1.0,
                "header_timestamp": 11.0,
                "x": 0.20,
                "y": 0.60,
                "z": 0.0,
                "frame_id": "table",
            },
            {
                "source_id": source_id,
                "episode_id": 0,
                "timestamp": 12.0,
                "t_rel_sec": 2.0,
                "header_timestamp": 12.0,
                "x": 0.10,
                "y": 0.80,
                "z": 0.0,
                "frame_id": "table",
            },
        ]
    )
    ball.to_csv(d / "ball_position_table.csv", index=False)

    if with_classic:
        classic = pd.DataFrame(
            [
                {
                    "source_id": source_id,
                    "episode_id": 0,
                    "timestamp": 10.4,
                    "t_rel_sec": 0.4,
                    "s": -0.05,
                },
                {
                    "source_id": source_id,
                    "episode_id": 0,
                    "timestamp": 11.4,
                    "t_rel_sec": 1.4,
                    "s": 0.03,
                },
            ]
        )
    else:
        classic = pd.DataFrame([], columns=["source_id", "episode_id", "timestamp", "t_rel_sec", "s"])
    classic.to_csv(d / "classic_selected_goto_s.csv", index=False)
    return d


def test_two_sources_can_share_local_episode_id_zero_without_collision(tmp_path: Path):
    d1 = _write_dataset_csv(tmp_path, "source_a")
    d2 = _write_dataset_csv(tmp_path, "source_b")

    combined = pid.combine_sources([str(d1), str(d2)])
    assert len(combined.manifest) == 2
    assert sorted(combined.manifest["global_episode_id"].tolist()) == [0, 1]
    assert len(combined.ball) == 6
    assert set(combined.ball["source_id"].unique()) == {"source_a", "source_b"}


def test_duplicate_source_ids_raise_clear_error(tmp_path: Path):
    d1 = _write_dataset_csv(tmp_path / "a", "dup_source")
    d2 = _write_dataset_csv(tmp_path / "b", "dup_source")
    try:
        pid.combine_sources([str(d1), str(d2)])
        assert False, "Expected duplicate source_id error"
    except RuntimeError as exc:
        msg = str(exc)
        assert "Duplicate source_id" in msg
        assert "--source-id" in msg


def test_global_episode_id_is_deterministic_and_contiguous(tmp_path: Path):
    d1 = _write_dataset_csv(tmp_path, "s1")
    d2 = _write_dataset_csv(tmp_path, "s2")

    combined1 = pid.combine_sources([str(d1), str(d2)])
    combined2 = pid.combine_sources([str(d1), str(d2)])
    assert combined1.manifest["global_episode_id"].tolist() == [0, 1]
    assert combined2.manifest["global_episode_id"].tolist() == [0, 1]
    assert len(combined1.ball) == 6
    assert len(combined1.classic) == 4


def test_plotter_accepts_dataset_csv_and_parent_bag_paths(tmp_path: Path):
    d1 = _write_dataset_csv(tmp_path, "src1")
    bag2 = tmp_path / "bag_parent"
    d2 = _write_dataset_csv(bag2, "src2")
    combined = pid.combine_sources([str(d1), str(bag2 / "src2")])
    assert len(combined.manifest) == 2


def test_plotter_module_does_not_import_rosbag_code():
    source = inspect.getsource(pid)
    assert "rosbag2_py" not in source


def test_no_combined_csv_is_written(tmp_path: Path):
    d1 = _write_dataset_csv(tmp_path, "src1")
    d2 = _write_dataset_csv(tmp_path, "src2")
    out_dir = tmp_path / "plots_out"

    args = types.SimpleNamespace(
        inputs=[str(d1), str(d2)],
        out_dir=str(out_dir),
        formats=["png"],
        dpi=100,
        color_mode="episode",
        show_classic_goto=True,
        classic_selection="last",
        mark_events=["all"],
        motion_speed_threshold_mps=0.1,
        motion_min_consecutive=2,
        motion_smoothing_window=3,
        motion_min_displacement_m=0.0,
        vision_intercept_mode="max-y",
        line_center_x=0.3,
        middle_line_y=0.6,
        s_sign=-1.0,
        table_x_limits=[0.0, 0.6],
        table_y_limits=[0.0, 1.2],
        interactive=False,
    )

    combined = pid.combine_sources(args.inputs)
    events = pid._resolve_enabled_events(args.mark_events)
    classic_map = pid._build_episode_classic_map(combined.classic, args.classic_selection)
    event_info, _stats = pid._collect_episode_events(combined, classic_map, args, events)

    out_dir.mkdir(parents=True, exist_ok=True)
    xy = out_dir / "ball_trajectory_xy_overlay.png"
    st = out_dir / "ball_lateral_s_vs_time.png"
    pid._plot_xy_overlay(combined, event_info, classic_map, events, args, [xy])
    pid._plot_s_time_overlay(combined, event_info, classic_map, events, args, [st])

    assert xy.exists()
    assert st.exists()
    assert not any(p.name.endswith(".csv") for p in out_dir.iterdir())


def test_table_x_to_s_and_inverse_agree_both_signs():
    x = np.array([0.1, 0.3, 0.5], dtype=float)
    for sign in [-1.0, 1.0]:
        s = table_x_to_s(x, line_center_x=0.3, s_sign=sign)
        x_back = s_to_table_x(s, line_center_x=0.3, s_sign=sign)
        assert np.allclose(x, x_back)


def test_classic_target_coordinate_is_correct_for_both_signs():
    s = np.array([-0.2, 0.0, 0.2], dtype=float)
    x_neg = s_to_table_x(s, line_center_x=0.3, s_sign=-1.0)
    x_pos = s_to_table_x(s, line_center_x=0.3, s_sign=1.0)
    assert np.allclose(x_neg, np.array([0.5, 0.3, 0.1]))
    assert np.allclose(x_pos, np.array([0.1, 0.3, 0.5]))


def test_first_last_classic_selection_is_deterministic():
    rows = pd.DataFrame(
        [
            {"timestamp": 2.0, "s": 0.2},
            {"timestamp": 1.0, "s": -0.1},
            {"timestamp": 3.0, "s": 0.4},
        ]
    )
    first = pick_representative_classic_row(rows, "first")
    last = pick_representative_classic_row(rows, "last")
    assert float(first["timestamp"]) == 1.0
    assert float(last["timestamp"]) == 3.0


def test_motion_onset_detects_stationary_then_moving_trajectory():
    t = np.linspace(0.0, 1.0, 11)
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.08, 0.11, 0.14, 0.17])
    y = np.zeros_like(x)
    onset = detect_motion_onset_time(
        timestamps=t,
        x=x,
        y=y,
        speed_threshold_mps=0.1,
        min_consecutive=2,
        smoothing_window=1,
        min_displacement_m=0.0,
    )
    assert onset is not None
    assert onset >= 0.5


def test_motion_jitter_below_threshold_has_no_false_onset():
    t = np.linspace(0.0, 1.0, 11)
    x = np.array([0.0, 0.001, -0.001, 0.0005, -0.0002, 0.0, 0.0007, -0.0006, 0.0, 0.0002, 0.0])
    y = np.zeros_like(x)
    onset = detect_motion_onset_time(
        timestamps=t,
        x=x,
        y=y,
        speed_threshold_mps=0.1,
        min_consecutive=3,
        smoothing_window=3,
        min_displacement_m=0.01,
    )
    assert onset is None


def test_max_y_time_is_correct():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 0.0, 0.0, 0.0])
    y = np.array([0.1, 0.5, 0.9, 0.4])
    tt, _x, fallback = detect_max_y_intercept(t, x, y, motion_onset_time=0.5)
    assert tt == 2.0
    assert fallback is False


def test_middle_line_crossing_interpolates_time_and_position():
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([0.4, 0.3, 0.2])
    y = np.array([0.4, 0.5, 0.7])
    tt, xx = detect_middle_line_crossing(
        t,
        x,
        y,
        middle_line_y=0.6,
        motion_onset_time=0.0,
    )
    assert np.isclose(tt, 1.5)
    assert np.isclose(xx, 0.25)


def test_missing_middle_line_crossing_returns_unavailable():
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([0.1, 0.1, 0.1])
    y = np.array([0.1, 0.2, 0.3])
    tt, xx = detect_middle_line_crossing(
        t,
        x,
        y,
        middle_line_y=0.6,
        motion_onset_time=0.0,
    )
    assert tt is None
    assert xx is None


def test_event_position_interpolation_does_not_extrapolate():
    t = np.array([1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    assert interpolate_position_at_time(t, x, y, 0.9) is None
    assert interpolate_position_at_time(t, x, y, 3.1) is None
    mid = interpolate_position_at_time(t, x, y, 1.5)
    assert mid is not None
    assert np.allclose(mid, (0.5, 0.5))


def test_s_time_draw_end_time_prefers_max_y():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.4, 0.3, 0.2, 0.1])
    y = np.array([0.2, 0.7, 0.6, 0.4])
    draw_end, mode = pid._compute_s_time_draw_end_time(
        timestamps=t,
        x=x,
        y=y,
        motion_onset_time=0.0,
        middle_line_y=0.6,
    )
    assert np.isclose(draw_end, 1.0)
    assert mode == "max-y"


def test_s_time_draw_end_time_falls_back_to_middle_line_when_max_y_unavailable():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.4, 0.3, 0.2, 0.1])
    y = np.array([0.1, 0.3, 0.7, 0.9])
    # onset after the last sample makes max-y unavailable in post-onset window.
    draw_end, mode = pid._compute_s_time_draw_end_time(
        timestamps=t,
        x=x,
        y=y,
        motion_onset_time=4.0,
        middle_line_y=0.6,
    )
    assert np.isclose(draw_end, 1.75)
    assert mode == "middle-line-fallback"


def test_episode_color_normalization_spans_combined_dataset():
    norm = pid.build_episode_color_normalization(total_episode_count=4)
    assert np.isclose(norm(0), 0.0)
    assert np.isclose(norm(3), 1.0)
    # Global id from a second source still maps into the same global normalization.
    assert np.isclose(norm(2), 2.0 / 3.0)
