from __future__ import annotations

import inspect
import types
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from intercept_dataset_common import (
    classify_lateral_intercept,
    detect_max_y_intercept,
    detect_middle_line_crossing,
    detect_motion_onset_time,
    directional_count_bias,
    interpolate_approach_on_y_grid,
    interpolate_position_at_time,
    pick_representative_classic_row,
    sample_descriptive_statistics,
    select_monotonic_approach_points,
    s_to_table_x,
    summarize_episode_grid,
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


def test_plotter_accepts_recording_parent_with_one_nested_bag(tmp_path: Path):
    direct = _write_dataset_csv(tmp_path / "direct", "src1")
    recording_parent = tmp_path / "recording_parent"
    _write_dataset_csv(recording_parent, "recording_nested_bag")
    combined = pid.combine_sources([str(direct), str(recording_parent)])
    assert len(combined.manifest) == 2
    assert set(combined.manifest["source_id"]) == {"src1", "recording_nested_bag"}


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


def test_l_r_deadband_classification_uses_strict_boundaries():
    epsilon = 0.005
    assert classify_lateral_intercept(0.006, epsilon) == "L"
    assert classify_lateral_intercept(-0.006, epsilon) == "R"
    assert classify_lateral_intercept(0.0, epsilon) == "deadband"
    assert classify_lateral_intercept(epsilon, epsilon) == "deadband"
    assert classify_lateral_intercept(-epsilon, epsilon) == "deadband"


def test_directional_bias_and_descriptive_statistics():
    assert np.isclose(directional_count_bias(6, 4), 0.2)
    assert np.isnan(directional_count_bias(0, 0))
    stats = sample_descriptive_statistics([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert np.isclose(stats["mean"], 2.5)
    assert np.isclose(stats["sample_std"], np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert np.isclose(stats["median"], 2.5)
    assert np.isclose(stats["q25"], 1.75)
    assert np.isclose(stats["q75"], 3.25)
    assert np.isclose(stats["iqr"], 1.5)


def test_monotonic_approach_drops_duplicates_and_mild_backsteps_deterministically():
    y = np.array([0.10, 0.20, 0.20, 0.1997, 0.30, 0.40])
    s = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    selected_y, selected_s = select_monotonic_approach_points(y, s)
    assert np.allclose(selected_y, [0.10, 0.20, 0.30, 0.40])
    assert np.allclose(selected_s, [0.00, 0.01, 0.04, 0.05])


def test_common_y_grid_interpolation_does_not_extrapolate():
    grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    values = interpolate_approach_on_y_grid(
        y=[0.1, 0.3], s=[-0.1, 0.1], y_grid=grid
    )
    assert np.isnan(values[0])
    assert np.allclose(values[1:4], [-0.1, 0.0, 0.1])
    assert np.isnan(values[4])


def test_episode_grid_statistics_weight_episodes_not_raw_samples():
    # Each row is already one episode, regardless of its original sample count/timing.
    values = np.array([[0.0, 1.0, np.nan], [2.0, 3.0, 4.0]])
    stats = summarize_episode_grid(values)
    assert stats["count"].tolist() == [2, 2, 1]
    assert np.allclose(stats["mean"], [1.0, 2.0, 4.0])
    assert np.allclose(stats["sample_std"][:2], [np.sqrt(2.0), np.sqrt(2.0)])
    assert np.isnan(stats["sample_std"][2])
    assert np.allclose(stats["q25"], [0.5, 1.5, 4.0])
    assert np.allclose(stats["q75"], [1.5, 2.5, 4.0])


def test_unequal_sample_counts_and_timing_still_give_one_value_per_episode():
    grid = np.array([0.1, 0.2, 0.3, 0.4])
    sparse = interpolate_approach_on_y_grid(
        y=[0.1, 0.4], s=[0.0, 0.3], y_grid=grid
    )
    dense_irregular = interpolate_approach_on_y_grid(
        y=[0.1, 0.13, 0.24, 0.31, 0.4],
        s=[0.2, 0.23, 0.34, 0.41, 0.5],
        y_grid=grid,
    )
    stats = summarize_episode_grid(np.vstack([sparse, dense_irregular]))
    assert stats["count"].tolist() == [2, 2, 2, 2]
    assert np.allclose(stats["mean"], [0.1, 0.2, 0.3, 0.4])


def test_analysis_records_episode_without_crossing_as_excluded():
    manifest = pd.DataFrame(
        [
            {
                "source_id": "source_a",
                "episode_id": 0,
                "global_episode_id": 0,
                "start_timestamp": 0.0,
                "end_timestamp": 1.0,
                "duration_sec": 1.0,
                "ball_sample_count": 6,
                "classic_goto_s_count": 0,
                "ball_frame_id": "table",
                "valid": True,
                "invalid_reason": "",
            }
        ]
    )
    ball = pd.DataFrame(
        {
            "source_id": ["source_a"] * 6,
            "episode_id": [0] * 6,
            "global_episode_id": [0] * 6,
            "timestamp": np.linspace(0.0, 1.0, 6),
            "t_rel_sec": np.linspace(0.0, 1.0, 6),
            "header_timestamp": np.linspace(0.0, 1.0, 6),
            "x": np.linspace(0.3, 0.2, 6),
            "y": np.linspace(0.1, 0.4, 6),
            "z": np.zeros(6),
            "frame_id": ["table"] * 6,
        }
    )
    combined = pid.CombinedDataset(
        manifest=manifest,
        ball=ball,
        classic=pd.DataFrame(),
        source_summaries=[],
        dataset_dirs=[],
        source_bags={"source_a": "/bags/source_a"},
    )
    args = types.SimpleNamespace(
        motion_speed_threshold_mps=0.1,
        motion_min_consecutive=2,
        motion_smoothing_window=1,
        motion_min_displacement_m=0.0,
        middle_line_y=0.6,
        line_center_x=0.3,
        s_sign=-1.0,
        lr_deadband_epsilon_m=0.005,
    )
    analysis = pid.analyze_trajectory_distribution(combined, args)
    assert len(analysis.episodes) == 1
    assert not bool(analysis.episodes.iloc[0]["included"])
    assert analysis.episodes.iloc[0]["exclusion_reason"] == "middle_line_crossing_unavailable"
    assert int(analysis.summary.iloc[0]["excluded_episodes"]) == 1


def test_distribution_figure_and_csv_outputs_are_created_and_closed(tmp_path: Path):
    episodes = pd.DataFrame(
        [
            {"source_bag": "/a", "source_id": "a", "episode_id": 0, "global_episode_id": 0, "s_onset": 0.0, "s_int": 0.05, "delta_s": 0.05, "classification": "L", "included": True, "exclusion_reason": ""},
            {"source_bag": "/b", "source_id": "b", "episode_id": 0, "global_episode_id": 1, "s_onset": 0.0, "s_int": -0.04, "delta_s": -0.04, "classification": "R", "included": True, "exclusion_reason": ""},
            {"source_bag": "/b", "source_id": "b", "episode_id": 1, "global_episode_id": 2, "s_onset": 0.0, "s_int": 0.0, "delta_s": 0.0, "classification": "deadband", "included": True, "exclusion_reason": ""},
        ]
    )
    summary = pd.DataFrame(
        [{"scope": "combined", "source_id": "", "source_bag": "", "usable_episodes": 3, "excluded_episodes": 0, "N_L": 1, "N_R": 1, "N_deadband": 1, "directional_count_bias_B": 0.0}]
    )
    grid_rows = []
    for group, offset in [("L", 0.05), ("R", -0.04)]:
        for y_value in np.linspace(0.1, 0.6, 5):
            grid_rows.append(
                {"scope": "combined", "source_id": "", "classification": group, "y": y_value, "contributing_episode_count": 1, "mean_s": offset, "sample_std_s": np.nan, "q25_s": offset, "q75_s": offset}
            )
    analysis = pid.TrajectoryDistributionAnalysis(
        episodes=episodes,
        summary=summary,
        mean_grid=pd.DataFrame(grid_rows),
        raw_segments={
            ("a", 0): (np.array([0.0, 0.05]), np.array([0.1, 0.6])),
            ("b", 0): (np.array([0.0, -0.04]), np.array([0.1, 0.6])),
            ("b", 1): (np.array([0.0, 0.0]), np.array([0.1, 0.6])),
        },
    )
    args = types.SimpleNamespace(
        table_x_limits=[0.0, 0.6], table_y_limits=[0.0, 0.8], line_center_x=0.3,
        s_sign=-1.0, middle_line_y=0.6, lr_deadband_epsilon_m=0.005, dpi=80,
    )
    figure_path = tmp_path / "distribution.png"
    pid._plot_trajectory_distribution(analysis, args, [figure_path])
    filenames = pid.write_trajectory_distribution_outputs(tmp_path, analysis)
    assert figure_path.exists()
    assert set(filenames) == {
        "ball_trajectory_distribution_lr_summary.csv",
        "ball_trajectory_distribution_lr_episodes.csv",
        "ball_trajectory_distribution_lr_mean_grid.csv",
    }
    episode_output = pd.read_csv(tmp_path / "ball_trajectory_distribution_lr_episodes.csv")
    assert {"source_bag", "episode_id", "s_onset", "s_int", "delta_s", "classification", "included", "exclusion_reason"}.issubset(episode_output.columns)
    assert pid.plt.get_fignums() == []
