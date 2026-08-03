from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd

from extract_intercept_dataset_csv import (
    ExtractedRows,
    _collect_topic_rows,
    extraction_filter_topics,
    write_output_atomic,
)
from intercept_dataset_common import (
    EventLogEntry,
    EpisodeWindow,
    assign_timestamps_to_episode_ids,
    parse_episode_control_events,
)


class FakeReader:
    def __init__(self, rows):
        self.rows = list(rows)
        self.idx = 0

    def has_next(self):
        return self.idx < len(self.rows)

    def read_next(self):
        row = self.rows[self.idx]
        self.idx += 1
        return row


def _ns(t_sec: float) -> int:
    return int(round(t_sec * 1e9))


def _ball_msg(x: float, y: float, z: float, frame_id: str = "table"):
    return types.SimpleNamespace(
        point=types.SimpleNamespace(x=float(x), y=float(y), z=float(z)),
        header=types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=int(x), nanosec=0),
            frame_id=frame_id,
        ),
    )


def _classic_msg(s: float):
    return types.SimpleNamespace(data=float(s))


def test_episode_normal_start_stop_commits_one_window():
    windows, _warnings = parse_episode_control_events(
        [
            EventLogEntry(1.0, 1),
            EventLogEntry(2.0, 2),
        ]
    )
    assert len(windows) == 1
    assert windows[0].episode_id == 0
    assert windows[0].start_timestamp == 1.0
    assert windows[0].end_timestamp == 2.0


def test_cancel_current_discards_unfinished_episode():
    windows, _warnings = parse_episode_control_events(
        [
            EventLogEntry(1.0, 1),
            EventLogEntry(1.5, 3),
            EventLogEntry(2.0, 2),
        ]
    )
    assert windows == []


def test_cancel_last_removes_last_committed_episode():
    windows, _warnings = parse_episode_control_events(
        [
            EventLogEntry(1.0, 1),
            EventLogEntry(2.0, 2),
            EventLogEntry(3.0, 1),
            EventLogEntry(4.0, 2),
            EventLogEntry(5.0, 4),
        ]
    )
    assert len(windows) == 1
    assert windows[0].start_timestamp == 1.0
    assert windows[0].end_timestamp == 2.0
    assert windows[0].episode_id == 0


def test_unfinished_final_episode_is_discarded():
    windows, _warnings = parse_episode_control_events(
        [
            EventLogEntry(1.0, 1),
            EventLogEntry(2.0, 2),
            EventLogEntry(3.0, 1),
        ]
    )
    assert len(windows) == 1
    assert windows[0].episode_id == 0


def test_surviving_episodes_receive_contiguous_local_ids():
    windows, _warnings = parse_episode_control_events(
        [
            EventLogEntry(0.0, 1),
            EventLogEntry(1.0, 2),
            EventLogEntry(2.0, 1),
            EventLogEntry(3.0, 2),
            EventLogEntry(4.0, 4),
        ]
    )
    assert [w.episode_id for w in windows] == [0]


def test_assignment_includes_boundaries_and_excludes_gap_and_canceled_window():
    windows, _warnings = parse_episode_control_events(
        [
            EventLogEntry(0.0, 1),
            EventLogEntry(1.0, 2),
            EventLogEntry(2.0, 1),
            EventLogEntry(2.5, 3),
            EventLogEntry(4.0, 1),
            EventLogEntry(5.0, 2),
        ]
    )
    timestamps = [0.0, 1.0, 1.5, 2.2, 4.0, 5.0, 5.1]
    assigned = assign_timestamps_to_episode_ids(timestamps, windows)
    assert assigned == [0, 0, None, None, 1, 1, None]


def test_collect_rows_preserves_raw_samples_relative_time_and_multiple_classic():
    windows = [EpisodeWindow(episode_id=0, start_timestamp=1.0, end_timestamp=2.0)]
    rows = [
        ("/ball", _ball_msg(0.1, 0.2, 0.3), _ns(1.0)),
        ("/classic", _classic_msg(-0.1), _ns(1.2)),
        ("/ball", _ball_msg(0.2, 0.3, 0.4), _ns(1.5)),
        ("/classic", _classic_msg(0.2), _ns(1.7)),
        ("/ball", _ball_msg(0.4, 0.5, 0.6), _ns(2.0)),
        ("/ball", _ball_msg(0.9, 1.0, 1.1), _ns(2.5)),
    ]
    reader = FakeReader(rows)

    ball_df, classic_df = _collect_topic_rows(
        reader=reader,
        deserialize_message=lambda raw, _cls: raw,
        msg_classes={"/ball": object(), "/classic": object()},
        windows=windows,
        source_id="src_a",
        ball_topic="/ball",
        classic_topic="/classic",
        progress_every=1000,
    )

    # Start/end boundaries are included and no resampling occurred.
    assert list(ball_df["timestamp"]) == [1.0, 1.5, 2.0]
    assert list(ball_df["t_rel_sec"]) == [0.0, 0.5, 1.0]

    # All classical messages in the window are preserved.
    assert list(classic_df["s"]) == [-0.1, 0.2]
    assert np.allclose(classic_df["timestamp"].to_numpy(dtype=float), np.array([1.2, 1.7]))


def test_empty_classic_data_writes_header_only_csv(tmp_path: Path):
    bag_dir = tmp_path / "bag_dir"
    bag_dir.mkdir()

    rows = ExtractedRows(
        manifest=pd.DataFrame(
            [
                {
                    "source_id": "src",
                    "episode_id": 0,
                    "start_timestamp": 1.0,
                    "end_timestamp": 2.0,
                    "duration_sec": 1.0,
                    "ball_sample_count": 0,
                    "classic_goto_s_count": 0,
                    "ball_frame_id": "",
                    "valid": False,
                    "invalid_reason": "insufficient_ball_samples",
                }
            ]
        ),
        ball=pd.DataFrame(
            [],
            columns=[
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
        ),
        classic=pd.DataFrame(
            [],
            columns=["source_id", "episode_id", "timestamp", "t_rel_sec", "s"],
        ),
        metadata={"source_id": "src"},
    )

    args = types.SimpleNamespace(
        path_to_target_bag=str(bag_dir),
        out_dir=None,
        force=False,
    )
    out_dir = write_output_atomic(args, rows)
    classic_csv = out_dir / "classic_selected_goto_s.csv"
    text = classic_csv.read_text(encoding="utf-8").strip()
    assert text == "source_id,episode_id,timestamp,t_rel_sec,s"


def test_extractor_storage_filters_include_only_required_topics():
    pass1, pass2 = extraction_filter_topics(
        episode_topic="/episode/control",
        ball_topic="/scene_localizer/top_cam/ball_3d_table",
        classic_goto_topic="/interception_controller/selected_goto_s",
    )
    assert pass1 == ["/episode/control"]
    assert pass2 == [
        "/scene_localizer/top_cam/ball_3d_table",
        "/interception_controller/selected_goto_s",
    ]
    all_topics = set(pass1 + pass2)
    assert "/top_cam/camera/color/image_raw" not in all_topics
    assert "/openmv_cam/image" not in all_topics
