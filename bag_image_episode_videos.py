#!/usr/bin/env python3
"""Create one side-by-side RGB/event-frame video per committed bag episode."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_intercept_plots.analysis.ball_reference import (  # noqa: E402
    ball_max_y_time_clipped_by_middle_line,
)
from ball_intercept_plots.geometry import derive_middle_line_intersection_table  # noqa: E402
from bag_to_csv import (  # noqa: E402
    TOPIC_EPISODE,
    EpisodeWindow,
    detect_storage_id_from_metadata,
    extract_episode_windows,
    get_topic_type_map,
    open_reader,
)


RGB_TOPIC = "/top_cam/camera/color/image_raw"
EVENT_TOPIC = "/openmv_cam/event_frame_3ch"
BALL_TOPIC_FILE = "scene_localizer__top_cam__ball_3d_table.csv"
GOTO_TOPIC_FILES = (
    "trajectory_executor__executed_goto_s.csv",
    "interception_controller__selected_goto_s.csv",
)
MIDDLE_TOPIC_FILE = "scene__middle_line_intersection_pose_robot_base.csv"
TABLE_POSE_TOPIC_FILE = "scene_localizer__table_pose_robot_base.csv"


@dataclass(frozen=True)
class TimeMarks:
    goto_frame: Optional[int] = None
    intercept_frame: Optional[int] = None


def log(message: str) -> None:
    print(message, flush=True)


def image_to_bgr(msg: Any) -> np.ndarray:
    """Convert common sensor_msgs/Image encodings without requiring cv_bridge."""
    encoding = str(msg.encoding).lower()
    channels_by_encoding = {
        "mono8": 1,
        "8uc1": 1,
        "rgb8": 3,
        "bgr8": 3,
        "8uc3": 3,
        "rgba8": 4,
        "bgra8": 4,
        "8uc4": 4,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")

    row_bytes = int(msg.width) * channels
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.step) * int(msg.height)
    if raw.size < expected or int(msg.step) < row_bytes:
        raise ValueError("malformed sensor_msgs/Image data/step")
    image = raw[:expected].reshape(int(msg.height), int(msg.step))[:, :row_bytes]
    image = image.reshape(int(msg.height), int(msg.width), channels) if channels > 1 else image

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image)


def episode_for_time(t: float, windows: list[EpisodeWindow]) -> Optional[int]:
    for pos, window in enumerate(windows):
        if window.start <= t <= window.end:
            return pos
        if t < window.start:
            break
    return None


def scan_rgb_timestamps(
    bag: Path,
    storage_id: str,
    windows: list[EpisodeWindow],
    rgb_topic: str,
) -> list[list[float]]:
    result: list[list[float]] = [[] for _ in windows]
    reader = open_reader(str(bag), storage_id)
    while reader.has_next():
        topic, _raw, t_ns = reader.read_next()
        if topic != rgb_topic:
            continue
        t = float(t_ns) * 1e-9
        pos = episode_for_time(t, windows)
        if pos is not None:
            result[pos].append(t)
    return result


def read_csv_episode(path: Path, episode_idx: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "episode_idx" not in df.columns:
        return pd.DataFrame()
    idx = pd.to_numeric(df["episode_idx"], errors="coerce")
    return df.loc[idx == episode_idx].copy()


def relative_times(df: pd.DataFrame, episode_start: float) -> np.ndarray:
    if "t_episode" in df.columns:
        return pd.to_numeric(df["t_episode"], errors="coerce").to_numpy(dtype=float)
    if "t_abs" in df.columns:
        return pd.to_numeric(df["t_abs"], errors="coerce").to_numpy(dtype=float) - episode_start
    return np.asarray([], dtype=float)


def nearest_frame(rgb_times: list[float], absolute_time: Optional[float]) -> Optional[int]:
    if not rgb_times or absolute_time is None or not math.isfinite(absolute_time):
        return None
    return int(np.argmin(np.abs(np.asarray(rgb_times) - absolute_time))) + 1


def compute_marks(
    csv_dir: Path,
    window: EpisodeWindow,
    rgb_times: list[float],
) -> TimeMarks:
    goto = pd.DataFrame()
    goto_source: Optional[str] = None
    for filename in GOTO_TOPIC_FILES:
        candidate = read_csv_episode(csv_dir / filename, window.idx)
        if not candidate.empty:
            goto = candidate
            goto_source = filename
            break
    goto_rel = relative_times(goto, window.start)
    goto_rel = goto_rel[np.isfinite(goto_rel)]
    goto_abs = window.start + float(goto_rel[0]) if goto_rel.size else None

    ball = read_csv_episode(csv_dir / BALL_TOPIC_FILE, window.idx)
    ball_t = relative_times(ball, window.start)
    if "point_y" in ball.columns:
        ball_y = pd.to_numeric(ball["point_y"], errors="coerce").to_numpy(dtype=float)
    else:
        ball_y = np.asarray([], dtype=float)
    finite = np.isfinite(ball_t) & np.isfinite(ball_y) if ball_t.size == ball_y.size else np.zeros(0, dtype=bool)
    ball_t, ball_y = ball_t[finite], ball_y[finite]

    middle_y_m: Optional[float] = None
    middle = read_csv_episode(csv_dir / MIDDLE_TOPIC_FILE, window.idx)
    table_pose = read_csv_episode(csv_dir / TABLE_POSE_TOPIC_FILE, window.idx)
    if not middle.empty and not table_pose.empty:
        try:
            derived = derive_middle_line_intersection_table(middle, table_pose)
            values = pd.to_numeric(derived.get("y_table"), errors="coerce").dropna()
            if not values.empty:
                middle_y_m = float(values.median())
        except (RuntimeError, ValueError, KeyError) as exc:
            log(f"[WARNING] episode {window.idx}: could not derive middle line: {exc}")

    intercept_rel = ball_max_y_time_clipped_by_middle_line(
        ball_t, ball_y, middle_line_y_m=middle_y_m
    )
    intercept_abs = window.start + intercept_rel if intercept_rel is not None else None
    marks = TimeMarks(
        goto_frame=nearest_frame(rgb_times, goto_abs),
        intercept_frame=nearest_frame(rgb_times, intercept_abs),
    )
    if marks.goto_frame is None:
        log(
            f"[WARNING] episode {window.idx}: no classical GOTO_S timestamp found "
            f"in {', '.join(GOTO_TOPIC_FILES)}"
        )
    else:
        log(
            f"[INFO] episode {window.idx}: classical GOTO_S frame "
            f"{marks.goto_frame} from {goto_source}"
        )
    if marks.intercept_frame is None:
        log(f"[WARNING] episode {window.idx}: no estimated-intercept frame could be computed")
    return marks


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Resize without letterboxing, preserving the image's aspect ratio."""
    scale = height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )


def overlay_text(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    line_h = 23
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    cv2.rectangle(frame, (8, 7), (20 + width, 13 + line_h * len(lines)), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (14, 27 + idx * line_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_videos(
    bag: Path,
    storage_id: str,
    windows: list[EpisodeWindow],
    rgb_times: list[list[float]],
    marks: list[TimeMarks],
    *,
    rgb_topic: str,
    event_topic: str,
    fps: float,
    out_dir: Path,
    rgb_cls: Any,
    event_cls: Any,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: list[Optional[cv2.VideoWriter]] = [None] * len(windows)
    latest_event: list[Optional[np.ndarray]] = [None] * len(windows)
    pending_rgb: list[list[np.ndarray]] = [[] for _ in windows]
    counts = [0] * len(windows)
    output_paths = [out_dir / f"episode_{w.idx:03d}_rgb_event.mp4" for w in windows]
    reader = open_reader(str(bag), storage_id)

    def write_frame(pos: int, rgb: np.ndarray, event: np.ndarray) -> None:
        panel_h = rgb.shape[0]
        rotated_event = cv2.rotate(event, cv2.ROTATE_90_COUNTERCLOCKWISE)
        divider = np.zeros((panel_h, 4, 3), dtype=np.uint8)
        combined = np.hstack(
            (
                resize_to_height(rgb, panel_h),
                divider,
                resize_to_height(rotated_event, panel_h),
            )
        )
        counts[pos] += 1
        lines = [f"{counts[pos]}/{len(rgb_times[pos])} Frames"]
        if marks[pos].intercept_frame is not None:
            lines.append(f"estimated intercept frame: {marks[pos].intercept_frame}")
        if marks[pos].goto_frame is not None:
            lines.append(f"classical GOTO_S frame: {marks[pos].goto_frame}")
        overlay_text(combined, lines)

        if writers[pos] is None:
            height, width = combined.shape[:2]
            writers[pos] = cv2.VideoWriter(
                str(output_paths[pos]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writers[pos].isOpened():
                raise RuntimeError(f"could not open video writer: {output_paths[pos]}")
        writers[pos].write(combined)

    try:
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            if topic not in {rgb_topic, event_topic}:
                continue
            pos = episode_for_time(float(t_ns) * 1e-9, windows)
            if pos is None:
                continue
            if topic == event_topic:
                event = image_to_bgr(deserialize_message(raw, event_cls))
                latest_event[pos] = event
                if pending_rgb[pos]:
                    for pending in pending_rgb[pos]:
                        write_frame(pos, pending, event)
                    pending_rgb[pos].clear()
                continue

            rgb = image_to_bgr(deserialize_message(raw, rgb_cls))
            event = latest_event[pos]
            if event is None:
                pending_rgb[pos].append(rgb)
                continue
            write_frame(pos, rgb, event)
    finally:
        for writer in writers:
            if writer is not None:
                writer.release()

    for pos, path in enumerate(output_paths):
        if counts[pos]:
            log(f"[INFO] wrote {path} ({counts[pos]} frames at {fps:g} Hz)")
        else:
            log(f"[WARNING] episode {windows[pos].idx} contains no RGB frames; no video written")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Path to a ROS 2 bag directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video frame rate (default: 30)")
    parser.add_argument("--num-episodes", type=int, default=None, help="Only render the first M committed episodes")
    timing_group = parser.add_mutually_exclusive_group()
    timing_group.add_argument(
        "--mark-times",
        dest="mark_times",
        action="store_true",
        help="Overlay timing frame numbers (default)",
    )
    timing_group.add_argument(
        "--no-mark-times",
        dest="mark_times",
        action="store_false",
        help="Only overlay the frame counter",
    )
    parser.set_defaults(mark_times=True)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="bag_to_csv.py output directory (default with --mark-times: BAG/csv)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: BAG/videos)")
    parser.add_argument("--storage-id", default=None, help="Override rosbag2 storage identifier")
    parser.add_argument("--episode-topic", default=TOPIC_EPISODE)
    parser.add_argument("--rgb-topic", default=RGB_TOPIC)
    parser.add_argument("--event-topic", default=EVENT_TOPIC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.num_episodes is not None and args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    bag = args.bag.expanduser().resolve()
    if not bag.is_dir():
        raise ValueError(f"bag path is not a directory: {bag}")
    storage_id = args.storage_id or detect_storage_id_from_metadata(str(bag))

    all_windows, marker_count = extract_episode_windows(str(bag), storage_id, args.episode_topic)
    if marker_count == 0 or not all_windows:
        raise RuntimeError("no committed episode windows found")
    windows = all_windows[: args.num_episodes]
    log(f"[INFO] rendering {len(windows)} of {len(all_windows)} committed episodes")

    reader = open_reader(str(bag), storage_id)
    topic_types = get_topic_type_map(reader)
    for topic in (args.rgb_topic, args.event_topic):
        if topic not in topic_types:
            raise RuntimeError(f"required image topic not found: {topic}")
        if topic_types[topic] != "sensor_msgs/msg/Image":
            raise RuntimeError(f"{topic} has unsupported type {topic_types[topic]}; expected sensor_msgs/msg/Image")

    rgb_times = scan_rgb_timestamps(bag, storage_id, windows, args.rgb_topic)
    time_marks = [TimeMarks() for _ in windows]
    if args.mark_times:
        csv_dir = args.csv_dir.expanduser().resolve() if args.csv_dir else bag / "csv"
        if not csv_dir.is_dir():
            log(
                f"[ERROR] timing CSV directory does not exist: {csv_dir}; "
                "creating videos without time annotations"
            )
        else:
            log(f"[INFO] using timing CSV directory: {csv_dir}")
            time_marks = [
                compute_marks(csv_dir, window, rgb_times[pos])
                for pos, window in enumerate(windows)
            ]

    render_videos(
        bag,
        storage_id,
        windows,
        rgb_times,
        time_marks,
        rgb_topic=args.rgb_topic,
        event_topic=args.event_topic,
        fps=args.fps,
        out_dir=(args.out_dir.expanduser().resolve() if args.out_dir else bag / "videos"),
        rgb_cls=get_message(topic_types[args.rgb_topic]),
        event_cls=get_message(topic_types[args.event_topic]),
    )


if __name__ == "__main__":
    main()
