#!/usr/bin/env python3
"""
Convert selected numeric ROS2 bag topics to CSV.

Typical usage:
    python3 bag_to_csv.py --bag /path/to/my_bag --out_dir /tmp/bag_csv
    python3 bag_to_csv.py --bag /path/to/my_bag --out_dir /tmp/bag_csv --use_episode_windows
    python3 bag_to_csv.py --bag /path/to/my_bag --out_dir /tmp/bag_csv \
        --topics /cartesian_cmd/twist /joint_states /teleop/gripper_state_cmd

Notes:
- This is meant for numeric plotting data, not image/event-frame data.
- sensor_msgs/Image topics are skipped by default.
- Run this inside a ROS2 environment where the relevant message packages are sourced.
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# Defaults copied from the current data-collection / HDF5 conversion workflow.
TOPIC_RGB = "/camera/camera/color/image_raw"
TOPIC_EVENT = "/openmv_cam/image"
TOPIC_JOINT = "/joint_states"
TOPIC_GRIPPER_STATE = "/teleop/gripper_state_cmd"
TOPIC_GRIPPER_CMD = "/teleop/gripper_cmd"
TOPIC_TWIST = "/cartesian_cmd/twist"
TOPIC_EPISODE = "/episode/control"

DEFAULT_TOPICS = [
    TOPIC_JOINT,
    TOPIC_GRIPPER_STATE,
    TOPIC_GRIPPER_CMD,
    TOPIC_TWIST,
    TOPIC_EPISODE,
]

IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}

FRANKA_ARM_JOINTS = [
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
]
FRANKA_FINGER_JOINTS = ["right_fr3_finger_joint1", "right_fr3_finger_joint2"]


@dataclass
class EpisodeWindow:
    idx: int
    start: float
    end: float


def log(msg: str) -> None:
    print(msg, flush=True)


def bag_timestamp_to_sec(ns: int) -> float:
    return float(ns) * 1e-9


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename_from_topic(topic: str) -> str:
    name = topic.strip("/") or "root"
    name = name.replace("/", "__")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return f"{name}.csv"


def safe_column_name(name: str) -> str:
    name = name.strip("/")
    name = name.replace("/", "__")
    name = name.replace(".", "_")
    name = name.replace("[", "_").replace("]", "")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return name.strip("_") or "field"


def open_reader(bag_path: str, storage_id: str = "sqlite3") -> rosbag2_py.SequentialReader:
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def get_topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    topic_types = reader.get_all_topics_and_types()
    return {x.name: x.type for x in topic_types}


def get_type_class_map(topic_type_map: Dict[str, str]) -> Dict[str, Any]:
    return {topic: get_message(msg_type) for topic, msg_type in topic_type_map.items()}


def print_topics(topic_type_map: Dict[str, str]) -> None:
    log("Available topics:")
    for topic in sorted(topic_type_map):
        log(f"  {topic:<45} {topic_type_map[topic]}")


def extract_episode_windows(
    bag_path: str,
    storage_id: str,
    episode_topic: str = TOPIC_EPISODE,
) -> List[EpisodeWindow]:
    """
    Scan /episode/control and apply the same start/stop/cancel semantics as the
    HDF5 converter:
      1 = start current episode
      2 = stop and commit current episode
      3 = cancel current unfinished episode
      4 = cancel last committed episode
    """
    log(f"[INFO] scanning {episode_topic} for episode boundaries")
    reader = open_reader(bag_path, storage_id=storage_id)
    topic_type_map = get_topic_type_map(reader)

    if episode_topic not in topic_type_map:
        raise RuntimeError(f"episode topic not found in bag: {episode_topic}")

    msg_types = get_type_class_map({episode_topic: topic_type_map[episode_topic]})
    current_start: Optional[float] = None
    committed_windows: List[EpisodeWindow] = []

    n_episode_msgs = 0
    n_start = 0
    n_stop = 0
    n_cancel_current = 0
    n_cancel_last = 0
    n_ignored = 0

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic != episode_topic:
            continue

        n_episode_msgs += 1
        msg = deserialize_message(raw, msg_types[topic])
        t = bag_timestamp_to_sec(t_ns)
        value = getattr(msg, "data", None)

        if value == 1:
            n_start += 1
            if current_start is None:
                current_start = t
            else:
                log("[WARNING] start received while already recording; ignoring duplicate start")

        elif value == 2:
            n_stop += 1
            if current_start is None:
                log("[WARNING] stop received while not recording; ignoring")
            else:
                committed_windows.append(
                    EpisodeWindow(idx=len(committed_windows), start=current_start, end=t)
                )
                current_start = None

        elif value == 3:
            n_cancel_current += 1
            if current_start is not None:
                current_start = None
            else:
                log("[WARNING] cancel_current received while not recording; ignoring")

        elif value == 4:
            n_cancel_last += 1
            if current_start is not None:
                log("[WARNING] cancel_last received while recording; ignoring")
            elif committed_windows:
                committed_windows.pop()
                for i, ep in enumerate(committed_windows):
                    ep.idx = i
            else:
                log("[WARNING] cancel_last received but no committed episode exists")

        else:
            n_ignored += 1
            log(f"[WARNING] unknown episode marker {value}; ignoring")

    if current_start is not None:
        log(f"[WARNING] bag ended during unfinished episode from {current_start:.9f}; discarding")

    log(
        "[INFO] episode marker counts: "
        f"messages={n_episode_msgs}, start={n_start}, stop={n_stop}, "
        f"cancel_current={n_cancel_current}, cancel_last={n_cancel_last}, ignored={n_ignored}"
    )

    return committed_windows


def filter_episode_windows(
    windows: Sequence[EpisodeWindow],
    min_duration: float,
    max_episodes: Optional[int],
) -> List[EpisodeWindow]:
    filtered: List[EpisodeWindow] = []
    for ep in windows:
        duration = ep.end - ep.start
        if duration < min_duration:
            log(f"[INFO] dropping episode {ep.idx}: too short ({duration:.3f}s)")
            continue
        filtered.append(EpisodeWindow(idx=len(filtered), start=ep.start, end=ep.end))

    if max_episodes is not None:
        filtered = filtered[:max_episodes]

    log(f"[INFO] using {len(filtered)} episode windows")
    for ep in filtered:
        log(f"       episode {ep.idx}: {ep.start:.9f} .. {ep.end:.9f}, dur={ep.end - ep.start:.3f}s")

    return filtered


def locate_episode(
    t: float,
    windows: Sequence[EpisodeWindow],
    start_idx: int,
) -> Tuple[Optional[EpisodeWindow], int]:
    """Return episode containing t, assuming bag timestamps are nondecreasing."""
    idx = start_idx
    n = len(windows)
    while idx < n and t > windows[idx].end:
        idx += 1
    if idx >= n:
        return None, idx
    ep = windows[idx]
    if ep.start <= t <= ep.end:
        return ep, idx
    return None, idx


def is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def is_scalar_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if not isinstance(value, (list, tuple)):
        return False
    return all(is_scalar(v) for v in value)


def add_header_fields(row: Dict[str, Any], msg: Any) -> None:
    header = getattr(msg, "header", None)
    if header is None:
        return
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        row["header_stamp"] = stamp_to_sec(stamp)
    frame_id = getattr(header, "frame_id", None)
    if frame_id:
        row["frame_id"] = frame_id


def flatten_twist_like(msg: Any) -> Optional[Dict[str, Any]]:
    """Handle geometry_msgs/Twist and geometry_msgs/TwistStamped."""
    row: Dict[str, Any] = {}
    add_header_fields(row, msg)

    twist = getattr(msg, "twist", msg)
    linear = getattr(twist, "linear", None)
    angular = getattr(twist, "angular", None)
    if linear is None or angular is None:
        return None

    row.update(
        {
            "linear_x": linear.x,
            "linear_y": linear.y,
            "linear_z": linear.z,
            "angular_x": angular.x,
            "angular_y": angular.y,
            "angular_z": angular.z,
        }
    )
    return row


def flatten_joint_state(
    msg: Any,
    joint_order: Optional[List[str]],
    include_velocity: bool,
    include_effort: bool,
    add_franka_qpos8: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    names = list(getattr(msg, "name", []))
    if not names:
        raise RuntimeError("JointState message has no names")

    if joint_order is None:
        joint_order = names

    name_to_idx = {name: i for i, name in enumerate(names)}
    positions = list(getattr(msg, "position", []))
    velocities = list(getattr(msg, "velocity", []))
    efforts = list(getattr(msg, "effort", []))

    row: Dict[str, Any] = {}
    add_header_fields(row, msg)

    for joint_name in joint_order:
        idx = name_to_idx.get(joint_name)
        suffix = safe_column_name(joint_name)

        if idx is not None and idx < len(positions):
            row[f"pos_{suffix}"] = positions[idx]
        else:
            row[f"pos_{suffix}"] = ""

        if include_velocity:
            if idx is not None and idx < len(velocities):
                row[f"vel_{suffix}"] = velocities[idx]
            else:
                row[f"vel_{suffix}"] = ""

        if include_effort:
            if idx is not None and idx < len(efforts):
                row[f"effort_{suffix}"] = efforts[idx]
            else:
                row[f"effort_{suffix}"] = ""

    if add_franka_qpos8:
        required = FRANKA_ARM_JOINTS + FRANKA_FINGER_JOINTS
        if all(name in name_to_idx for name in required):
            for i, joint_name in enumerate(FRANKA_ARM_JOINTS):
                row[f"qpos_{i}"] = positions[name_to_idx[joint_name]]
            row["gripper_width"] = (
                positions[name_to_idx[FRANKA_FINGER_JOINTS[0]]]
                + positions[name_to_idx[FRANKA_FINGER_JOINTS[1]]]
            )
        else:
            for i in range(7):
                row[f"qpos_{i}"] = ""
            row["gripper_width"] = ""

    return row, joint_order


def flatten_generic(
    obj: Any,
    prefix: str = "",
    max_array_len: int = 32,
) -> Dict[str, Any]:
    """
    Recursively flatten scalar ROS message fields.
    Large arrays are skipped, because they are usually images or dense buffers.
    """
    row: Dict[str, Any] = {}

    if is_scalar(obj):
        row[safe_column_name(prefix)] = obj
        return row

    if isinstance(obj, str):
        row[safe_column_name(prefix)] = obj
        return row

    if is_scalar_sequence(obj):
        if len(obj) <= max_array_len:
            for i, value in enumerate(obj):
                row[safe_column_name(f"{prefix}_{i}")] = value
        else:
            row[safe_column_name(f"{prefix}_len")] = len(obj)
        return row

    if isinstance(obj, (bytes, bytearray, memoryview)):
        row[safe_column_name(f"{prefix}_len")] = len(obj)
        return row

    if hasattr(obj, "get_fields_and_field_types"):
        for field_name in obj.get_fields_and_field_types().keys():
            if field_name == "header":
                # Header is handled separately in add_header_fields().
                continue
            value = getattr(obj, field_name)
            child_prefix = f"{prefix}_{field_name}" if prefix else field_name
            row.update(flatten_generic(value, child_prefix, max_array_len=max_array_len))
        return row

    return row


def flatten_message(
    msg: Any,
    msg_type: str,
    topic: str,
    joint_orders: Dict[str, List[str]],
    include_velocity: bool,
    include_effort: bool,
    add_franka_qpos8: bool,
    max_array_len: int,
) -> Dict[str, Any]:
    if msg_type in IMAGE_TYPES:
        return {}

    if msg_type == "sensor_msgs/msg/JointState":
        row, order = flatten_joint_state(
            msg,
            joint_orders.get(topic),
            include_velocity=include_velocity,
            include_effort=include_effort,
            add_franka_qpos8=add_franka_qpos8,
        )
        joint_orders[topic] = order
        return row

    twist_row = flatten_twist_like(msg)
    if twist_row is not None:
        return twist_row

    row: Dict[str, Any] = {}
    add_header_fields(row, msg)
    row.update(flatten_generic(msg, max_array_len=max_array_len))
    return row


class CsvTopicWriter:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer: Optional[csv.DictWriter] = None
        self.columns: Optional[List[str]] = None
        self.n_rows = 0
        self.warned_extra_columns = False

    def write(self, row: Dict[str, Any]) -> None:
        if self.writer is None:
            self.columns = list(row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
            self.writer.writeheader()

        assert self.writer is not None
        assert self.columns is not None

        extra_columns = [c for c in row.keys() if c not in self.columns]
        if extra_columns and not self.warned_extra_columns:
            log(
                f"[WARNING] {self.path}: later rows contain new columns that are not in "
                f"the first row schema and will be ignored: {extra_columns[:10]}"
            )
            self.warned_extra_columns = True

        self.writer.writerow({c: row.get(c, "") for c in self.columns})
        self.n_rows += 1

    def close(self) -> None:
        self.file.close()


def resolve_selected_topics(
    topic_type_map: Dict[str, str],
    topics: Optional[List[str]],
    all_numeric: bool,
    include_image_topics: bool,
) -> List[str]:
    available = set(topic_type_map.keys())

    if all_numeric:
        selected = []
        for topic, msg_type in topic_type_map.items():
            if msg_type in IMAGE_TYPES and not include_image_topics:
                continue
            selected.append(topic)
        return sorted(selected)

    if topics:
        missing = [topic for topic in topics if topic not in available]
        if missing:
            log(f"[WARNING] requested topics not found and will be skipped: {missing}")
        return [topic for topic in topics if topic in available]

    selected = [topic for topic in DEFAULT_TOPICS if topic in available]
    if not selected:
        log("[WARNING] none of the default numeric topics were found")
    return selected


def write_metadata(out_dir: str, topic_type_map: Dict[str, str], selected_topics: Sequence[str]) -> None:
    metadata_path = os.path.join(out_dir, "topics_metadata.csv")
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["topic", "type", "selected"])
        writer.writeheader()
        for topic in sorted(topic_type_map):
            writer.writerow(
                {
                    "topic": topic,
                    "type": topic_type_map[topic],
                    "selected": int(topic in selected_topics),
                }
            )
    log(f"[INFO] wrote metadata: {metadata_path}")


def convert_bag_to_csv(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    reader = open_reader(args.bag, storage_id=args.storage_id)
    topic_type_map = get_topic_type_map(reader)

    if args.list_topics:
        print_topics(topic_type_map)
        return

    selected_topics = resolve_selected_topics(
        topic_type_map,
        topics=args.topics,
        all_numeric=args.all_numeric,
        include_image_topics=args.include_image_topics,
    )

    selected_topics = [
        topic
        for topic in selected_topics
        if args.include_image_topics or topic_type_map[topic] not in IMAGE_TYPES
    ]

    if not selected_topics:
        print_topics(topic_type_map)
        raise RuntimeError(
            "No topics selected. Use --topics, --all_numeric, or check the available topics above."
        )

    log("[INFO] selected topics:")
    for topic in selected_topics:
        log(f"       {topic} :: {topic_type_map[topic]}")

    windows: Optional[List[EpisodeWindow]] = None
    if args.use_episode_windows:
        raw_windows = extract_episode_windows(args.bag, args.storage_id, args.episode_topic)
        windows = filter_episode_windows(raw_windows, args.min_duration, args.max_episodes)
        if not windows:
            raise RuntimeError("No episode windows left after filtering")

    # Re-open after optional episode pass.
    reader = open_reader(args.bag, storage_id=args.storage_id)
    msg_types = get_type_class_map({topic: topic_type_map[topic] for topic in selected_topics})

    write_metadata(args.out_dir, topic_type_map, selected_topics)

    selected_set = set(selected_topics)
    writers: Dict[str, CsvTopicWriter] = {}
    joint_orders: Dict[str, List[str]] = {}
    first_selected_t: Optional[float] = None
    episode_idx_cursor = 0
    n_scanned = 0
    n_written = 0
    n_skipped_non_numeric = 0

    try:
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            n_scanned += 1

            if n_scanned % args.progress_every == 0:
                log(f"[INFO] scanned {n_scanned} messages, wrote {n_written} rows")

            if topic not in selected_set:
                continue

            msg_type = topic_type_map[topic]
            if msg_type in IMAGE_TYPES and not args.include_image_topics:
                n_skipped_non_numeric += 1
                continue

            t_abs = bag_timestamp_to_sec(t_ns)

            if args.start_sec is not None and t_abs < args.start_sec:
                continue
            if args.end_sec is not None and t_abs > args.end_sec:
                continue

            current_ep: Optional[EpisodeWindow] = None
            if windows is not None:
                current_ep, episode_idx_cursor = locate_episode(t_abs, windows, episode_idx_cursor)
                if current_ep is None:
                    if episode_idx_cursor >= len(windows):
                        # Bag is time-ordered. Once all selected episode windows are past,
                        # nothing else can be written.
                        break
                    continue

            if first_selected_t is None:
                first_selected_t = t_abs

            msg = deserialize_message(raw, msg_types[topic])
            data_row = flatten_message(
                msg,
                msg_type=msg_type,
                topic=topic,
                joint_orders=joint_orders,
                include_velocity=args.include_joint_velocity,
                include_effort=args.include_joint_effort,
                add_franka_qpos8=args.franka_qpos8,
                max_array_len=args.max_array_len,
            )

            if not data_row:
                n_skipped_non_numeric += 1
                continue

            row: Dict[str, Any] = {
                "t_abs": f"{t_abs:.9f}",
                "t_rel": f"{t_abs - first_selected_t:.9f}",
            }
            if current_ep is not None:
                row["episode_idx"] = current_ep.idx
                row["t_episode"] = f"{t_abs - current_ep.start:.9f}"
            row.update(data_row)

            if topic not in writers:
                out_path = os.path.join(args.out_dir, safe_filename_from_topic(topic))
                writers[topic] = CsvTopicWriter(out_path)
                log(f"[INFO] writing {topic} -> {out_path}")

            writers[topic].write(row)
            n_written += 1

    finally:
        for writer in writers.values():
            writer.close()

    log("")
    log("[INFO] conversion summary")
    log(f"       scanned messages:       {n_scanned}")
    log(f"       written rows:           {n_written}")
    log(f"       skipped non-numeric:    {n_skipped_non_numeric}")
    for topic in sorted(writers):
        log(f"       {topic}: {writers[topic].n_rows} rows -> {writers[topic].path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected numeric ROS2 bag topics to one CSV file per topic."
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to ROS2 bag directory. For sqlite3 bags this is usually the directory containing metadata.yaml.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--storage_id",
        default="sqlite3",
        help="rosbag2 storage id. Default: sqlite3. Use mcap for MCAP bags.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Specific topics to export. Default: common numeric topics from the thesis workflow.",
    )
    parser.add_argument(
        "--all_numeric",
        action="store_true",
        help="Try exporting all non-image topics. Large arrays are truncated/skipped.",
    )
    parser.add_argument(
        "--list_topics",
        action="store_true",
        help="Print available topics and exit.",
    )
    parser.add_argument(
        "--include_image_topics",
        action="store_true",
        help="Do not use this for normal plotting. Allows Image topics to be touched, but their data arrays are not fully exported.",
    )
    parser.add_argument(
        "--start_sec",
        type=float,
        default=None,
        help="Optional absolute bag timestamp lower bound in seconds.",
    )
    parser.add_argument(
        "--end_sec",
        type=float,
        default=None,
        help="Optional absolute bag timestamp upper bound in seconds.",
    )
    parser.add_argument(
        "--use_episode_windows",
        action="store_true",
        help="Only export messages inside committed /episode/control windows. Adds episode_idx and t_episode columns.",
    )
    parser.add_argument(
        "--episode_topic",
        default=TOPIC_EPISODE,
        help=f"Episode marker topic. Default: {TOPIC_EPISODE}",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.0,
        help="Minimum episode duration in seconds when --use_episode_windows is active.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional maximum number of committed episodes to export.",
    )
    parser.add_argument(
        "--include_joint_velocity",
        action="store_true",
        help="Include velocity_* columns in joint_states CSV if present.",
    )
    parser.add_argument(
        "--include_joint_effort",
        action="store_true",
        help="Include effort_* columns in joint_states CSV if present.",
    )
    parser.add_argument(
        "--franka_qpos8",
        action="store_true",
        help="Add qpos_0..qpos_6 and gripper_width columns for right_fr3 joint names.",
    )
    parser.add_argument(
        "--max_array_len",
        type=int,
        default=32,
        help="Maximum primitive array length to expand for generic messages. Default: 32.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Print progress every N scanned bag messages. Default: 50000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_sec is not None and args.end_sec is not None and args.end_sec < args.start_sec:
        raise RuntimeError("--end_sec must be >= --start_sec")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise RuntimeError("--max_episodes must be positive")
    if args.max_array_len < 0:
        raise RuntimeError("--max_array_len must be >= 0")
    convert_bag_to_csv(args)


if __name__ == "__main__":
    main()
