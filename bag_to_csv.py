#!/usr/bin/env python3
"""
Convert ROS 2 bag topics to one CSV per topic.

Default behavior:
- requires only --bag
- writes CSV files into <bag>/csv/
- exports all discovered topics unless --topics is provided
- auto-detects storage_identifier from metadata.yaml unless --storage_id is provided

Notes:
- sensor_msgs/Image and sensor_msgs/CompressedImage are exported as metadata-only rows.
- raw binary payload buffers are never expanded into CSV cells.
- run inside a sourced ROS 2 workspace that provides all message definitions used in the bag.
"""

import argparse
import csv
import os
import re
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPIC_EPISODE = "/episode/control"

IMAGE_TYPE = "sensor_msgs/msg/Image"
COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
IMAGE_TYPES = {IMAGE_TYPE, COMPRESSED_IMAGE_TYPE}

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


@dataclass
class TopicPass1Info:
    message_columns: Set[str]
    rows_candidate: int


@dataclass
class Pass1Result:
    first_selected_t: Optional[float]
    all_topic_message_count: Dict[str, int]
    per_topic: Dict[str, TopicPass1Info]
    topic_errors: Dict[str, str]


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


def open_reader(bag_path: str, storage_id: str) -> rosbag2_py.SequentialReader:
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


def print_topics(topic_type_map: Dict[str, str]) -> None:
    log("Available topics:")
    for topic in sorted(topic_type_map):
        log(f"  {topic:<55} {topic_type_map[topic]}")


def detect_storage_id_from_metadata(bag_path: str) -> str:
    metadata_path = os.path.join(bag_path, "metadata.yaml")
    if not os.path.exists(metadata_path):
        raise RuntimeError(
            "metadata.yaml not found and --storage_id was not provided; "
            "cannot determine bag storage format"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        storage_id = (
            data.get("rosbag2_bagfile_information", {})
            .get("storage_identifier", None)
        )
        if storage_id:
            return str(storage_id).strip()
    except Exception:
        pass

    # Fallback lightweight parser.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("storage_identifier:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            if value:
                return value

    raise RuntimeError(
        "Unable to read storage_identifier from metadata.yaml and --storage_id was not provided"
    )


def resolve_selected_topics(
    topic_type_map: Dict[str, str],
    topics: Optional[List[str]],
    all_numeric: bool,
    include_image_topics: bool,
) -> List[str]:
    available = set(topic_type_map.keys())

    if all_numeric:
        log("[WARNING] --all_numeric is deprecated; selecting all discovered topics by default")

    if include_image_topics:
        log("[WARNING] --include_image_topics is now redundant; image topics are exported as metadata-only")

    if topics:
        missing = [topic for topic in topics if topic not in available]
        if missing:
            log(f"[WARNING] requested topics not found and will be skipped: {missing}")
        return [topic for topic in topics if topic in available]

    return sorted(topic_type_map.keys())


def resolve_message_classes_for_topics(
    selected_topics: Sequence[str],
    topic_type_map: Dict[str, str],
) -> Dict[str, Any]:
    classes: Dict[str, Any] = {}
    failures: List[str] = []

    for topic in selected_topics:
        msg_type = topic_type_map[topic]
        try:
            classes[topic] = get_message(msg_type)
        except Exception as exc:
            failures.append(
                f"{topic} :: {msg_type} :: {exc}"
            )

    if failures:
        lines = [
            "Failed to load message classes for selected topics.",
            "Make sure the workspace containing these interfaces is sourced.",
            "Failures:",
        ]
        lines.extend([f"  - {x}" for x in failures])
        raise RuntimeError("\n".join(lines))

    return classes


def extract_episode_windows(
    bag_path: str,
    storage_id: str,
    episode_topic: str = TOPIC_EPISODE,
) -> Tuple[List[EpisodeWindow], int]:
    """
    Scan /episode/control semantics:
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

    msg_cls = get_message(topic_type_map[episode_topic])
    current_start: Optional[float] = None
    committed_windows: List[EpisodeWindow] = []

    n_episode_msgs = 0

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic != episode_topic:
            continue

        n_episode_msgs += 1
        msg = deserialize_message(raw, msg_cls)
        t = bag_timestamp_to_sec(t_ns)
        value = getattr(msg, "data", None)

        if value == 1:
            if current_start is None:
                current_start = t
            else:
                log("[WARNING] start received while already recording; ignoring duplicate start")

        elif value == 2:
            if current_start is None:
                log("[WARNING] stop received while not recording; ignoring")
            else:
                committed_windows.append(
                    EpisodeWindow(idx=len(committed_windows), start=current_start, end=t)
                )
                current_start = None

        elif value == 3:
            if current_start is not None:
                current_start = None
            else:
                log("[WARNING] cancel_current received while not recording; ignoring")

        elif value == 4:
            if current_start is not None:
                log("[WARNING] cancel_last received while recording; ignoring")
            elif committed_windows:
                committed_windows.pop()
                for i, ep in enumerate(committed_windows):
                    ep.idx = i
            else:
                log("[WARNING] cancel_last received but no committed episode exists")

        else:
            log(f"[WARNING] unknown episode marker {value}; ignoring")

    if current_start is not None:
        log(f"[WARNING] bag ended during unfinished episode from {current_start:.9f}; discarding")

    return committed_windows, n_episode_msgs


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


def is_binary_buffer(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


def is_sequence_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return False
    return isinstance(value, AbcSequence)


def is_uint8_like_sequence(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if not is_sequence_like(value):
        return False

    typecode = getattr(value, "typecode", None)
    if isinstance(typecode, str) and typecode in {"B", "b"}:
        return True

    # Fallback heuristic for generic uint8 buffers.
    try:
        seq = list(value)
    except Exception:
        return False
    if not seq:
        return False
    return all(isinstance(x, int) and 0 <= x <= 255 for x in seq)


def add_header_fields(row: Dict[str, Any], msg: Any) -> None:
    header = getattr(msg, "header", None)
    if header is None:
        return
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        row["header_stamp"] = stamp_to_sec(stamp)
    frame_id = getattr(header, "frame_id", None)
    if frame_id is not None:
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


def flatten_image_metadata(msg: Any, msg_type: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    add_header_fields(row, msg)

    if msg_type == IMAGE_TYPE:
        row["height"] = getattr(msg, "height", "")
        row["width"] = getattr(msg, "width", "")
        row["encoding"] = getattr(msg, "encoding", "")
        row["is_bigendian"] = getattr(msg, "is_bigendian", "")
        row["step"] = getattr(msg, "step", "")
        row["data_len"] = len(getattr(msg, "data", b""))
        return row

    if msg_type == COMPRESSED_IMAGE_TYPE:
        row["format"] = getattr(msg, "format", "")
        row["data_len"] = len(getattr(msg, "data", b""))
        return row

    return row


def flatten_generic(
    obj: Any,
    prefix: str = "",
    max_array_len: int = 32,
    skip_top_level_header: bool = False,
    _depth: int = 0,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    if is_scalar(obj):
        row[safe_column_name(prefix)] = obj
        return row

    if isinstance(obj, str):
        row[safe_column_name(prefix)] = obj
        return row

    if is_binary_buffer(obj):
        row[safe_column_name(f"{prefix}_len")] = len(obj)
        return row

    if is_uint8_like_sequence(obj):
        row[safe_column_name(f"{prefix}_len")] = len(obj)
        return row

    if hasattr(obj, "get_fields_and_field_types"):
        for field_name in obj.get_fields_and_field_types().keys():
            if skip_top_level_header and _depth == 0 and field_name == "header":
                continue
            value = getattr(obj, field_name)
            child_prefix = f"{prefix}_{field_name}" if prefix else field_name
            row.update(
                flatten_generic(
                    value,
                    prefix=child_prefix,
                    max_array_len=max_array_len,
                    skip_top_level_header=False,
                    _depth=_depth + 1,
                )
            )
        return row

    if is_sequence_like(obj):
        seq = list(obj)
        len_key = safe_column_name(f"{prefix}_len") if prefix else "len"
        row[len_key] = len(seq)

        if len(seq) > max_array_len:
            return row

        for i, value in enumerate(seq):
            child_prefix = f"{prefix}_{i}" if prefix else str(i)
            row.update(
                flatten_generic(
                    value,
                    prefix=child_prefix,
                    max_array_len=max_array_len,
                    skip_top_level_header=False,
                    _depth=_depth + 1,
                )
            )
        return row

    # Unknown object type: keep string repr for visibility.
    if prefix:
        row[safe_column_name(prefix)] = str(obj)
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
        return flatten_image_metadata(msg, msg_type)

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
    row.update(
        flatten_generic(
            msg,
            max_array_len=max_array_len,
            skip_top_level_header=True,
        )
    )
    return row


class CsvTopicWriter:
    def __init__(self, path: str, columns: Sequence[str]):
        self.path = path
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.columns = list(columns)
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()
        self.n_rows = 0

    def write(self, row: Dict[str, Any]) -> None:
        self.writer.writerow({c: row.get(c, "") for c in self.columns})
        self.n_rows += 1

    def close(self) -> None:
        self.file.close()


def pass1_scan(
    bag_path: str,
    storage_id: str,
    topic_type_map: Dict[str, str],
    selected_topics: Sequence[str],
    msg_types: Dict[str, Any],
    windows: Optional[Sequence[EpisodeWindow]],
    args: argparse.Namespace,
) -> Pass1Result:
    reader = open_reader(bag_path, storage_id=storage_id)
    selected_set = set(selected_topics)

    per_topic: Dict[str, TopicPass1Info] = {
        topic: TopicPass1Info(message_columns=set(), rows_candidate=0)
        for topic in selected_topics
    }
    all_topic_message_count: Dict[str, int] = {topic: 0 for topic in topic_type_map}
    topic_errors: Dict[str, str] = {}

    first_selected_t: Optional[float] = None
    episode_idx_cursor = 0
    n_scanned = 0

    joint_orders: Dict[str, List[str]] = {}

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        n_scanned += 1

        if topic in all_topic_message_count:
            all_topic_message_count[topic] += 1

        if n_scanned % args.progress_every == 0:
            log(f"[INFO] pass1 scanned {n_scanned} messages")

        if topic not in selected_set:
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
                    break
                continue

        if first_selected_t is None:
            first_selected_t = t_abs

        try:
            msg = deserialize_message(raw, msg_types[topic])
            data_row = flatten_message(
                msg,
                msg_type=topic_type_map[topic],
                topic=topic,
                joint_orders=joint_orders,
                include_velocity=args.include_joint_velocity,
                include_effort=args.include_joint_effort,
                add_franka_qpos8=args.franka_qpos8,
                max_array_len=args.max_array_len,
            )
            per_topic[topic].message_columns.update(data_row.keys())
            per_topic[topic].rows_candidate += 1
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            topic_errors.setdefault(topic, detail)
            log(f"[ERROR] pass1 flatten failed for {topic} ({topic_type_map[topic]}): {detail}")

    return Pass1Result(
        first_selected_t=first_selected_t,
        all_topic_message_count=all_topic_message_count,
        per_topic=per_topic,
        topic_errors=topic_errors,
    )


def pass2_write(
    bag_path: str,
    storage_id: str,
    topic_type_map: Dict[str, str],
    selected_topics: Sequence[str],
    msg_types: Dict[str, Any],
    windows: Optional[Sequence[EpisodeWindow]],
    pass1: Pass1Result,
    out_dir: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
    reader = open_reader(bag_path, storage_id=storage_id)
    selected_set = set(selected_topics)

    context_columns = ["t_abs", "t_rel"]
    if windows is not None:
        context_columns.extend(["episode_idx", "t_episode"])

    writers: Dict[str, CsvTopicWriter] = {}
    rows_written: Dict[str, int] = {topic: 0 for topic in selected_topics}
    output_files: Dict[str, str] = {}
    topic_errors: Dict[str, str] = {}

    for topic in selected_topics:
        message_columns = sorted(
            c for c in pass1.per_topic[topic].message_columns if c not in set(context_columns)
        )
        columns = context_columns + message_columns

        out_name = safe_filename_from_topic(topic)
        out_path = os.path.join(out_dir, out_name)
        writers[topic] = CsvTopicWriter(out_path, columns)
        output_files[topic] = out_name
        log(f"[INFO] writing {topic} -> {out_path}")

    first_selected_t = pass1.first_selected_t
    episode_idx_cursor = 0
    n_scanned = 0

    joint_orders: Dict[str, List[str]] = {}

    try:
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            n_scanned += 1

            if n_scanned % args.progress_every == 0:
                total_written = sum(rows_written.values())
                log(f"[INFO] pass2 scanned {n_scanned} messages, wrote {total_written} rows")

            if topic not in selected_set:
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
                        break
                    continue

            if first_selected_t is None:
                first_selected_t = t_abs

            try:
                msg = deserialize_message(raw, msg_types[topic])
                data_row = flatten_message(
                    msg,
                    msg_type=topic_type_map[topic],
                    topic=topic,
                    joint_orders=joint_orders,
                    include_velocity=args.include_joint_velocity,
                    include_effort=args.include_joint_effort,
                    add_franka_qpos8=args.franka_qpos8,
                    max_array_len=args.max_array_len,
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                log(f"[ERROR] pass2 flatten failed for {topic} ({topic_type_map[topic]}): {detail}")
                topic_errors.setdefault(topic, detail)
                continue

            t_rel = 0.0 if first_selected_t is None else (t_abs - first_selected_t)
            row: Dict[str, Any] = {
                "t_abs": f"{t_abs:.9f}",
                "t_rel": f"{t_rel:.9f}",
            }
            if current_ep is not None:
                row["episode_idx"] = current_ep.idx
                row["t_episode"] = f"{t_abs - current_ep.start:.9f}"
            row.update(data_row)

            writers[topic].write(row)
            rows_written[topic] += 1

    finally:
        for writer in writers.values():
            writer.close()

    return rows_written, output_files, topic_errors


def write_topics_metadata(
    out_dir: str,
    topic_type_map: Dict[str, str],
    selected_topics: Sequence[str],
    message_count: Dict[str, int],
    rows_written: Dict[str, int],
    output_files: Dict[str, str],
    topic_errors: Dict[str, str],
) -> None:
    selected_set = set(selected_topics)

    metadata_path = os.path.join(out_dir, "topics_metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "topic",
            "type",
            "selected",
            "message_count",
            "rows_written",
            "output_file",
            "export_mode",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for topic in sorted(topic_type_map):
            msg_type = topic_type_map[topic]
            selected = topic in selected_set
            count = int(message_count.get(topic, 0))
            wrote = int(rows_written.get(topic, 0))
            out_file = output_files.get(topic, "")
            err = topic_errors.get(topic, "")

            if not selected:
                mode = "not_selected"
            elif err:
                mode = "error"
            elif wrote == 0:
                mode = "empty"
            elif msg_type in IMAGE_TYPES:
                mode = "metadata_only"
            else:
                mode = "full"

            writer.writerow(
                {
                    "topic": topic,
                    "type": msg_type,
                    "selected": int(selected),
                    "message_count": count,
                    "rows_written": wrote,
                    "output_file": out_file,
                    "export_mode": mode,
                    "error": err,
                }
            )

    log(f"[INFO] wrote metadata: {metadata_path}")


def convert_bag_to_csv(args: argparse.Namespace) -> None:
    bag_path = os.path.abspath(args.bag)
    if not os.path.isdir(bag_path):
        raise RuntimeError(f"bag directory does not exist: {bag_path}")

    storage_id: str
    if args.storage_id:
        storage_id = args.storage_id
        log(f"[INFO] using explicit storage id: {storage_id}")
    else:
        storage_id = detect_storage_id_from_metadata(bag_path)
        log(f"[INFO] detected storage id: {storage_id}")

    out_dir = args.out_dir
    ensure_dir(out_dir)

    reader = open_reader(bag_path, storage_id=storage_id)
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

    if not selected_topics:
        print_topics(topic_type_map)
        raise RuntimeError("No topics selected")

    log("[INFO] selected topics:")
    for topic in selected_topics:
        log(f"       {topic} :: {topic_type_map[topic]}")

    # Fail before conversion if selected types are not importable.
    msg_types = resolve_message_classes_for_topics(selected_topics, topic_type_map)

    windows: Optional[List[EpisodeWindow]] = None
    if args.use_episode_windows:
        raw_windows, n_episode_msgs = extract_episode_windows(
            bag_path,
            storage_id,
            args.episode_topic,
        )

        if n_episode_msgs == 0:
            raise RuntimeError(
                "--use_episode_windows requested, but /episode/control contains no messages"
            )

        windows = filter_episode_windows(raw_windows, args.min_duration, args.max_episodes)
        if not windows:
            raise RuntimeError("No episode windows left after filtering")

    pass1 = pass1_scan(
        bag_path=bag_path,
        storage_id=storage_id,
        topic_type_map=topic_type_map,
        selected_topics=selected_topics,
        msg_types=msg_types,
        windows=windows,
        args=args,
    )

    rows_written, output_files, pass2_errors = pass2_write(
        bag_path=bag_path,
        storage_id=storage_id,
        topic_type_map=topic_type_map,
        selected_topics=selected_topics,
        msg_types=msg_types,
        windows=windows,
        pass1=pass1,
        out_dir=out_dir,
        args=args,
    )

    merged_errors = dict(pass1.topic_errors)
    for topic, detail in pass2_errors.items():
        merged_errors.setdefault(topic, detail)

    write_topics_metadata(
        out_dir=out_dir,
        topic_type_map=topic_type_map,
        selected_topics=selected_topics,
        message_count=pass1.all_topic_message_count,
        rows_written=rows_written,
        output_files=output_files,
        topic_errors=merged_errors,
    )

    log("")
    log("[INFO] conversion summary")
    log(f"       selected topics:         {len(selected_topics)}")
    log(f"       total topics in bag:     {len(topic_type_map)}")
    log(f"       total rows written:      {sum(rows_written.values())}")

    for topic in sorted(selected_topics):
        mode = "error" if topic in merged_errors else (
            "metadata_only" if topic_type_map[topic] in IMAGE_TYPES and rows_written[topic] > 0 else (
                "empty" if rows_written[topic] == 0 else "full"
            )
        )
        log(
            f"       {topic}: messages={pass1.all_topic_message_count.get(topic, 0)}, "
            f"rows={rows_written.get(topic, 0)}, mode={mode}, file={output_files.get(topic, '')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bag topics to one CSV file per topic."
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to ROS2 bag directory (containing metadata.yaml).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional CSV output directory. Default: <bag>/csv",
    )
    parser.add_argument(
        "--storage_id",
        default=None,
        help="Optional rosbag2 storage id override (e.g. sqlite3, mcap). Default: auto-detect from metadata.yaml.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Optional specific topics to export. Default: all discovered topics.",
    )
    parser.add_argument(
        "--all_numeric",
        action="store_true",
        help="Deprecated. No longer needed; all topics are selected by default.",
    )
    parser.add_argument(
        "--list_topics",
        action="store_true",
        help="Print available topics and exit.",
    )
    parser.add_argument(
        "--include_image_topics",
        action="store_true",
        help="Deprecated. Image topics are exported as metadata-only rows by default.",
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
        help="Maximum array length to recursively expand in generic flattening. Default: 32.",
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

    args.out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.join(os.path.abspath(args.bag), "csv")
    )

    convert_bag_to_csv(args)


if __name__ == "__main__":
    main()
