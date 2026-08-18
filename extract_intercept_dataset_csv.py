#!/usr/bin/env python3
"""Extract lightweight per-bag CSVs for intercept dataset analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from intercept_dataset_common import (
    EventLogEntry,
    EpisodeWindow,
    build_output_rows_sorted,
    default_source_id_from_bag_path,
    find_episode_index_for_timestamp,
    parse_episode_control_events,
    summarize_classic_counts,
)


DEFAULT_EPISODE_TOPIC = "/episode/control"
DEFAULT_BALL_TOPIC = "/scene_localizer/top_cam/ball_3d_table"
DEFAULT_CLASSIC_GOTO_TOPIC = "/interception_controller/selected_goto_s"


def log(message: str) -> None:
    print(message, flush=True)


def bag_timestamp_to_sec(timestamp_ns: int) -> float:
    return float(timestamp_ns) * 1e-9


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def detect_storage_id_from_metadata(bag_path: Path) -> str:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        raise RuntimeError(
            f"metadata.yaml not found in {bag_path}; pass --storage-id explicitly"
        )

    text = metadata_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        value = (
            data.get("rosbag2_bagfile_information", {})
            .get("storage_identifier", "")
        )
        value = str(value).strip()
        if value:
            return value
    except Exception:
        pass

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("storage_identifier:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value

    raise RuntimeError(
        f"Could not detect storage_identifier from {metadata_path}; pass --storage-id"
    )


def _import_ros_modules() -> Tuple[Any, Any, Any]:
    try:
        import rosbag2_py  # type: ignore
        from rclpy.serialization import deserialize_message  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ROS bag dependencies are unavailable. Source the ROS 2 workspace "
            "and ensure rosbag2_py/rclpy/rosidl_runtime_py are importable."
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def open_filtered_reader(
    rosbag2_py: Any,
    bag_path: Path,
    storage_id: str,
    topics: Sequence[str],
) -> Any:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    filtered_topics = list(dict.fromkeys(topics))
    if not filtered_topics:
        raise RuntimeError("Topic filter must be non-empty")
    reader.set_filter(rosbag2_py.StorageFilter(topics=filtered_topics))
    return reader


def list_topic_types(
    rosbag2_py: Any,
    bag_path: Path,
    storage_id: str,
) -> Dict[str, str]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return {x.name: x.type for x in reader.get_all_topics_and_types()}


def extraction_filter_topics(
    episode_topic: str,
    ball_topic: str,
    classic_goto_topic: str,
) -> Tuple[List[str], List[str]]:
    """Return pass1 and pass2 topic filters."""
    return [episode_topic], [ball_topic, classic_goto_topic]


@dataclass
class ExtractedRows:
    manifest: pd.DataFrame
    ball: pd.DataFrame
    classic: pd.DataFrame
    metadata: Dict[str, Any]


def _scan_episode_events(
    reader: Any,
    deserialize_message: Any,
    msg_cls: Any,
    episode_topic: str,
    progress_every: int,
) -> List[EventLogEntry]:
    events: List[EventLogEntry] = []
    scanned = 0
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        scanned += 1
        if scanned % progress_every == 0:
            log(f"[INFO] pass1 scanned {scanned} filtered messages")
        if topic != episode_topic:
            continue
        msg = deserialize_message(raw, msg_cls)
        events.append(
            EventLogEntry(
                timestamp=bag_timestamp_to_sec(t_ns),
                value=int(getattr(msg, "data", 0)),
            )
        )
    return events


def _read_ball_fields(msg: Any) -> Tuple[Optional[float], Optional[float], Optional[float], float, str]:
    point = getattr(msg, "point", None)
    if point is None:
        return None, None, None, np.nan, ""

    x = float(getattr(point, "x", np.nan))
    y = float(getattr(point, "y", np.nan))
    z = float(getattr(point, "z", np.nan))

    header = getattr(msg, "header", None)
    if header is None:
        return x, y, z, np.nan, ""
    stamp = getattr(header, "stamp", None)
    frame_id = str(getattr(header, "frame_id", ""))
    header_ts = np.nan if stamp is None else stamp_to_sec(stamp)
    return x, y, z, header_ts, frame_id


def _collect_topic_rows(
    reader: Any,
    deserialize_message: Any,
    msg_classes: Dict[str, Any],
    windows: Sequence[EpisodeWindow],
    source_id: str,
    ball_topic: str,
    classic_topic: str,
    progress_every: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ball_rows: List[Dict[str, Any]] = []
    classic_rows: List[Dict[str, Any]] = []
    cursor = 0
    scanned = 0

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        scanned += 1
        if scanned % progress_every == 0:
            log(f"[INFO] pass2 scanned {scanned} filtered messages")

        ts = bag_timestamp_to_sec(t_ns)
        ep_idx, cursor = find_episode_index_for_timestamp(ts, windows, cursor)
        if ep_idx is None:
            continue
        episode = windows[ep_idx]

        if topic == ball_topic:
            msg = deserialize_message(raw, msg_classes[topic])
            x, y, z, header_ts, frame_id = _read_ball_fields(msg)
            if x is None or y is None or z is None:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            ball_rows.append(
                {
                    "source_id": source_id,
                    "episode_id": int(episode.episode_id),
                    "timestamp": float(ts),
                    "t_rel_sec": float(ts - episode.start_timestamp),
                    "header_timestamp": float(header_ts),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "frame_id": frame_id,
                }
            )

        elif topic == classic_topic and classic_topic in msg_classes:
            msg = deserialize_message(raw, msg_classes[topic])
            s_val = float(getattr(msg, "data", np.nan))
            if not np.isfinite(s_val):
                continue
            classic_rows.append(
                {
                    "source_id": source_id,
                    "episode_id": int(episode.episode_id),
                    "timestamp": float(ts),
                    "t_rel_sec": float(ts - episode.start_timestamp),
                    "s": float(s_val),
                }
            )

    ball_df = pd.DataFrame(
        ball_rows,
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
    )
    classic_df = pd.DataFrame(
        classic_rows,
        columns=["source_id", "episode_id", "timestamp", "t_rel_sec", "s"],
    )
    return build_output_rows_sorted(ball_df), build_output_rows_sorted(classic_df)


def _build_manifest(
    source_id: str,
    windows: Sequence[EpisodeWindow],
    ball_df: pd.DataFrame,
    classic_df: pd.DataFrame,
) -> pd.DataFrame:
    ball_counts = (
        ball_df.groupby(["source_id", "episode_id"], as_index=False)
        .size()
        .rename(columns={"size": "ball_sample_count"})
    )
    classic_counts = (
        classic_df.groupby(["source_id", "episode_id"], as_index=False)
        .size()
        .rename(columns={"size": "classic_goto_s_count"})
    )

    manifest_rows: List[Dict[str, Any]] = []
    for ep in windows:
        frame_values: List[str] = []
        if not ball_df.empty:
            frame_values = (
                ball_df.loc[ball_df["episode_id"] == ep.episode_id, "frame_id"]
                .astype(str)
                .str.strip()
                .tolist()
            )
        nonempty_frames = sorted({f for f in frame_values if f})

        invalid_reasons: List[str] = []
        if len(nonempty_frames) > 1:
            invalid_reasons.append("inconsistent_ball_frame_id")

        if not ball_counts.empty:
            match = ball_counts.loc[ball_counts["episode_id"] == ep.episode_id, "ball_sample_count"]
            sample_count = int(match.iloc[0]) if not match.empty else 0
        else:
            sample_count = 0
        if sample_count < 2:
            invalid_reasons.append("insufficient_ball_samples")

        frame_id = nonempty_frames[0] if len(nonempty_frames) == 1 else ""

        if not classic_counts.empty:
            match = classic_counts.loc[
                classic_counts["episode_id"] == ep.episode_id,
                "classic_goto_s_count",
            ]
            classic_count = int(match.iloc[0]) if not match.empty else 0
        else:
            classic_count = 0

        valid = len(invalid_reasons) == 0
        manifest_rows.append(
            {
                "source_id": source_id,
                "episode_id": int(ep.episode_id),
                "start_timestamp": float(ep.start_timestamp),
                "end_timestamp": float(ep.end_timestamp),
                "duration_sec": float(ep.duration_sec),
                "ball_sample_count": int(sample_count),
                "classic_goto_s_count": int(classic_count),
                "ball_frame_id": frame_id,
                "valid": bool(valid),
                "invalid_reason": "" if valid else ";".join(invalid_reasons),
            }
        )

    return pd.DataFrame(
        manifest_rows,
        columns=[
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
    )


def extract_dataset(args: argparse.Namespace) -> ExtractedRows:
    bag_path = Path(args.path_to_target_bag).expanduser().resolve()
    if not bag_path.is_dir():
        raise RuntimeError(f"Bag path is not a directory: {bag_path}")

    storage_id = args.storage_id or detect_storage_id_from_metadata(bag_path)
    source_id = args.source_id or default_source_id_from_bag_path(str(bag_path))

    rosbag2_py, deserialize_message, get_message = _import_ros_modules()
    topic_types = list_topic_types(rosbag2_py, bag_path, storage_id)

    if args.episode_topic not in topic_types:
        raise RuntimeError(
            f"Missing required episode topic: {args.episode_topic}. "
            "Check topic names or run with --episode-topic."
        )
    if args.ball_topic not in topic_types:
        raise RuntimeError(
            f"Missing required ball topic: {args.ball_topic}. "
            "Check topic names or run with --ball-topic."
        )

    has_classic_topic = args.classic_goto_topic in topic_types
    if not has_classic_topic:
        log(
            f"[WARNING] Optional classical GOTO_S topic not found: {args.classic_goto_topic}. "
            "classic_selected_goto_s.csv will contain only a header."
        )

    pass1_topics, pass2_topics = extraction_filter_topics(
        args.episode_topic,
        args.ball_topic,
        args.classic_goto_topic,
    )
    log(f"[INFO] pass1 storage filter topics: {pass1_topics}")
    pass1_reader = open_filtered_reader(rosbag2_py, bag_path, storage_id, pass1_topics)

    ep_cls = get_message(topic_types[args.episode_topic])
    events = _scan_episode_events(
        reader=pass1_reader,
        deserialize_message=deserialize_message,
        msg_cls=ep_cls,
        episode_topic=args.episode_topic,
        progress_every=args.progress_every,
    )
    if not events:
        raise RuntimeError(
            f"No /episode/control messages found on {args.episode_topic}. "
            "Extraction cannot continue."
        )

    windows, warnings = parse_episode_control_events(events)
    for warning in warnings:
        log(f"[WARNING] {warning}")

    log(f"[INFO] committed episodes after state-machine parsing: {len(windows)}")

    if has_classic_topic:
        pass2_active_topics = pass2_topics
    else:
        pass2_active_topics = [args.ball_topic]
    log(f"[INFO] pass2 storage filter topics: {pass2_active_topics}")
    pass2_reader = open_filtered_reader(
        rosbag2_py,
        bag_path,
        storage_id,
        pass2_active_topics,
    )

    msg_classes = {
        args.ball_topic: get_message(topic_types[args.ball_topic]),
    }
    if has_classic_topic:
        msg_classes[args.classic_goto_topic] = get_message(topic_types[args.classic_goto_topic])

    ball_df, classic_df = _collect_topic_rows(
        reader=pass2_reader,
        deserialize_message=deserialize_message,
        msg_classes=msg_classes,
        windows=windows,
        source_id=source_id,
        ball_topic=args.ball_topic,
        classic_topic=args.classic_goto_topic,
        progress_every=args.progress_every,
    )

    manifest_df = _build_manifest(source_id, windows, ball_df, classic_df)

    valid_count = int(manifest_df["valid"].sum()) if not manifest_df.empty else 0
    metadata = {
        "schema_version": "1.0",
        "source_id": source_id,
        "source_bag_path": str(bag_path),
        "source_bag_basename": bag_path.name,
        "storage_id": storage_id,
        "episode_topic": args.episode_topic,
        "ball_topic": args.ball_topic,
        "classic_goto_topic": args.classic_goto_topic,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "committed_episode_count": int(len(manifest_df)),
        "valid_trajectory_count": int(valid_count),
        "ball_sample_count": int(len(ball_df)),
        "classic_goto_s_count": int(len(classic_df)),
    }

    return ExtractedRows(
        manifest=manifest_df,
        ball=ball_df,
        classic=classic_df,
        metadata=metadata,
    )


def _write_csv_with_headers(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = []
    out = out.loc[:, list(columns)]
    out.to_csv(path, index=False)


def write_output_atomic(args: argparse.Namespace, rows: ExtractedRows) -> Path:
    bag_path = Path(args.path_to_target_bag).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (bag_path / "dataset_csv")
    )
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise RuntimeError(
            f"Output directory already exists and is non-empty: {out_dir}. "
            "Use --force to overwrite safely."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_csv_tmp_", dir=str(parent)))
    try:
        _write_csv_with_headers(
            rows.manifest,
            temp_dir / "manifest.csv",
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
        )
        _write_csv_with_headers(
            rows.ball,
            temp_dir / "ball_position_table.csv",
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
        )
        _write_csv_with_headers(
            rows.classic,
            temp_dir / "classic_selected_goto_s.csv",
            ["source_id", "episode_id", "timestamp", "t_rel_sec", "s"],
        )
        (temp_dir / "extraction_metadata.json").write_text(
            json.dumps(rows.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if out_dir.exists() and args.force:
            shutil.rmtree(out_dir)
        if out_dir.exists() and not any(out_dir.iterdir()):
            out_dir.rmdir()
        shutil.move(str(temp_dir), str(out_dir))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return out_dir


def print_summary(rows: ExtractedRows, resolved_out_dir: Path) -> None:
    manifest = rows.manifest
    ball = rows.ball
    classic = rows.classic

    source_id = rows.metadata["source_id"]
    committed = int(len(manifest))
    valid = int(manifest["valid"].sum()) if not manifest.empty else 0
    invalid = committed - valid
    raw_ball = int(len(ball))

    zero, one, multiple = summarize_classic_counts(manifest, classic)

    log("")
    log("[SUMMARY]")
    log(f"source_id: {source_id}")
    log(f"committed episodes: {committed}")
    log(f"valid trajectories: {valid}")
    log(f"invalid trajectories: {invalid}")
    log(f"raw ball sample count: {raw_ball}")
    log(f"episodes with zero classical GOTO_S messages: {zero}")
    log(f"episodes with exactly one classical GOTO_S message: {one}")
    log(f"episodes with multiple classical GOTO_S messages: {multiple}")
    log(f"resolved output directory: {resolved_out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-bag intercept dataset CSV files without deserializing image topics.",
    )
    parser.add_argument(
        "path_to_target_bag",
        help="Path to target ROS 2 bag directory containing metadata.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <PATH_TO_TARGET_BAG>/dataset_csv/",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Override source identifier. Default: bag basename with one trailing _bag removed.",
    )
    parser.add_argument(
        "--storage-id",
        default=None,
        help="Override rosbag2 storage identifier (e.g., mcap, sqlite3).",
    )
    parser.add_argument(
        "--episode-topic",
        default=DEFAULT_EPISODE_TOPIC,
        help=f"Episode-control topic. Default: {DEFAULT_EPISODE_TOPIC}",
    )
    parser.add_argument(
        "--ball-topic",
        default=DEFAULT_BALL_TOPIC,
        help=f"Table-frame ball topic. Default: {DEFAULT_BALL_TOPIC}",
    )
    parser.add_argument(
        "--classic-goto-topic",
        default=DEFAULT_CLASSIC_GOTO_TOPIC,
        help=f"Classical selected GOTO_S topic. Default: {DEFAULT_CLASSIC_GOTO_TOPIC}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing non-empty output directory safely via temporary output then replace.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print filtered-pass progress every N messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise RuntimeError("--progress-every must be positive")

    rows = extract_dataset(args)
    out_dir = write_output_atomic(args, rows)
    print_summary(rows, out_dir)


if __name__ == "__main__":
    main()
