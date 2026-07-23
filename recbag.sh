#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./record_sic_vs_ric.sh
#   ./record_sic_vs_ric.sh --name test_ric_wetrun
#   ./record_sic_vs_ric.sh --name test_ric_wetrun --out-root /home/jau/dyros/data/bags
#   ./record_sic_vs_ric.sh --name quantitative_only --no-images

OUT_ROOT="/home/jau/dyros/data/bags"
BAG_NAME="sic_vs_ric_$(date +%Y%m%d_%H%M%S)"
RECORD_IMAGES=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      BAG_NAME="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --no-images)
      RECORD_IMAGES=false
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--name BAG_NAME] [--out-root DIRECTORY] [--no-images]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

OUT="${OUT_ROOT}/${BAG_NAME}"
mkdir -p "$OUT_ROOT"

TOPICS=(
  # Episode boundaries
  /episode/control

  # ACT policy output
  /act/intercept_prediction

  # Classical scene estimation
  /ball_tracker2/ball_2d_px
  /scene_localizer/top_cam/ball_3d_table
  /scene/ball_trajectory_table
  /scene/middle_line_intersection_pose_robot_base

  # Scene interception controller (SIC)
  /interception_controller/status
  /interception_controller/selected_goto_s
  /interception_controller/commanded_target_table

  # Rollout interception controller (RIC)
  /rollout_interception_controller/status
  /rollout_interception_controller/selected_goto_s
  /rollout_interception_controller/commanded_target_table

  # Command actually executed
  /trajectory_executor/executed_goto_s
  /trajectory_executor/executed_goto_s_target_base
  /trajectory_executor/middle_line_state
  /cartesian_executor/status

  # Robot state
  /joint_states

  # Experimental configuration
  /teleop/interception_arm_mode
  /teleop/interception_arm_inhibit

  # Coordinate transforms
  /scene_localizer/table_pose_robot_base
  /scene_localizer/top_cam/camera_pose_robot_base
  /tf
  /tf_static
)

# Camera images are recorded by default.
if [[ "$RECORD_IMAGES" == true ]]; then
  TOPICS+=(
    /top_cam/camera/color/image_raw/compressed
    /top_cam/camera/color/camera_info
  )
fi

echo "Recording SIC-vs-RIC experiment:"
echo "  Bag:    $OUT"
echo "  Images: $RECORD_IMAGES"
echo
echo "Stop recording with Ctrl-C."

ros2 bag record \
  --storage mcap \
  --output "$OUT" \
  "${TOPICS[@]}"