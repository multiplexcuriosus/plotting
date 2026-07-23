#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./record_sic_vs_ric.sh
#   ./record_sic_vs_ric.sh --name cont_tracker_debug
#   ./record_sic_vs_ric.sh --name cont_tracker_debug \
#     --out-root /home/jau/dyros/data/bags
#   ./record_sic_vs_ric.sh --name quantitative_only --no-images
#
# Camera images are recorded by default.

OUT_ROOT="/home/jau/dyros/data/bags"
BAG_NAME="sic_ric_cont_tracker_$(date +%Y%m%d_%H%M%S)"
RECORD_IMAGES=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 ]] || {
        echo "Error: --name requires a value." >&2
        exit 1
      }
      BAG_NAME="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || {
        echo "Error: --out-root requires a value." >&2
        exit 1
      }
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

  # Continuous tracker:
  # requested, accepted-for-analysis, telemetry, and Cartesian target
  /cont_tracker/target_s
  /cont_tracker/track_s
  /cont_tracker/accepted_target_s
  /cont_tracker/accepted_target_base

  # Executor/controller state
  /trajectory_executor/executed_goto_s
  /trajectory_executor/executed_goto_s_target_base
  /trajectory_executor/middle_line_state
  /cartesian_executor/status
  /cartesian_cmd/twist

  # Robot measurements and low-level diagnostics
  /joint_states
  /right_fr3/joint_states
  /right_franka_robot_state_broadcaster/robot_state
  /right_franka/external_wrenches

  # Experimental configuration
  /teleop/interception_arm_mode
  /teleop/interception_arm_inhibit

  # Coordinate transforms
  /scene_localizer/table_pose_robot_base
  /scene_localizer/top_cam/camera_pose_robot_base
  /tf
  /tf_static

  # Logs: useful for controller errors, cancellations, and reflexes
  /rosout
)

# Camera images are recorded by default.
if [[ "$RECORD_IMAGES" == true ]]; then
  TOPICS+=(
    /top_cam/camera/color/image_raw/compressed
    /top_cam/camera/color/camera_info
  )
fi

# Add hidden cont_tracker action topics when the action server is active.
# Explicitly checking avoids warnings if the action server is not running.
mapfile -t AVAILABLE_TOPICS < <(ros2 topic list --include-hidden-topics)

topic_exists() {
  local wanted="$1"
  local available

  for available in "${AVAILABLE_TOPICS[@]}"; do
    if [[ "$available" == "$wanted" ]]; then
      return 0
    fi
  done

  return 1
}

HIDDEN_ACTION_TOPICS=(
  /cont_tracker/_action/feedback
  /cont_tracker/_action/status
)

RECORD_HIDDEN=false

for topic in "${HIDDEN_ACTION_TOPICS[@]}"; do
  if topic_exists "$topic"; then
    TOPICS+=("$topic")
    RECORD_HIDDEN=true
  fi
done

echo "Recording SIC/RIC/continuous-tracker experiment:"
echo "  Bag:             $OUT"
echo "  Images:          $RECORD_IMAGES"
echo "  Hidden action:   $RECORD_HIDDEN"
echo "  Topic count:     ${#TOPICS[@]}"
echo
echo "Important continuous-tracker streams:"
echo "  /cont_tracker/target_s"
echo "  /cont_tracker/track_s"
echo "  /cont_tracker/accepted_target_s"
echo "  /cont_tracker/accepted_target_base"
echo
echo "Stop recording with Ctrl-C."

BAG_ARGS=(
  ros2 bag record
  --storage mcap
  --output "$OUT"
)

if [[ "$RECORD_HIDDEN" == true ]]; then
  BAG_ARGS+=(--include-hidden-topics)
fi

"${BAG_ARGS[@]}" "${TOPICS[@]}"