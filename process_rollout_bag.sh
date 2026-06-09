#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 <top_level_folder> [--xlim <xmin> <xmax>]"
    echo "  $0 <top_level_folder> [<xmin> <xmax>]"
    echo
    echo "Example:"
    echo "  $0 /home/jau/dyros/data/bags/debug/rollout_comparison/recording_20260601_163655/ --xlim 0 25"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

TOP_DIR="$1"
shift

# Remove trailing slash, if present
TOP_DIR="${TOP_DIR%/}"

# Extract recording base name, e.g. recording_20260601_163655
# Works for names like recording_20260601_163655_rgb_rollout_dark_center.
INPUT_NAME="$(basename "$TOP_DIR")"
if [[ "$INPUT_NAME" =~ ^(recording_[0-9]{8}_[0-9]{6}) ]]; then
    RECORDING_NAME="${BASH_REMATCH[1]}"
else
    echo "Error: folder name must start with recording_YYYYMMDD_HHMMSS"
    echo "  Received: $INPUT_NAME"
    exit 1
fi

TITLE_SUFFIX=""
if [[ "$INPUT_NAME" =~ ^${RECORDING_NAME}_(.+)$ ]]; then
    TITLE_SUFFIX="${BASH_REMATCH[1]}"
fi

format_plot_title() {
    local raw="$1"
    local -a tokens=()
    local -a words=()
    local has_rollout=0
    local title=""

    if [[ -z "$raw" ]]; then
        echo ""
        return
    fi

    IFS='_' read -r -a words <<< "$raw"

    for w in "${words[@]}"; do
        [[ "$w" == "rollout" ]] && has_rollout=1
    done

    if [[ $has_rollout -eq 1 ]]; then
        tokens+=("Rollout")
    fi

    for w in "${words[@]}"; do
        [[ "$w" == "rollout" ]] && continue
        case "$w" in
            rgb)
                tokens+=("RGB")
                ;;
            event)
                tokens+=("Event")
                ;;
            *)
                tokens+=("${w^}")
                ;;
        esac
    done

    for tok in "${tokens[@]}"; do
        if [[ -z "$title" ]]; then
            title="$tok"
        else
            title+=" - $tok"
        fi
    done

    echo "$title"
}

PLOT_TITLE="$(format_plot_title "$TITLE_SUFFIX")"
if [[ -z "$PLOT_TITLE" ]]; then
    PLOT_TITLE="$INPUT_NAME"
fi

BAG_PATH="${TOP_DIR}/${RECORDING_NAME}_bag"
CSV_DIR="${TOP_DIR}/csv_data"

# Defaults
XLIM_MIN=0
XLIM_MAX=25

if [[ $# -eq 0 ]]; then
    :
elif [[ $# -eq 2 ]]; then
    XLIM_MIN="$1"
    XLIM_MAX="$2"
elif [[ $# -eq 3 && "$1" == "--xlim" ]]; then
    XLIM_MIN="$2"
    XLIM_MAX="$3"
else
    usage
fi

OUTPUT_PATH="${TOP_DIR}/twist_traj_xlim_${XLIM_MIN}_${XLIM_MAX}.png"

if [[ ! -d "$TOP_DIR" ]]; then
    echo "Error: top-level folder does not exist:"
    echo "  $TOP_DIR"
    exit 1
fi

if [[ ! -d "$BAG_PATH" ]]; then
    echo "Error: expected bag folder does not exist:"
    echo "  $BAG_PATH"
    exit 1
fi

NEED_EXTRACT=1
if [[ -d "$CSV_DIR" ]]; then
    if find "$CSV_DIR" -mindepth 1 -print -quit | grep -q .; then
        echo "[INFO] Existing CSV directory detected; reusing:"
        echo "  $CSV_DIR"
        NEED_EXTRACT=0
    else
        echo "[INFO] Existing CSV directory is empty; extracting bag again:"
        echo "  $CSV_DIR"
    fi
else
    mkdir -p "$CSV_DIR"
fi

if [[ $NEED_EXTRACT -eq 1 ]]; then
    echo "[1/2] Extracting bag to CSV..."
    echo "  Bag:     $BAG_PATH"
    echo "  Out dir: $CSV_DIR"

    python3 src/plotting/bag_to_csv.py \
        --bag "$BAG_PATH" \
        --out_dir "$CSV_DIR"
else
    echo "[1/2] Skipping bag extraction (CSV already available)."
fi

echo "[2/2] Plotting CSV topics..."
echo "  CSV dir: $CSV_DIR"
echo "  xlim:    $XLIM_MIN $XLIM_MAX"
echo "  title:   $PLOT_TITLE"
echo "  output:  $OUTPUT_PATH"

python3 src/plotting/plot_multi_topic_csv.py \
    "$CSV_DIR" \
    --save "$OUTPUT_PATH" \
    --title "$PLOT_TITLE" \
    --negate-twist \
    --min-label-duration "250" \
    --xlim "$XLIM_MIN" "$XLIM_MAX"

echo "Done."