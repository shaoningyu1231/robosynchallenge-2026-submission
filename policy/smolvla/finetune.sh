#!/bin/bash
# ----------------------------------------------------------------------------
# bash finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]
# bash finetune.sh click_bell datasets/cobotmagic_Sim_click_bell \
#   outputs/train/cobotmagic_smolvla_click_bell_run1 3 --steps=50000
#
# Set SMOLVLA_NOHUP=1 to launch in the background and write a log under
# $LEROBOT_ROOT/outputs/train/logs by default.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TASK_NAME="${1:?Usage: bash policy/smolvla/finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]}"
DATASET_ROOT="${2:?Usage: bash policy/smolvla/finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]}"
OUTPUT_DIR="${3:?Usage: bash policy/smolvla/finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

CONDA_ROOT="${CONDA_ROOT:-}"
SMOLVLA_CONDA_ENV="${SMOLVLA_CONDA_ENV:-smolvla}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${SMOLVLA_LEROBOT_ROOT:-}}"
DATASET_REPO_ID="${SMOLVLA_DATASET_REPO_ID:-RoboSynChallenge/cobotmagic_Sim_${TASK_NAME}}"
JOB_NAME="${SMOLVLA_JOB_NAME:-$(basename "$OUTPUT_DIR")}"
# Feature-key rename map passed to lerobot-train.
#
# Default is empty. Datasets from server/normalize_v21_dataset.py already use
# observation.state and observation.images.cam_high / cam_left_wrist /
# cam_right_wrist, so there is nothing to rename -- and a non-empty map is not
# harmless: LeRobot b6ec006 (configs/train.py) rejects any rename_map when the
# policy is built fresh with --policy.type rather than loaded from a checkpoint,
# failing with "`rename_map` requires a pretrained policy checkpoint". The old
# default did exactly that, so the pilot would have died at config validation.
#
# SMOLVLA_RENAME_MAP=legacy restores the original mapping from the raw v2.1 key
# names to camera1/2/3. It only makes sense on LeRobot versions that allow it, or
# together with a pretrained --policy.path; b6ec006 cannot read v2.1 data at all,
# so normalising first is the path that works here. Any other value is taken as
# literal JSON.
LEGACY_RENAME_MAP='{"observation.qpos":"observation.state","cam_high.color":"observation.images.camera1","cam_left_wrist.color":"observation.images.camera2","cam_right_wrist.color":"observation.images.camera3"}'
case "${SMOLVLA_RENAME_MAP:-}" in
    "")     RENAME_MAP='{}' ;;
    legacy) RENAME_MAP="$LEGACY_RENAME_MAP" ;;
    *)      RENAME_MAP="$SMOLVLA_RENAME_MAP" ;;
esac
if [[ "$RENAME_MAP" != "{}" ]]; then
    has_policy_path=0
    for arg in "${EXTRA_ARGS[@]}"; do
        [[ "$arg" == --policy.path* ]] && has_policy_path=1
    done
    if (( ! has_policy_path )); then
        echo "Warning: a non-empty rename_map with --policy.type=smolvla and no --policy.path is" >&2
        echo "         rejected by LeRobot b6ec006 and newer. Expect a config validation error." >&2
    fi
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Error: dataset root does not exist: $DATASET_ROOT" >&2
    exit 1
fi

if [[ "${SMOLVLA_USE_CONDA:-auto}" != "0" && -n "$CONDA_ROOT" && -f "$CONDA_ROOT/bin/activate" ]]; then
    source "$CONDA_ROOT/bin/activate"
    if conda env list | awk '{print $1}' | grep -qx "$SMOLVLA_CONDA_ENV"; then
        conda activate "$SMOLVLA_CONDA_ENV"
    else
        echo "Warning: conda env '$SMOLVLA_CONDA_ENV' not found; using current Python environment." >&2
    fi
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [[ -z "$LEROBOT_ROOT" && -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    LEROBOT_ROOT="$SCRIPT_DIR/lerobot"
fi

if [[ -n "$LEROBOT_ROOT" && ! -d "$LEROBOT_ROOT" ]]; then
    echo "Error: configured LeRobot root does not exist: $LEROBOT_ROOT" >&2
    exit 1
fi

if [[ -z "$LEROBOT_ROOT" && ! "$(command -v lerobot-train || true)" ]]; then
    if [[ "${SMOLVLA_AUTO_SETUP:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        bash "$SCRIPT_DIR/setup_lerobot.sh" "$SCRIPT_DIR/lerobot"
        LEROBOT_ROOT="$SCRIPT_DIR/lerobot"
    else
        echo "Error: cannot find lerobot-train and no LeRobot source is configured." >&2
        echo "Set SMOLVLA_LEROBOT_ROOT=policy/smolvla/lerobot, install lerobot in this env, or run:" >&2
        echo "  bash policy/smolvla/setup_lerobot.sh" >&2
        exit 1
    fi
fi

echo "========================================="
echo "  SmolVLA Policy Finetune"
echo "  Task:       $TASK_NAME"
echo "  Dataset:    $DATASET_ROOT"
echo "  Repo ID:    $DATASET_REPO_ID"
echo "  Output:     $OUTPUT_DIR"
echo "  GPU:        $GPU_ID"
echo "  LeRobot:    ${LEROBOT_ROOT:-installed package}"
echo "  Conda env:  ${CONDA_DEFAULT_ENV:-current environment}"
echo "========================================="

if [[ -n "$LEROBOT_ROOT" ]]; then
    cd "$LEROBOT_ROOT"
else
    cd "$REPO_ROOT"
fi

TRAIN_CMD=(
    lerobot-train
    --policy.type=smolvla
    --policy.load_vlm_weights=true
    --policy.push_to_hub=false
    --policy.device=cuda
    --batch_size=32
    --steps=50000
    --num_workers=8
    --persistent_workers=true
    --wandb.enable=false
    "--dataset.repo_id=$DATASET_REPO_ID"
    "--dataset.root=$DATASET_ROOT"
    "--rename_map=$RENAME_MAP"
    "--output_dir=$OUTPUT_DIR"
    "--job_name=$JOB_NAME"
    "${EXTRA_ARGS[@]}"
)

case "${SMOLVLA_NOHUP:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        LOG_DIR="${SMOLVLA_LOG_DIR:-${LEROBOT_ROOT:-$REPO_ROOT}/outputs/train/logs}"
        mkdir -p "$LOG_DIR"
        LOG_PATH="$LOG_DIR/$JOB_NAME.log"
        nohup "${TRAIN_CMD[@]}" > "$LOG_PATH" 2>&1 &
        echo "Started SmolVLA finetune in background."
        echo "  PID: $!"
        echo "  Log: $LOG_PATH"
        ;;
    *)
        "${TRAIN_CMD[@]}"
        ;;
esac
