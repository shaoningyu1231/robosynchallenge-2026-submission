#!/bin/bash
# ----------------------------------------------------------------------------
# Evaluate the SmolVLA multi-task submission on one task.
#
#   bash policy/smolvla_multitask/eval.sh <task_name> <setting> <gpu_id> [--key value ...]
#
#   bash policy/smolvla_multitask/eval.sh click_bell random 0 --max_episodes 100 --seed 0
#
# The checkpoint comes from deploy_policy.yml (checkpoints.tasks.<task>, then
# checkpoints.default) and is downloaded from Hugging Face at a fixed revision.
# To evaluate a local directory instead, pass --checkpoint_dir <dir>.
# Any other --key value pair overrides the same key in deploy_policy.yml.
#
# Environment:
#   SMOLVLA_PYTHON         Python of the SmolVLA worker (default: .venv here, made by setup_lerobot.sh)
#   SMOLVLA_DEPLOY_CONFIG  alternative config, e.g. test_configs/official_baseline.yml
#   PYTHON_BIN             Python of the simulator (default: EmbodiChain's .venv)
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="${ROBOSYN_VENV_DIR:-$EMBODICHAIN_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

POLICY_NAME=smolvla_multitask
USAGE="Usage: bash policy/$POLICY_NAME/eval.sh <task_name> <setting> <gpu_id> [--key value ...]"

TASK_NAME="${1:?$USAGE}"
SETTING="${2:?$USAGE}"
GPU_ID="${3:?$USAGE}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    # The official smolvla/eval.sh takes a checkpoint path third; this one does not.
    echo "Error: gpu_id must be an integer, got '$GPU_ID'. $USAGE" >&2
    echo "       To use a local checkpoint, pass --checkpoint_dir <dir> after gpu_id." >&2
    exit 1
fi
shift 3
EXTRA_ARGS=("$@")
if (( ${#EXTRA_ARGS[@]} % 2 )); then
    echo "Error: overrides must be --key value pairs, got: ${EXTRA_ARGS[*]}" >&2
    exit 1
fi

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

DEPLOY_CONFIG="${SMOLVLA_DEPLOY_CONFIG:-policy/$POLICY_NAME/deploy_policy.yml}"

if [[ -z "${SMOLVLA_PYTHON:-}" && -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    export SMOLVLA_PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi
# Placed before EXTRA_ARGS on the command line, so an explicit override still wins.
ENV_ARGS=()
if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    ENV_ARGS+=(--smolvla_python "$SMOLVLA_PYTHON")
fi
if [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]]; then
    ENV_ARGS+=(--lerobot_root "$SMOLVLA_LEROBOT_ROOT")
fi

# SimulationManager.destroy() otherwise ends the process with os._exit(0) during
# teardown, which reports success even when the evaluation raised. With 0,
# scripts/eval_policy.py drains the cleanup queue itself and real exit codes
# survive. (The official act/eval.sh sets the same.)
export EMBODICHAIN_SIM_EXIT_PROCESS="${EMBODICHAIN_SIM_EXIT_PROCESS:-0}"

# DexSim uses physical GPU ordinals from --gpu_id. Keep the simulator parent
# unmasked, then mask only the SmolVLA worker via --smolvla_cuda_visible_devices.
unset CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  SmolVLA Multi-task Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Config:     $DEPLOY_CONFIG"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
echo "  Worker Py:  ${SMOLVLA_PYTHON:-current eval Python}"
echo "========================================="

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: cannot find executable Python: $PYTHON_BIN" >&2
    exit 1
fi

cd "$REPO_ROOT"
if [[ ! -f "$DEPLOY_CONFIG" ]]; then
    echo "Error: config not found: $DEPLOY_CONFIG" >&2
    exit 1
fi

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config "$DEPLOY_CONFIG" \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --gpu_id "$GPU_ID" \
    --smolvla_cuda_visible_devices "$GPU_ID" \
    "${ENV_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
