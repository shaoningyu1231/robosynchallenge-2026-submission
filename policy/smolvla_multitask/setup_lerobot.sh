#!/bin/bash
# ----------------------------------------------------------------------------
# Build the SmolVLA worker environment for this policy.
#
#   bash policy/smolvla_multitask/setup_lerobot.sh
#
# Creates policy/smolvla_multitask/.venv (Python 3.12) with LeRobot at a pinned
# commit, installed editable from policy/smolvla_multitask/lerobot, and every
# other package at the versions in requirements.lock. eval.sh picks up .venv
# automatically.
#
# This must be a separate environment from the simulator's: lerobot[smolvla]
# needs numpy>=2 (through rerun-sdk), while the simulator's prebuilt binaries
# need numpy<2. The two only talk through smolvla_worker.py's JSON protocol.
#
# Why pinned: the official policy/smolvla/setup_lerobot.sh follows LeRobot
# main. LeRobot 0.4.4 cannot load the released SmolVLA checkpoints at all
# (its SmolVLAConfig has no pretrained_revision), so "whatever main is today"
# is not reproducible in either direction.
#
# Environment:
#   PYTHON312         Python 3.12 interpreter to build the venv from (default: uv's python 3.12)
#   UV_DEFAULT_INDEX  package index (default: PyPI), e.g. a local PyPI mirror
#   UV_CACHE_DIR      uv's cache; on the same disk as .venv, packages are hard-linked, not copied
#   LEROBOT_REPO      LeRobot git URL or local clone (default: GitHub)
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
LEROBOT_COMMIT=b6ec0060779550c0a157ae34feb89e0cf86012a8
LEROBOT_DIR="$SCRIPT_DIR/lerobot"
VENV="$SCRIPT_DIR/.venv"

command -v uv >/dev/null || {
    echo "Error: uv is required (https://docs.astral.sh/uv/getting-started/installation/)." >&2
    exit 1
}

if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
    git init -q "$LEROBOT_DIR"
    git -C "$LEROBOT_DIR" remote add origin "$LEROBOT_REPO"
fi
git -C "$LEROBOT_DIR" fetch -q --depth=1 origin "$LEROBOT_COMMIT"
git -C "$LEROBOT_DIR" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$LEROBOT_DIR" rev-parse HEAD)" == "$LEROBOT_COMMIT" ]] || {
    echo "Error: LeRobot checkout is not at $LEROBOT_COMMIT" >&2
    exit 1
}

if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python "${PYTHON312:-3.12}" "$VENV"
fi

# [dataset] is included so the same environment can also train (lerobot-train
# imports lerobot.datasets); inference alone would need only [smolvla].
uv pip install --python "$VENV/bin/python" \
    --constraint "$SCRIPT_DIR/requirements.lock" \
    -e "$LEROBOT_DIR[smolvla,dataset]"

"$VENV/bin/python" - <<'EOF'
import sys
import lerobot, torch, transformers
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401

print(f"python {sys.version.split()[0]}, lerobot {lerobot.__version__}, torch {torch.__version__}, "
      f"transformers {transformers.__version__}, cuda available: {torch.cuda.is_available()}")
EOF
echo "SmolVLA worker environment ready: $VENV"
