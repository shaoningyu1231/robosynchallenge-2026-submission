#!/bin/bash
# ----------------------------------------------------------------------------
# Environment for the tasks routed to pi0 (checkpoint entries with backend: pi0).
#
#   bash policy/smolvla_multitask/setup_pi0.sh
#
# 1. policy/pi05/.venv: the official pi0/pi0.5 environment of this repository,
#    built from its own lock file exactly as policy/pi05/README.md describes,
#    with --frozen so nothing outside uv.lock is resolved. The pi0 server
#    (pi0_serve.py) runs in it.
# 2. policy/smolvla_multitask/.pi0_client_deps: msgpack for the client side,
#    which runs inside the simulator's Python (PYTHON_BIN, as in eval.sh); the
#    version and hashes are taken from the same uv.lock and enforced. Nothing is
#    installed into the simulator's environment itself.
#
# 3. the PaliGemma tokenizer, downloaded once from its original source
#    (storage.googleapis.com/big_vision) and cached only if its sha256 is the
#    one pinned in pi0_backend.py (not shipped with the weights).
#
# Needs uv (https://docs.astral.sh/uv/) and network access to the sources
# named in policy/pi05/uv.lock and to storage.googleapis.com. About 12 GB on disk.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
PYTHON_BIN="${PYTHON_BIN:-${ROBOSYN_VENV_DIR:-$EMBODICHAIN_ROOT/.venv}/bin/python}"
UV="${UV:-uv}"
PI05_DIR="$REPO_ROOT/policy/pi05"
DEPS_DIR="$SCRIPT_DIR/.pi0_client_deps"

command -v "$UV" >/dev/null || { echo "Error: uv not found (set UV=/path/to/uv)" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Error: simulator Python not found: $PYTHON_BIN" >&2; exit 1; }

echo "== policy/pi05 environment (uv sync --frozen, lock $(sha256sum "$PI05_DIR/uv.lock" | cut -c1-16))"
(cd "$PI05_DIR" && GIT_LFS_SKIP_SMUDGE=1 "$UV" sync --frozen)

echo "== client dependency for the simulator side -> $DEPS_DIR"
req="$(mktemp)"
"$PI05_DIR/.venv/bin/python" - "$PI05_DIR/uv.lock" "$req" <<'PY'
import sys, tomllib
lock = tomllib.load(open(sys.argv[1], "rb"))
pkg = [p for p in lock["package"] if p["name"] == "msgpack"]
assert len(pkg) == 1, pkg
p = pkg[0]
hashes = [w["hash"] for w in p.get("wheels", [])] + ([p["sdist"]["hash"]] if "sdist" in p else [])
open(sys.argv[2], "w").write(f"msgpack=={p['version']} " + " ".join(f"--hash={h}" for h in hashes) + "\n")
print("msgpack", p["version"], len(hashes), "hashes from the lock")
PY
"$UV" pip install --python "$PYTHON_BIN" --target "$DEPS_DIR" --no-deps --require-hashes -r "$req"
rm -f "$req"

echo "== PaliGemma tokenizer from its original source (sha256 pinned in pi0_backend.py) -> data home"
(cd "$REPO_ROOT" && "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '.'); \
from pathlib import Path; from policy.smolvla_multitask import pi0_backend as B; \
print('tokenizer:', B.prepare_data_home(Path('/nonexistent'), Path('${PI0_OPENPI_DATA_HOME:-$SCRIPT_DIR/.pi0_openpi_data}')))")

echo "== checks"
"$PYTHON_BIN" -c "import sys; sys.path[:0] = ['$DEPS_DIR', '$PI05_DIR/packages/openpi-client/src']; \
import msgpack, websockets; from openpi_client import websocket_client_policy; \
print('client side: msgpack', msgpack.version, 'websockets', websockets.__version__)"
(cd "$PI05_DIR" && .venv/bin/python -c "from openpi.training import config; c = config.get_config('pi0_drawer_lora_b1_tomato'); \
print('server side: openpi config', c.name, 'horizon', c.model.action_horizon)")
echo "PI0_SETUP=OK"
