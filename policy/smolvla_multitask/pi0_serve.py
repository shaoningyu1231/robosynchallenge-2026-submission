"""pi0 backend: the policy server, run in policy/pi05's own environment by pi0_backend.py.

    <policy/pi05/.venv python> policy/smolvla_multitask/pi0_serve.py --port P policy:checkpoint \
        --policy.config C --policy.dir D

openpi's scripts/serve_policy.py (its create_policy is reused; the pi05 tree is only patched with the named task configs)
with the flow-matching noise of every call supplied by the client, as in the research server the drawer checkpoint was
evaluated with (policy/pi0_research/serve_noise.py of the research repository): each observation must carry the noise
block under NOISE_KEY -- shape (action_horizon, action_dim) float32, finite -- which is removed before the policy's input
transforms and passed to openpi's Policy.infer(obs, noise=...). A call without it is refused. The reply carries the
sha256 (first 16 hex) of the noise used, for the client to check.

Added for the submission: a watchdog thread that ends this process when the evaluator that started it is gone (its
parent pid changes), so a killed evaluation never leaves a model on the GPU.
"""

import hashlib
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tyro

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pi05" / "scripts"))
import serve_policy as _serve  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402

NOISE_KEY = "research/noise"


def noise_digest(noise: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(noise, dtype=np.float32).tobytes()).hexdigest()[:16]


class NoisePolicy:
    """Wraps an openpi Policy: infer(obs) takes the noise out of obs and passes it to Policy.infer."""

    def __init__(self, policy, shape):
        self._policy = policy
        self._shape = tuple(shape)

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        if NOISE_KEY not in obs:
            raise ValueError(f"this server requires the noise of every call under {NOISE_KEY!r}")
        noise = np.asarray(obs.pop(NOISE_KEY))
        if noise.shape != self._shape or noise.dtype != np.float32 or not np.isfinite(noise).all():
            raise ValueError(f"noise must be finite float32 {self._shape}, got {noise.dtype} {noise.shape}")
        out = self._policy.infer(obs, noise=noise)
        out["research_noise_digest"] = noise_digest(noise)
        return out

    def reset(self) -> None:
        self._policy.reset()


def watch_parent(interval_s: float = 2.0) -> None:
    parent = os.getppid()

    def loop():
        while True:
            time.sleep(interval_s)
            if os.getppid() != parent:
                logging.warning("the evaluator (pid %s) is gone; stopping the pi0 server", parent)
                os._exit(0)

    threading.Thread(target=loop, daemon=True, name="parent-watchdog").start()


def main(args: _serve.Args) -> None:
    if not isinstance(args.policy, _serve.Checkpoint):
        raise SystemExit("pi0_serve serves a trained checkpoint only")
    watch_parent()
    model = _config.get_config(args.policy.config).model
    shape = (int(model.action_horizon), int(model.action_dim))
    policy = _serve.create_policy(args)
    metadata = dict(policy.metadata or {})
    metadata["research_noise"] = {"key": NOISE_KEY, "shape": list(shape), "dtype": "float32", "required": True}
    logging.info("pi0 server (host: %s), noise %s", socket.gethostname(), shape)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=NoisePolicy(policy, shape), host="127.0.0.1", port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(_serve.Args))
