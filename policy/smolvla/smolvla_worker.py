#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set SMOLVLA_LEROBOT_ROOT."
            )

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import PolicyProcessorPipeline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot. Install LeRobot in the SmolVLA Python env, "
            "or run policy/smolvla/setup_lerobot.sh and set SMOLVLA_LEROBOT_ROOT."
        ) from exc

    # The converters live in lerobot.processor.converters and are re-exported
    # from the package root only on newer LeRobot. Release 0.4.4 exports just
    # two of the four, so importing them all from the root raises ImportError --
    # which the handler above does not catch, since a missing *name* is not a
    # ModuleNotFoundError. Take the root first, fall back to the module that has
    # held them all along.
    try:
        from lerobot.processor import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ImportError:
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    policy.to(args.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "reset":
                policy.reset()
                _send({"ok": True})
                continue

            if cmd != "infer":
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                obs, task = _load_obs(request["obs_path"])
                obs = {key: torch.from_numpy(value) for key, value in obs.items()}
                obs["task"] = task
                proc_obs = preprocessor(obs)

                num_actions = int(request.get("n_action_steps", 1))
                actions = []
                forwards, forward_s = 0, 0.0
                if request.get("replan", False):
                    # Predict a fresh chunk from this observation and return its first
                    # num_actions steps, bypassing the LeRobot action queue.
                    chunk_size = int(policy.config.chunk_size)
                    if not 1 <= num_actions <= chunk_size:
                        raise ValueError(f"replan steps must be in [1, {chunk_size}], got {num_actions}")
                    started = time.perf_counter()
                    with torch.no_grad():
                        chunk = policy.predict_action_chunk(proc_obs)
                    _synchronize(args.device)
                    forwards, forward_s = 1, time.perf_counter() - started
                    for index in range(num_actions):
                        action = postprocessor(chunk[:, index])
                        actions.append(action.squeeze(0).cpu().numpy().astype(np.float32).tolist())
                else:
                    for _ in range(num_actions):
                        # select_action only runs the model when its action queue is empty.
                        runs_forward = policy._check_get_actions_condition()
                        started = time.perf_counter()
                        action = policy.select_action(proc_obs)
                        if runs_forward:
                            _synchronize(args.device)
                            forwards += 1
                            forward_s += time.perf_counter() - started
                        action = postprocessor(action)
                        actions.append(action.squeeze(0).cpu().numpy().astype(np.float32).tolist())

            _send({"ok": True, "actions": actions, "forwards": forwards, "forward_s": forward_s})
        except Exception as exc:  # noqa: BLE001
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
