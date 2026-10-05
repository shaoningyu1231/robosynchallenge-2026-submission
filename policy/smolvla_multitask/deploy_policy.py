# ----------------------------------------------------------------------------
# SmolVLA multi-task submission adapter for RoboSynChallenge (checkpoint choice: checkpoints.py)
# ----------------------------------------------------------------------------

from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from policy.inference_timing import finish_inference, start_inference
from policy.smolvla_multitask import noise_seed, pi0_backend, task_norm as task_norm_module
from policy.smolvla_multitask.checkpoints import VLM_REQUIRED_FILES, materialize, resolve_source, resolve_vlm


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_env(value: np.ndarray, env_index: int = 0) -> np.ndarray:
    if value.ndim == 0:
        return value
    return value[env_index]


def _extract_image(obs, sensor_name: str, env_index: int = 0) -> np.ndarray:
    image = _to_numpy(obs["sensor"][sensor_name]["color"])
    image = _select_env(image, env_index=env_index)
    if image.ndim != 3:
        raise ValueError(f"Expected image for {sensor_name} to be 3D, got {image.shape}")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        chw = image
    else:
        chw = np.moveaxis(image, -1, 0)

    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)
    elif chw.shape[0] > 3:
        chw = chw[:3]

    if np.issubdtype(chw.dtype, np.floating):
        if float(np.nanmax(chw)) <= 1.0:
            chw = chw * 255.0
        chw = np.clip(chw, 0, 255).astype(np.uint8)
    elif chw.dtype != np.uint8:
        chw = np.clip(chw, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(chw)


def _extract_joint(obs, key: str, fallback: np.ndarray | None = None, env_index: int = 0) -> np.ndarray:
    robot_obs = obs["robot"]
    if key in robot_obs:
        value = _to_numpy(robot_obs[key])
        value = _select_env(value, env_index=env_index).astype(np.float32, copy=False).reshape(-1)
        return np.ascontiguousarray(value)
    if fallback is None:
        raise KeyError(f"Missing robot observation key: {key}")
    return np.zeros_like(fallback, dtype=np.float32)


def encode_obs(obs, image_keys: dict[str, str] | None = None):
    state = _extract_joint(obs, "qpos")
    qvel = _extract_joint(obs, "qvel", fallback=state)
    qf = _extract_joint(obs, "qf", fallback=state)

    encoded = {
        "observation.state": state,
        "observation.qvel": qvel,
        "observation.qf": qf,
    }
    image_keys = image_keys or {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
    }
    for obs_key, sensor_name in image_keys.items():
        encoded[obs_key] = _extract_image(obs, sensor_name)
    return encoded


def _resolve_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"auto", "none", ""}:
            return None
        return normalized in {"1", "true", "yes", "y", "on"}
    return bool(value)


def encode_action(
    action,
    env,
    rescale_gripper: bool = True,
    gripper_indices: tuple[int, ...] = (6, 13),
    gripper_source_range: tuple[float, float] = (0.0, 1.0),
):
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    action_array = action_array[:env_action_dim]
    if rescale_gripper:
        action_array = action_array.copy()
        action_space = env.unwrapped.single_action_space
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        src_low, src_high = gripper_source_range
        src_span = max(src_high - src_low, np.finfo(np.float32).eps)
        for gripper_idx in gripper_indices:
            if gripper_idx >= env_action_dim:
                continue
            gripper_01 = np.clip((action_array[gripper_idx] - src_low) / src_span, 0.0, 1.0)
            action_array[gripper_idx] = low[gripper_idx] + gripper_01 * (
                high[gripper_idx] - low[gripper_idx]
            )
    action_tensor = torch.as_tensor(action_array, dtype=torch.float32, device=env.unwrapped.device)
    return action_tensor.unsqueeze(0)


def _empty_timing() -> dict:
    return {"rpc_calls": 0, "forwards": 0, "rpc_s": 0.0, "forward_s": 0.0}


def _empty_episode_record() -> dict:
    # Kept apart from episode_timing, which log_episode_timing clears when the
    # episode ends -- before eval_policy.py asks for this record.
    return {
        "env_seed": None, "noise_seed": None, "forwards": 0, "noise_digests": [],
        "calls": [], "actions": hashlib.sha256(),
    }


def _digest(arrays) -> str:
    """First 16 hex digits of SHA-256 over the arrays' raw bytes, in order."""
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()[:16]


def _episode_env_seed(env) -> int:
    """The environment seed scripts/eval_policy.py reset this episode with.

    prepare_episode_reset stores it on the unwrapped env; RecordingEnvProxy
    forwards attribute access, so this works with and without video recording.
    """
    seed = getattr(getattr(env, "unwrapped", env), "_robosyn_episode_seed", None)
    if seed is None:
        raise RuntimeError(
            "smolvla_noise_seed is set, but the evaluator recorded no episode seed "
            "(_robosyn_episode_seed); refusing to fall back to unseeded noise"
        )
    return int(seed)


class SmolVLAWorkerClient:
    def __init__(self, usr_args: dict):
        policy_root = Path(__file__).resolve().parent
        self.checkpoint_dir = Path(usr_args["checkpoint_dir"]).resolve()
        lerobot_root = (
            usr_args.get("lerobot_root")
            or os.environ.get("SMOLVLA_LEROBOT_ROOT")
            or os.environ.get("LEROBOT_ROOT")
            or ""
        )
        self.lerobot_root = Path(lerobot_root).expanduser().resolve() if lerobot_root else None
        self.python_bin = usr_args.get("smolvla_python") or os.environ.get("SMOLVLA_PYTHON") or sys.executable
        self.device = usr_args.get("pytorch_device", "cuda")
        self.smolvla_steps = int(usr_args.get("smolvla_steps", 10))
        replan_steps = usr_args.get("smolvla_replan_steps")
        self.replan_steps = None if replan_steps in (None, "", "None", "null") else int(replan_steps)
        if self.replan_steps is not None and self.replan_steps < 1:
            # The upper bound (chunk_size) is checked by the worker, which knows the model.
            raise ValueError(f"smolvla_replan_steps must be >= 1, got {self.replan_steps}")
        self.episode_timing = _empty_timing()
        self.noise_base = noise_seed.parse_base(usr_args.get("smolvla_noise_seed"))
        self.episode_record = _empty_episode_record()
        self.check_success_each_step = bool(
            _resolve_optional_bool(usr_args.get("smolvla_check_success_each_step", False))
        )
        # required: the checkpoint's own per-task statistics decide the transform
        # and the units. legacy: exactly the behaviour Q1a and the official
        # checkpoints were evaluated with, heuristic included.
        self.task_norm = usr_args.get("smolvla_task_norm")
        self._action_space_checked = False
        self._state_range_checked = False
        if self.task_norm is not None:
            self.gripper_indices = self.task_norm.gripper_indices
            self.gripper_source_range = self.task_norm.gripper_source_range
            self.rescale_gripper = self.task_norm.rescale_gripper
        else:
            # A mixed routing cannot state these: they are refused for the
            # required-mode task, so the config leaves them out and the legacy
            # tasks take the same defaults the official baseline config spells out.
            indices = usr_args.get("smolvla_gripper_indices")
            self.gripper_indices = tuple(int(idx) for idx in ([6, 13] if indices is None else indices))
            source_range = usr_args.get("smolvla_gripper_source_range") or [0.0, 1.0]
            self.gripper_source_range = (float(source_range[0]), float(source_range[1]))
            self.rescale_gripper = self._resolve_gripper_rescale(
                usr_args.get("smolvla_rescale_gripper") or "auto")
        self.debug_actions = bool(usr_args.get("smolvla_debug_actions", False))
        self.debug_action_chunks = int(usr_args.get("smolvla_debug_action_chunks", 1))
        self.debug_action_log_path = usr_args.get("smolvla_debug_action_log_path")
        self._debug_chunks_printed = 0
        # Debugging reproducibility: save what the first calls of each episode
        # sent and got back, as episodeNNN_callNNN.npz (off by default).
        dump_dir = usr_args.get("smolvla_dump_obs_dir")
        self.dump_obs_dir = None if dump_dir in (None, "", "None", "null") else Path(dump_dir)
        self.dump_obs_calls = int(usr_args.get("smolvla_dump_obs_calls", 5))
        self._episodes_started = 0
        self.cuda_visible_devices = str(
            usr_args.get("smolvla_cuda_visible_devices", usr_args.get("gpu_id", 0))
        )
        self.image_keys = self._resolve_image_keys()
        worker_path = policy_root / "smolvla_worker.py"
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        worker_cmd = [
            self.python_bin,
            str(worker_path),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--device",
            self.device,
        ]
        if self.lerobot_root is not None:
            worker_cmd.extend(["--lerobot-root", str(self.lerobot_root)])
        vlm_path = usr_args.get("smolvla_vlm_path")
        if vlm_path:
            worker_cmd.extend(["--vlm-path", str(vlm_path), "--vlm-repo-id", str(usr_args["smolvla_vlm_repo_id"])])
            # Everything the worker needs is local by now (checkpoint and pinned
            # VLM). Offline mode turns any remaining lookup by repo id into an
            # error instead of a quiet download of the Hub's current main.
            worker_env["HF_HUB_OFFLINE"] = "1"
            worker_env["TRANSFORMERS_OFFLINE"] = "1"
        self.proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        atexit.register(self.close)

    def _resolve_gripper_rescale(self, override) -> bool:
        override_value = _resolve_optional_bool(override)
        if override_value is not None:
            return override_value

        stats_path = self.checkpoint_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        try:
            from safetensors.torch import load_file

            stats = load_file(str(stats_path))
            action_max = _to_numpy(stats["action.max"]).reshape(-1)
        except Exception:
            # Most RoboSyn CobotMagic tasks use 0-1 gripper data, while the env
            # expects physical qpos. Prefer the safer conversion if stats are unavailable.
            return True

        valid_indices = [idx for idx in self.gripper_indices if idx < action_max.shape[0]]
        gripper_max = action_max[valid_indices]
        return bool(gripper_max.size and float(np.nanmax(gripper_max)) > 0.5)

    def standardize_state(self, state):
        """The observation in the checkpoint's training coordinates.

        The raw gripper reading is range-checked on the first observation of an
        episode only. It answers "is this environment the one these statistics
        describe", which is a property of the episode, not of the frame: checking
        every step would end a run mid-trajectory over a transient overshoot,
        and it cannot prove a unit anyway (metres [0, 0.05] sit inside the ratio
        range), so it is a sanity check, not the unit proof.
        """
        if not self._state_range_checked:
            self._state_range_checked = True
            problem = self.task_norm.state_range_problem(state)
            if problem:
                raise task_norm_module.TaskNormalizationError(
                    f"{problem}; the observation does not match the statistics in the checkpoint")
        return self.task_norm.standardize_state(state)

    def check_action_space(self, env) -> None:
        """Once per episode: does the env's gripper range match the metadata?"""
        if getattr(self, "_action_space_checked", False):
            return
        self._action_space_checked = True
        action_space = env.unwrapped.single_action_space
        low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
        problem = self.task_norm.action_space_problem(low, high)
        if problem:
            raise task_norm_module.TaskNormalizationError(problem)

    def close(self) -> None:
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def _resolve_image_keys(self) -> dict[str, str]:
        config_path = self.checkpoint_dir / "config.json"
        config = json.loads(config_path.read_text())
        input_features = config.get("input_features", {})

        if "observation.images.camera1" in input_features:
            return {
                "observation.images.camera1": "cam_high",
                "observation.images.camera2": "cam_left_wrist",
                "observation.images.camera3": "cam_right_wrist",
            }

        return {
            "observation.images.cam_high": "cam_high",
            "observation.images.cam_left_wrist": "cam_left_wrist",
            "observation.images.cam_right_wrist": "cam_right_wrist",
        }

    def _rpc(self, payload: dict) -> dict:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("SmolVLA worker pipes are not available.")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("SmolVLA worker exited unexpectedly.")
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "SmolVLA worker error"))
            return response

    def infer(self, obs: dict[str, np.ndarray], task: str) -> list[np.ndarray]:
        fd, obs_path = tempfile.mkstemp(prefix="smolvla_obs_", suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(obs_path, **obs, task=np.asarray(task))
            response = self._rpc(
                {
                    "cmd": "infer",
                    "obs_path": obs_path,
                    "n_action_steps": self.replan_steps if self.replan_steps is not None else self.smolvla_steps,
                    "replan": self.replan_steps is not None,
                    "seeded_noise": self.noise_base is not None,
                }
            )
        finally:
            if os.path.exists(obs_path):
                os.remove(obs_path)

        self.episode_timing["forwards"] += int(response.get("forwards", 0))
        self.episode_timing["forward_s"] += float(response.get("forward_s", 0.0))
        actions = [np.asarray(action, dtype=np.float32) for action in response["actions"]]
        record = self.episode_record
        record["forwards"] += int(response.get("forwards", 0))
        record["noise_digests"].extend(response.get("noise_digests", []))
        for action in actions:
            record["actions"].update(action.tobytes())
        if self.dump_obs_dir is not None and len(record["calls"]) < self.dump_obs_calls:
            self.dump_obs_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.dump_obs_dir / f"episode{self._episodes_started - 1:03d}_call{len(record['calls']):03d}.npz",
                **obs,
                task=np.asarray(task),
                actions=np.asarray(actions, dtype=np.float32),
                noise_digests=np.asarray(response.get("noise_digests", [])),
                env_seed=np.asarray(-1 if record["env_seed"] is None else record["env_seed"]),
                forward_indices=np.asarray(response.get("forward_indices", []), dtype=np.int64),
                chunk_positions=np.asarray(response.get("chunk_positions", []), dtype=np.int64),
            )
        # Per call, so two runs that drift apart show where: in the observation
        # (the simulator), or in the actions given the same observation and noise.
        # Only calls with a forward (forward_indices non-empty) computed actions from
        # this call's observation; the others popped the queue of an earlier chunk.
        record["calls"].append({
            "obs": _digest(obs[key] for key in sorted(obs)),
            "actions": _digest(actions),
            "forwards": int(response.get("forwards", 0)),
            "forward_indices": list(response.get("forward_indices", [])),
            "chunk_positions": list(response.get("chunk_positions", [])),
        })
        return actions

    def seed_noise(self, env_seed: int) -> None:
        """Seed this episode's noise generator in the worker from the environment seed."""
        seed = noise_seed.policy_noise_seed(self.noise_base, env_seed)
        self._rpc({"cmd": "seed_noise", "seed": seed})
        self.episode_record["env_seed"] = int(env_seed)
        self.episode_record["noise_seed"] = seed

    def episode_metadata(self) -> dict:
        record = self.episode_record
        return {
            "env_seed": record["env_seed"],
            "noise_seed": record["noise_seed"],
            "forwards": record["forwards"],
            "noise_digests": list(record["noise_digests"]),
            # Every action the policy returned, in order, as float32 bytes.
            "action_digest": record["actions"].hexdigest()[:16],
            # One entry per worker call: digests of the encoded observation sent
            # and of the actions returned, the forwards that call ran (count and
            # episode-wide indices) and each returned action's position in its chunk.
            "calls": [dict(call) for call in record["calls"]],
        }

    def record_call(self, seconds: float) -> None:
        self.episode_timing["rpc_calls"] += 1
        self.episode_timing["rpc_s"] += seconds

    def log_episode_timing(self) -> None:
        """One line per episode for tools/parse_eval_log.py: worker calls vs actual model forwards."""
        timing = self.episode_timing
        print(
            f"[SmolVLA timing] rpc_calls={timing['rpc_calls']} forwards={timing['forwards']} "
            f"rpc_s={timing['rpc_s']:.6f} forward_s={timing['forward_s']:.6f}",
            flush=True,
        )
        self.episode_timing = _empty_timing()

    def reset(self) -> None:
        # Both checks are per-episode: the action space and the observation's
        # units are properties of the environment this episode runs in.
        self._action_space_checked = False
        self._state_range_checked = False
        self._episodes_started += 1
        self._debug_chunks_printed = 0
        self.episode_timing = _empty_timing()
        self.episode_record = _empty_episode_record()
        self._rpc({"cmd": "reset"})

    def maybe_print_action_debug(self, actions: list[np.ndarray]) -> None:
        if not self.debug_actions or self._debug_chunks_printed >= self.debug_action_chunks:
            return
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[1] <= max(self.gripper_indices, default=-1):
            print(f"[SmolVLA action debug] unexpected action shape: {action_array.shape}")
            self._debug_chunks_printed += 1
            return

        gripper_values = {idx: action_array[:, idx] for idx in self.gripper_indices}
        gripper_summary = " ".join(
            f"gripper[{idx}] first10={np.round(values[:10], 4).tolist()} "
            f"min/max=({values.min():.4f}, {values.max():.4f})"
            for idx, values in gripper_values.items()
        )
        summary = (
            "[SmolVLA action debug] "
            f"chunk={self._debug_chunks_printed} shape={action_array.shape} "
            f"{gripper_summary}"
        )
        detail = (
            "[SmolVLA action debug] "
            f"first action={np.round(action_array[0], 4).tolist()} "
            f"last action={np.round(action_array[-1], 4).tolist()}"
        )
        print(summary, flush=True)
        print(detail, flush=True)
        if self.debug_action_log_path:
            with open(self.debug_action_log_path, "a", encoding="utf-8") as debug_file:
                debug_file.write(summary + "\n")
                debug_file.write(detail + "\n")
        self._debug_chunks_printed += 1

    def maybe_print_env_action_debug(self, action_tensor: torch.Tensor) -> None:
        if not self.debug_actions or self._debug_chunks_printed > self.debug_action_chunks:
            return
        action_array = _to_numpy(action_tensor).reshape(-1)
        if action_array.shape[0] <= max(self.gripper_indices, default=-1):
            return
        gripper_summary = " ".join(
            f"gripper[{idx}]={action_array[idx]:.6f}" for idx in self.gripper_indices
        )
        line = (
            "[SmolVLA env action debug] "
            f"rescale_gripper={self.rescale_gripper} "
            f"source_range={self.gripper_source_range} "
            f"{gripper_summary} "
            f"first env action={np.round(action_array, 6).tolist()}"
        )
        print(line, flush=True)
        if self.debug_action_log_path:
            with open(self.debug_action_log_path, "a", encoding="utf-8") as debug_file:
                debug_file.write(line + "\n")


def get_model(usr_args):
    # Order and rules live in checkpoints.py: explicit checkpoint_dir, then the
    # task's entry, then the default; never the official checkpoints by accident.
    # It comes first because the entry is also where a mixed routing declares
    # this task's normalization.
    source = resolve_source(usr_args)
    python_bin = usr_args.get("smolvla_python") or os.environ.get("SMOLVLA_PYTHON") or sys.executable
    if source.backend == "pi0":
        # A task routed to pi0 (pi0_backend.py); nothing below applies to it.
        return pi0_backend.get_model(usr_args, source, materialize, python_bin)
    # Declared, never inferred: a checkpoint that lost its statistics must fail
    # here rather than quietly evaluate as a legacy one.
    mode = task_norm_module.resolve_mode(usr_args, source.mode)
    vlm = resolve_vlm(usr_args)
    checkpoint_dir = materialize(source, python_bin)
    # Both directions, before the VLM is fetched or the worker started: a Q1c
    # checkpoint under a legacy config is refused just like a required config
    # without statistics.
    task_norm_module.check_mode_matches_checkpoint(mode, checkpoint_dir)
    vlm_dir = materialize(vlm, python_bin, required=VLM_REQUIRED_FILES)
    print(
        f"Checkpoint for {usr_args['task_name']}: {source.describe()} "
        f"(from {source.origin}) -> {checkpoint_dir}\n"
        f"VLM backbone: {vlm.describe()} -> {vlm_dir}",
        flush=True,
    )
    if mode == "required":
        task_norm_module.reject_legacy_overrides(usr_args)
        document = task_norm_module.load_document(checkpoint_dir)
        # The canonical task name the evaluator passes, not the instruction:
        # instructions are model input, and two tasks could share wording.
        slug = usr_args.get("task_name")
        if not slug:
            raise task_norm_module.TaskNormalizationError(
                "the evaluator did not pass task_name; task-conditioned statistics "
                "cannot be selected without it")
        norm = task_norm_module.TaskNorm(document, str(slug))
        if source.content_sha256 is not None and norm.content_sha256 != source.content_sha256:
            # The entry names one statistics document; a checkpoint carrying
            # another would run, and its numbers would not be this submission's.
            raise task_norm_module.TaskNormalizationError(
                f"{source.origin}.content_sha256 is {source.content_sha256}, but the checkpoint "
                f"carries {norm.content_sha256}")
        usr_args["smolvla_task_norm"] = norm
        print(f"Task-conditioned normalization: {norm.slug} "
              f"({norm.transform_version}, content {norm.content_sha256[:16]}, "
              f"task {norm.task_sha256[:16]})", flush=True)
    usr_args["checkpoint_dir"] = str(checkpoint_dir)
    usr_args["smolvla_vlm_path"] = str(vlm_dir)
    usr_args["smolvla_vlm_repo_id"] = vlm.repo_id
    # eval_policy.py copies these into evaluation_metrics.json, so the metrics say
    # exactly which weights produced them.
    usr_args["checkpoint_path"] = source.describe()
    base = noise_seed.parse_base(usr_args.get("smolvla_noise_seed"))
    usr_args["checkpoint_metadata"] = {
        "task_normalization": (usr_args["smolvla_task_norm"].metadata()
                               if mode == "required" else {"mode": "legacy"}),
        "policy": {
            "origin": source.origin,
            "repo_id": source.repo_id,
            "revision": source.revision,
            "path": str(checkpoint_dir),
            "mode": mode,
            "mode_declared_by": source.origin if source.mode else "smolvla_task_normalization",
            "expected_content_sha256": source.content_sha256,
        },
        "vlm": {"repo_id": vlm.repo_id, "revision": vlm.revision, "path": str(vlm_dir)},
        "policy_noise": None if base is None else {
            "base_seed": base,
            "episode_seed": noise_seed.DERIVATION,
            "per_forward": noise_seed.PER_FORWARD,
        },
    }
    return SmolVLAWorkerClient(usr_args)


def eval(env, model, obs):
    if isinstance(model, pi0_backend.Pi0Model):
        return pi0_backend.eval(env, model, obs)
    if model.noise_base is not None and model.episode_record["noise_seed"] is None:
        model.seed_noise(_episode_env_seed(env))
    instruction = getattr(env, "_current_instruction", None) or "click the bell"
    # One sample per worker call: observation encoding, IPC and the returned
    # actions (a model forward runs only when the worker needs a new chunk).
    inference_times_s = []
    started_at = start_inference()
    encoded_obs = encode_obs(obs, image_keys=model.image_keys)
    if model.task_norm is not None:
        model.check_action_space(env)
        encoded_obs["observation.state"] = model.standardize_state(encoded_obs["observation.state"])
    actions = model.infer(encoded_obs, task=instruction)
    if model.task_norm is not None:
        # Back to training units before anything env-facing touches them, so
        # encode_action's gripper conversion sees the same values it would for a
        # checkpoint trained without task-conditioned normalization.
        actions = [model.task_norm.unstandardize_action(action) for action in actions]
    finish_inference(started_at, inference_times_s)
    model.record_call(inference_times_s[-1])
    model.maybe_print_action_debug(actions)

    final_obs = obs
    info = {}
    truncated = False
    for action in actions:
        action_tensor = encode_action(
            action,
            env,
            rescale_gripper=model.rescale_gripper,
            gripper_indices=model.gripper_indices,
            gripper_source_range=model.gripper_source_range,
        )
        model.maybe_print_env_action_debug(action_tensor)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if bool(_to_numpy(truncated).reshape(-1)[0]):
            break
        if model.check_success_each_step and env.get_wrapper_attr("is_task_success")():
            break

    # Same end-of-episode test as scripts/eval_policy.py, which runs right after this returns.
    if bool(_to_numpy(truncated).reshape(-1)[0]) or env.get_wrapper_attr("is_task_success")():
        model.log_episode_timing()

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()


def episode_metadata(model):
    """Per-episode record for evaluation_metrics.json: seeds, forwards, noise and action digests."""
    if isinstance(model, pi0_backend.Pi0Model):
        return pi0_backend.episode_metadata(model)
    return model.episode_metadata()


def close_model(model):
    model.close()
