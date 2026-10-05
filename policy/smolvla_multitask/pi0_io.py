"""pi0 backend: observation and action mapping between EmbodiChain and the pi0 policy. numpy only.

Copied unchanged in substance from the research adapter the drawer checkpoint was evaluated with
(policy/pi0_research/robosyn_io.py of the research repository); only this header differs, and the research-only
"duck" prompt variant is left out.

encode(obs, prompt) -> the dict the research data config's input transforms expect, with the keys the training repack
  produces: the three cameras' colour images (HWC uint8, alpha dropped: the recorder stores color[..., :3]) and the
  14-dim qpos exactly as the recorder stores observation.state (obs["robot"]["qpos"]); plus the prompt.
decode(actions, low, high, n_exec) -> the first n_exec of the policy's (horizon, 14) absolute actions for env.step:
  the 12 arm joints unchanged (no scaling); the 2 grippers (6, 13), trained as a 0-1 ratio, mapped linearly onto the
  action space's [low, high] (the environment's gripper action is the physical opening, [0, 0.05]); a report says
  whether everything is finite, how many gripper values lay outside [0, 1] (clipped before mapping) and how many arm
  values lie outside the action space (not clipped here: reported).
Prompt variants are explicit: "original" is the instruction the evaluator gives (the organisers' text, identical to
the dataset's task text), "duck" names the object actually in the drawer scene. Training and inference must use the
same variant.
"""

from __future__ import annotations

import numpy as np

GRIPPERS = (6, 13)
ARMS = tuple(i for i in range(14) if i not in GRIPPERS)
CAMERAS = {"observation/image": "cam_high", "observation/left_wrist_image": "cam_left_wrist",
           "observation/right_wrist_image": "cam_right_wrist"}
PROMPT_VARIANTS = {
    "original": None,   # the evaluator's instruction, as given (the only variant the submission uses)
}


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_prompt(variant: str, env_instruction: str | None) -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose one of {sorted(PROMPT_VARIANTS)}")
    text = PROMPT_VARIANTS[variant]
    if text is None:
        if not env_instruction:
            raise ValueError("prompt variant 'original' needs the evaluator's instruction")
        return env_instruction
    return text


def encode(obs, prompt: str) -> dict:
    out = {}
    for key, cam in CAMERAS.items():
        img = _np(obs["sensor"][cam]["color"])
        img = img[0] if img.ndim == 4 else img
        if img.dtype != np.uint8:
            raise TypeError(f"{cam}: colour image is {img.dtype}, expected uint8 (what the recorder stored)")
        out[key] = np.ascontiguousarray(img[..., :3])
    state = _np(obs["robot"]["qpos"]).reshape(-1).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"qpos has shape {state.shape}, expected (14,)")
    out["observation/state"] = state
    out["prompt"] = prompt
    return out


def decode(actions, low, high, n_exec: int):
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 14:
        raise ValueError(f"actions have shape {a.shape}, expected (horizon, >=14)")
    a = a[:n_exec, :14].copy()
    low = np.asarray(low, dtype=np.float64).reshape(-1)[:14]
    high = np.asarray(high, dtype=np.float64).reshape(-1)[:14]
    g = list(GRIPPERS)
    arms = list(ARMS)
    report = {"finite": bool(np.isfinite(a).all()),
              "gripper_outside_unit": int(((a[:, g] < 0) | (a[:, g] > 1)).sum())}
    a[:, g] = low[g] + np.clip(a[:, g], 0.0, 1.0) * (high[g] - low[g])
    report["arm_outside_action_space"] = int(((a[:, arms] < low[arms]) | (a[:, arms] > high[arms])).sum())
    return a.astype(np.float32), report
