"""Task-conditioned normalization for Q1c checkpoints, on the inference side.

A Q1c checkpoint was trained on data standardized per task: each task's state
and action were shifted and scaled by that task's own statistics before LeRobot's
own normalizer ever saw them. Acting with such a checkpoint therefore has to
apply the same transform to the observation and undo it on the action, using the
statistics of the task being evaluated -- and the task has to come from the
evaluator's canonical name, never from the language instruction, which is text a
model consumes rather than an identifier.

Everything here fails loudly. There is no fallback: a missing file, an unknown
task, a digest that does not match, statistics of the wrong width -- each ends
the run at model construction rather than producing actions in a unit nobody
checked. The alternative to failing is emitting plausible-looking motion in the
wrong coordinates, which no downstream check would catch.

The statistics live in <checkpoint>/task_normalization.json, written by
tools/task_norm.py and shipped inside the checkpoint, because a deployment
config can be edited independently of the weights it describes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

STATISTICS_FILE = "task_normalization.json"
FEATURES = ("observation.state", "action")
DIM = 14
# What this adapter knows how to undo. A checkpoint standardized some other way
# must not be run through this one just because the file parses.
SUPPORTED_TRANSFORMS = frozenset({"task_conditioned_mean_std"})
SUPPORTED_TRANSFORM_VERSIONS = frozenset({"q1c-1"})


class TaskNormalizationError(RuntimeError):
    """Anything that makes the statistics unusable. Never caught internally."""


def canonical(obj) -> bytes:
    """The one serialization digests are taken over (tools/task_norm.py agrees)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def _no_duplicate_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            # json.loads would otherwise keep the last one, so two entries for a
            # task would silently pick a winner.
            raise TaskNormalizationError(f"duplicate key in task_normalization.json: {key}")
        seen[key] = value
    return seen


def _vector(block: dict, feature: str, field: str, slug: str) -> np.ndarray:
    values = block.get(feature, {}).get(field)
    if not isinstance(values, list) or len(values) != DIM:
        raise TaskNormalizationError(
            f"{slug}/{feature}.{field}: expected {DIM} values, got "
            f"{len(values) if isinstance(values, list) else type(values).__name__}")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise TaskNormalizationError(f"{slug}/{feature}.{field} contains a non-finite value")
    return array


def load_document(checkpoint_dir: Path) -> dict:
    """Read and fully validate the statistics file inside a checkpoint.

    The structure is demanded, not sniffed: an empty object is valid JSON and
    must not pass for a set of statistics.
    """
    path = Path(checkpoint_dir) / STATISTICS_FILE
    if not path.is_file():
        raise TaskNormalizationError(
            f"no {path}; this checkpoint was not trained with task-conditioned "
            f"normalization, or the file was not shipped with it")
    try:
        document = json.loads(path.read_text(), object_pairs_hook=_no_duplicate_keys)
    except TaskNormalizationError:
        raise
    except ValueError as error:
        raise TaskNormalizationError(f"{path} does not parse: {error}") from error

    if document.get("schema_version") != 1:
        raise TaskNormalizationError(
            f"{path}: schema_version {document.get('schema_version')!r} is not 1")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise TaskNormalizationError(f"{path}: no payload object")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise TaskNormalizationError(f"{path}: payload has no tasks")
    if _digest(payload) != document.get("content_sha256"):
        raise TaskNormalizationError(f"{path}: content_sha256 does not match the payload")
    digests = document.get("task_sha256")
    if not isinstance(digests, dict) or set(digests) != set(tasks):
        raise TaskNormalizationError(f"{path}: task_sha256 does not cover the same tasks as the payload")
    for slug, block in tasks.items():
        if _digest(block) != digests[slug]:
            raise TaskNormalizationError(f"{path}: task_sha256 does not match the block for {slug}")
    if list(payload.get("features") or []) != list(FEATURES):
        raise TaskNormalizationError(
            f"{path}: statistics cover {payload.get('features')!r}, this adapter transforms "
            f"{list(FEATURES)}")
    if payload.get("transform") not in SUPPORTED_TRANSFORMS:
        raise TaskNormalizationError(
            f"{path}: transform {payload.get('transform')!r} is not one this adapter implements "
            f"({sorted(SUPPORTED_TRANSFORMS)})")
    if payload.get("transform_version") not in SUPPORTED_TRANSFORM_VERSIONS:
        raise TaskNormalizationError(
            f"{path}: transform_version {payload.get('transform_version')!r} is not supported "
            f"({sorted(SUPPORTED_TRANSFORM_VERSIONS)})")
    threshold = payload.get("constant_std_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold < 0:
        raise TaskNormalizationError(
            f"{path}: constant_std_threshold {threshold!r} is not a non-negative number")
    return document


def _checked_gripper(gripper) -> dict:
    """The gripper metadata, with nothing defaulted.

    A default here is the failure mode this whole file exists to prevent: the
    adapter would silently assume a unit, and wrong units produce motion that
    looks reasonable and is scaled wrong. If the checkpoint does not say, the
    run stops.
    """
    if not isinstance(gripper, dict):
        raise TaskNormalizationError("the statistics carry no gripper metadata")
    dims = gripper.get("dims")
    if (not isinstance(dims, list) or not dims
            or not all(isinstance(d, int) and not isinstance(d, bool) for d in dims)):
        raise TaskNormalizationError("gripper.dims must list the gripper dimensions")
    if len(set(dims)) != len(dims):
        raise TaskNormalizationError(f"gripper.dims repeats a dimension: {dims}")
    if any(d < 0 or d >= DIM for d in dims):
        raise TaskNormalizationError(f"gripper.dims must lie in [0, {DIM - 1}]: {dims}")
    tolerance = gripper.get("sanity_tolerance")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise TaskNormalizationError(
            f"gripper.sanity_tolerance {tolerance!r} is not a non-negative finite number")
    for side in ("state", "action"):
        block = gripper.get(side)
        if not isinstance(block, dict):
            raise TaskNormalizationError(f"gripper.{side} is missing; both sides must be described")
        for field in ("env_units", "training_units"):
            if not isinstance(block.get(field), str) or not block[field]:
                raise TaskNormalizationError(f"gripper.{side}.{field} is missing")
        for field in ("env_range", "training_range"):
            values = block.get(field)
            if (not isinstance(values, list) or len(values) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               for v in values)):
                raise TaskNormalizationError(f"gripper.{side}.{field} must be a pair of numbers")
            if not all(np.isfinite(v) for v in values):
                raise TaskNormalizationError(f"gripper.{side}.{field} must be finite: {values}")
            if values[1] <= values[0]:
                # A degenerate range cannot define a conversion, and a reversed
                # one would convert with the wrong sign.
                raise TaskNormalizationError(
                    f"gripper.{side}.{field} {values} is empty or reversed")
    return gripper


class TaskNorm:
    """One task's statistics, and the transform in both directions."""

    def __init__(self, document: dict, slug: str):
        payload = document["payload"]
        tasks = payload["tasks"]
        if slug not in tasks:
            raise TaskNormalizationError(
                f"the checkpoint has no statistics for task {slug!r}; it knows "
                f"{sorted(tasks)}")
        block = tasks[slug]
        self.slug = slug
        self.content_sha256 = document["content_sha256"]
        self.task_sha256 = document["task_sha256"][slug]
        self.transform_version = payload.get("transform_version")
        self.gripper = _checked_gripper(payload.get("gripper"))
        threshold = float(payload["constant_std_threshold"])
        self.stats = {}
        for feature in FEATURES:
            mean = _vector(block, feature, "mean", slug)
            std = _vector(block, feature, "std", slug)
            mask_values = block.get(feature, {}).get("constant_mask")
            if not isinstance(mask_values, list) or len(mask_values) != DIM:
                raise TaskNormalizationError(f"{slug}/{feature}: constant_mask must have {DIM} entries")
            if not all(isinstance(v, bool) for v in mask_values):
                # 0/1 or "false" would work by accident today and differently
                # after any change to how the file is written.
                raise TaskNormalizationError(
                    f"{slug}/{feature}: constant_mask must be true/false, got "
                    f"{sorted({type(v).__name__ for v in mask_values})}")
            mask = np.asarray(mask_values, dtype=bool)
            expected = std <= threshold
            if not np.array_equal(mask, expected):
                disagreeing = [int(d) for d in np.nonzero(mask != expected)[0]]
                raise TaskNormalizationError(
                    f"{slug}/{feature}: constant_mask disagrees with std <= "
                    f"{threshold} at {disagreeing}")
            if (std[~mask] <= 0).any():
                raise TaskNormalizationError(
                    f"{slug}/{feature}: a dimension has std <= 0 but is not marked constant")
            self.stats[feature] = (mean, std, mask)

    # ---- the transform ------------------------------------------------------------

    def state_in_training_units(self, state: np.ndarray) -> np.ndarray:
        """The observation converted to the unit the statistics were built in.

        Today every measured task reports the gripper as a 0-1 ratio, the same
        as the training data, so this is the identity. It exists because the
        remaining nine tasks are unverified: if one of them turns out to report
        metres, the conversion is defined by the two ranges in the metadata
        rather than left to whoever notices the mismatch.
        """
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        block = self.gripper["state"]
        if block["env_units"] == block["training_units"]:
            return values
        env_low, env_high = (float(v) for v in block["env_range"])
        train_low, train_high = (float(v) for v in block["training_range"])
        span = env_high - env_low
        if span <= 0:
            raise TaskNormalizationError(
                f"gripper state env_range {block['env_range']} cannot be converted to "
                f"{block['training_range']}")
        converted = values.copy()
        for index in self.gripper_indices:
            if index < converted.shape[0]:
                ratio = (converted[index] - env_low) / span
                converted[index] = train_low + ratio * (train_high - train_low)
        return converted

    def standardize_state(self, state: np.ndarray) -> np.ndarray:
        mean, std, mask = self.stats["observation.state"]
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        if values.shape[0] != DIM:
            raise TaskNormalizationError(f"state has {values.shape[0]} values, expected {DIM}")
        values = self.state_in_training_units(values)
        safe = np.where(mask, 1.0, std)
        return np.where(mask, 0.0, (values - mean) / safe).astype(np.float32)

    def unstandardize_action(self, action: np.ndarray) -> np.ndarray:
        """Back to training units. Constant dimensions return the task's mean:
        the model was trained to emit zero there and its output carries no
        information about them."""
        mean, std, mask = self.stats["action"]
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape[0] < DIM:
            raise TaskNormalizationError(f"action has {values.shape[0]} values, expected {DIM}")
        head = np.where(mask, mean, values[:DIM] * std + mean)
        return np.concatenate([head, values[DIM:]]).astype(np.float32)

    # ---- what the environment needs ------------------------------------------------

    @property
    def gripper_indices(self) -> tuple[int, ...]:
        return tuple(int(i) for i in self.gripper["dims"])

    @property
    def rescale_gripper(self) -> bool:
        """Whether the environment's unit differs from the training unit.

        Explicit metadata, not the old heuristic on the checkpoint's action.max:
        a Q1c checkpoint's statistics are standardized, so that test would read
        a few standard deviations and answer nonsense.
        """
        action = self.gripper["action"]
        return action["env_units"] != action["training_units"]

    @property
    def gripper_source_range(self) -> tuple[float, float]:
        low, high = self.gripper["action"]["training_range"]
        return float(low), float(high)

    def state_range_problem(self, state: np.ndarray) -> str | None:
        """A sanity check on the raw observation, before standardizing.

        It cannot prove a unit -- metres [0, 0.05] sit inside the ratio range --
        but it does catch an observation that is nowhere near what the training
        data held.
        """
        low, high = (float(v) for v in self.gripper["state"]["env_range"])
        tolerance = float(self.gripper["sanity_tolerance"])
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        for index in self.gripper_indices:
            if index >= values.shape[0]:
                continue
            value = values[index]
            if not (low - tolerance <= value <= high + tolerance):
                return (f"gripper state dim {index} is {value:.4f}, outside the environment range "
                        f"[{low}, {high}] with tolerance {tolerance}")
        return None

    def action_space_problem(self, low: np.ndarray, high: np.ndarray) -> str | None:
        """Does the environment's action space match what the metadata claims?"""
        expected_low, expected_high = self.gripper["action"]["env_range"]
        for index in self.gripper_indices:
            if index >= low.shape[0]:
                continue
            if not (np.isclose(low[index], expected_low, atol=1e-6)
                    and np.isclose(high[index], expected_high, atol=1e-6)):
                return (f"the environment's gripper action space is "
                        f"[{low[index]:.6g}, {high[index]:.6g}], the statistics say "
                        f"[{expected_low}, {expected_high}]")
        return None

    def metadata(self) -> dict:
        """What the evaluation metrics should record about this run."""
        return {
            "mode": "required",
            "task": self.slug,
            "transform_version": self.transform_version,
            "content_sha256": self.content_sha256,
            "task_sha256": self.task_sha256,
        }


LEGACY_ONLY_ARGS = ("smolvla_rescale_gripper", "smolvla_gripper_source_range",
                    "smolvla_gripper_indices")


def reject_legacy_overrides(usr_args: dict) -> None:
    """In required mode the metadata decides the units, and nothing else may.

    Leaving these available would allow a second conversion on top of the
    task-conditioned inverse, which produces motion that looks reasonable and is
    scaled wrong.
    """
    present = [name for name in LEGACY_ONLY_ARGS if usr_args.get(name) not in (None, "auto")]
    if present:
        raise TaskNormalizationError(
            f"{', '.join(present)} cannot be used with smolvla_task_normalization=required; "
            f"the units come from the checkpoint's statistics")


def resolve_mode(usr_args: dict) -> str:
    """required or legacy, declared by the config -- never inferred from a file.

    Inferring it would mean a checkpoint that lost its statistics quietly
    evaluates as a legacy one.
    """
    mode = usr_args.get("smolvla_task_normalization")
    if mode is None:
        raise TaskNormalizationError(
            "smolvla_task_normalization must be set to 'required' (Q1c checkpoints, which "
            "carry per-task statistics) or 'legacy' (Q1a and the official checkpoints)")
    mode = str(mode).strip().lower()
    if mode not in ("required", "legacy"):
        raise TaskNormalizationError(
            f"smolvla_task_normalization={mode!r} is not 'required' or 'legacy'")
    return mode


def check_mode_matches_checkpoint(mode: str, checkpoint_dir) -> None:
    """The declared mode and the checkpoint must agree, in both directions.

    required without statistics: there is nothing to standardize with.
    legacy with statistics: the checkpoint was trained on state and action
    standardized per task, and legacy mode would feed it the globally normalized
    observation and execute its output as if it were in training units -- motion
    that looks plausible and is wrong. That second case is not hypothetical: on
    2026-09-19 a curve point read a config that had been switched back to legacy
    and evaluated a Q1c checkpoint to the end (2/30, discarded) without a word.

    The mode is still declared, never inferred; this only refuses a declaration
    the checkpoint contradicts. get_model calls it before anything is started --
    no VLM download, no worker. Presence is lexists: a dangling link or a
    directory under that name is still a claim the checkpoint makes.
    """
    path = Path(checkpoint_dir) / STATISTICS_FILE
    present = os.path.lexists(path)
    if mode == "required":
        if not present:
            raise TaskNormalizationError(
                f"smolvla_task_normalization=required, but {path} does not exist: this "
                f"checkpoint was not trained with task-conditioned normalization, or the file "
                f"was not shipped with it")
    elif mode == "legacy":
        if present:
            raise TaskNormalizationError(
                f"smolvla_task_normalization=legacy, but the checkpoint carries {path}: it was "
                f"trained with task-conditioned normalization, and legacy mode would run it "
                f"without the transform. Use a required config "
                f"(policy/smolvla_multitask/test_configs/q1c_taskstd.yml)")
    else:
        raise TaskNormalizationError(f"unknown mode {mode!r}; resolve_mode first")
