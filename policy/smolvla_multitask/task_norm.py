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
from pathlib import Path

import numpy as np

FEATURES = ("observation.state", "action")
DIM = 14


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
    path = Path(checkpoint_dir) / "task_normalization.json"
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
    if payload.get("features") and list(payload["features"]) != list(FEATURES):
        raise TaskNormalizationError(
            f"{path}: statistics cover {payload['features']}, this adapter transforms {list(FEATURES)}")
    return document


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
        self.gripper = payload.get("gripper") or {}
        if not self.gripper.get("state") or not self.gripper.get("action"):
            raise TaskNormalizationError(
                "the statistics do not describe the gripper units for both state and action; "
                "without that the action unit would have to be guessed")
        self.stats = {}
        for feature in FEATURES:
            mean = _vector(block, feature, "mean", slug)
            std = _vector(block, feature, "std", slug)
            mask_values = block.get(feature, {}).get("constant_mask")
            if not isinstance(mask_values, list) or len(mask_values) != DIM:
                raise TaskNormalizationError(f"{slug}/{feature}: constant_mask must have {DIM} entries")
            mask = np.asarray(mask_values, dtype=bool)
            if (std[~mask] <= 0).any():
                raise TaskNormalizationError(
                    f"{slug}/{feature}: a dimension has std <= 0 but is not marked constant")
            self.stats[feature] = (mean, std, mask)

    # ---- the transform ------------------------------------------------------------

    def standardize_state(self, state: np.ndarray) -> np.ndarray:
        mean, std, mask = self.stats["observation.state"]
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        if values.shape[0] != DIM:
            raise TaskNormalizationError(f"state has {values.shape[0]} values, expected {DIM}")
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
        return tuple(int(i) for i in self.gripper.get("dims", (6, 13)))

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
        low, high = self.gripper["action"].get("training_range", (0.0, 1.0))
        return float(low), float(high)

    def state_range_problem(self, state: np.ndarray) -> str | None:
        """A sanity check on the raw observation, before standardizing.

        It cannot prove a unit -- metres [0, 0.05] sit inside the ratio range --
        but it does catch an observation that is nowhere near what the training
        data held.
        """
        low, high = self.gripper["state"].get("training_range", (0.0, 1.0))
        tolerance = float(self.gripper.get("sanity_tolerance", 0.05))
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        for index in self.gripper_indices:
            if index >= values.shape[0]:
                continue
            value = values[index]
            if not (low - tolerance <= value <= high + tolerance):
                return (f"gripper state dim {index} is {value:.4f}, outside the training range "
                        f"[{low}, {high}] with tolerance {tolerance}")
        return None

    def action_space_problem(self, low: np.ndarray, high: np.ndarray) -> str | None:
        """Does the environment's action space match what the metadata claims?"""
        expected_low, expected_high = self.gripper["action"].get("env_range", (0.0, 0.05))
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
