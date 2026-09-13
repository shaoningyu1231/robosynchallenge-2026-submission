# ----------------------------------------------------------------------------
# Which checkpoint a task is evaluated with
# ----------------------------------------------------------------------------
#
# Resolution order, fixed:
#   1. checkpoint_dir, a local directory given explicitly (eval.sh --checkpoint_dir)
#   2. checkpoints.tasks.<task_name> in deploy_policy.yml
#   3. checkpoints.default in deploy_policy.yml
#   4. nothing: stop with an error
#
# There is deliberately no fallback to the official per-task checkpoints. If a
# task were silently missing from the table, the evaluation would score the
# official model under this submission's name; failing is the only safe answer.
#
# The VLM backbone is pinned the same way (the `vlm` entry). A SmolVLA checkpoint
# names its backbone only by repo id, and LeRobot loads that id -- weights,
# config, processor and tokenizer -- from the Hub's current main. Without the
# pin an empty cache would build the model on whatever main is that day.
#
# Standard library only, so it can be tested without torch or the simulator.

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent
REPO_ROOT = POLICY_DIR.parents[1]
FETCH_SCRIPT = POLICY_DIR / "fetch_checkpoint.py"

FULL_REVISION = re.compile(r"[0-9a-f]{40}")
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
ENTRY_KEYS = {"repo_id", "revision"}
# Files a LeRobot SmolVLA checkpoint cannot load without.
REQUIRED_FILES = ("config.json", "model.safetensors")
# Files the pinned VLM snapshot must provide: the checkpoints set
# load_vlm_weights, so the backbone weights are read before the checkpoint's
# own weights replace them, and the processor and tokenizer come from here too.
VLM_REQUIRED_FILES = ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json")


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointSource:
    origin: str                  # which config entry decided it
    local_dir: Path | None = None
    repo_id: str | None = None
    revision: str | None = None

    def describe(self) -> str:
        if self.local_dir is not None:
            return str(self.local_dir)
        return f"hf://{self.repo_id}@{self.revision}"


def known_tasks(configs_dir: Path = REPO_ROOT / "configs") -> set[str]:
    """Task names with a gym config, e.g. configs/click_bell/random/gym_config.json."""
    return {p.parent.parent.name for p in configs_dir.glob("*/*/gym_config.json")}


def _hf_entry(entry, where: str) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise CheckpointError(f"{where} must be a mapping with repo_id and revision, got {entry!r}")
    keys = set(entry)
    if keys != ENTRY_KEYS:
        # A misspelt key ("revison") must not quietly become "no revision".
        raise CheckpointError(
            f"{where} must have exactly the keys {sorted(ENTRY_KEYS)}, got {sorted(keys)}"
        )
    repo_id, revision = entry["repo_id"], entry["revision"]
    if not isinstance(repo_id, str) or not REPO_ID.fullmatch(repo_id):
        raise CheckpointError(f"{where}.repo_id must look like 'org/name', got {repo_id!r}")
    if not isinstance(revision, str) or not FULL_REVISION.fullmatch(revision):
        # Branch names and short hashes can move or become ambiguous; the
        # submission must name exactly the weights that were evaluated.
        raise CheckpointError(
            f"{where}.revision must be a full 40-character commit hash, got {revision!r}"
        )
    return repo_id, revision


def _local_dir(value) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        # A LeRobot step directory (checkpoints/050000) holds the model one level down.
        path = path / "pretrained_model"
    if not path.is_dir():
        raise CheckpointError(f"checkpoint_dir does not exist: {path}")
    return path


def resolve_source(usr_args: dict, tasks: set[str] | None = None) -> CheckpointSource:
    task = usr_args.get("task_name")
    if not task:
        raise CheckpointError("task_name is not set")

    explicit = usr_args.get("checkpoint_dir")
    if explicit not in (None, "", "None", "null"):
        return CheckpointSource(origin="checkpoint_dir", local_dir=_local_dir(explicit))

    table = usr_args.get("checkpoints") or {}
    if not isinstance(table, dict) or not set(table) <= {"default", "tasks"}:
        raise CheckpointError(f"checkpoints must be a mapping with 'default' and/or 'tasks', got {table!r}")
    per_task = table.get("tasks") or {}
    if not isinstance(per_task, dict):
        raise CheckpointError(f"checkpoints.tasks must be a mapping, got {per_task!r}")

    tasks = known_tasks() if tasks is None else tasks
    unknown = sorted(set(per_task) - tasks)
    if unknown:
        # A typo in a task key would otherwise fall through to the default model.
        raise CheckpointError(f"checkpoints.tasks names unknown tasks: {unknown}")
    if task not in tasks:
        raise CheckpointError(f"unknown task_name {task!r}")

    if per_task.get(task) is not None:
        where = f"checkpoints.tasks.{task}"
        repo_id, revision = _hf_entry(per_task[task], where)
        return CheckpointSource(origin=where, repo_id=repo_id, revision=revision)

    if table.get("default") is not None:
        repo_id, revision = _hf_entry(table["default"], "checkpoints.default")
        return CheckpointSource(origin="checkpoints.default", repo_id=repo_id, revision=revision)

    raise CheckpointError(
        f"no checkpoint configured for task {task!r}: checkpoint_dir is empty, "
        f"checkpoints.tasks has no {task!r} entry and checkpoints.default is null. "
        "Refusing to guess a model."
    )


def resolve_vlm(usr_args: dict) -> CheckpointSource:
    entry = usr_args.get("vlm")
    if entry is None:
        raise CheckpointError("vlm is not set; the VLM backbone must be pinned to a repo_id and full revision")
    repo_id, revision = _hf_entry(entry, "vlm")
    return CheckpointSource(origin="vlm", repo_id=repo_id, revision=revision)


def pin_vlm(config, repo_id: str, local_dir) -> None:
    """Point a loaded SmolVLA config at the pinned local VLM snapshot.

    Refuses a checkpoint trained on a different backbone than the one pinned:
    loading it would silently pair its expert with the wrong VLM.
    """
    if config.vlm_model_name != repo_id:
        raise CheckpointError(
            f"checkpoint uses VLM {config.vlm_model_name!r}, but the pinned VLM is {repo_id!r}"
        )
    config.vlm_model_name = str(local_dir)


def check_checkpoint_dir(path: Path, required=REQUIRED_FILES) -> Path:
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise CheckpointError(f"{path} is incomplete: missing {missing}")
    return path


def materialize(source: CheckpointSource, python_bin: str, required=REQUIRED_FILES) -> Path:
    """Return a local directory holding the checkpoint, downloading it if needed.

    The download runs in the worker's Python, which is the one guaranteed to
    have huggingface_hub; the simulator's Python may not.
    """
    if source.local_dir is not None:
        return check_checkpoint_dir(source.local_dir, required)
    cmd = [python_bin, str(FETCH_SCRIPT), "--repo-id", source.repo_id, "--revision", source.revision]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise CheckpointError(
            f"downloading {source.describe()} failed (exit {proc.returncode}); "
            "the repository must be public and the revision must exist"
        )
    return check_checkpoint_dir(parse_fetch_output(proc.stdout, source.revision), required)


def parse_fetch_output(stdout: str, revision: str) -> Path:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise CheckpointError("fetch_checkpoint.py printed nothing")
    result = json.loads(lines[-1])
    if result.get("revision") != revision:
        raise CheckpointError(f"asked for revision {revision}, got {result.get('revision')}")
    return Path(result["path"])
