"""Checks how policy/smolvla_multitask picks the checkpoint for a task.

Needs neither torch, the simulator nor network access: checkpoints.py is loaded
straight from its file, and downloads are replaced by a fake worker Python.
The tests that read the shipped YAML files are skipped where PyYAML is missing.

    python -m unittest discover -s tests -v
"""

import importlib.util
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO_ROOT / "policy" / "smolvla_multitask"

spec = importlib.util.spec_from_file_location("smolvla_multitask_checkpoints", POLICY_DIR / "checkpoints.py")
checkpoints = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checkpoints  # dataclasses resolves annotations through sys.modules
spec.loader.exec_module(checkpoints)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

TASKS = {"click_bell", "handle_basket", "sample_loading"}
REV_A = "a" * 40
REV_B = "b" * 40
OFFICIAL_TASKS = {
    "table_rearrangement", "click_bell", "water_pouring", "handle_basket", "items_handover",
    "drawer_open_place", "mixer_operating", "item_assembly", "manipulate_pipette", "sample_loading",
}


def entry(repo="team/multitask", revision=REV_A):
    return {"repo_id": repo, "revision": revision}


def resolve(**usr_args):
    usr_args.setdefault("task_name", "click_bell")
    return checkpoints.resolve_source(usr_args, tasks=TASKS)


class ResolutionOrder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.local = Path(self.tmp.name) / "ckpt"
        self.local.mkdir()

    def test_explicit_checkpoint_dir_wins_over_both_tables(self):
        source = resolve(
            checkpoint_dir=str(self.local),
            checkpoints={"default": entry(), "tasks": {"click_bell": entry(revision=REV_B)}},
        )
        self.assertEqual(source.origin, "checkpoint_dir")
        self.assertEqual(source.local_dir, self.local.resolve())

    def test_step_directory_resolves_to_pretrained_model(self):
        (self.local / "pretrained_model").mkdir()
        self.assertEqual(resolve(checkpoint_dir=str(self.local)).local_dir, (self.local / "pretrained_model").resolve())

    def test_missing_checkpoint_dir_is_an_error_not_a_fallback(self):
        with self.assertRaisesRegex(checkpoints.CheckpointError, "does not exist"):
            resolve(checkpoint_dir=str(self.local / "nope"), checkpoints={"default": entry()})

    def test_task_entry_wins_over_default(self):
        source = resolve(checkpoints={"default": entry(), "tasks": {"click_bell": entry("team/bell", REV_B)}})
        self.assertEqual((source.origin, source.repo_id, source.revision), ("checkpoints.tasks.click_bell", "team/bell", REV_B))
        self.assertEqual(source.describe(), f"hf://team/bell@{REV_B}")

    def test_default_covers_tasks_without_an_entry(self):
        source = resolve(task_name="sample_loading", checkpoints={"default": entry(), "tasks": {"click_bell": entry(revision=REV_B)}})
        self.assertEqual((source.origin, source.revision), ("checkpoints.default", REV_A))

    def test_nothing_configured_fails_instead_of_guessing(self):
        for table in (None, {}, {"default": None, "tasks": {}}):
            with self.subTest(table=table), self.assertRaisesRegex(checkpoints.CheckpointError, "Refusing to guess"):
                resolve(checkpoint_dir="", checkpoints=table)


class EntryValidation(unittest.TestCase):
    def test_revision_must_be_a_full_commit_hash(self):
        for revision in ("main", "27965b7e", REV_A.upper(), REV_A + "0", None):
            with self.subTest(revision=revision), self.assertRaisesRegex(checkpoints.CheckpointError, "40-character"):
                resolve(checkpoints={"default": entry(revision=revision)})

    def test_misspelt_or_extra_keys_are_rejected(self):
        for bad in ({"repo_id": "team/m", "revison": REV_A}, {**entry(), "subfolder": "x"}, "team/m"):
            with self.subTest(bad=bad), self.assertRaises(checkpoints.CheckpointError):
                resolve(checkpoints={"default": bad})

    def test_repo_id_must_name_org_and_repo(self):
        for repo in ("multitask", "https://huggingface.co/team/m", "team/m/extra"):
            with self.subTest(repo=repo), self.assertRaisesRegex(checkpoints.CheckpointError, "org/name"):
                resolve(checkpoints={"default": entry(repo=repo)})

    def test_typo_in_task_key_does_not_fall_through_to_default(self):
        with self.assertRaisesRegex(checkpoints.CheckpointError, "unknown tasks.*click_bel"):
            resolve(checkpoints={"default": entry(), "tasks": {"click_bel": entry(revision=REV_B)}})

    def test_unknown_task_name_and_unknown_table_keys(self):
        with self.assertRaisesRegex(checkpoints.CheckpointError, "unknown task_name"):
            resolve(task_name="open_door", checkpoints={"default": entry()})
        with self.assertRaises(checkpoints.CheckpointError):
            resolve(checkpoints={"defaults": entry()})

    def test_known_tasks_come_from_the_repository_configs(self):
        self.assertLessEqual(OFFICIAL_TASKS, checkpoints.known_tasks())


class VlmPin(unittest.TestCase):
    PINNED = {"repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "revision": REV_A}

    def test_vlm_must_be_pinned_to_a_full_revision(self):
        self.assertEqual(checkpoints.resolve_vlm({"vlm": self.PINNED}).describe(), f"hf://{self.PINNED['repo_id']}@{REV_A}")
        with self.assertRaisesRegex(checkpoints.CheckpointError, "must be pinned"):
            checkpoints.resolve_vlm({})
        with self.assertRaisesRegex(checkpoints.CheckpointError, "40-character"):
            checkpoints.resolve_vlm({"vlm": {**self.PINNED, "revision": "main"}})

    def test_pin_replaces_the_repo_id_with_the_local_snapshot(self):
        config = type("Config", (), {"vlm_model_name": self.PINNED["repo_id"]})()
        checkpoints.pin_vlm(config, self.PINNED["repo_id"], "/cache/snapshots/" + REV_A)
        self.assertEqual(config.vlm_model_name, "/cache/snapshots/" + REV_A)

    def test_pin_refuses_a_checkpoint_trained_on_another_backbone(self):
        config = type("Config", (), {"vlm_model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"})()
        with self.assertRaisesRegex(checkpoints.CheckpointError, "pinned VLM"):
            checkpoints.pin_vlm(config, self.PINNED["repo_id"], "/cache/x")
        self.assertEqual(config.vlm_model_name, "HuggingFaceTB/SmolVLM2-2.2B-Instruct")


class Materialize(unittest.TestCase):
    """Downloads go through the worker Python; a fake one stands in for it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp))
        self.snapshot = self.tmp / "snapshots" / REV_A
        self.snapshot.mkdir(parents=True)

    def fake_python(self, printed, exit_code=0):
        script = self.tmp / "fake_python"
        script.write_text(f"#!/bin/sh\necho {json.dumps(json.dumps(printed))}\nexit {exit_code}\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return str(script)

    def source(self):
        return checkpoints.CheckpointSource(origin="checkpoints.default", repo_id="team/m", revision=REV_A)

    def test_downloaded_snapshot_is_used_when_complete(self):
        for name in checkpoints.REQUIRED_FILES:
            (self.snapshot / name).write_text("{}")
        python = self.fake_python({"path": str(self.snapshot), "revision": REV_A})
        self.assertEqual(checkpoints.materialize(self.source(), python), self.snapshot)

    def test_incomplete_snapshot_is_rejected(self):
        (self.snapshot / "config.json").write_text("{}")
        python = self.fake_python({"path": str(self.snapshot), "revision": REV_A})
        with self.assertRaisesRegex(checkpoints.CheckpointError, "model.safetensors"):
            checkpoints.materialize(self.source(), python)

    def test_revision_mismatch_and_failed_download_are_errors(self):
        with self.assertRaisesRegex(checkpoints.CheckpointError, "asked for revision"):
            checkpoints.materialize(self.source(), self.fake_python({"path": str(self.snapshot), "revision": REV_B}))
        with self.assertRaisesRegex(checkpoints.CheckpointError, "must be public"):
            checkpoints.materialize(self.source(), self.fake_python({}, exit_code=1))


@unittest.skipIf(yaml is None, "PyYAML not installed")
class ShippedConfigs(unittest.TestCase):
    def load(self, relative):
        return yaml.safe_load((POLICY_DIR / relative).read_text())

    def test_submission_config_never_names_the_official_checkpoints(self):
        text = (POLICY_DIR / "deploy_policy.yml").read_text()
        self.assertNotIn("SmolVLA_sim_", text)
        config = self.load("deploy_policy.yml")
        self.assertEqual(config["policy_name"], "smolvla_multitask")
        self.assertEqual(config["checkpoint_dir"], "")
        # Whatever is configured must resolve cleanly, or fail loudly as "not configured".
        for task in sorted(OFFICIAL_TASKS):
            try:
                source = checkpoints.resolve_source({**config, "task_name": task})
            except checkpoints.CheckpointError as exc:
                self.assertIn("Refusing to guess", str(exc))
            else:
                self.assertNotIn("SmolVLA_sim_", source.describe())

    def test_both_configs_seed_the_noise(self):
        for f in ("deploy_policy.yml", "test_configs/official_baseline.yml"):
            with self.subTest(config=f):
                self.assertEqual(self.load(f)["smolvla_noise_seed"], 0)

    def test_both_configs_pin_the_same_vlm(self):
        pins = [checkpoints.resolve_vlm(self.load(f)) for f in ("deploy_policy.yml", "test_configs/official_baseline.yml")]
        self.assertEqual(pins[0], pins[1])
        self.assertEqual(pins[0].repo_id, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    def test_official_baseline_config_covers_every_task_at_a_full_revision(self):
        config = self.load("test_configs/official_baseline.yml")
        self.assertIsNone(config["checkpoints"]["default"])
        self.assertEqual(set(config["checkpoints"]["tasks"]), OFFICIAL_TASKS)
        for task in OFFICIAL_TASKS:
            source = checkpoints.resolve_source({**config, "task_name": task})
            self.assertEqual(source.repo_id, f"RoboSynChallenge/SmolVLA_sim_{task}")

    def test_baseline_config_differs_from_the_submission_only_in_checkpoints_and_name(self):
        submission = self.load("deploy_policy.yml")
        baseline = self.load("test_configs/official_baseline.yml")
        for config in (submission, baseline):
            config.pop("checkpoints")
            config.pop("model_name")
        self.assertEqual(submission, baseline)


if __name__ == "__main__":
    unittest.main()
