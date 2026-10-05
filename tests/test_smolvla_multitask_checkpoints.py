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


class PerTaskMode(unittest.TestCase):
    """A mixed routing declares each task's normalization beside its checkpoint."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.local = Path(self.tmp.name) / "ckpt"
        self.local.mkdir()

    def table(self, **tasks):
        return {"checkpoints": {"default": None, "tasks": tasks}}

    def test_each_entry_carries_its_own_mode(self):
        config = self.table(
            click_bell={**entry(), "mode": "required", "content_sha256": "a" * 64},
            handle_basket={**entry(repo="RoboSynChallenge/SmolVLA_sim_handle_basket"), "mode": "legacy"},
            sample_loading={**entry(), "mode": "legacy"})
        first = resolve(task_name="click_bell", **config)
        self.assertEqual((first.mode, first.content_sha256), ("required", "a" * 64))
        self.assertEqual(resolve(task_name="handle_basket", **config).mode, "legacy")

    def test_a_half_declared_table_is_refused_on_every_task(self):
        # The entries without a mode would fall back to the single scalar, which
        # under a mixed routing is by definition wrong for one of the groups.
        config = self.table(click_bell={**entry(), "mode": "required"},
                            handle_basket=entry(), sample_loading=entry())
        for task in ("click_bell", "handle_basket", "sample_loading"):
            with self.subTest(task=task), self.assertRaises(checkpoints.CheckpointError) as cm:
                resolve(task_name=task, **config)
            self.assertIn("either every checkpoint entry declares a mode", str(cm.exception))

    def test_per_task_modes_forbid_a_default(self):
        config = {"checkpoints": {"default": {**entry(), "mode": "legacy"},
                                  "tasks": {"click_bell": {**entry(), "mode": "required"}}}}
        with self.assertRaises(checkpoints.CheckpointError) as cm:
            resolve(task_name="handle_basket", **config)
        self.assertIn("cannot be combined with per-task modes", str(cm.exception))

    def test_an_unlisted_or_misspelt_task_has_nowhere_to_fall(self):
        config = self.table(click_bell={**entry(), "mode": "legacy"})
        with self.assertRaises(checkpoints.CheckpointError) as cm:
            resolve(task_name="handle_basket", **config)
        self.assertIn("Refusing to guess", str(cm.exception))
        with self.assertRaises(checkpoints.CheckpointError) as cm:
            resolve(task_name="click_bel", **config)
        self.assertIn("unknown task_name", str(cm.exception))
        with self.assertRaises(checkpoints.CheckpointError) as cm:
            resolve(task_name="click_bell", **self.table(click_bel={**entry(), "mode": "legacy"}))
        self.assertIn("unknown tasks", str(cm.exception))

    def test_an_invalid_mode_or_digest_is_refused(self):
        for bad, message in (({"mode": "Required "}, None),      # accepted, normalised
                             ({"mode": "strict"}, ".mode must be one of"),
                             ({"mode": True}, ".mode must be one of"),
                             ({"mode": "required", "content_sha256": "abc"}, ".content_sha256 must be"),
                             ({"mode": "legacy", "content_sha256": "a" * 64},
                              "only a required-mode entry")):
            config = self.table(click_bell={**entry(), **bad})
            with self.subTest(entry=bad):
                if message is None:
                    self.assertEqual(resolve(task_name="click_bell", **config).mode, "required")
                else:
                    with self.assertRaises(checkpoints.CheckpointError) as cm:
                        resolve(task_name="click_bell", **config)
                    self.assertIn(message, str(cm.exception))

    def test_a_misspelt_optional_key_does_not_become_no_mode(self):
        config = self.table(click_bell={**entry(), "mod": "required"})
        with self.assertRaises(checkpoints.CheckpointError) as cm:
            resolve(task_name="click_bell", **config)
        self.assertIn("may add", str(cm.exception))

    def test_a_local_entry_routes_one_task_to_a_directory(self):
        source = resolve(task_name="click_bell", **self.table(
            click_bell={"local_dir": str(self.local), "mode": "required",
                        "content_sha256": "b" * 64}))
        self.assertEqual(source.local_dir, self.local.resolve())
        self.assertEqual((source.mode, source.content_sha256), ("required", "b" * 64))
        self.assertIsNone(source.repo_id)

    def test_a_local_entry_still_rejects_unknown_keys_and_missing_directories(self):
        with self.assertRaises(checkpoints.CheckpointError):
            resolve(task_name="click_bell",
                    **self.table(click_bell={"local_dir": str(self.local), "revision": REV_A}))
        with self.assertRaises(checkpoints.CheckpointError):
            resolve(task_name="click_bell",
                    **self.table(click_bell={"local_dir": str(self.local / "nope")}))

    def test_the_table_is_validated_even_for_a_task_that_resolves(self):
        # A broken entry for another task must not wait until that task runs.
        config = self.table(click_bell={**entry(), "mode": "legacy"},
                            handle_basket={**entry(), "mode": "sideways"})
        with self.assertRaises(checkpoints.CheckpointError) as cm:
            resolve(task_name="click_bell", **config)
        self.assertIn("checkpoints.tasks.handle_basket.mode", str(cm.exception))


class Pi0Backend(unittest.TestCase):
    """A task routed to pi0 (backend: pi0): its own keys, outside SmolVLA's mode rule, never the default."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.local = Path(self.tmp.name) / "pi0"
        self.local.mkdir()

    def table(self, **tasks):
        return {"checkpoints": {"default": None, "tasks": tasks}}

    def pi0(self, **extra):
        return {**entry(repo="team/pi0_drawer"), "backend": "pi0", "pi0_config": "pi0_drawer_lora_b1_tomato", **extra}

    def test_a_pi0_entry_beside_mode_declaring_smolvla_entries(self):
        config = self.table(click_bell=self.pi0(), handle_basket={**entry(), "mode": "legacy"})
        source = resolve(task_name="click_bell", **config)
        self.assertEqual((source.backend, source.pi0_config, source.repo_id), ("pi0", "pi0_drawer_lora_b1_tomato", "team/pi0_drawer"))
        self.assertIsNone(source.mode)
        other = resolve(task_name="handle_basket", **config)
        self.assertEqual((other.backend, other.pi0_config, other.mode), ("smolvla", None, "legacy"))

    def test_a_local_pi0_entry(self):
        config = self.table(click_bell={"local_dir": str(self.local), "backend": "pi0", "pi0_config": "pi0_x_lora_b2"})
        source = resolve(task_name="click_bell", **config)
        self.assertEqual((source.local_dir, source.backend), (self.local.resolve(), "pi0"))

    def test_a_pi0_entry_must_name_its_config(self):
        for bad in (None, "", "drawer", "pi0 drawer"):
            with self.subTest(pi0_config=bad), self.assertRaises(checkpoints.CheckpointError):
                resolve(task_name="click_bell", **self.table(click_bell=self.pi0(pi0_config=bad)))

    def test_a_pi0_entry_takes_no_smolvla_mode(self):
        for extra in ({"mode": "legacy"}, {"mode": "required", "content_sha256": "a" * 64}):
            with self.subTest(extra=extra), self.assertRaises(checkpoints.CheckpointError):
                resolve(task_name="click_bell", **self.table(click_bell=self.pi0(**extra)))

    def test_unknown_backend_and_stray_pi0_config_are_refused(self):
        with self.assertRaises(checkpoints.CheckpointError):
            resolve(task_name="click_bell", **self.table(click_bell=self.pi0(backend="pi05")))
        with self.assertRaises(checkpoints.CheckpointError):
            resolve(task_name="click_bell", **self.table(click_bell={**entry(), "pi0_config": "pi0_x"}))

    def test_a_broken_pi0_entry_fails_every_task(self):
        config = self.table(click_bell={**entry(), "mode": "legacy"}, handle_basket=self.pi0(pi0_config=None))
        with self.assertRaises(checkpoints.CheckpointError) as cm:
            resolve(task_name="click_bell", **config)
        self.assertIn("checkpoints.tasks.handle_basket", str(cm.exception))

    def test_the_default_cannot_route_to_pi0(self):
        with self.assertRaises(checkpoints.CheckpointError):
            resolve(task_name="click_bell", checkpoints={"default": self.pi0(), "tasks": {}})

    def test_the_pi0_entry_still_needs_a_full_revision(self):
        with self.assertRaises(checkpoints.CheckpointError):
            resolve(task_name="click_bell", **self.table(click_bell=self.pi0(revision="main")))


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

    def test_submission_config_matches_the_final_mapping(self):
        # The mapping was settled by holdout on 2026-09-26
        # (experiments/holdout_DECISION.txt in the research repository): all ten
        # tasks on the official single-task checkpoints. items_handover was the
        # one task this project had chosen its own model for, on a 30-scene
        # development result that a hundred untouched scenes did not reproduce.
        #
        # Naming the official checkpoints is therefore deliberate -- but only
        # task by task. What must never happen is one arriving through a default
        # or a fallback, which would score the official model under this
        # submission's name on a task nobody chose it for.
        config = self.load("deploy_policy.yml")
        self.assertEqual(config["policy_name"], "smolvla_multitask")
        self.assertEqual(config["checkpoint_dir"], "")
        self.assertIsNone(config["checkpoints"]["default"])
        self.assertEqual(set(config["checkpoints"]["tasks"]), OFFICIAL_TASKS)
        for task in sorted(OFFICIAL_TASKS):
            source = checkpoints.resolve_source({**config, "task_name": task})
            self.assertEqual(source.origin, f"checkpoints.tasks.{task}")
            self.assertEqual(source.repo_id, f"RoboSynChallenge/SmolVLA_sim_{task}")
            self.assertEqual(source.mode, "legacy")
            self.assertIsNone(source.content_sha256)

    def test_the_submission_pins_the_revisions_it_was_evaluated_at(self):
        # The holdout numbers belong to these exact revisions; a branch name or a
        # moved pin would make them describe something else.
        submission = self.load("deploy_policy.yml")["checkpoints"]["tasks"]
        baseline = self.load("test_configs/official_baseline.yml")["checkpoints"]["tasks"]
        self.assertEqual({t: e["revision"] for t, e in submission.items()},
                         {t: e["revision"] for t, e in baseline.items()})

    def test_submission_config_declares_no_global_mode_and_no_gripper_overrides(self):
        # One scalar cannot describe a routing where nine tasks are legacy and
        # one is required, and in required mode the gripper keys are refused.
        config = self.load("deploy_policy.yml")
        self.assertIsNone(config["smolvla_task_normalization"])
        for key in ("smolvla_rescale_gripper", "smolvla_gripper_indices",
                    "smolvla_gripper_source_range"):
            self.assertIsNone(config[key], key)

    def test_nothing_unpublished_is_shipped(self):
        # There was a placeholder here while items_handover pointed at Q2-A,
        # which is not published. The holdout sent that task back to the
        # official checkpoint, so nothing in the submission is unpublished --
        # and nothing may quietly become so again.
        text = (POLICY_DIR / "deploy_policy.yml").read_text()
        for marker in ("PLACEHOLDER", "0" * 40):
            self.assertNotIn(marker, text)

    def test_a_local_entry_cannot_be_shipped(self):
        text = (POLICY_DIR / "deploy_policy.yml").read_text()
        self.assertNotIn("local_dir", text)

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

    def test_baseline_config_differs_from_the_submission_only_where_it_must(self):
        # The two configs are the same evaluation. They differ in the routing,
        # the name, and -- since the submission is mixed -- where the mode is
        # declared: the baseline is one legacy model for every task and keeps
        # the scalar, the submission declares a mode per entry instead.
        submission = self.load("deploy_policy.yml")
        baseline = self.load("test_configs/official_baseline.yml")
        differ = ("checkpoints", "model_name", "smolvla_task_normalization",
                  "smolvla_rescale_gripper", "smolvla_gripper_indices",
                  "smolvla_gripper_source_range")
        for config in (submission, baseline):
            for key in differ:
                config.pop(key)
        self.assertEqual(submission, baseline)
        # and the values the submission leaves unset are the ones the adapter
        # falls back to, so the nine legacy tasks behave exactly as the baseline.
        baseline_again = self.load("test_configs/official_baseline.yml")
        self.assertEqual(baseline_again["smolvla_task_normalization"], "legacy")
        self.assertEqual(baseline_again["smolvla_rescale_gripper"], "auto")
        self.assertEqual(baseline_again["smolvla_gripper_indices"], [6, 13])
        self.assertEqual(baseline_again["smolvla_gripper_source_range"], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
