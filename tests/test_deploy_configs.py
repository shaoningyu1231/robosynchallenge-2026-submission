"""Every shipped deployment config, checked against the adapter's own rules.

    python3 -m unittest tests.test_deploy_configs -v

A config is the one place where a Q1c run can be set up wrong without any code
being wrong. It happened on 2026-09-19: the first evaluation of the Q1c
checkpoint was refused at startup because the config still carried
smolvla_gripper_indices and smolvla_gripper_source_range, which required mode
forbids -- they would convert the gripper a second time on top of the
task-conditioned inverse, producing motion that looks reasonable and is scaled
wrong. The guard did its job and no metrics were written, but nothing had said
in advance which configs were meant to be usable in which mode.

So the rules are asserted here, against the real files and the real guard.

Since the mapping was frozen (2026-09-24) the submission config is mixed: one
task on a checkpoint with per-task statistics, nine on the official ones. Such
a config has no single mode, so each checkpoint entry declares its own and the
scalar is null. Every config here is therefore one of two shapes -- one scalar
and no entry modes, or an entry mode on every entry and no scalar -- and the
rules below are checked per task, the way the adapter resolves them.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "policy" / "smolvla_multitask"

try:
    import yaml
except ImportError:                                    # PyYAML is not everywhere
    yaml = None

# Loaded by path: the package's __init__ imports deploy_policy, which needs torch.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "q1c_task_norm_for_configs", CONFIG_DIR / "task_norm.py")
task_norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task_norm)


_ck_spec = importlib.util.spec_from_file_location(
    "q1c_checkpoints_for_configs", CONFIG_DIR / "checkpoints.py")
checkpoints = importlib.util.module_from_spec(_ck_spec)
sys.modules[_ck_spec.name] = checkpoints          # dataclasses resolves through sys.modules
_ck_spec.loader.exec_module(checkpoints)

TASKS = sorted(checkpoints.known_tasks())
# The ten the competition scores. known_tasks() also yields "other_tasks", the
# name of the directory holding the extra scenes -- not a task, and not routed.
COMPETITION_TASKS = [
    "click_bell", "drawer_open_place", "handle_basket", "item_assembly", "items_handover",
    "manipulate_pipette", "mixer_operating", "sample_loading", "table_rearrangement",
    "water_pouring",
]


def modes_of(config: dict) -> dict:
    """{task: mode} exactly as the adapter would resolve it, for every task.

    A task the config does not route is left out: not configured is a separate
    failure (checkpoints.py refuses to guess), not a mode question.
    """
    resolved = {}
    for task in TASKS:
        try:
            source = checkpoints.resolve_source({**config, "task_name": task})
        except checkpoints.CheckpointError:
            continue
        resolved[task] = task_norm.resolve_mode(config, source.mode)
    return resolved


def shipped_configs():
    paths = [CONFIG_DIR / "deploy_policy.yml"]
    paths += sorted((CONFIG_DIR / "test_configs").glob("*.yml"))
    return [p for p in paths if p.is_file()]


@unittest.skipIf(yaml is None, "PyYAML is not installed here")
class DeployConfigs(unittest.TestCase):
    def test_every_config_declares_a_mode_exactly_one_way(self):
        for path in shipped_configs():
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text()) or {}
                self.assertIn("smolvla_task_normalization", config,
                              "the key is always present, so its absence is never the reason "
                              "a mode went undeclared")
                scalar = config["smolvla_task_normalization"]
                resolved = modes_of(config)
                if scalar is None:
                    # No scalar: the modes must come from the entries, so the
                    # config has to route the tasks itself.
                    self.assertTrue(resolved, f"{path.name} declares no mode anywhere")
                    # Mixed: every entry declares its own, which resolve_source
                    # has already checked is all-or-none.
                    tasks = (config.get("checkpoints") or {}).get("tasks") or {}
                    self.assertTrue(tasks, "a config without a scalar must route per task")
                    for task, entry in tasks.items():
                        self.assertIn(entry.get("mode"), ("legacy", "required"),
                                      f"{path.name}: {task} declares no mode")
                else:
                    # A scalar config may route nothing at all (the research
                    # configs take --checkpoint_dir on the command line); what
                    # it must not do is route something under another mode.
                    self.assertLessEqual(set(resolved.values()), {str(scalar).lower()})
                    self.assertEqual(task_norm.resolve_mode(config), str(scalar).lower())

    def test_required_routes_pass_the_adapters_own_guard(self):
        # Not a re-implementation of the rule: the guard the adapter runs.
        for path in shipped_configs():
            config = yaml.safe_load(path.read_text()) or {}
            if "required" not in modes_of(config).values():
                continue
            with self.subTest(config=path.name):
                try:
                    task_norm.reject_legacy_overrides(config)
                except task_norm.TaskNormalizationError as error:
                    self.fail(f"{path.name} would be refused at startup: {error}")

    def test_a_required_route_with_a_legacy_key_is_caught(self):
        # The reverse: the test would be worthless if it passed anything.
        for path in shipped_configs():
            config = yaml.safe_load(path.read_text()) or {}
            if "required" not in modes_of(config).values():
                continue
            config["smolvla_gripper_indices"] = [6, 13]
            with self.assertRaises(task_norm.TaskNormalizationError):
                task_norm.reject_legacy_overrides(config)
            return
        self.skipTest("no config routes a task in required mode")

    def test_there_is_a_required_route_and_a_legacy_one(self):
        modes = {}
        for path in shipped_configs():
            config = yaml.safe_load(path.read_text()) or {}
            for mode in set(modes_of(config).values()):
                modes.setdefault(mode, []).append(path.name)
        self.assertIn("legacy", modes, "Q1a and the official checkpoints need a legacy route")
        self.assertIn("required", modes, "a task-conditioned checkpoint needs a required route")

    def test_the_submission_config_is_the_frozen_mixed_routing(self):
        # Nine tasks on the official legacy checkpoints, items_handover on this
        # project's Q2-A -- the mapping frozen on 2026-09-24. A single scalar
        # would be wrong for one group or the other, so there is none.
        config = yaml.safe_load((CONFIG_DIR / "deploy_policy.yml").read_text())
        self.assertIsNone(config["smolvla_task_normalization"])
        resolved = modes_of(config)
        self.assertEqual(set(resolved), set(COMPETITION_TASKS))
        self.assertEqual(resolved["items_handover"], "required")
        for task in COMPETITION_TASKS:
            if task != "items_handover":
                self.assertEqual(resolved[task], "legacy", task)
        # and nothing outside the ten is routed, mode or not
        for task in set(TASKS) - set(COMPETITION_TASKS):
            with self.assertRaises(checkpoints.CheckpointError):
                checkpoints.resolve_source({**config, "task_name": task})

    def test_an_unrouted_task_is_not_silently_given_a_mode(self):
        config = yaml.safe_load((CONFIG_DIR / "deploy_policy.yml").read_text())
        config["checkpoints"]["tasks"].pop("click_bell")
        self.assertNotIn("click_bell", modes_of(config))
        with self.assertRaises(checkpoints.CheckpointError):
            checkpoints.resolve_source({**config, "task_name": "click_bell"})


if __name__ == "__main__":
    unittest.main()
