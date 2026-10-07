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
        # get_model dispatches pi0 before SmolVLA's normalization guard. pi0
        # uses its own checkpoint assets; assigning it a SmolVLA mode is wrong.
        resolved[task] = "pi0" if source.backend == "pi0" else task_norm.resolve_mode(config, source.mode)
    return resolved


def shipped_configs():
    paths = [CONFIG_DIR / "deploy_policy.yml"]
    paths += sorted((CONFIG_DIR / "test_configs").glob("*.yml"))
    return [p for p in paths if p.is_file()]


@unittest.skipIf(yaml is None, "PyYAML is not installed here")
class DeployConfigs(unittest.TestCase):
    def test_pi0_local_config_keeps_nine_legacy_routes_and_own_assets(self):
        config = yaml.safe_load((CONFIG_DIR / "test_configs/pi0_drawer_local.yml").read_text())
        resolved = modes_of(config)
        self.assertEqual(resolved["drawer_open_place"], "pi0")
        self.assertEqual(set(resolved), set(COMPETITION_TASKS))
        for task in set(COMPETITION_TASKS) - {"drawer_open_place"}:
            self.assertEqual(resolved[task], "legacy", task)

    def test_pi0_drawer_handover_config_routes_exactly_those_two_and_keeps_eight_legacy(self):
        config = yaml.safe_load((CONFIG_DIR / "test_configs/pi0_drawer_handover_local.yml").read_text())
        resolved = modes_of(config)
        self.assertEqual(set(resolved), set(COMPETITION_TASKS))
        self.assertEqual({t for t, m in resolved.items() if m == "pi0"}, {"drawer_open_place", "items_handover"})
        for task in set(COMPETITION_TASKS) - {"drawer_open_place", "items_handover"}:
            self.assertEqual(resolved[task], "legacy", task)
        entry = config["checkpoints"]["tasks"]["items_handover"]
        self.assertEqual(entry["pi0_config"], "pi0_items_handover_lora_b2")
        self.assertTrue(entry["local_dir"].endswith("/b2_items_handover/15999"))  # the B6 locked label (step 16000)
        pi05_config = (ROOT / "policy/pi05/src/openpi/training/config.py").read_text()
        self.assertIn('name="pi0_items_handover_lora_b2"', pi05_config)

    def test_pi0_drawer_handover_config_differs_from_the_drawer_one_only_in_the_handover_entry(self):
        a = yaml.safe_load((CONFIG_DIR / "test_configs/pi0_drawer_local.yml").read_text())
        b = yaml.safe_load((CONFIG_DIR / "test_configs/pi0_drawer_handover_local.yml").read_text())
        ta, tb = a["checkpoints"].pop("tasks"), b["checkpoints"].pop("tasks")
        self.assertEqual(a, b)
        self.assertEqual({k: v for k, v in ta.items() if k != "items_handover"},
                         {k: v for k, v in tb.items() if k != "items_handover"})

    def test_pi0_cannot_be_assigned_a_smolvla_mode(self):
        config = yaml.safe_load((CONFIG_DIR / "test_configs/pi0_drawer_local.yml").read_text())
        config["checkpoints"]["tasks"]["drawer_open_place"]["mode"] = "legacy"
        with self.assertRaises(checkpoints.CheckpointError):
            checkpoints.resolve_source({**config, "task_name": "drawer_open_place"})

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
                        if entry.get("backend", "smolvla") == "pi0":
                            self.assertNotIn("mode", entry, "pi0 must not use SmolVLA normalization")
                            self.assertTrue(entry.get("pi0_config"), "pi0 requires its own transform config")
                            self.assertEqual(resolved.get(task), "pi0")
                            continue
                        self.assertIn(entry.get("mode"), ("legacy", "required"),
                                      f"{path.name}: {task} declares no mode")
                else:
                    # A scalar config may route nothing at all (the research
                    # configs take --checkpoint_dir on the command line); what
                    # it must not do is route something under another mode.
                    self.assertLessEqual(set(resolved.values()) - {"pi0"}, {str(scalar).lower()})
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

    def test_there_is_a_required_config_and_a_legacy_one(self):
        # The submission ships ten legacy routes, but the required path is still
        # live code and still has to be exercised by a config that declares it
        # -- test_configs/q1c_taskstd.yml, which every task-conditioned
        # evaluation in this project used.
        modes = {}
        for path in shipped_configs():
            config = yaml.safe_load(path.read_text()) or {}
            scalar = config.get("smolvla_task_normalization")
            found = set(modes_of(config).values()) or ({str(scalar).lower()} if scalar else set())
            for mode in found:
                modes.setdefault(mode, []).append(path.name)
        self.assertIn("legacy", modes, "the official checkpoints need a legacy config")
        self.assertIn("required", modes, "a task-conditioned checkpoint needs a required config")

    def test_the_submission_config_is_the_final_mapping(self):
        # All ten on the official legacy checkpoints, settled by holdout on
        # 2026-09-26. The scalar stays unset even though every entry now says
        # the same thing: a scalar states the mode once, and a checkpoint that
        # turned up carrying statistics would be run without them, while a
        # per-task declaration refuses that one task.
        config = yaml.safe_load((CONFIG_DIR / "deploy_policy.yml").read_text())
        self.assertIsNone(config["smolvla_task_normalization"])
        resolved = modes_of(config)
        self.assertEqual(set(resolved), set(COMPETITION_TASKS))
        for task in COMPETITION_TASKS:
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
