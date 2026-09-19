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


def shipped_configs():
    paths = [CONFIG_DIR / "deploy_policy.yml"]
    paths += sorted((CONFIG_DIR / "test_configs").glob("*.yml"))
    return [p for p in paths if p.is_file()]


@unittest.skipIf(yaml is None, "PyYAML is not installed here")
class DeployConfigs(unittest.TestCase):
    def test_every_config_declares_a_mode(self):
        for path in shipped_configs():
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text()) or {}
                self.assertIn("smolvla_task_normalization", config,
                              "the mode is declared, never inferred from a file's presence")
                self.assertEqual(task_norm.resolve_mode(config),
                                 str(config["smolvla_task_normalization"]).lower())

    def test_required_configs_pass_the_adapters_own_guard(self):
        # Not a re-implementation of the rule: the guard the adapter runs.
        for path in shipped_configs():
            config = yaml.safe_load(path.read_text()) or {}
            if task_norm.resolve_mode(config) != "required":
                continue
            with self.subTest(config=path.name):
                try:
                    task_norm.reject_legacy_overrides(config)
                except task_norm.TaskNormalizationError as error:
                    self.fail(f"{path.name} would be refused at startup: {error}")

    def test_a_required_config_with_a_legacy_key_is_caught(self):
        # The reverse: the test would be worthless if it passed anything.
        for path in shipped_configs():
            config = yaml.safe_load(path.read_text()) or {}
            if task_norm.resolve_mode(config) != "required":
                continue
            config["smolvla_gripper_indices"] = [6, 13]
            with self.assertRaises(task_norm.TaskNormalizationError):
                task_norm.reject_legacy_overrides(config)
            return
        self.skipTest("no config declares required mode")

    def test_there_is_a_required_config_and_a_legacy_one(self):
        modes = {}
        for path in shipped_configs():
            config = yaml.safe_load(path.read_text()) or {}
            modes.setdefault(task_norm.resolve_mode(config), []).append(path.name)
        self.assertIn("legacy", modes, "Q1a and the official checkpoints need a legacy config")
        self.assertIn("required", modes, "Q1c needs a config that declares required")

    def test_the_default_config_stays_legacy(self):
        # The submission's default evaluates the checkpoints measured so far;
        # switching it to required would refuse every one of them.
        config = yaml.safe_load((CONFIG_DIR / "deploy_policy.yml").read_text())
        self.assertEqual(task_norm.resolve_mode(config), "legacy")


if __name__ == "__main__":
    unittest.main()
