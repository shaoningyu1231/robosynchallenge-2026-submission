"""The task-conditioned transform, and everything it refuses to do.

    python3 -m unittest tests.test_task_norm -v

No torch and no simulator: this is the arithmetic and the validation, which are
where a training-inference asymmetry would hide. The end-to-end checks that need
a real worker live in the acceptance scripts.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Loaded by path, not through the package: policy/smolvla_multitask/__init__.py
# imports deploy_policy, which needs torch, and this file is deliberately
# runnable without it -- and without a GPU.
import importlib.util  # noqa: E402

_MODULE = Path(__file__).resolve().parents[1] / "policy" / "smolvla_multitask" / "task_norm.py"
_spec = importlib.util.spec_from_file_location("q1c_task_norm_under_test", _MODULE)
task_norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task_norm)
assert Path(task_norm.__file__).resolve() == _MODULE, "loaded another checkout's task_norm"

DIM = 14


def block(mean, std, constant=()):
    mask = [i in constant for i in range(DIM)]
    return {
        "observation.state": {"mean": list(mean), "std": list(std), "constant_mask": mask},
        "action": {"mean": list(mean), "std": list(std), "constant_mask": mask},
        "task_index": 0,
        "frames": 10,
    }


GRIPPER = {
    "dims": [6, 13],
    "sanity_tolerance": 0.05,
    "state": {"env_units": "ratio_0_1", "training_units": "ratio_0_1",
              "env_range": [0.0, 1.0], "training_range": [0.0, 1.0]},
    "action": {"env_units": "meter", "training_units": "ratio_0_1",
               "env_range": [0.0, 0.05], "training_range": [0.0, 1.0]},
}


def document(tasks, gripper=None, **overrides):
    payload = {
        "transform": "task_conditioned_mean_std",
        "transform_version": "q1c-1",
        "features": list(task_norm.FEATURES),
        "gripper": GRIPPER if gripper is None else gripper,
        "tasks": tasks,
    }
    payload.update(overrides.pop("payload", {}))
    out = {
        "schema_version": 1,
        "payload": payload,
        "content_sha256": hashlib.sha256(task_norm.canonical(payload)).hexdigest(),
        "task_sha256": {slug: hashlib.sha256(task_norm.canonical(b)).hexdigest()
                        for slug, b in tasks.items()},
    }
    out.update(overrides)
    return out


def written(document_dict) -> Path:
    directory = Path(tempfile.mkdtemp())
    (directory / "task_normalization.json").write_text(json.dumps(document_dict))
    return directory


class Transform(unittest.TestCase):
    def setUp(self):
        self.mean = np.arange(DIM, dtype=float) * 0.1
        self.std = np.full(DIM, 0.25)
        self.norm = task_norm.TaskNorm(document({"table_rearrangement": block(self.mean, self.std)}),
                                       "table_rearrangement")

    def test_state_is_standardized_by_the_task_statistics(self):
        state = self.mean + self.std * np.array([1.0, -2.0] + [0.0] * (DIM - 2))
        got = self.norm.standardize_state(state)
        want = np.array([1.0, -2.0] + [0.0] * (DIM - 2), dtype=np.float32)
        np.testing.assert_allclose(got, want, atol=1e-6)

    def test_action_returns_to_training_units(self):
        standardized = np.linspace(-1.5, 1.5, DIM)
        got = self.norm.unstandardize_action(standardized)
        np.testing.assert_allclose(got, standardized * self.std + self.mean, rtol=1e-6)

    def test_round_trip(self):
        state = self.mean + self.std * np.random.default_rng(0).normal(size=DIM)
        back = self.norm.unstandardize_action(self.norm.standardize_state(state))
        np.testing.assert_allclose(back, state, atol=1e-6)

    def test_constant_dimensions_return_the_task_mean(self):
        std = self.std.copy()
        std[6] = 0.0
        norm = task_norm.TaskNorm(
            document({"click_bell": block(self.mean, std, constant=(6,))}), "click_bell")
        # Whatever the model emits there, the action is the task's own mean.
        for emitted in (-5.0, 0.0, 17.0):
            action = np.zeros(DIM)
            action[6] = emitted
            # float32 output, so compare at float32 precision.
            self.assertEqual(norm.unstandardize_action(action)[6], np.float32(self.mean[6]))
        state = self.mean.copy()
        state[6] = 999.0
        self.assertEqual(float(norm.standardize_state(state)[6]), 0.0)

    def test_extra_action_dimensions_pass_through(self):
        padded = np.concatenate([np.zeros(DIM), [7.0, 8.0]])
        got = self.norm.unstandardize_action(padded)
        np.testing.assert_allclose(got[DIM:], [7.0, 8.0])

    def test_gripper_rules_come_from_the_metadata(self):
        self.assertEqual(self.norm.gripper_indices, (6, 13))
        self.assertTrue(self.norm.rescale_gripper)          # metre env, ratio training
        self.assertEqual(self.norm.gripper_source_range, (0.0, 1.0))
        same_units = json.loads(json.dumps(GRIPPER))
        same_units["action"]["env_units"] = "ratio_0_1"
        norm = task_norm.TaskNorm(
            document({"t": block(self.mean, self.std)}, gripper=same_units), "t")
        self.assertFalse(norm.rescale_gripper)

    def test_action_space_must_match_the_metadata(self):
        low, high = np.zeros(DIM), np.full(DIM, 3.14)
        high[6] = high[13] = 0.05
        self.assertIsNone(self.norm.action_space_problem(low, high))
        high[13] = 1.0
        self.assertIn("gripper action space", self.norm.action_space_problem(low, high))

    def test_state_range_is_a_sanity_check_with_tolerance(self):
        state = np.zeros(DIM)
        state[6] = 1.0131                                   # measured on the real env
        self.assertIsNone(self.norm.state_range_problem(state))
        state[6] = 3.5
        self.assertIn("outside the training range", self.norm.state_range_problem(state))


class Validation(unittest.TestCase):
    def setUp(self):
        self.tasks = {"table_rearrangement": block(np.zeros(DIM), np.ones(DIM))}

    def load(self, document_dict):
        return task_norm.load_document(written(document_dict))

    def test_valid_document_loads(self):
        self.assertEqual(self.load(document(self.tasks))["schema_version"], 1)

    def test_missing_file_fails(self):
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.load_document(Path(tempfile.mkdtemp()))

    def test_empty_object_is_not_statistics(self):
        with self.assertRaises(task_norm.TaskNormalizationError):
            self.load({})

    def test_a_changed_number_breaks_the_digest(self):
        doc = document(self.tasks)
        doc["payload"]["tasks"]["table_rearrangement"]["action"]["std"][3] = 2.0
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            self.load(doc)
        self.assertIn("content_sha256", str(caught.exception))

    def test_a_task_digest_that_does_not_cover_the_payload(self):
        doc = document(self.tasks)
        doc["payload"]["tasks"]["table_rearrangement"]["action"]["std"][3] = 2.0
        doc["content_sha256"] = hashlib.sha256(task_norm.canonical(doc["payload"])).hexdigest()
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            self.load(doc)
        self.assertIn("task_sha256", str(caught.exception))

    def test_duplicate_task_keys_are_refused(self):
        directory = Path(tempfile.mkdtemp())
        doc = json.dumps(document(self.tasks), separators=(",", ":"))
        marker = '"tasks":{'
        self.assertIn(marker, doc)
        doubled = doc.replace(marker, marker + '"table_rearrangement":{},', 1)
        (directory / "task_normalization.json").write_text(doubled)
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.load_document(directory)

    def test_unknown_task_fails_loudly(self):
        doc = self.load(document(self.tasks))
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            task_norm.TaskNorm(doc, "drawer_open_place")
        self.assertIn("drawer_open_place", str(caught.exception))

    def test_a_misspelled_task_is_not_matched_loosely(self):
        doc = self.load(document(self.tasks))
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.TaskNorm(doc, "table_rearragement")

    def test_wrong_width_statistics_fail(self):
        tasks = {"t": block(np.zeros(DIM), np.ones(DIM))}
        tasks["t"]["action"]["mean"] = [0.0] * (DIM - 1)
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.TaskNorm(self.load(document(tasks)), "t")

    def test_zero_std_without_a_constant_mask_fails(self):
        std = np.ones(DIM)
        std[2] = 0.0
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.TaskNorm(self.load(document({"t": block(np.zeros(DIM), std)})), "t")

    def test_gripper_units_must_be_described_for_both_sides(self):
        gripper = {"dims": [6, 13], "action": GRIPPER["action"]}
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.TaskNorm(self.load(document(self.tasks, gripper=gripper)), "table_rearrangement")


class Mode(unittest.TestCase):
    def test_the_mode_must_be_declared(self):
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.resolve_mode({})
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.resolve_mode({"smolvla_task_normalization": "yes"})
        self.assertEqual(task_norm.resolve_mode({"smolvla_task_normalization": "required"}), "required")
        self.assertEqual(task_norm.resolve_mode({"smolvla_task_normalization": "Legacy"}), "legacy")

    def test_required_refuses_the_old_gripper_overrides(self):
        task_norm.reject_legacy_overrides({"smolvla_rescale_gripper": "auto"})     # the default
        for name in task_norm.LEGACY_ONLY_ARGS:
            with self.assertRaises(task_norm.TaskNormalizationError) as caught:
                task_norm.reject_legacy_overrides({name: True})
            self.assertIn(name, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
