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


def block(mean, std, constant=None):
    # By default the mask is what the threshold says, which is what the loader
    # now insists on; `constant` overrides it to build inconsistent documents.
    mask = ([i in constant for i in range(DIM)] if constant is not None
            else [bool(s <= 1e-06) for s in std])
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
        "constant_std_threshold": 1e-06,
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
        # The check happens before any conversion, so it speaks of the env range.
        self.assertIn("outside the environment range", self.norm.state_range_problem(state))


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
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            # constant=() forces an all-false mask, contradicting the zero std.
            task_norm.TaskNorm(self.load(document({"t": block(np.zeros(DIM), std, constant=())})), "t")
        self.assertIn("constant_mask", str(caught.exception))

    def test_a_mask_that_marks_a_moving_dimension_fails(self):
        std = np.ones(DIM)
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            task_norm.TaskNorm(self.load(document({"t": block(np.zeros(DIM), std, constant=(4,))})), "t")
        self.assertIn("disagrees with std", str(caught.exception))

    def test_a_mask_of_zeros_and_ones_is_not_a_mask(self):
        tasks = {"t": block(np.zeros(DIM), np.ones(DIM))}
        tasks["t"]["action"]["constant_mask"] = [0] * DIM
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            task_norm.TaskNorm(self.load(document(tasks)), "t")
        self.assertIn("true/false", str(caught.exception))

    def test_the_transform_and_its_version_must_be_supported(self):
        for field, value in (("transform", "minmax"), ("transform_version", "q1c-99"),
                             ("features", ["action"]), ("constant_std_threshold", "small")):
            doc = document(self.tasks)
            doc["payload"][field] = value
            doc["content_sha256"] = hashlib.sha256(task_norm.canonical(doc["payload"])).hexdigest()
            with self.assertRaises(task_norm.TaskNormalizationError, msg=f"{field}={value!r}"):
                self.load(doc)

    def test_gripper_dimensions_must_be_unique_and_in_range(self):
        for dims in ([6, 6], [6, 14], [-1, 6], [6, True]):
            gripper = json.loads(json.dumps(GRIPPER))
            gripper["dims"] = dims
            with self.assertRaises(task_norm.TaskNormalizationError, msg=str(dims)):
                task_norm.TaskNorm(self.load(document(self.tasks, gripper=gripper)), "table_rearrangement")

    def test_ranges_must_be_finite_and_non_degenerate(self):
        for value in ([0.0, 0.0], [1.0, 0.0], [0.0, float("inf")]):
            gripper = json.loads(json.dumps(GRIPPER))
            gripper["action"]["env_range"] = value
            payload_json = json.dumps(gripper)          # inf would not serialize canonically
            if "Infinity" in payload_json:
                with self.assertRaises((task_norm.TaskNormalizationError, ValueError)):
                    task_norm.TaskNorm(self.load(document(self.tasks, gripper=gripper)), "table_rearrangement")
                continue
            with self.assertRaises(task_norm.TaskNormalizationError, msg=str(value)):
                task_norm.TaskNorm(self.load(document(self.tasks, gripper=gripper)), "table_rearrangement")

    def test_tolerance_must_be_a_finite_non_negative_number(self):
        for value in (-0.1, "wide", True):
            gripper = json.loads(json.dumps(GRIPPER))
            gripper["sanity_tolerance"] = value
            with self.assertRaises(task_norm.TaskNormalizationError, msg=repr(value)):
                task_norm.TaskNorm(self.load(document(self.tasks, gripper=gripper)), "table_rearrangement")

    def test_gripper_units_must_be_described_for_both_sides(self):
        gripper = {"dims": [6, 13], "sanity_tolerance": 0.05, "action": GRIPPER["action"]}
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.TaskNorm(self.load(document(self.tasks, gripper=gripper)), "table_rearrangement")

    def test_nothing_about_the_gripper_is_defaulted(self):
        # Every field, removed one at a time, must stop the run: a default here
        # would let the adapter assume a unit.
        for path in (("dims",), ("sanity_tolerance",), ("state", "env_units"),
                     ("state", "training_range"), ("action", "env_range"),
                     ("action", "training_units")):
            gripper = json.loads(json.dumps(GRIPPER))
            target = gripper
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            with self.assertRaises(task_norm.TaskNormalizationError, msg=f"{path} was defaulted"):
                task_norm.TaskNorm(self.load(document(self.tasks, gripper=gripper)), "table_rearrangement")
        with self.assertRaises(task_norm.TaskNormalizationError):
            task_norm.TaskNorm(self.load(document(self.tasks, gripper={})), "table_rearrangement")


class StateUnits(unittest.TestCase):
    """What happens if a task turns out to report the gripper in metres."""

    def setUp(self):
        self.mean = np.zeros(DIM)
        self.std = np.ones(DIM)
        self.metres = json.loads(json.dumps(GRIPPER))
        self.metres["state"] = {"env_units": "meter", "training_units": "ratio_0_1",
                                "env_range": [0.0, 0.05], "training_range": [0.0, 1.0]}

    def norm(self, gripper):
        return task_norm.TaskNorm(document({"t": block(self.mean, self.std)}, gripper=gripper), "t")

    def test_matching_units_pass_through_untouched(self):
        norm = self.norm(GRIPPER)
        state = np.linspace(0, 1, DIM)
        np.testing.assert_array_equal(norm.state_in_training_units(state), state)

    def test_metre_state_is_converted_by_the_declared_ranges(self):
        norm = self.norm(self.metres)
        state = np.zeros(DIM)
        state[6] = 0.025          # half open, in metres
        state[13] = 0.05          # fully open
        converted = norm.state_in_training_units(state)
        self.assertAlmostEqual(converted[6], 0.5)
        self.assertAlmostEqual(converted[13], 1.0)
        # Only the gripper dimensions move.
        self.assertTrue(np.array_equal(np.delete(converted, [6, 13]), np.delete(state, [6, 13])))

    def test_the_range_check_uses_the_environment_unit(self):
        norm = self.norm(self.metres)
        state = np.zeros(DIM)
        state[6] = 0.05                                  # in range for metres
        self.assertIsNone(norm.state_range_problem(state))
        state[6] = 0.9                                   # would be fine as a ratio, not as metres
        self.assertIn("outside the environment range", norm.state_range_problem(state))

    def test_a_degenerate_range_cannot_be_converted(self):
        broken = json.loads(json.dumps(self.metres))
        broken["state"]["env_range"] = [0.05, 0.05]
        with self.assertRaises(task_norm.TaskNormalizationError):
            self.norm(broken).state_in_training_units(np.zeros(DIM))


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


class ModeContract(unittest.TestCase):
    """Declared mode against what the checkpoint carries: both directions refuse."""

    def checkpoint(self, statistics=None):
        directory = Path(tempfile.mkdtemp())
        path = directory / task_norm.STATISTICS_FILE
        if statistics == "file":
            path.write_text(json.dumps(document({"t": block(np.zeros(DIM), np.ones(DIM))})))
        elif statistics == "dangling":
            path.symlink_to(directory / "gone.json")
        elif statistics == "directory":
            path.mkdir()
        return directory

    def test_required_with_statistics_passes(self):
        task_norm.check_mode_matches_checkpoint("required", self.checkpoint("file"))

    def test_required_without_statistics_is_refused(self):
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            task_norm.check_mode_matches_checkpoint("required", self.checkpoint())
        self.assertIn("does not exist", str(caught.exception))

    def test_legacy_without_statistics_passes(self):
        task_norm.check_mode_matches_checkpoint("legacy", self.checkpoint())

    def test_legacy_with_statistics_is_refused(self):
        # What happened on 2026-09-19, when nothing refused it.
        with self.assertRaises(task_norm.TaskNormalizationError) as caught:
            task_norm.check_mode_matches_checkpoint("legacy", self.checkpoint("file"))
        self.assertIn("legacy", str(caught.exception))
        self.assertIn(task_norm.STATISTICS_FILE, str(caught.exception))

    def test_legacy_refuses_anything_under_that_name(self):
        # Not "a readable file": the name alone is the checkpoint's claim.
        for kind in ("dangling", "directory"):
            with self.subTest(kind=kind), self.assertRaises(task_norm.TaskNormalizationError):
                task_norm.check_mode_matches_checkpoint("legacy", self.checkpoint(kind))

    def test_an_unresolved_mode_is_refused(self):
        for mode in ("Legacy", "auto", None):
            with self.subTest(mode=mode), self.assertRaises(task_norm.TaskNormalizationError):
                task_norm.check_mode_matches_checkpoint(mode, self.checkpoint())


if __name__ == "__main__":
    unittest.main()
