"""load_frozen_episode_seeds refuses a list that does not belong to this run.

Needs the simulator environment (scripts/eval_policy.py imports EmbodiChain), so
it skips elsewhere:

    python -m unittest discover -s tests -p test_frozen_episode_seeds.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts import eval_policy
except Exception:  # noqa: BLE001  (no simulator here)
    eval_policy = None


@unittest.skipIf(eval_policy is None, "scripts/eval_policy.py needs the simulator environment")
class FrozenEpisodeSeeds(unittest.TestCase):
    SEEDS = [209652396, 398764591, 924231285]

    def write(self, directory, drop_detail=False, **overrides):
        payload = {
            "task": "item_assembly", "setting": "random", "base_seed": 0, "episodes": 3,
            "accepted": list(self.SEEDS),
            "accepted_detail": [{"candidate_index": i, "seed": s, "expert_plan_steps": 5,
                                 "reason": "expert_planning_succeeded"} for i, s in enumerate(self.SEEDS)],
            "skipped": [],
        }
        payload.update(overrides)
        if drop_detail:
            payload.pop("accepted_detail")
        path = Path(directory) / "seeds.json"
        path.write_text(json.dumps(payload))
        return path

    def config(self, path, **overrides):
        config = {"eval_episode_seeds": str(path), "task_name": "item_assembly", "setting": "random",
                  "seed": 0, "max_episodes": 3}
        config.update(overrides)
        return config

    def test_a_matching_list_is_accepted_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp)
            frozen = eval_policy.load_frozen_episode_seeds(self.config(path))
            self.assertEqual(frozen["seeds"], self.SEEDS)
            self.assertEqual(frozen["kind"], "frozen_list")
            self.assertEqual(len(frozen["file_sha256"]), 64)
            self.assertEqual(len(frozen["used_sha256"]), 64)
            # Only the episodes that run are hashed.
            fewer = eval_policy.load_frozen_episode_seeds(self.config(path, max_episodes=2))
            self.assertEqual(fewer["seeds"], self.SEEDS[:2])
            self.assertNotEqual(fewer["used_sha256"], frozen["used_sha256"])

    def test_no_list_means_the_organisers_protocol(self):
        for value in (None, "", "null"):
            self.assertIsNone(eval_policy.load_frozen_episode_seeds({"eval_episode_seeds": value}))

    def test_every_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = {
                "selected for task": (dict(task="click_bell"), {}),
                "selected for setting": (dict(setting="clear"), {}),
                "base seed": (dict(base_seed=1), {}),
                "accepted_detail does not match": (dict(accepted_detail=[{"seed": 1}, {"seed": 2}, {"seed": 3}]), {}),
                "outside the evaluator's range": (dict(accepted=[-1, 2, 3], drop_detail=True), {}),
                "repeats a seed": (dict(accepted=[7, 7, 8], drop_detail=True), {}),
                "seeds for": (dict(), {"max_episodes": 5}),
            }
            for expected, (payload, config) in cases.items():
                with self.subTest(expected):
                    path = self.write(tmp, **payload)
                    with self.assertRaises(ValueError) as caught:
                        eval_policy.load_frozen_episode_seeds(self.config(path, **config))
                    self.assertIn(expected, str(caught.exception))

    def test_it_cannot_be_combined_with_a_single_fixed_seed(self):
        self.assertIn("eval_episode_seeds and eval_fixed_episode_seed cannot both be set",
                      Path(REPO_ROOT / "scripts" / "eval_policy.py").read_text())


if __name__ == "__main__":
    unittest.main()
