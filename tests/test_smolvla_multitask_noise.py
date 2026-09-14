"""Checks the seeded flow-matching noise of policy/smolvla_multitask.

The seed rule needs only the standard library; the worker generator and the
adapter bookkeeping need torch and numpy (no simulator, no LeRobot: the worker
is replaced by a fake that draws from the real EpisodeNoise).

    python -m unittest discover -s tests -p test_smolvla_multitask_noise.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO_ROOT / "policy" / "smolvla_multitask"
sys.path.insert(0, str(REPO_ROOT))


def load(name, filename, directory=POLICY_DIR):
    # By path: in the simulator's venv `import tests` finds an unrelated package
    # (draccus ships one), and `tests.<module>` another checkout's file.
    spec = importlib.util.spec_from_file_location(name, directory / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


noise_seed = load("smolvla_multitask_noise_seed", "noise_seed.py")


def call_action_digest(actions):
    """The adapter's per-call action digest, recomputed from saved actions."""
    import hashlib

    h = hashlib.sha256()
    for action in actions:
        h.update(action.tobytes())
    return h.hexdigest()[:16]

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    torch = None


class SeedRule(unittest.TestCase):
    def test_deterministic_and_in_generator_range(self):
        a = noise_seed.policy_noise_seed(0, 398764591)
        self.assertEqual(a, noise_seed.policy_noise_seed(0, 398764591))
        self.assertTrue(0 <= a < 2**63)

    def test_differs_by_environment_seed_and_base(self):
        seeds = {noise_seed.policy_noise_seed(base, env) for base in (0, 1) for env in (1, 2, 398764591)}
        self.assertEqual(len(seeds), 6)

    def test_parse_base(self):
        self.assertIsNone(noise_seed.parse_base(None))
        self.assertIsNone(noise_seed.parse_base("null"))
        self.assertEqual(noise_seed.parse_base(0), 0)
        self.assertEqual(noise_seed.parse_base("12"), 12)
        for bad in (-1, "-1", 1.5, "1.5", True, "abc"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                noise_seed.parse_base(bad)


@unittest.skipIf(torch is None, "torch not installed")
class WorkerNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load("smolvla_multitask_worker", "smolvla_worker.py")

    def test_same_seed_same_sequence_and_each_draw_differs(self):
        shape = (1, 50, 32)
        a, b = self.worker.EpisodeNoise(7), self.worker.EpisodeNoise(7)
        draws_a = [a.draw(shape, "cpu") for _ in range(3)]
        draws_b = [b.draw(shape, "cpu") for _ in range(3)]
        for (na, da), (nb, db) in zip(draws_a, draws_b):
            self.assertTrue(torch.equal(na, nb))
            self.assertEqual(da, db)
        self.assertEqual(len({d for _, d in draws_a}), 3, "successive chunks must get different noise")
        self.assertEqual((a.draws, draws_a[0][0].dtype, tuple(draws_a[0][0].shape)), (3, torch.float32, shape))

    def test_different_seed_different_noise(self):
        shape = (1, 50, 32)
        _, d1 = self.worker.EpisodeNoise(7).draw(shape, "cpu")
        _, d2 = self.worker.EpisodeNoise(8).draw(shape, "cpu")
        self.assertNotEqual(d1, d2)

    def test_does_not_touch_the_global_rng(self):
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        self.worker.EpisodeNoise(7).draw((1, 50, 32), "cpu")
        self.assertTrue(torch.equal(torch.rand(3), expected))


@unittest.skipIf(torch is None, "torch not installed")
class AdapterSeeding(unittest.TestCase):
    """The real SmolVLAWorkerClient bookkeeping, with a fake worker behind _rpc."""

    @classmethod
    def setUpClass(cls):
        from policy.smolvla_multitask import deploy_policy

        fakes = load("smolvla_adapter_fakes", "test_smolvla_adapter.py", REPO_ROOT / "tests")
        cls.dp, cls.fakes = deploy_policy, fakes
        if Path(deploy_policy.__file__).resolve().parents[2] != REPO_ROOT:
            raise RuntimeError(f"deploy_policy imported from another checkout: {deploy_policy.__file__}")
        cls.worker = load("smolvla_multitask_worker_for_adapter", "smolvla_worker.py")

    def client(self, base=0):
        dp, worker = self.dp, self.worker
        client = object.__new__(dp.SmolVLAWorkerClient)
        client.noise_base = base
        client.episode_record = dp._empty_episode_record()
        client.episode_timing = dp._empty_timing()
        client.replan_steps, client.smolvla_steps = None, 10
        client.image_keys = self.fakes.FakeModel.image_keys
        client.rescale_gripper, client.gripper_indices, client.gripper_source_range = False, (6, 13), (0.0, 1.0)
        client.check_success_each_step = False
        client.debug_actions, client._debug_chunks_printed, client.debug_action_log_path = False, 0, None
        client.dump_obs_dir, client.dump_obs_calls, client._episodes_started = None, 5, 0
        client.sent = []
        state = {"noise": None}

        def fake_rpc(payload):
            client.sent.append(payload["cmd"])
            if payload["cmd"] == "reset":
                state["noise"] = None
                return {"ok": True}
            if payload["cmd"] == "seed_noise":
                state["noise"] = worker.EpisodeNoise(payload["seed"])
                return {"ok": True}
            assert payload["seeded_noise"] == (base is not None)
            if payload["seeded_noise"] and state["noise"] is None:
                raise RuntimeError("never seeded")
            # One forward per call; the actions depend on the noise, like the real policy's.
            if state["noise"] is not None:
                noise, digest = state["noise"].draw((1, 50, 32), "cpu")
                digests = [digest]
            else:
                noise, digests = torch.randn(1, 50, 32), []
            actions = noise[0, :10, :14].numpy().tolist()
            return {"ok": True, "actions": actions, "forwards": 1, "forward_s": 0.0, "noise_digests": digests}

        client._rpc = fake_rpc
        return client

    def episode(self, client, env_seed, calls=2):
        env = self.fakes.FakeEnv()
        if env_seed is not None:
            env.unwrapped._robosyn_episode_seed = env_seed
        self.dp.reset_model(client)
        for _ in range(calls):
            self.dp.eval(env, client, self.fakes.make_obs())
        return self.dp.episode_metadata(client)

    def test_seeded_once_per_episode_and_recorded(self):
        client = self.client()
        meta = self.episode(client, 398764591)
        self.assertEqual(client.sent, ["reset", "seed_noise", "infer", "infer"])
        self.assertEqual(meta["env_seed"], 398764591)
        self.assertEqual(meta["noise_seed"], noise_seed.policy_noise_seed(0, 398764591))
        self.assertEqual(meta["forwards"], 2)
        self.assertEqual(len(set(meta["noise_digests"])), 2)
        self.assertEqual([call["forwards"] for call in meta["calls"]], [1, 1])
        self.assertEqual(len({call["obs"] for call in meta["calls"]}), 1, "the fake env sends one observation")
        self.assertEqual(len({call["actions"] for call in meta["calls"]}), 2)

    def test_same_scene_reproduces_and_another_scene_does_not(self):
        client = self.client()
        first = self.episode(client, 398764591)
        again = self.episode(client, 398764591)
        other = self.episode(client, 924231285)
        self.assertEqual(first, again)
        self.assertNotEqual(first["noise_digests"], other["noise_digests"])
        self.assertNotEqual(first["action_digest"], other["action_digest"])

    def test_another_base_changes_the_noise(self):
        a = self.episode(self.client(base=0), 398764591)
        b = self.episode(self.client(base=1), 398764591)
        self.assertNotEqual(a["noise_digests"], b["noise_digests"])

    def test_missing_episode_seed_is_an_error_not_unseeded_noise(self):
        with self.assertRaisesRegex(RuntimeError, "no episode seed"):
            self.episode(self.client(), None)

    def test_dump_saves_the_first_calls_of_each_episode(self):
        import tempfile

        client = self.client()
        with tempfile.TemporaryDirectory() as tmp:
            client.dump_obs_dir, client.dump_obs_calls = Path(tmp), 1
            self.episode(client, 398764591)
            meta = self.episode(client, 924231285)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["episode000_call000.npz", "episode001_call000.npz"])
            saved = np.load(Path(tmp) / "episode001_call000.npz")
            self.assertEqual(int(saved["env_seed"]), 924231285)
            self.assertEqual(saved["noise_digests"].tolist(), meta["noise_digests"][:1])
            self.assertEqual(call_action_digest(saved["actions"]), meta["calls"][0]["actions"])

    def test_off_sends_no_seed_and_records_none(self):
        client = self.client(base=None)
        meta = self.episode(client, 398764591)
        self.assertNotIn("seed_noise", client.sent)
        self.assertEqual((meta["noise_seed"], meta["noise_digests"]), (None, []))


if __name__ == "__main__":
    unittest.main()
