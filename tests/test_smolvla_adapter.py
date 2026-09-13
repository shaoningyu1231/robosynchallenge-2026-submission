"""Checks the SmolVLA adapter against the contract of scripts/eval_policy.py.

Runs without the simulator or LeRobot: the env and the worker client are fakes.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from policy.smolvla import deploy_policy  # noqa: E402

ACTION_DIM = 14


class FakeSpace:
    shape = (ACTION_DIM,)
    low = np.full(ACTION_DIM, -1.0, dtype=np.float32)
    high = np.full(ACTION_DIM, 1.0, dtype=np.float32)


class FakeUnwrapped:
    single_action_space = FakeSpace()
    device = "cpu"


def make_obs():
    image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    return {
        "robot": {"qpos": np.zeros((1, ACTION_DIM), dtype=np.float32)},
        "sensor": {name: {"color": image} for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
    }


class FakeEnv:
    def __init__(self, success_at=None, truncate_at=1000):
        self.unwrapped = FakeUnwrapped()
        self.steps = 0
        self.success_at = success_at
        self.truncate_at = truncate_at
        self._current_instruction = "Click the bell"

    def step(self, action):
        assert tuple(action.shape) == (1, ACTION_DIM)
        self.steps += 1
        truncated = np.array([self.steps >= self.truncate_at])
        return make_obs(), 0.0, np.array([False]), truncated, {"elapsed_steps": np.array(self.steps)}

    def get_wrapper_attr(self, name):
        assert name == "is_task_success"
        return lambda: self.success_at is not None and self.steps >= self.success_at


class FakeModel:
    """Stands in for SmolVLAWorkerClient without starting the worker process."""

    image_keys = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
    }
    rescale_gripper = False
    gripper_indices = (6, 13)
    gripper_source_range = (0.0, 1.0)

    def __init__(self, n_actions=10, check_success_each_step=False):
        self.n_actions = n_actions
        self.check_success_each_step = check_success_each_step
        self.tasks = []
        self.calls = 0
        self.logged = 0

    def infer(self, obs, task):
        self.tasks.append(task)
        return [np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(self.n_actions)]

    def maybe_print_action_debug(self, actions):
        pass

    def maybe_print_env_action_debug(self, action_tensor):
        pass

    def record_call(self, seconds):
        self.calls += 1

    def log_episode_timing(self):
        self.logged += 1


class EvalContract(unittest.TestCase):
    def test_returns_four_values_like_eval_policy_expects(self):
        env, model = FakeEnv(), FakeModel()
        _, info, _, times = deploy_policy.eval(env, model, make_obs())
        self.assertEqual(len(times), 1)
        self.assertGreaterEqual(times[0], 0.0)
        self.assertEqual(int(info["elapsed_steps"]), 10)
        self.assertEqual(model.tasks, ["Click the bell"])

    def test_official_behavior_runs_whole_batch_after_success(self):
        env = FakeEnv(success_at=3)
        deploy_policy.eval(env, FakeModel(check_success_each_step=False), make_obs())
        self.assertEqual(env.steps, 10)

    def test_stops_at_success_when_enabled(self):
        env = FakeEnv(success_at=3)
        deploy_policy.eval(env, FakeModel(check_success_each_step=True), make_obs())
        self.assertEqual(env.steps, 3)

    def test_stops_at_truncation(self):
        env = FakeEnv(truncate_at=4)
        _, _, truncated, _ = deploy_policy.eval(env, FakeModel(), make_obs())
        self.assertEqual(env.steps, 4)
        self.assertTrue(bool(truncated[0]))


class EpisodeTiming(unittest.TestCase):
    def test_logged_once_when_the_episode_ends(self):
        env, model = FakeEnv(success_at=15), FakeModel(check_success_each_step=True)
        deploy_policy.eval(env, model, make_obs())  # steps 1-10, still running
        self.assertEqual((model.calls, model.logged), (1, 0))
        deploy_policy.eval(env, model, make_obs())  # succeeds at step 15
        self.assertEqual((model.calls, model.logged, env.steps), (2, 1, 15))

    def test_logged_on_truncation(self):
        env, model = FakeEnv(truncate_at=4), FakeModel()
        deploy_policy.eval(env, model, make_obs())
        self.assertEqual(model.logged, 1)


class ReplanValidation(unittest.TestCase):
    def test_rejects_non_positive_replan_steps(self):
        for value in (0, -3):
            with self.assertRaises(ValueError):
                deploy_policy.SmolVLAWorkerClient({"checkpoint_dir": "/nonexistent", "smolvla_replan_steps": value})


class WorkerRequest(unittest.TestCase):
    def client(self, smolvla_steps, replan_steps):
        client = object.__new__(deploy_policy.SmolVLAWorkerClient)
        client.smolvla_steps = smolvla_steps
        client.replan_steps = replan_steps
        client.episode_timing = deploy_policy._empty_timing()
        client.requests = []
        client._rpc = lambda payload: client.requests.append(payload) or {
            "ok": True,
            "actions": [],
            "forwards": 1,
            "forward_s": 0.25,
        }
        return client

    def test_worker_forwards_accumulate_per_episode(self):
        client = self.client(10, 25)
        for _ in range(2):
            client.infer({"observation.state": np.zeros(ACTION_DIM, dtype=np.float32)}, "task")
        client.record_call(0.5)
        self.assertEqual(client.episode_timing, {"rpc_calls": 1, "forwards": 2, "rpc_s": 0.5, "forward_s": 0.5})
        client.log_episode_timing()
        self.assertEqual(client.episode_timing, deploy_policy._empty_timing())

    def request_for(self, smolvla_steps, replan_steps):
        client = self.client(smolvla_steps, replan_steps)
        client.infer({"observation.state": np.zeros(ACTION_DIM, dtype=np.float32)}, "task")
        request = client.requests[0]
        return request["n_action_steps"], request["replan"]

    def test_default_uses_action_queue(self):
        self.assertEqual(self.request_for(10, None), (10, False))

    def test_replan_mode(self):
        self.assertEqual(self.request_for(10, 25), (25, True))


try:
    import torch

    from policy.smolvla import smolvla_worker
except ImportError:      # torch lives in the SmolVLA env, not the sim env
    torch = None
    smolvla_worker = None


@unittest.skipIf(smolvla_worker is None, "needs torch and the worker module")
class WorkerImageDtype(unittest.TestCase):
    """The camera frames the worker hands to the policy.

    Frames cross IPC as uint8 and the checkpoint's preprocessor only moves
    tensors to the device without casting, so an uncast frame reaches
    F.interpolate(mode="bilinear") and CUDA raises
    '"upsample_bilinear2d_out_frame" not implemented for Byte'. That happened on
    the first real rollout, and none of the tests above could catch it: they
    stop at the adapter and never inspect an image's dtype.
    """

    def obs(self, image):
        return {
            "observation.state": np.zeros(ACTION_DIM, dtype=np.float32),
            "observation.images.cam_high": image,
        }

    def test_uint8_frames_become_float32_zero_to_one(self):
        image = np.array([[[0, 128, 255]]], dtype=np.uint8)
        out = smolvla_worker._obs_to_tensors(self.obs(image))["observation.images.cam_high"]
        self.assertEqual(out.dtype, torch.float32)
        self.assertAlmostEqual(out.flatten()[0].item(), 0.0, places=6)
        self.assertAlmostEqual(out.flatten()[1].item(), 128 / 255, places=6)
        self.assertAlmostEqual(out.flatten()[2].item(), 1.0, places=6)

    def test_frame_shape_is_preserved(self):
        image = np.zeros((2, 8, 6, 3), dtype=np.uint8)
        out = smolvla_worker._obs_to_tensors(self.obs(image))["observation.images.cam_high"]
        self.assertEqual(tuple(out.shape), (2, 8, 6, 3))

    def test_float_frames_are_not_divided_twice(self):
        image = np.full((1, 2, 2, 3), 0.5, dtype=np.float32)
        out = smolvla_worker._obs_to_tensors(self.obs(image))["observation.images.cam_high"]
        self.assertEqual(out.dtype, torch.float32)
        self.assertAlmostEqual(out.flatten()[0].item(), 0.5, places=6)

    def test_non_image_keys_keep_their_dtype(self):
        obs = self.obs(np.zeros((1, 2, 2, 3), dtype=np.uint8))
        obs["observation.qvel"] = np.zeros(ACTION_DIM, dtype=np.float64)
        out = smolvla_worker._obs_to_tensors(obs)
        self.assertEqual(out["observation.state"].dtype, torch.float32)
        self.assertEqual(out["observation.qvel"].dtype, torch.float64)

    def test_every_camera_is_converted(self):
        obs = {f"observation.images.{name}": np.zeros((1, 2, 2, 3), dtype=np.uint8)
               for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
        out = smolvla_worker._obs_to_tensors(obs)
        self.assertEqual({t.dtype for t in out.values()}, {torch.float32})


if __name__ == "__main__":
    unittest.main()
