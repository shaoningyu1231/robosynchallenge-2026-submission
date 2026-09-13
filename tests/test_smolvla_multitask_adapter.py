"""Checks that policy/smolvla_multitask starts its worker on the pinned VLM, offline.

Needs torch and numpy (the adapter imports them), not the simulator or LeRobot:
the worker's Python is a fake that records how it was started.

    python -m unittest tests.test_smolvla_multitask_adapter -v
"""

import json
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from policy.smolvla_multitask import deploy_policy  # noqa: E402
from policy.smolvla_multitask.checkpoints import CheckpointError  # noqa: E402

REV = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


class WorkerStart(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.checkpoint = self.tmp / "ckpt"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps({"input_features": {}}))
        self.record = self.tmp / "record.json"
        fake = self.tmp / "fake_python"
        # Records argv and the offline variables, then idles like a worker waiting on stdin.
        fake.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > {self.record}.args\n"
            f"echo \"$HF_HUB_OFFLINE $TRANSFORMERS_OFFLINE\" > {self.record}.env\n"
            "exec cat > /dev/null\n"
        )
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.python = str(fake)

    def start(self, **extra):
        usr_args = {
            "checkpoint_dir": str(self.checkpoint),
            "smolvla_python": self.python,
            "smolvla_rescale_gripper": True,
            **extra,
        }
        client = deploy_policy.SmolVLAWorkerClient(usr_args)
        self.addCleanup(client.close)
        for _ in range(100):
            if Path(f"{self.record}.env").exists():
                break
            time.sleep(0.05)
        args = Path(f"{self.record}.args").read_text().split("\n")
        env = Path(f"{self.record}.env").read_text().split()
        return args, env

    def test_worker_gets_the_pinned_snapshot_and_runs_offline(self):
        snapshot = f"/hf/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/{REV}"
        args, env = self.start(
            smolvla_vlm_path=snapshot, smolvla_vlm_repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        self.assertEqual(args[args.index("--vlm-path") + 1], snapshot)
        self.assertEqual(args[args.index("--vlm-repo-id") + 1], "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self.assertEqual(env, ["1", "1"])

    def test_get_model_refuses_to_start_without_a_vlm_pin(self):
        config = {"task_name": "click_bell", "checkpoint_dir": str(self.checkpoint), "smolvla_python": self.python}
        with self.assertRaisesRegex(CheckpointError, "must be pinned"):
            deploy_policy.get_model(config)


if __name__ == "__main__":
    unittest.main()
