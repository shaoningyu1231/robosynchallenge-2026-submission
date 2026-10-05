"""policy/smolvla_multitask/pi0_backend.py without a GPU, torch or the simulator.

    python -m unittest tests.test_pi0_backend -v

The noise derivation reproduces digests recorded by the drawer independent test (research repository,
experiments/B3_DRAWER_TEST_RESULT.txt); checkpoint and tokenizer checks refuse what they must; and the server's
lifecycle -- started, stopped with its whole process group, or failing to start without leaving anything behind -- is
exercised with a stand-in for policy/pi05's Python that listens on the given port and starts a child of its own.
"""

import hashlib
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# the package's __init__ loads deploy_policy (torch); register the package without running it
if "policy.smolvla_multitask" not in sys.modules:
    import types
    for name, path in (("policy", REPO_ROOT / "policy"), ("policy.smolvla_multitask", REPO_ROOT / "policy" / "smolvla_multitask")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules.setdefault(name, pkg)
from policy.smolvla_multitask import pi0_backend as B  # noqa: E402

FAKE_SERVER = textwrap.dedent("""\
    #!/usr/bin/env python3
    import os, socket, subprocess, sys, time
    if os.environ.get("FAKE_MODE") == "exit":
        print("the model failed to load", flush=True)
        sys.exit(3)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    child = subprocess.Popen(["sleep", "300"])
    open(os.environ["FAKE_PIDS"], "w").write(f"{os.getpid()} {child.pid} {os.environ['CUDA_VISIBLE_DEVICES']} "
                                             f"{os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen()
    while True:
        time.sleep(1)
""")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie still answers kill(0); it is gone for our purpose
        return Path(f"/proc/{pid}/status").read_text().find("State:\tZ") < 0
    except OSError:
        return True


class Noise(unittest.TestCase):
    def test_reproduces_the_drawer_test_digests(self):
        # pi0 block 1, scene 0 of the frozen test list: seed 653853283, noise base 0, its first three calls
        got = [B.noise_digest(B.call_noise(0, 653853283, k, (50, 32))) for k in range(3)]
        self.assertEqual(got, ["cbb5d64b900df9ae", "cbb1acb7ec0d1e1a", "c12490eb2013ccde"])

    def test_noise_base_must_be_a_non_negative_integer(self):
        self.assertEqual(B._noise_base("0"), 0)
        for bad in (None, -1, "x", True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(B.Pi0Error):
                B._noise_base(bad)


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, normalizers=1):
        c = self.root / "ckpt"
        (c / "params").mkdir(parents=True)
        for name in ("_METADATA", "manifest.ocdbt"):
            (c / "params" / name).write_text("x")
        for i in range(normalizers):
            d = c / "assets" / "RoboSynChallenge" / f"task{i}"
            d.mkdir(parents=True)
            (d / "norm_stats.json").write_text("{}")
        return c

    def test_a_servable_checkpoint(self):
        c = self.checkpoint()
        self.assertEqual(B.check_checkpoint(c), c)

    def test_missing_params_or_not_exactly_one_normalizer(self):
        c = self.checkpoint(normalizers=2)
        with self.assertRaises(B.Pi0Error):
            B.check_checkpoint(c)
        (c / "params" / "_METADATA").unlink()
        with self.assertRaises(B.Pi0Error):
            B.check_checkpoint(self.root / "ckpt")

    def test_tokenizer_comes_from_the_snapshot_only_when_its_hash_is_right(self):
        c = self.checkpoint()
        home = self.root / "home"
        self.assertTrue(B.prepare_data_home(c, home).startswith("absent"))
        (c / "big_vision").mkdir()
        (c / B.TOKENIZER).write_bytes(b"not the tokenizer")
        with self.assertRaises(B.Pi0Error):
            B.prepare_data_home(c, home)
        saved = B.TOKENIZER_SHA256
        B.TOKENIZER_SHA256 = hashlib.sha256(b"not the tokenizer").hexdigest()
        try:
            self.assertEqual(B.prepare_data_home(c, home), "from the checkpoint snapshot")
            self.assertEqual((home / B.TOKENIZER).read_bytes(), b"not the tokenizer")
            self.assertEqual(B.prepare_data_home(c, home), "present")
        finally:
            B.TOKENIZER_SHA256 = saved


class ServerLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.python = root / "python"
        self.python.write_text(FAKE_SERVER)
        self.python.chmod(self.python.stat().st_mode | stat.S_IXUSR)
        self.pids = root / "pids"
        os.environ["FAKE_PIDS"] = str(self.pids)
        self.addCleanup(os.environ.pop, "FAKE_PIDS", None)
        self.addCleanup(os.environ.pop, "FAKE_MODE", None)
        self.server = B.Pi0Server(str(self.python), "pi0_x", root, 3, root / "home", root / "server.log")
        self.addCleanup(self.server.stop)

    def test_started_then_stopped_with_its_whole_process_group(self):
        self.server.start()
        pid, child, gpu, prealloc = self.pids.read_text().split()
        self.assertEqual((gpu, prealloc), ("3", "false"))
        self.assertTrue(alive(int(pid)) and alive(int(child)))
        self.server.stop()
        time.sleep(0.5)
        self.assertFalse(alive(int(pid)))
        self.assertFalse(alive(int(child)), "the server's own child outlived the stop")
        self.assertIsNone(self.server.proc)

    def test_a_server_that_exits_is_an_error_with_its_log(self):
        os.environ["FAKE_MODE"] = "exit"
        with self.assertRaises(B.Pi0Error) as cm:
            self.server.start()
        self.assertIn("the model failed to load", str(cm.exception))
        self.assertIsNone(self.server.proc)

    def test_a_missing_environment_is_named(self):
        self.server.python = str(Path(self.tmp.name) / "nope" / "python")
        with self.assertRaises(B.Pi0Error) as cm:
            self.server.start()
        self.assertIn("setup_pi0.sh", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
