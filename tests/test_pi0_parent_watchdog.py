"""Test the actual stdlib-only watchdog without importing JAX or opening sockets.

The function is compiled from its source AST. These tests cover its branches;
real evaluator/server death remains a separate GPU lifecycle acceptance.
"""
import ast
import logging
import signal
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


class Exited(Exception):
    pass


class Watchdog(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "policy/smolvla_multitask/pi0_serve.py"
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "watch_parent")
        self.os = Mock()
        self.os.environ = {"PI0_EVALUATOR_PID": "123"}
        self.os.getppid.return_value = 123
        self.os.getpid.return_value = 500
        self.os.getpgrp.return_value = 500
        self.os._exit.side_effect = Exited
        self.thread = Mock()
        ns = {"os": self.os, "signal": signal, "time": Mock(), "logging": logging,
              "threading": types.SimpleNamespace(Thread=self.thread)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
        self.watch = ns["watch_parent"]

    def test_original_parent_alive_starts_daemon(self):
        self.watch()
        self.os.killpg.assert_not_called()
        self.thread.assert_called_once()
        self.assertTrue(self.thread.call_args.kwargs["daemon"])
        self.thread.return_value.start.assert_called_once()

    def test_parent_died_during_import_kills_owned_group(self):
        self.os.getppid.return_value = 1
        with self.assertRaises(Exited):
            self.watch()
        self.os.killpg.assert_called_once_with(500, signal.SIGKILL)
        self.thread.assert_not_called()

    def test_manually_launched_server_never_kills_shell_group(self):
        self.os.getppid.return_value = 1
        self.os.getpgrp.return_value = 400
        with self.assertRaises(Exited):
            self.watch()
        self.os.killpg.assert_not_called()

    def test_parent_dies_after_thread_started(self):
        self.watch()
        self.os.getppid.return_value = 1
        target = self.thread.call_args.kwargs["target"]
        with self.assertRaises(Exited):
            target()
        self.os.killpg.assert_called_once_with(500, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
