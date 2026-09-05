"""Host-safety checks using fixed harmless programs, never model output."""

from __future__ import annotations

import os
import shutil
import unittest

from integrations.simpletir_qwen35.simpletir_sandbox import run_python


@unittest.skipUnless(os.name == "posix" and shutil.which("bwrap") and shutil.which("systemd-run"), "Linux sandbox unavailable")
class TestSimpleTIRSandbox(unittest.TestCase):
    def test_calculation(self):
        result = run_python("print(6 * 7)")
        self.assertTrue(result.ok, result.stderr)
        self.assertEqual(result.stdout.strip(), "42")

    def test_host_data_disk_is_invisible(self):
        result = run_python("import os; print(os.path.exists('/data2'))")
        self.assertTrue(result.ok, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_network_namespace_has_no_external_interface(self):
        code = "import socket\ns=socket.socket(); s.settimeout(.2)\ntry:\n s.connect(('1.1.1.1',53)); print('unexpected')\nexcept OSError:\n print('blocked')"
        result = run_python(code)
        self.assertTrue(result.ok, result.stderr)
        self.assertEqual(result.stdout.strip(), "blocked")

    def test_infinite_loop_is_stopped(self):
        result = run_python("while True: pass", timeout_seconds=1.0)
        self.assertTrue(result.timed_out, (result.returncode, result.stderr, result.elapsed_seconds))
        self.assertLess(result.elapsed_seconds, 4.5)

    def test_reward_output_can_exceed_student_observation_budget(self):
        code = "print('x' * 600)\nprint('\\\\boxed{42}')"
        result = run_python(code, output_limit=2048)
        self.assertTrue(result.ok, result.stderr)
        self.assertIn(r"\boxed{42}", result.stdout)


if __name__ == "__main__":
    unittest.main()
