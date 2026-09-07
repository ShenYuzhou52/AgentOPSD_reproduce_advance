"""CPU-only regression tests for the SimpleTIR trajectory boundary."""

from __future__ import annotations

import contextlib
import io
import threading
import unittest

from integrations.simpletir_qwen35.trajectory import (
    extract_last_boxed,
    extract_python_fence,
    format_observation,
    is_only_final_answer,
    parse_turn,
    requires_sandbox,
    score_simpletir_math,
    with_final_answer_helper,
)


class TestSimpleTIRTrajectory(unittest.TestCase):
    def test_first_python_fence_matches_simpletir(self):
        text = "reasoning\n```python\nprint(1)\n```\n```python\nprint(2)\n```"
        self.assertEqual(extract_python_fence(text), "print(1)")

    def test_nested_last_boxed_is_preserved(self):
        text = r"first \boxed{1}; final \boxed{\frac{3}{4}}"
        self.assertEqual(extract_last_boxed(text), r"\boxed{\frac{3}{4}}")

    def test_void_turn_is_not_a_synthetic_action(self):
        self.assertTrue(parse_turn("I am unsure.").is_void)
        self.assertFalse(parse_turn(r"\boxed{7}").is_void)

    def test_boxed_answer_with_code_still_requires_execution(self):
        parsed = parse_turn("```python\nfinal_answer(7)\n```\n\\boxed{7}")
        self.assertTrue(parsed.has_boxed_answer)
        self.assertTrue(requires_sandbox(parsed))

    def test_helper_prints_a_boxed_answer(self):
        namespace: dict[str, object] = {}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(with_final_answer_helper("final_answer(42)"), namespace, namespace)
        self.assertEqual(stdout.getvalue().strip(), r"\boxed{42}")

    def test_substantive_tool_check_matches_simpletir(self):
        self.assertTrue(is_only_final_answer("final_answer(7)"))
        self.assertFalse(is_only_final_answer("x = 7\nfinal_answer(x)"))

    def test_correct_boxed_stdout_receives_reward(self):
        reward = score_simpletir_math(r"\boxed{118}", "118", substantive_tool_use=True)
        self.assertEqual(reward["answer_accuracy"], 1.0)
        self.assertEqual(reward["score"], 1.0)

    def test_python_style_expression_still_scores(self):
        # final_answer(str(sympy_expr)) prints Python syntax; the latex-first
        # parser keeps it as a bare string and verify() alone returned False.
        reward = score_simpletir_math(r"\boxed{5*x**2/2}", r"\frac{5x^2}{2}", substantive_tool_use=True)
        self.assertEqual(reward["answer_accuracy"], 1.0)

    def test_python_style_fallback_rejects_wrong_expression(self):
        reward = score_simpletir_math(r"\boxed{5*x**2/3}", r"\frac{5x^2}{2}", substantive_tool_use=True)
        self.assertEqual(reward["answer_accuracy"], 0.0)

    def test_scoring_works_in_worker_threads(self):
        # Verl runs agent loops on worker threads; math_verify's SIGALRM
        # timeout raises ValueError there and the guard silently zeroed every
        # reward, including correct answers.
        result = {}

        def score() -> None:
            result["reward"] = score_simpletir_math("a + b = 8\n\\boxed{8}\n", "8", substantive_tool_use=True)

        thread = threading.Thread(target=score)
        thread.start()
        thread.join(timeout=30)
        self.assertEqual(result["reward"]["answer_accuracy"], 1.0)

    def test_observation_is_bounded_and_has_no_hidden_context(self):
        obs = format_observation(stdout="abc" * 500, stderr="", timed_out=False, limit=32)
        self.assertTrue(obs.startswith("\nCode execution result: "))
        self.assertLessEqual(len(obs), len("\nCode execution result: ") + 32 + 1)


if __name__ == "__main__":
    unittest.main()
