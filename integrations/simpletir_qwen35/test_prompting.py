"""Regression tests for the reference SimpleTIR prompt adapter."""

from __future__ import annotations

import unittest

from integrations.simpletir_qwen35.prompting import SIMPLETIR_PROMPT_PREFIX, with_simpletir_prompt


class TestSimpleTIRPrompt(unittest.TestCase):
    def test_prepends_reference_tool_contract_without_mutating_source(self):
        raw = [{"role": "user", "content": "What is 2 + 2?"}]
        result = with_simpletir_prompt(raw)
        self.assertEqual(raw[0]["content"], "What is 2 + 2?")
        self.assertEqual(result[0]["content"], SIMPLETIR_PROMPT_PREFIX + "What is 2 + 2?")
        self.assertIn("final_answer()", result[0]["content"])

    def test_is_idempotent_for_already_prepared_rows(self):
        raw = [{"role": "user", "content": SIMPLETIR_PROMPT_PREFIX + "Question"}]
        self.assertEqual(with_simpletir_prompt(raw), raw)


if __name__ == "__main__":
    unittest.main()
