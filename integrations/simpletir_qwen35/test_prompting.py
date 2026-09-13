"""Regression tests for the reference SimpleTIR prompt adapter.

中文说明：前缀只拼一次（幂等）、不改动数据集原行、格式错误立即报错。
前缀文本与上游逐字一致，改一个字都会影响可比性。
"""

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
