"""Regression checks for the text-only Qwen3.5 agent-loop contract.

中文说明：守护"Qwen3.5 文本-only"约定——位置编码必须是一维文本 RoPE，
多模态输入必须被拒绝。这是当初 actor old-logprob 阶段位置张量形状炸掉后
加的回归。
"""

from __future__ import annotations

import unittest

import torch

from integrations.simpletir_qwen35.agent_loop_manager import text_only_multi_modal_inputs, text_only_position_ids


class TestTextOnlyPositions(unittest.TestCase):
    def test_qwen_processor_is_not_allowed_to_inject_visual_mrope(self):
        position_ids = text_only_position_ids(
            torch.tensor([[1, 1, 1]]),
            {},
        )
        self.assertEqual(position_ids.shape, (1, 3))
        self.assertTrue(torch.equal(position_ids, torch.tensor([[0, 1, 2]])))

    def test_multimodal_inputs_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "text-only"):
            text_only_position_ids(
                torch.tensor([[1]]),
                {"image_grid_thw": torch.tensor([[1, 1, 1]])},
            )

    def test_worker_discards_processor_auxiliary_fields_for_text_only_rows(self):
        self.assertEqual(text_only_multi_modal_inputs(None), {})

    def test_worker_rejects_real_media(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected multimodal"):
            text_only_multi_modal_inputs({"images": [object()]})


if __name__ == "__main__":
    unittest.main()
