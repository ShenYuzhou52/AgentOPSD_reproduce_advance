"""Regression tests for the pinned Verl prompt-logprob wire format."""
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from integrations.simpletir_qwen35.simpletir_agent_loop import SimpleTIRPythonAgentLoop


class TestTeacherAlignment(IsolatedAsyncioTestCase):
    async def score(self, ids):
        loop = object.__new__(SimpleTIRPythonAgentLoop)
        loop.ct_build_initial_tokens = AsyncMock(return_value=[10, 20])
        loop.server_manager = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
            extra_fields={"prompt_ids": ids, "prompt_logprobs": [[-9.], [-0.3], [-0.4], [0.]]}
        )))
        return await loop._same_policy_teacher_logprobs(
            student_messages=[{"role": "user", "content": "test"}],
            response_ids=[30, 40], ground_truth="test-only", priority=0,
        )

    async def test_singleton_axis_and_shift_exclude_dummy(self):
        self.assertEqual(await self.score([[20], [30], [40], [0]]), [-0.3, -0.4])

    async def test_wrong_token_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "token ids do not match"):
            await self.score([[20], [31], [40], [0]])

    async def test_topk_ids_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one token"):
            await self.score([[20], [30, 31], [40], [0]])
