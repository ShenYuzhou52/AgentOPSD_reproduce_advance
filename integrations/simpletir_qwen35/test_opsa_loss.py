"""OPSA 优势纯函数与损失反向传播的单测（CPU）。

覆盖论文式 (4) 的关键性质：
- 选点 = 每行采样 log-prob 最低的 20%（至少 1 个），padding 不参与；
- 熵自适应端点：选点内熵最高 → A=-1，最低 → A=-0.5，常熵/δ=0 → 固定 -0.75；
- fail-loud：非有限熵、非法 frac、形状不一致、缺 entropy 字段都必须报错；
- 反向传播：梯度只落在选点，padding 行/未选点严格零梯度。

注意 ``no_padding_2_padding`` 的契约：传入的 flat 张量必须覆盖 prompt+response
的总 token 数（每行 = 1 个 prompt 位置 + 整段 response），函数内部按 jagged
offsets 切出 response 部分——因此 backward 测试的 flat 长度是 18 而不是 15。
"""
from types import SimpleNamespace
from unittest import TestCase

import torch
from tensordict import TensorDict

from integrations.simpletir_qwen35.opsa_loss import opsa_loss, opsa_token_advantage


class TestOPSATokenAdvantage(TestCase):
    def test_selects_lowest_logprob_fraction(self):
        old_logp = torch.tensor([[-1.0, -9.0, -2.0, -8.0, -3.0, -4.0, -5.0, -6.0, -7.0, 0.0]])
        entropy = torch.zeros_like(old_logp)
        mask = torch.ones_like(old_logp, dtype=torch.bool)
        selected, advantage = opsa_token_advantage(old_logp, entropy, mask)
        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(selected[0].nonzero().flatten().tolist(), [1, 3])
        # 常熵 → r=0 → 固定优势 -0.75。
        self.assertTrue(torch.equal(advantage[selected], torch.full((2,), -0.75)))

    def test_min_one_token_for_short_rows(self):
        old_logp = torch.tensor([[-1.0, -2.0, -3.0]])
        entropy = torch.zeros_like(old_logp)
        mask = torch.ones_like(old_logp, dtype=torch.bool)
        selected, _ = opsa_token_advantage(old_logp, entropy, mask, lowest_frac=0.2)
        self.assertEqual(int(selected.sum()), 1)
        self.assertEqual(selected[0].nonzero().flatten().tolist(), [2])

    def test_entropy_adaptive_endpoints(self):
        # 10 选 2：位置 1(-9.0) 与 3(-8.0)。熵分别为 1.0 与 0.0 → r=+1/-1。
        old_logp = torch.tensor([[-1.0, -9.0, -2.0, -8.0, -3.0, -4.0, -5.0, -6.0, -7.0, 0.0]])
        entropy = torch.zeros_like(old_logp)
        entropy[0, 1] = 1.0
        entropy[0, 3] = 0.0
        mask = torch.ones_like(old_logp, dtype=torch.bool)
        _, advantage = opsa_token_advantage(old_logp, entropy, mask)
        self.assertEqual(advantage[0, 1].item(), -1.0)   # 选点内最高熵 → 最强负优势
        self.assertEqual(advantage[0, 3].item(), -0.5)   # 选点内最低熵 → 最弱负优势

    def test_delta_zero_gives_fixed_advantage(self):
        old_logp = torch.tensor([[-1.0, -9.0, -2.0, -8.0, -3.0, -4.0, -5.0, -6.0, -7.0, 0.0]])
        entropy = torch.randn_like(old_logp)
        mask = torch.ones_like(old_logp, dtype=torch.bool)
        _, advantage = opsa_token_advantage(old_logp, entropy, mask, delta=0.0)
        self.assertTrue(torch.equal(advantage[advantage != 0.0], torch.full((2,), -0.75)))

    def test_padding_tokens_never_selected(self):
        old_logp = torch.tensor([[-1.0, -9.0, -99.0, -99.0, -3.0]])
        entropy = torch.zeros_like(old_logp)
        mask = torch.tensor([[True, True, False, False, True]])
        selected, advantage = opsa_token_advantage(old_logp, entropy, mask)
        self.assertEqual(int(selected.sum()), 1)  # 3 个真实 token → k=1
        self.assertEqual(selected[0].nonzero().flatten().tolist(), [1])
        self.assertEqual(advantage[~selected].abs().sum().item(), 0.0)

    def test_all_padding_row_is_safe(self):
        old_logp = torch.zeros(1, 3)
        entropy = torch.zeros_like(old_logp)
        mask = torch.zeros_like(old_logp, dtype=torch.bool)
        selected, advantage = opsa_token_advantage(old_logp, entropy, mask)
        self.assertEqual(int(selected.sum()), 0)
        self.assertFalse(torch.isnan(advantage).any())

    def test_nonfinite_entropy_raises(self):
        old_logp = torch.zeros(1, 4)
        entropy = torch.tensor([[0.0, 1.0, float("nan"), 0.5]])
        mask = torch.ones_like(old_logp, dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            opsa_token_advantage(old_logp, entropy, mask)

    def test_invalid_fraction_and_shapes_raise(self):
        old_logp = torch.zeros(1, 4)
        mask = torch.ones_like(old_logp, dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            opsa_token_advantage(old_logp, torch.zeros_like(old_logp), mask, lowest_frac=0.0)
        with self.assertRaises(RuntimeError):
            opsa_token_advantage(old_logp, torch.zeros(1, 5), mask)


def _nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(r) for r in rows], layout=torch.jagged)


def _batch_data(mask_rows, logp_rows):
    """三行批次：两行真实响应 + 一行零损失 padding 行（与 OPSD 测试同款布局）。

    ``old_log_probs`` 必须携带真实采样值（verl 批次如此）：OPSA 用它选最低
    20% token，填零会让选点在并列值里任意挑，梯度断言失去确定性。
    """
    responses = [[0] * len(row) for row in mask_rows]
    data = TensorDict(
        {
            "prompts": _nested([[10]] * len(mask_rows)),
            "responses": _nested(responses),
            "response_mask": _nested([list(map(float, row)) for row in mask_rows]),
            # 真实 verl 批次携带的 rollout 采样 log-prob（opsa_loss 只读取该字段）。
            "old_log_probs": _nested([list(row) for row in logp_rows]),
        },
        batch_size=len(mask_rows),
    )
    data.set_non_tensor("dp_size", 1)
    data.set_non_tensor("batch_num_tokens", None)
    data.set_non_tensor("global_batch_size", None)
    return data


def _flat(rows):
    """prompt(1) + response 段拼接：no_padding_2_padding 期望的 total_nnz 布局。"""
    return [v for row in rows for v in [0.0] + list(row)]


def _shifted_flat(resp_rows):
    """verl log_probs 布局：token 的 logp 写在前一位置（"left-shift by one"）。

    no_padding_2_padding 对每行取 ``flat[row_start : row_start+resp_len]``，行块
    之后跳过 prompt_len 个位置——因此每行先放 resp 值、再放 prompt_len 个占位。
    本测试全部 prompt_len=1，故每行 = resp 值 + 1 个占位。
    """
    out = []
    for row in resp_rows:
        out.extend(row)
        out.append(0.0)
    return out


class TestOPSALoss(TestCase):
    def _config(self):
        return SimpleNamespace(
            global_batch_info={},
            loss_scale_factor=None,
            loss_agg_mode="token-mean",
            entropy_coeff=0.0,
            use_kl_loss=True,
            kl_loss_type="low_var_kl",
            kl_loss_coef=0.01,
        )

    def test_backward_gradient_only_on_selected_tokens(self):
        mask_rows = [[1] * 10, [1] * 4, [0]]
        logp_resp = [[-1.0, -9.0, -2.0, -8.0, -3.0, -4.0, -5.0, -6.0, -7.0, -0.5],
                     [-1.0, -9.0, -2.0, -8.0],
                     [0.0]]
        data = _batch_data(mask_rows, logp_resp)
        entropy_resp = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.0]]
        flat_logp = torch.tensor(_shifted_flat(logp_resp), requires_grad=True)
        flat_entropy = torch.tensor(_shifted_flat(entropy_resp))
        config = self._config()
        loss, metrics = opsa_loss(config, {"log_probs": flat_logp, "entropy": flat_entropy}, data)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("opsa/pg_loss", metrics)
        loss.backward()
        grad = flat_logp.grad
        self.assertTrue(torch.isfinite(grad).all())
        # 选点（flat 索引，行块内偏移 = 响应位置）：row0 响应 1、3 → 1、3；row1 响应 1 → 12。
        for idx in (1, 3, 12):
            self.assertGreater(grad[idx].abs().item(), 0.0, f"selected token {idx} must receive gradient")
        # 未选点、行块尾占位与 padding 行梯度严格为零。
        for idx in (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17):
            self.assertEqual(grad[idx].abs().item(), 0.0, f"token {idx} must have zero gradient")

    def test_missing_entropy_fails_loud(self):
        mask_rows = [[1] * 10, [1] * 4, [0]]
        logp_rows = [[0.0] * 10, [0.0] * 4, [0.0]]
        data = _batch_data(mask_rows, logp_rows)
        flat_logp = torch.tensor(_flat(logp_rows))
        with self.assertRaises(RuntimeError):
            opsa_loss(self._config(), {"log_probs": flat_logp}, data)

    def test_all_padding_micro_batch_returns_zero_loss(self):
        # ppo_micro_batch_size_per_gpu=1 时可能抽到只含平衡补行的 micro-batch：
        # 响应 mask 全零 → 必须静默返回零损失（零损失 padding 不变量），不能抛错。
        mask_rows = [[0] * 4, [0] * 4]
        logp_rows = [[0.0] * 4, [0.0] * 4]
        data = _batch_data(mask_rows, logp_rows)
        flat_logp = torch.tensor(_flat(logp_rows), requires_grad=True)
        flat_entropy = torch.tensor(_flat(logp_rows))
        config = self._config()
        loss, metrics = opsa_loss(config, {"log_probs": flat_logp, "entropy": flat_entropy}, data)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["opsa/padding_micro_batch"], 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(flat_logp.grad).all())
