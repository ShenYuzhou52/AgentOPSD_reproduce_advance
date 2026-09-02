"""Regression tests for masked GRPO sequence rewards."""

from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "SDAR"))

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_masked_sentinel_cannot_change_grpo_advantage():
    rewards = torch.tensor([[10.0, -100000.0], [0.0, -100000.0]])
    response_mask = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.asarray(["group", "group"], dtype=object),
        traj_index=np.asarray(["traj0", "traj1"], dtype=object),
        max_abs_advantage=10.0,
    )

    assert torch.allclose(advantages, returns)
    assert torch.count_nonzero(advantages[:, 1]) == 0
    assert float(advantages[:, 0].abs().max()) < 2.0
