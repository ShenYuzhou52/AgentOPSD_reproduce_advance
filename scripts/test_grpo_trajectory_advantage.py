import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_cross_step_false_uses_episode_outcome_and_keeps_action_penalty_finite():
    # Two trajectories, each flattened into two turns. The second turn of the
    # failed trajectory has an invalid-action penalty. Previously it was
    # normalized against a zero-variance outcome group as -0.1 / 1e-6.
    rewards = torch.tensor([[0.0], [-0.1], [10.0], [10.0]])
    mask = torch.ones_like(rewards)
    outcomes = torch.tensor([0.0, 0.0, 10.0, 10.0])
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.array(["group"] * 4, dtype=object),
        traj_index=np.array(["fail", "fail", "success", "success"], dtype=object),
        trajectory_scores=outcomes,
        compute_mean_std_cross_steps=False,
        max_abs_advantage=100,
    )

    assert torch.isfinite(advantages).all()
    assert advantages.abs().max().item() < 100
    assert torch.allclose(advantages[1] - advantages[0], torch.tensor([-0.1]))
