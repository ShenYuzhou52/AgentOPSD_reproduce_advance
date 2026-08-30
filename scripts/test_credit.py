"""Unit tests for the AgentOPSD credit-assignment math (CPU, no SDAR needed).

Run:  uv run pytest scripts/test_credit.py -q
"""

import numpy as np
import pytest
import torch

from agentopsd.credit import AgentOPSDConfig, reshape_advantages


def _grpo_advantages(traj_rewards, group_ids):
    """Group-normalize per-trajectory outcomes like verl's GRPO."""
    rewards = np.asarray(traj_rewards, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    adv = np.zeros_like(rewards)
    for g in np.unique(group_ids):
        sel = group_ids == g
        mu = rewards[sel].mean()
        sigma = rewards[sel].std()
        adv[sel] = (rewards[sel] - mu) / (sigma + 1e-6)
    return adv


def _make_batch(n_traj=6, n_turns=3, seed=0, group_success_frac=0.5):
    rng = np.random.RandomState(seed)
    uid = []
    traj_uid = []
    turn_step = []
    traj_rewards = []
    episode_rewards = []
    for i in range(n_traj):
        u = f"task_{i // 2}"
        t = f"traj_{i}"
        r = 10.0 if rng.rand() < group_success_frac else 0.0
        traj_rewards.append(r)
        for k in range(n_turns):
            uid.append(u)
            traj_uid.append(t)
            turn_step.append(k)
            episode_rewards.append(r)
    B = len(uid)
    T = 8
    group_ids = [f"task_{i // 2}" for i in range(n_traj)]
    traj_adv = _grpo_advantages(traj_rewards, group_ids)
    torch.manual_seed(seed)
    teacher = torch.randn(B, T)
    student = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    # GRPO broadcasts the trajectory-level advantage to every token/turn row.
    traj_adv_per_row = np.repeat(np.asarray(traj_adv), n_turns)
    advantages = torch.tensor(np.repeat(traj_adv_per_row[:, None], T, axis=1), dtype=torch.float32)
    return {
        "advantages": advantages,
        "teacher_log_probs": teacher,
        "student_log_probs": student,
        "response_mask": mask.float(),
        "uid": np.array(uid, dtype=object),
        "traj_uid": np.array(traj_uid, dtype=object),
        "turn_step": np.array(turn_step, dtype=np.int64),
        "episode_rewards": np.array(episode_rewards, dtype=np.float64),
    }


def test_lambda_zero_recovers_grpo():
    kw = _make_batch()
    out, diag = reshape_advantages(**kw, cfg=AgentOPSDConfig(lam=0.0))
    assert torch.allclose(out, kw["advantages"], atol=1e-6)
    assert 1.0 - 0.05 <= diag["agentopsd/multiplier_mean"] <= 1.0 + 0.05


def test_single_turn_recovers_grpo():
    kw = _make_batch(n_turns=1)
    out, _ = reshape_advantages(**kw, cfg=AgentOPSDConfig())
    assert torch.allclose(out, kw["advantages"], atol=1e-6)


def test_sign_preservation_and_bounds():
    kw = _make_batch(group_success_frac=0.5)
    cfg = AgentOPSDConfig(lam=0.5, b=0.2)
    out, diag = reshape_advantages(**kw, cfg=cfg)
    adv = kw["advantages"]
    nonzero = adv != 0
    assert torch.all(torch.sign(out[nonzero]) == torch.sign(adv[nonzero]))
    lo = (1.0 - cfg.lam) * adv.abs() + cfg.lam * (1.0 - cfg.b) * adv.abs()
    hi = (1.0 - cfg.lam) * adv.abs() + cfg.lam * (1.0 + cfg.b) * adv.abs()
    assert torch.all(out[nonzero].abs() >= lo[nonzero] - 1e-5)
    assert torch.all(out[nonzero].abs() <= hi[nonzero] + 1e-5)
    assert diag["agentopsd/multiplier_min"] >= 1.0 - cfg.b - 1e-6
    assert diag["agentopsd/multiplier_max"] <= 1.0 + cfg.b + 1e-6


def test_recursion_prefers_positive_evidence():
    # Two trajectories in one group (one success, one fail → A_traj0 > 0).
    # traj0: turn0 carries strong positive teacher evidence, turn1 is neutral;
    # the recursive belief update must amplify turn0 and attenuate turn1.
    rewards = [10.0, 0.0]
    uid = np.array(["task_0"] * 4, dtype=object)
    traj_uid = np.array(["t0", "t0", "t1", "t1"], dtype=object)
    turn_step = np.array([0, 1, 0, 1], dtype=np.int64)
    episode_rewards = np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float64)
    traj_adv = _grpo_advantages(rewards, ["task_0", "task_0"])
    advantages = torch.tensor(
        np.repeat(np.asarray([traj_adv[0], traj_adv[0], traj_adv[1], traj_adv[1]])[:, None], 8, axis=1),
        dtype=torch.float32,
    )
    teacher = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0],  # t0 turn0: strong positive evidence
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0],  # t0 turn1: neutral
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0],  # t1 turn0
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0],  # t1 turn1
        ]
    )
    out, _ = reshape_advantages(
        advantages=advantages,
        teacher_log_probs=teacher,
        student_log_probs=torch.zeros_like(teacher),
        response_mask=torch.ones(4, 8),
        uid=uid,
        traj_uid=traj_uid,
        turn_step=turn_step,
        episode_rewards=episode_rewards,
        cfg=AgentOPSDConfig(),
    )
    turn0 = out[0, 0].item()
    turn1 = out[1, 0].item()
    assert turn0 > turn1
    assert traj_adv[0] > 0


def test_all_fail_group_has_zero_advantage():
    kw = _make_batch(n_traj=4, n_turns=2, seed=7, group_success_frac=0.0)
    out, diag = reshape_advantages(**kw, cfg=AgentOPSDConfig())
    assert torch.all(out == 0)
    assert diag["agentopsd/adv_std_after"] == 0.0


def test_two_turn_normalization_uses_population_std():
    # With B0=.5, gamma=0, and evidence [logit(.6), logit(.9)], the two
    # revisions are [.1, .3]. Population normalization gives z=[-1, +1] and
    # therefore w=[.8, 1.2]. The sample std would incorrectly weaken this.
    uid = np.array(["task", "task", "task", "task"], dtype=object)
    traj_uid = np.array(["positive", "positive", "negative", "negative"], dtype=object)
    turn_step = np.array([0, 1, 0, 1], dtype=np.int64)
    rewards = np.array([10.0, 10.0, 0.0, 0.0])
    advantages = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]])
    teacher = torch.tensor([[np.log(1.5)], [np.log(9.0)], [0.0], [0.0]])

    out, _ = reshape_advantages(
        advantages=advantages,
        teacher_log_probs=teacher,
        student_log_probs=torch.zeros_like(teacher),
        response_mask=torch.ones_like(teacher),
        uid=uid,
        traj_uid=traj_uid,
        turn_step=turn_step,
        episode_rewards=rewards,
        cfg=AgentOPSDConfig(lam=1.0, b=0.2, gamma=0.0),
    )

    assert out[0, 0].item() == pytest.approx(0.8, abs=5e-4)
    assert out[1, 0].item() == pytest.approx(1.2, abs=5e-4)


def test_config_from_dict():
    cfg = AgentOPSDConfig.from_dict({"lam": 0.25, "gamma": 0.9, "nope": 1})
    assert cfg.lam == 0.25
    assert cfg.gamma == 0.9
    assert cfg.b == 0.2
    assert AgentOPSDConfig.from_dict({"enable": False}).enabled is False
