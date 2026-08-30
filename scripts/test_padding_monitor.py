"""Regression tests for zero-loss multi-turn padding and collapse monitoring."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "SDAR"))

from agent_system.multi_turn_rollout.utils import adjust_batch
from agentopsd.trainer.monitor import AgentOPSDMonitor
from verl import DataProto


def _config():
    return SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=4),
            ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=4),
            actor=SimpleNamespace(
                use_kl_loss=False,
                ppo_micro_batch_size_per_gpu=4,
            ),
        ),
        algorithm=SimpleNamespace(use_kl_in_reward=False),
    )


def _batch():
    n, length = 3, 5
    tensors = {
        "input_ids": torch.arange(n * length).reshape(n, length),
        "attention_mask": torch.ones(n, length, dtype=torch.long),
        "position_ids": torch.arange(length).repeat(n, 1),
        "responses": torch.ones(n, 2, dtype=torch.long),
        "loss_mask": torch.ones(n, length, dtype=torch.long),
        "token_level_scores": torch.full((n, length), 3.0),
        "token_level_rewards": torch.full((n, length), 4.0),
        "advantages": torch.full((n, length), 5.0),
        "returns": torch.full((n, length), 6.0),
    }
    non_tensors = {
        "uid": np.array(["g0", "g0", "g1"], dtype=object),
        "traj_uid": np.array(["t0", "t1", "t2"], dtype=object),
        "turn_step": np.array([0, 1, 0], dtype=np.int64),
        "episode_rewards": np.array([10.0, 0.0, 10.0], dtype=np.float32),
        "episode_lengths": np.array([2.0, 2.0, 1.0], dtype=np.float32),
        "rewards": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        "active_masks": np.array([True, True, True]),
        "is_action_valid": np.array([False, True, True]),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_padding_is_not_a_real_turn_and_has_zero_training_signal():
    out = adjust_batch(_config(), _batch())

    assert len(out) == 4
    assert out.non_tensor_batch["is_padding"].tolist() == [False, False, False, True]
    assert out.non_tensor_batch["uid"][3].startswith("__padding_group_")
    assert out.non_tensor_batch["traj_uid"][3].startswith("__padding_traj_")
    assert out.non_tensor_batch["uid"][3] not in {"g0", "g1"}
    assert out.non_tensor_batch["traj_uid"][3] not in {"t0", "t1", "t2"}
    assert torch.equal(out.batch["attention_mask"][3, -2:], torch.zeros(2, dtype=torch.long))
    for key in ("loss_mask", "token_level_scores", "token_level_rewards", "advantages", "returns"):
        assert torch.count_nonzero(out.batch[key][3]) == 0
    assert out.non_tensor_batch["is_action_valid"][3]


def test_delete_padding_is_rejected():
    with pytest.raises(ValueError, match="must not delete"):
        adjust_batch(_config(), _batch(), mode="delete")


def test_collapse_alert_requires_three_steps_and_tracks_late_training():
    monitor = AgentOPSDMonitor(total_steps=10)
    diag = {
        "agentopsd/belief_saturation_ratio": 1.0,
        "agentopsd/belief_revision_abs_mean": 0.0,
        "agentopsd/credit_abs_mean": 0.0,
        "agentopsd/adv_nonfinite_ratio": 0.0,
        "agentopsd/group_success_mixed_ratio": 1.0,
        "agentopsd/reshape_applied": 1.0,
    }
    records = [monitor.record(diag, step, {}) for step in range(1, 11)]

    assert records[0]["agentopsd/collapse/signal"] == 1.0
    assert records[1]["agentopsd/collapse/alert"] == 0.0
    assert records[2]["agentopsd/collapse/alert"] == 1.0
    assert records[7]["agentopsd/is_late_training"] == 1.0
    assert records[-1]["agentopsd/collapse/late_alert"] == 1.0
