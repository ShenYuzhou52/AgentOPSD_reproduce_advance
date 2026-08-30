"""Plug AgentOPSD credit reshaping into SDAR's verl trainer.

SDAR's ``SkillSDRayTrainer.fit`` computes the group-relative (GRPO) advantage by
calling the module-level function ``verl.trainer.ppo.skillsd_ray_trainer.compute_advantage``
and later passes the batch to the actor's policy update.  We wrap that call:
after the GRPO advantages are produced, AgentOPSD's recursive turn-level credit
reshapes ``data.batch["advantages"]`` in place, so the policy update sees the
turn-level advantages Ã_k without touching any other trainer code.

The teacher forward pass (skill-conditioned log-probs) is still performed by
``SkillSDRayTrainer``; AgentOPSD only re-weights advantages and does not add a
distillation loss (matching the paper: "No separate distillation loss is
introduced; the detached self-teacher signal acts only through Ã").
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import torch

from agentopsd.credit import AgentOPSDConfig, reshape_advantages

_ORIGINAL_COMPUTE_ADVANTAGE = None
_RUNTIME: Dict[str, Any] = {"cfg": AgentOPSDConfig()}
_INSTALLED = False


def install(cfg_dict: Optional[dict] = None) -> None:
    """Wrap SDAR's GRPO advantage computation with AgentOPSD reshaping.

    Must be called on the Ray driver before the trainer's ``fit()`` starts
    (i.e. inside ``main_agentopsd.run``).  ``cfg_dict`` is the composed
    ``+algorithm.agentopsd.*`` Hydra block, e.g.
    ``{"lam": 0.5, "b": 0.2, "gamma": 0.95, "enabled": True}``.
    """
    global _ORIGINAL_COMPUTE_ADVANTAGE, _RUNTIME, _INSTALLED
    import verl.trainer.ppo.skillsd_ray_trainer as skillsd_module

    if not _INSTALLED:
        _ORIGINAL_COMPUTE_ADVANTAGE = skillsd_module.compute_advantage
    _RUNTIME = {"cfg": AgentOPSDConfig.from_dict(cfg_dict)}
    skillsd_module.compute_advantage = _wrapped_compute_advantage
    _INSTALLED = True
    print(f"[agentopsd] advantage-reshaping hook installed: {_RUNTIME['cfg']}", flush=True)


def uninstall() -> None:
    """Restore SDAR's original ``compute_advantage`` (used by tests only)."""
    global _INSTALLED
    if _INSTALLED:
        import verl.trainer.ppo.skillsd_ray_trainer as skillsd_module

        skillsd_module.compute_advantage = _ORIGINAL_COMPUTE_ADVANTAGE
        _INSTALLED = False


def _wrapped_compute_advantage(data, *args, **kwargs):
    data = _ORIGINAL_COMPUTE_ADVANTAGE(data, *args, **kwargs)
    cfg = _RUNTIME["cfg"]
    if not cfg.enabled:
        return data
    if "teacher_log_probs" not in data.batch:
        raise RuntimeError("AgentOPSD requires teacher_log_probs; refusing to silently run GRPO")

    batch = data.batch
    non_tensor = data.non_tensor_batch
    required = ("uid", "traj_uid", "turn_step", "episode_rewards")
    missing = [key for key in required if non_tensor.get(key) is None]
    if missing:
        raise RuntimeError(
            "AgentOPSD requires multi-turn metadata "
            f"{required}; missing {missing}. Refusing to silently run GRPO."
        )

    uid = np.asarray(non_tensor["uid"], dtype=object)
    traj_uid = np.asarray(non_tensor["traj_uid"], dtype=object)
    turn_step = np.asarray(non_tensor["turn_step"], dtype=np.int64)
    episode_rewards = np.asarray(non_tensor["episode_rewards"], dtype=np.float64)
    is_padding = np.asarray(non_tensor.get("is_padding", np.zeros(len(uid), dtype=bool)), dtype=bool)
    if is_padding.shape != (len(uid),):
        raise RuntimeError(f"invalid is_padding shape {is_padding.shape}; expected {(len(uid),)}")
    for name, values in (
        ("uid", uid),
        ("traj_uid", traj_uid),
        ("turn_step", turn_step),
        ("episode_rewards", episode_rewards),
    ):
        if values.shape != (len(uid),):
            raise RuntimeError(f"invalid {name} shape {values.shape}; expected {(len(uid),)}")
    active = ~is_padding
    if not active.any():
        raise RuntimeError("AgentOPSD received a batch containing only padding rows")
    active_t = torch.as_tensor(active, dtype=torch.bool, device=batch["advantages"].device)

    new_adv, diag = reshape_advantages(
        advantages=batch["advantages"][active_t],
        teacher_log_probs=batch["teacher_log_probs"][active_t],
        student_log_probs=batch["old_log_probs"][active_t],
        response_mask=batch["response_mask"][active_t],
        uid=uid[active],
        traj_uid=traj_uid[active],
        turn_step=turn_step[active],
        episode_rewards=episode_rewards[active],
        cfg=cfg,
    )
    batch["advantages"][active_t] = new_adv
    if (~active_t).any():
        batch["advantages"][~active_t] = 0
        if "returns" in batch:
            batch["returns"][~active_t] = 0
    diag["agentopsd/padding_turn_count"] = float(is_padding.sum())
    diag["agentopsd/active_turn_count"] = float(active.sum())
    diag["agentopsd/reshape_applied"] = 1.0
    diag["agentopsd/trajectory_metadata_validated"] = 1.0
    data.meta_info["agentopsd"] = diag
    print("[agentopsd] " + json.dumps(diag, sort_keys=True), flush=True)
    return data

