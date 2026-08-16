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

from agentopsd.credit import AgentOPSDConfig, reshape_advantages

_ORIGINAL_COMPUTE_ADVANTAGE = None
_RUNTIME: Dict[str, Any] = {"cfg": AgentOPSDConfig(), "multi_turn": True}
_INSTALLED = False


def install(cfg_dict: Optional[dict] = None, *, multi_turn: bool = True) -> None:
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
    _RUNTIME = {"cfg": AgentOPSDConfig.from_dict(cfg_dict), "multi_turn": multi_turn}
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
    try:
        if "teacher_log_probs" not in data.batch:
            return data
        batch = data.batch
        non_tensor = data.non_tensor_batch
        uid = non_tensor.get("uid")
        traj_uid = non_tensor.get("traj_uid")
        turn_step = non_tensor.get("turn_step")
        episode_rewards = non_tensor.get("episode_rewards")
        if uid is None or traj_uid is None or turn_step is None or episode_rewards is None:
            print("[agentopsd] WARNING: uid/traj_uid/turn_step/episode_rewards missing; skipping reshape", flush=True)
            return data

        new_adv, diag = reshape_advantages(
            advantages=batch["advantages"],
            teacher_log_probs=batch["teacher_log_probs"],
            student_log_probs=batch["old_log_probs"],
            response_mask=batch["response_mask"],
            uid=np.asarray(uid, dtype=object),
            traj_uid=np.asarray(traj_uid, dtype=object),
            turn_step=np.asarray(turn_step, dtype=np.int64),
            episode_rewards=np.asarray(episode_rewards, dtype=np.float64),
            cfg=cfg,
        )
        batch["advantages"] = new_adv
        data.meta_info["agentopsd"] = diag
        print("[agentopsd] " + json.dumps(diag, sort_keys=True), flush=True)
        _maybe_wandb_log(diag)
    except Exception as exc:  # never let credit bookkeeping kill a training step
        print(f"[agentopsd] WARNING: reshaping skipped this step: {exc!r}", flush=True)
    return data


def _maybe_wandb_log(diag: Dict[str, float]) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(diag)
    except Exception:
        pass

