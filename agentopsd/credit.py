"""AgentOPSD recursive turn-level credit assignment (arXiv:2608.05987).

The module is deliberately framework-agnostic.  The verl/SDAR trainer hands us
a batch whose rows are *turns*: one row per (trajectory, turn), with

* ``traj_uid``          -- which trajectory the row belongs to,
* ``turn_step``         -- the turn index inside that trajectory,
* ``episode_rewards``   -- the terminal outcome of that trajectory (repeated on
  every turn row of the same trajectory),
* ``uid``               -- the GRPO group (task) the trajectory belongs to,
* ``teacher_log_probs`` / ``student_log_probs`` -- token-level log-probabilities
  of the skill-conditioned (teacher) and unconditioned (student) branches,
* ``response_mask``     -- which response tokens are real,
* ``advantages``        -- the sequence-level GRPO advantage (identical for all
  tokens of one trajectory before reshaping).

``reshape_advantages`` converts the sequence-level advantages into turn-level
reshaped advantages following Algorithm 1 of the paper:

1. token gap            δ_{k,t} = log π_θ(y_{k,t} | h+_{k,t}) - log π_θ(y_{k,t} | h_{k,t})
2. turn evidence        e_k     = Σ_t δ_{k,t}
3. recursive belief     c_k = γ c_{k-1} + e_k,  ℓ_k = logit(B_0) + c_k,
                         B_k = σ(ℓ_k),  ΔB_k = B_k - B_{k-1}
                         with B_0 = clip(group success rate S/G, ε0, 1-ε0)
4. outcome-aligned      q_k = sign(A_seq) · ΔB_k
5. bounded reshaping    z_k = (q_k - μ_q)/(σ_q + ε),  w_k = clip(1 + b·z_k, 1-b, 1+b),
                         Ã_k = A_seq · ((1-λ) + λ·w_k)

Every token of turn k inherits Ã_k.  The teacher signal is detached; there is
no critic and no extra rollout -- the only overhead is the teacher forward pass
already performed by the SDAR trainer for its skill-conditioned log-probs.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np
import torch

EPS: float = 1e-4


@dataclasses.dataclass
class AgentOPSDConfig:
    """AgentOPSD hyperparameters (paper Table 3 / Appendix F).

    One shared setting is used across environments and model scales in the
    paper: λ=0.5, b=0.2, γ=0.95, ε0=1e-4.
    """

    enabled: bool = True
    lam: float = 0.5  # λ: reshaping weight ∈ [0, 1]; 0 recovers vanilla GRPO
    b: float = 0.2  # b: multiplier band ∈ (0, 1); w_k ∈ [1-b, 1+b]
    gamma: float = 0.95  # γ: evidence decay ∈ (0, 1]
    eps0: float = 1e-4  # ε0: clip of the group-success prior B0
    eps_q: float = EPS  # ε: stabilizer in within-trajectory standardization
    success_threshold: float = 0.5  # episode reward > threshold → success

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AgentOPSDConfig":
        cfg = cls()
        if not d:
            return cfg
        for key, value in d.items():
            # ``enable`` was used by the first shell wrapper. Keep accepting
            # it so a manual Hydra invocation cannot silently enable credit.
            if key == "enable":
                key = "enabled"
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
        return cfg


def _logit(p: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def reshape_advantages(
    *,
    advantages: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    uid: np.ndarray,
    traj_uid: np.ndarray,
    turn_step: np.ndarray,
    episode_rewards: np.ndarray,
    cfg: Optional[AgentOPSDConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reshape sequence-level GRPO advantages into AgentOPSD turn-level ones.

    Args:
        advantages: (B, T) token-level GRPO advantages (same value on every
            token of one turn row).
        teacher_log_probs: (B, T) skill-conditioned log-probs (detached).
        student_log_probs: (B, T) policy log-probs of the same tokens.
        response_mask: (B, T) 0/1 mask of real response tokens.
        uid: (B,) GRPO group (task) id of each row.
        traj_uid: (B,) trajectory id of each row.
        turn_step: (B,) turn index inside the trajectory (only order matters).
        episode_rewards: (B,) terminal outcome of the trajectory, repeated on
            every turn row (e.g. 0/10 for ALFWorld, 0..1 for WebShop).
        cfg: AgentOPSD hyperparameters.

    Returns:
        reshaped_advantages: same shape as ``advantages``; token t of turn k
            inherits Ã_k.
        diagnostics: scalar metrics prefixed ``agentopsd/`` for logging.
    """
    cfg = cfg or AgentOPSDConfig()
    if advantages.ndim != 2:
        raise ValueError(f"advantages must be (B, T), got {tuple(advantages.shape)}")
    B, T = advantages.shape
    device, dtype = advantages.device, advantages.dtype

    mask = response_mask.bool()
    gap = (teacher_log_probs - student_log_probs).detach() * mask.to(dtype)
    evidence = gap.sum(dim=-1)  # (B,) per-turn evidence e_k

    # Sequence-level advantage of each row: GRPO broadcasts one scalar to all
    # tokens of a row, so reading the first valid token is enough.
    valid = mask.any(dim=-1)
    first_valid = mask.float().argmax(dim=-1)
    adv_row = advantages.gather(dim=1, index=first_valid.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    adv_row = torch.where(valid, adv_row, torch.zeros_like(adv_row))

    uid_np = np.asarray(uid, dtype=object)
    traj_np = np.asarray(traj_uid, dtype=object)
    turn_np = np.asarray(turn_step, dtype=np.int64)
    rew_np = np.asarray(episode_rewards, dtype=np.float64)

    unique_traj, traj_inv = np.unique(traj_np, return_inverse=True)
    first_row = np.unique(traj_inv, return_index=True)[1]
    uid_of_traj = uid_np[first_row]
    traj_reward = rew_np[first_row]

    # B0 = clip(S/G, ε0, 1-ε0): group success rate over trajectories (Prop. 7).
    group_success: Dict[object, float] = {}
    for u in np.unique(uid_np):
        group_success[u] = float(np.mean(traj_reward[uid_of_traj == u] > cfg.success_threshold))

    group_success_values = np.asarray(list(group_success.values()), dtype=np.float64)
    mixed_group_count = int(np.sum((group_success_values > 0.0) & (group_success_values < 1.0)))
    group_success_std = (
        float(group_success_values.std(ddof=0)) if group_success_values.size else 0.0
    )

    rows_by_traj: Dict[object, np.ndarray] = {}
    for i, t in enumerate(unique_traj):
        rows_by_traj[t] = np.nonzero(traj_inv == i)[0]

    reshaped = advantages.clone()
    n_traj = unique_traj.size
    total_turns = 0
    multi_turn_traj = 0
    ev_abs_sum = 0.0
    rev_sum = 0.0
    q_abs_sum = 0.0
    w_sum = 0.0
    w_min = float("inf")
    w_max = float("-inf")
    pivotal = 0
    belief_saturated = 0
    success_sum = 0.0
    belief_min = float("inf")
    belief_max = float("-inf")
    raw_w_sum = 0.0
    raw_w_min = float("inf")
    raw_w_max = float("-inf")
    raw_multiplier_clipped = 0

    for t in unique_traj:
        rows = rows_by_traj[t]
        if rows.size == 0:
            continue
        order = np.argsort(turn_np[rows], kind="stable")
        rows = rows[order]
        K = rows.size
        total_turns += K
        if K > 1:
            multi_turn_traj += 1

        e = evidence[rows].to(device)  # (K,)
        a_seq = adv_row[rows[0]].to(device)  # broadcast value of this trajectory
        u = uid_np[rows[0]]
        success_sum += group_success[u]
        b0 = float(np.clip(group_success[u], cfg.eps0, 1.0 - cfg.eps0))
        b0 = torch.tensor(b0, dtype=dtype, device=device)

        c = torch.zeros((), dtype=dtype, device=device)
        b_prev = b0
        qs = []
        for k in range(K):
            c = cfg.gamma * c + e[k]  # c_k = γ c_{k-1} + e_k
            ell = _logit(b0, eps=cfg.eps0) + c  # ℓ_k = logit(B0) + c_k
            b_k = torch.sigmoid(ell)
            belief_saturated += int(((b_k <= 0.01) | (b_k >= 0.99)).item())
            belief_min = min(belief_min, b_k.item())
            belief_max = max(belief_max, b_k.item())
            delta_b = b_k - b_prev  # ΔB_k = B_k - B_{k-1}
            b_prev = b_k
            qs.append(torch.sign(a_seq) * delta_b)  # q_k = sign(A_seq)·ΔB_k
            rev_sum += delta_b.abs().item()
            ev_abs_sum += e[k].abs().item()
        q = torch.stack(qs)
        q_abs_sum += q.abs().sum().item()

        mu_q = q.mean()
        # Algorithm 1 uses the population std over the K turns of one
        # trajectory. The unbiased/sample estimate over-amplifies K=2.
        sigma_q = q.std(unbiased=False) if K > 1 else torch.zeros((), dtype=dtype, device=device)
        z = (q - mu_q) / (sigma_q + cfg.eps_q)
        w_lo, w_hi = 1.0 - cfg.b, 1.0 + cfg.b
        raw_w = 1.0 + cfg.b * z
        raw_multiplier_clipped += int(((raw_w < w_lo) | (raw_w > w_hi)).sum().item())
        raw_w_sum += raw_w.sum().item()
        raw_w_min = min(raw_w_min, raw_w.min().item())
        raw_w_max = max(raw_w_max, raw_w.max().item())
        w = torch.clamp(raw_w, w_lo, w_hi)
        w_sum += w.sum().item()
        w_min = min(w_min, w.min().item())
        w_max = max(w_max, w.max().item())
        pivotal += (z.abs() > 1.0).sum().item()

        adv_turn = a_seq * ((1.0 - cfg.lam) + cfg.lam * w)  # Ã_k
        for idx, row in enumerate(rows):
            reshaped[row] = adv_turn[idx]

    valid_adv = advantages[mask]
    valid_reshaped = reshaped[mask]
    adv_finite = torch.isfinite(valid_adv) & torch.isfinite(valid_reshaped)
    adv_nonfinite_count = int((~adv_finite).sum().item()) if adv_finite.numel() else 0
    adv_abs_sum = valid_reshaped[adv_finite].abs().sum().item() if adv_finite.any() else 0.0
    denom = max(total_turns, 1)
    diag = {
        "agentopsd/traj_count": float(n_traj),
        "agentopsd/turn_count": float(total_turns),
        "agentopsd/turns_per_traj_mean": float(total_turns / max(n_traj, 1)),
        "agentopsd/multi_turn_traj_ratio": float(multi_turn_traj / max(n_traj, 1)),
        "agentopsd/group_success_mean": float(success_sum / max(n_traj, 1)),
        "agentopsd/group_success_std": group_success_std,
        "agentopsd/group_success_mixed_ratio": float(
            mixed_group_count / max(len(group_success_values), 1)
        ),
        "agentopsd/evidence_abs_mean": float(ev_abs_sum / denom),
        "agentopsd/belief_revision_abs_mean": float(rev_sum / denom),
        "agentopsd/credit_abs_mean": float(q_abs_sum / denom),
        "agentopsd/multiplier_mean": float(w_sum / denom),
        "agentopsd/multiplier_min": float(w_min if w_min < float("inf") else 1.0),
        "agentopsd/multiplier_max": float(w_max if w_max > float("-inf") else 1.0),
        "agentopsd/pivotal_turn_ratio": float(pivotal / denom),
        "agentopsd/belief_saturation_ratio": float(belief_saturated / denom),
        "agentopsd/belief_min": float(belief_min if belief_min < float("inf") else 0.0),
        "agentopsd/belief_max": float(belief_max if belief_max > float("-inf") else 1.0),
        "agentopsd/multiplier_raw_mean": float(raw_w_sum / denom),
        "agentopsd/multiplier_raw_min": float(raw_w_min if raw_w_min < float("inf") else 1.0),
        "agentopsd/multiplier_raw_max": float(raw_w_max if raw_w_max > float("-inf") else 1.0),
        "agentopsd/multiplier_clip_ratio": float(raw_multiplier_clipped / denom),
        "agentopsd/adv_abs_mean": float(adv_abs_sum / max(int(adv_finite.sum().item()), 1)),
        "agentopsd/adv_nonfinite_ratio": float(adv_nonfinite_count / max(valid_reshaped.numel(), 1)),
        "agentopsd/adv_std_before": float(valid_adv[adv_finite].std(unbiased=False).item()) if adv_finite.any() else 0.0,
        "agentopsd/adv_std_after": float(valid_reshaped[adv_finite].std(unbiased=False).item()) if adv_finite.any() else 0.0,
    }
    return reshaped, diag

