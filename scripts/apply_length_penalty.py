from pathlib import Path

ROOT = Path("/data2/ssd/yixinshen/AgentOPSD-tir")

def replace_once(rel, old, new):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{rel}: expected one match, found {count}: {old[:70]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {rel}")

rel = "integrations/simpletir_qwen35/simpletir_agent_loop.py"
replace_once(rel,
'''        self.max_episode_response_tokens = int(simpletir_cfg.get("max_episode_response_tokens", 16 * 1024))
''',
'''        self.max_episode_response_tokens = int(simpletir_cfg.get("max_episode_response_tokens", 16 * 1024))
        # Length/overlong penalty knobs.  Default lambda=0 keeps the reward
        # identical to the plain SimpleTIR score (ablations stay comparable);
        # "quota" mode charges linearly above a free token quota, "overlong"
        # mode is a flat charge on budget-exhausted episodes.
        self.length_penalty_lambda = float(simpletir_cfg.get("length_penalty_lambda", 0.0))
        self.length_penalty_mode = str(simpletir_cfg.get("length_penalty_mode", "quota"))
        self.length_penalty_free_frac = float(simpletir_cfg.get("length_penalty_free_frac", 0.5))
        if self.length_penalty_lambda < 0 or self.length_penalty_free_frac < 0:
            raise ValueError("simpletir.length_penalty_* must be non-negative")
''')
replace_once(rel,
'''        final_output = outputs[-1]
        final_output.extra_fields.update(
''',
'''        # Training-only length penalty, applied BEFORE the score reaches
        # GRPO's group normalization so it acts as within-group relative
        # pressure ("shorter correct beats longer correct").  Validation
        # keeps the pure task score for cross-run comparability.
        length_penalty = 0.0
        if not is_validation and self.length_penalty_lambda > 0.0:
            if self.length_penalty_mode == "overlong":
                length_penalty = self.length_penalty_lambda * float(episode_overlong)
            else:
                free = self.length_penalty_free_frac * self.max_episode_response_tokens
                overshoot = max(0.0, episode_response_tokens - free)
                span = max(1.0, self.max_episode_response_tokens - free)
                length_penalty = self.length_penalty_lambda * (overshoot / span)
        if length_penalty > 0.0:
            reward["score"] = max(0.0, reward["score"] - length_penalty)
        reward["length_penalty"] = length_penalty

        final_output = outputs[-1]
        final_output.extra_fields.update(
''')

rel = "scripts/run_simpletir_qwen35_4b.sh"
replace_once(rel,
'''  "+simpletir.max_episode_response_tokens=${MAX_EPISODE_RESPONSE_TOKENS}"
''',
'''  "+simpletir.max_episode_response_tokens=${MAX_EPISODE_RESPONSE_TOKENS}"
  "+simpletir.length_penalty_lambda=${LENGTH_PENALTY_LAMBDA:-0.0}"
  "+simpletir.length_penalty_mode=${LENGTH_PENALTY_MODE:-quota}"
  "+simpletir.length_penalty_free_frac=${LENGTH_PENALTY_FREE_FRAC:-0.5}"
''')
print("length penalty wired (default off)")
