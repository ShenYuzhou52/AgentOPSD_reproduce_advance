# Qwen3.5-4B × SimpleTIR experiment overlay

This directory is an overlay for the pinned upstream Verl revision
`5b79827b04cee6e4b7b5ff737e047f1d72d50433`; it does not modify the existing
ALFWorld/SDAR checkout.

## Fixed comparison contract

All three runs use the same Qwen3.5-4B initialization, SimpleLR Math 35
training split, rollout group size, sampler, five-turn budget, local Python
tool, binary reward, validation cadence, seed, and checkpoint policy:

| Label | Policy-gradient loss | Teacher signal |
|---|---|---|
| `grpo` | GRPO | none |
| `opsd_author_code` | disabled | confidence-gated self-distillation, matching the public AgentOPSD author's OPSD script |
| `agentopsd` | GRPO | gold-answer-conditioned self-teacher gap used only for bounded turn-level advantage reshaping |

`opsd_author_code` is deliberately named after the public code path.  It is
not silently described as paper-exact OPSD: the paper's Appendix D describes
distribution matching, whereas the public script configures a zero PG
coefficient plus gated SDAR loss.

The development validation split is `simplelr_math_35/test`.  Final results
are reported only after the three arms are frozen, on the disjoint
`deepscaler/aime` and `deepscaler/aime25` files.  We do not merge a
DeepScaler training split into this first comparison: that keeps the initial
three-way collapse test small, reproducible, and clearly attributable to the
TIR loop rather than a mixture change.

## Reward and leakage boundary

The student receives only the math question and its own bounded sandbox output.
`reward_model.ground_truth` is read after the final student turn solely by the
training-side scorer, and is never added to `raw_prompt`, the tool observation,
`extra_fields`, traces, JSONL metrics, or validation prompts.  The teacher
branch will receive that answer only in its separately constructed training
forward pass.

The default reward follows SimpleTIR math semantics: answer correctness is
binary; a correct response with no substantive Python tool use receives half
credit.  JSONL records aggregate execution and format rates but never raw gold
answers, student code, or stdout.

More precisely, the correctness parser receives the full episode text: every
assistant turn followed by its bounded observation, mirroring upstream
SimpleTIR's `hf_math_verify.compute_score`, whose `extract_solution` reads the
whole multi-turn output.  The last `\boxed{}` anywhere in that text is the
prediction, so `final_answer()` output inside an observation and a boxed final
answer stated by the model score identically.  (An earlier stdout-only
reading accidentally applied the LeetCode reward path to math data and zeroed
most correct episodes; measured at temperature 0 it scored 11% where the
correct semantics score 78%.)

The private teacher branch is skipped in validation, including initial
validation.  The launch script forces `trace.token2text=False`, disables both
generic rollout and validation dumps, and writes only aggregate JSONL metrics.

## P0 invariant

One emitted TransferQueue row represents one actual generated assistant turn.
The loop returns no synthetic rows.  Group-relative advantage is calculated on
the terminal row and broadcast only to the existing turns in the same session;
invalid/partial sessions are rejected, never padded by duplicating a turn.

## Execution layout

`scripts/run_simpletir_qwen35_4b.sh` is the only launcher.  It puts Ray,
temporary files, caches, checkpoints and logs below `/data2`, keeps at most
three actor checkpoints, evaluates every five optimizer steps, and writes a
collapse-monitor record every optimizer step.  Set `PREFLIGHT_ONLY=1` to run
the model/data/config checks without starting Ray or allocating a GPU.
