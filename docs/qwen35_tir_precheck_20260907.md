# Qwen3.5 SimpleTIR server precheck — 2026-09-07

Status: in progress. This is a two-step execution smoke test, not an ablation benchmark result.

- Server: `124.128.251.62:16022`, user `yixinshen`.
- Overlay: `/data2/ssd/yixinshen/AgentOPSD-tir`, base commit `7be30267eb6e5f5f48cbce5cd714edbf141da61e`.
- Verl: `/data2/ssd/yixinshen/benchmarks/verl-qwen35-base`, commit `5b79827b04cee6e4b7b5ff737e047f1d72d50433`.
- Python: Verl `.venv/bin/python`; model: `/data2/ssd/yixinshen/models/Qwen3.5-4B`.
- Data: `/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/simplelr_math_35`.
- Results root: `/data2/ssd/yixinshen/experiments/qwen35-simpletir/precheck_20260907`.
- Ray: `/data2/ssd/yixinshen/r/<experiment hash>`; temp: `/data2/ssd/yixinshen/t/<experiment hash>`.
- Initially free A800 80 GB GPUs: 0,1,2,3,6,7. Existing tasks on 4,5 were preserved.

## Configuration

Two GPUs per run, 2 training steps, 8 prompts/batch, 4 rollouts/prompt, actor minibatch 4, 5 tool turns maximum, response limit 1024, model context 12288, GPU rollout memory utilization 0.30, dataset seed 42. Initial validation disabled. Four original development examples are evaluated at step 2; checkpoint is saved at step 2. No final AIME evaluation. Fresh runs use `resume_mode=disable`.

## Findings

- GRPO completed 2 steps, validation, checkpoint saving, and exited 0. Both steps had zero reward, zero advantages, and zero gradient norm; this establishes execution only, not effective learning.
- AgentOPSD initially failed before an optimizer update: teacher response token ID alignment check compared singleton lists from Verl (`[[id], ...]`) against flat student IDs. The existing shifted logprob slice was correct. The fix unwraps singleton IDs and keeps strict token/length checks.
- Original OPSD attempt was deliberately terminated during initialization after identifying the shared teacher-path bug. It is not a completed OPSD test.
- CPU/sandbox tests: 19 passed before fix; 22 passed after fix, including singleton shape, shifted logprob alignment excluding the trailing dummy, wrong-token rejection, and top-k rejection.
- Tool sandbox uses `/usr/bin/python3` (3.10.12). `numpy`, `sympy`, and `scipy` are unavailable inside it. Standard-library execution and isolation tests pass. Missing scientific packages can reduce tool success; individual sampled failures have not been classified.
- Training reports unavailable fast-path libraries and falls back to torch. Optional DeepGEMM import reports missing CUDA_HOME; GRPO still completes.

## Artifacts

- GRPO: `grpo_smoke2_s42/metrics.jsonl`, `train.log`, `config_overrides.txt`, `checkpoints/global_step_2` (51 GB, both ranks plus optimizer/data state).
- Fixed AgentOPSD: `teacherfix_agentopsd/agentopsd_teacherfix2_s42/` (GPUs 0,1).
- Fixed OPSD: `teacherfix_opsd/opsd_author_code_teacherfix2_s42/` (GPUs 2,3).

OPSD here means `opsd_author_code`: public-author-code gated SDAR loss with no task policy-gradient term, plus configured reference KL. It is not claimed to be paper-exact distribution-matching OPSD.
