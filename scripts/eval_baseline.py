"""TIR-Bench baseline eval for Qwen3.5-4B without training.

Modes:
  cot  — "without CI": single-turn, no tools, boxed CoT answer (official 29.9).
  tir  — "with CI":    multi-turn fenced-Python loop, the exact sandbox/scoring
                       modules used by the training agent loop (official 38.9).

Dataset: test_fixed100_s42.parquet (seed-42 sample of the 500-question
simplelr_math_35 test split).  Headline metric is binary answer_accuracy from
score_simpletir_math (thread-safe, sympify fallback); score applies the
SimpleTIR half-credit rule and is reported for reference only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OVERLAY = "/data2/ssd/yixinshen/AgentOPSD-tir"
sys.path.insert(0, OVERLAY)

MODEL_DIR = "/data2/ssd/yixinshen/models/Qwen3.5-4B"
DATA = "/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/simplelr_math_35/test_fixed100_s42.parquet"

from integrations.simpletir_qwen35.prompting import with_simpletir_prompt
from integrations.simpletir_qwen35.simpletir_sandbox import run_python
from integrations.simpletir_qwen35.trajectory import (
    format_observation,
    is_only_final_answer,
    parse_turn,
    requires_sandbox,
    score_simpletir_math,
    with_final_answer_helper,
)

COT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cot", "tir"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    rows = pd.read_parquet(DATA)
    questions = [list(messages) for messages in rows["prompt"]]
    golds = [reward["ground_truth"] for reward in rows["reward_model"]]

    def render(messages):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        return text

    llm = LLM(
        model=MODEL_DIR,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=16384,
        enable_prefix_caching=(args.mode == "tir"),
        enforce_eager=False,
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=1.0, top_k=-1,
        max_tokens=args.max_tokens,
    )

    records = []
    if args.mode == "cot":
        prompts = [render([{"role": "user", "content": m[0]["content"] + COT_SUFFIX}]) for m in questions]
        outs = llm.generate(prompts, sampling)
        for messages, gold, out in zip(questions, golds, outs):
            text = out.outputs[0].text
            reward = score_simpletir_math(text, gold, substantive_tool_use=False)
            records.append({
                "question": messages[0]["content"],
                "gold": None,
                "response_tail": text[-400:],
                "finish": out.outputs[0].finish_reason,
                "n_tokens": len(out.outputs[0].token_ids),
                **{k: float(v) for k, v in reward.items()},
            })
    else:
        # Lockstep multi-turn loop over all questions, same semantics as the
        # training agent loop: parse fence, execute, observe, terminal rules.
        active = list(range(len(questions)))
        histories = {i: [m | {} for m in with_simpletir_prompt(questions[i])] for i in active}
        episode_text = {i: [] for i in active}
        substantive = {i: False for i in active}
        done = {}
        for turn in range(args.max_turns):
            if not active:
                break
            prompts = [render(histories[i]) for i in active]
            outs = llm.generate(prompts, sampling)
            still = []
            codes = {}
            for i, out in zip(active, outs):
                text = out.outputs[0].text
                parsed = parse_turn(text)
                histories[i].append({"role": "assistant", "content": text})
                episode_text[i].append(text)
                if requires_sandbox(parsed):
                    codes[i] = (parsed.code, with_final_answer_helper(parsed.code))
                    still.append(i)
                else:
                    done[i] = (text, parsed)
            if codes:
                def exec_one(item):
                    i, (_, helper_code) = item
                    return i, run_python(helper_code, timeout_seconds=5.0, output_limit=16384)
                with ThreadPoolExecutor(max_workers=16) as pool:
                    for i, sb in pool.map(exec_one, codes.items()):
                        obs = format_observation(stdout=sb.stdout, stderr=sb.stderr,
                                                 timed_out=sb.timed_out, limit=512)
                        episode_text[i].append(obs)
                        substantive[i] = substantive[i] or (sb.ok and not is_only_final_answer(codes[i][0]))
                        histories[i].append({"role": "user", "content": obs})
            # terminal: boxed in text or observation
            next_active = []
            for i in still:
                last_user = histories[i][-1]["content"]
                obs_boxed = "Code execution result: " in last_user and parse_turn(last_user).has_boxed_answer
                asst_text = histories[i][-2]["content"]
                if parse_turn(asst_text).has_boxed_answer or obs_boxed:
                    done[i] = (asst_text, parse_turn(asst_text))
                else:
                    next_active.append(i)
            active = next_active
        # questions that ran out of turns without a terminal condition
        for i in active:
            done[i] = (histories[i][-1]["content"] if histories[i][-1]["role"] == "assistant" else "", None)
        for i in range(len(questions)):
            joined = "\n".join(episode_text[i])
            reward = score_simpletir_math(joined, golds[i], substantive_tool_use=substantive[i])
            records.append({
                "question": questions[i][0]["content"],
                "gold": None,
                "turns": sum(1 for m in histories[i] if m["role"] == "assistant"),
                "response_tail": (done.get(i, ("", None))[0] or "")[-400:],
                "stdout_tail": joined[-200:],
                **{k: float(v) for k, v in reward.items()},
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    n = len(records)
    acc = sum(r["answer_accuracy"] for r in records) / n
    boxed = sum(r["is_boxed_ratio"] for r in records) / n
    score = sum(r["score"] for r in records) / n
    summary = {
        "mode": args.mode, "thinking": args.thinking, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "n": n,
        "answer_accuracy": acc, "is_boxed_ratio": boxed, "score_with_halving": score,
    }
    print("SUMMARY", json.dumps(summary))
    (out_path.parent / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
