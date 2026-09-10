"""Build the formal GRPO training/validation data from DeepMath-103K.

Calibration protocol: the Qwen3.5-4B (thinking, TIR, 15 turns, greedy,
episode token budget B) pilot reward is measured per 0.5-step difficulty
slice.  Two adjacent slices whose rewards bracket the target mean are mixed
so the expected dataset reward lands inside [0.40, 0.45] (aim 0.425).

Outputs (under --out-dir):
  train_deepmath_mix_s42.parquet   ~10000 rows, data_source=deepmath
  val_deepmath100_s42.parquet      100 rows,  data_source=deepmath_val100
  val_aime24.parquet / val_aime25.parquet (normalized data_source labels)
  build_manifest.json              every number used to take the decisions

The 100 validation questions are drawn from the same mixture as the training
pool (fixed seed), then removed from the training parquet.  Every question is
decontaminated against AIME24/25 by normalized-text containment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_REWARD = 0.425
TRAIN_N = 10000
VAL_N = 100
SEED = 42


def norm_text(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def to_simpletir(df: pd.DataFrame, data_source: str, split: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": r["question"]}],
            "ability": "math",
            "reward_model": {"ground_truth": str(r["final_answer"])},
            "extra_info": {
                "difficulty": float(r["difficulty"]),
                "topic": r["topic"],
                "split": split,
            },
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmath", default="/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepmath/deepmath_103k.parquet")
    parser.add_argument("--pilot-root", default="/data2/ssd/yixinshen/experiments/qwen35-simpletir/dataset_select_20260910")
    parser.add_argument("--aime24", default="/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepscaler/aime.parquet")
    parser.add_argument("--aime25", default="/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepscaler/aime25.parquet")
    parser.add_argument("--out-dir", default="/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepmath")
    parser.add_argument("--budget", type=int, default=32768)
    parser.add_argument("--target", type=float, default=TARGET_REWARD)
    parser.add_argument("--train-n", type=int, default=TRAIN_N)
    parser.add_argument("--val-n", type=int, default=VAL_N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    pilot = Path(args.pilot_root)
    difficulties = [5.0, 5.5, 6.0, 6.5]
    reward, overlong, n_pilot = {}, {}, {}
    for d in difficulties:
        summary = pilot / f"eval_d{str(d).replace('.', '_')}_b{args.budget // 1024}k" / "summary.json"
        data = json.loads(summary.read_text())
        reward[d] = float(data["score_with_halving"])
        overlong[d] = float(data.get("episode_overlong_ratio", -1.0))
        n_pilot[d] = int(data["n"])

    # choose adjacent anchors bracketing the target
    anchors = None
    for lo, hi in zip(difficulties[:-1], difficulties[1:]):
        if reward[lo] >= args.target >= reward[hi]:
            anchors = (lo, hi)
            break
    if anchors is None:
        raise SystemExit(f"no adjacent difficulty pair brackets target {args.target}: {reward}")
    lo, hi = anchors
    w = (args.target - reward[hi]) / (reward[lo] - reward[hi])
    if not (0.0 <= w <= 1.0):
        raise SystemExit(f"mix weight out of range: {w} for rewards {reward}")

    deepmath = pd.read_parquet(args.deepmath)
    rng = np.random.default_rng(args.seed)
    n_total = args.train_n + args.val_n
    n_lo = int(round(w * n_total))
    n_hi = n_total - n_lo

    pool_lo = deepmath[np.isclose(deepmath["difficulty"], lo)].reset_index(drop=True)
    pool_hi = deepmath[np.isclose(deepmath["difficulty"], hi)].reset_index(drop=True)
    if len(pool_lo) < n_lo or len(pool_hi) < n_hi:
        raise SystemExit(f"pool too small: lo={len(pool_lo)}<{n_lo} hi={len(pool_hi)}<{n_hi}")

    idx_lo = np.sort(rng.choice(len(pool_lo), size=n_lo, replace=False))
    idx_hi = np.sort(rng.choice(len(pool_hi), size=n_hi, replace=False))
    picked = pd.concat([pool_lo.iloc[idx_lo], pool_hi.iloc[idx_hi]], ignore_index=True)
    picked = picked.iloc[rng.permutation(len(picked))].reset_index(drop=True)

    # decontamination against AIME24/25 (normalized containment, both directions)
    aimes = []
    for path, source in ((args.aime24, "aime24"), (args.aime25, "aime25")):
        df = pd.read_parquet(path)
        aimes.append((df, source))
    aime_keys = [norm_k for df, _ in aimes for norm_k in (norm_text(df.iloc[i]["prompt"][0]["content"])[:180] for i in range(len(df)))]

    def contaminated(question: str) -> bool:
        key = norm_text(question)
        return any(k in key or key[:180] in k for k in aime_keys if len(k) > 40)

    contam_mask = picked["question"].map(contaminated)
    n_dropped = int(contam_mask.sum())
    picked = picked[~contam_mask].reset_index(drop=True)
    while len(picked) < n_total:  # top up if contamination dropped any
        extra_pool = pd.concat([pool_lo, pool_hi])
        extra = extra_pool.iloc[rng.choice(len(extra_pool), size=64, replace=False)]
        extra = extra[~extra["question"].map(contaminated)]
        picked = pd.concat([picked, extra], ignore_index=True).drop_duplicates("question")
        picked = picked.iloc[rng.permutation(len(picked))].reset_index(drop=True)
    picked = picked.iloc[:n_total].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_df = to_simpletir(picked.iloc[: args.val_n], "deepmath_val100", "val")
    train_df = to_simpletir(picked.iloc[args.val_n:], "deepmath", "train")
    train_path = out_dir / f"train_deepmath_mix_r{int(round(args.target*1000))}_s{args.seed}.parquet"
    val_path = out_dir / f"val_deepmath{args.val_n}_s{args.seed}.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    aime_paths = {}
    for df, source in aimes:
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "data_source": source,
                "prompt": r["prompt"],
                "ability": r.get("ability", "math"),
                "reward_model": r["reward_model"],
                "extra_info": r.get("extra_info", {"split": "val"}),
            })
        path = out_dir / f"val_{source}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        aime_paths[source] = str(path)

    manifest = {
        "protocol": {
            "model": "/data2/ssd/yixinshen/models/Qwen3.5-4B",
            "mode": "tir", "thinking": True, "temperature": 0.0,
            "max_turns": 15, "max_tokens_total": args.budget,
            "max_model_len": 49152, "seed": args.seed,
        },
        "pilot_rewards": {str(k): v for k, v in reward.items()},
        "pilot_overlong": {str(k): v for k, v in overlong.items()},
        "anchors": {"lo": lo, "hi": hi, "weight_lo": w},
        "expected_mix_reward": w * reward[lo] + (1 - w) * reward[hi],
        "mix_counts": {"lo": n_lo, "hi": n_hi},
        "contamination_dropped": n_dropped,
        "train_file": str(train_path), "train_rows": len(train_df),
        "val_file": str(val_path), "val_rows": len(val_df),
        "aime_files": aime_paths,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "val_sha256": hashlib.sha256(val_path.read_bytes()).hexdigest(),
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
