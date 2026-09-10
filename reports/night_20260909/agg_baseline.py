import json

base = "/data2/ssd/yixinshen/experiments/qwen35-simpletir/base_eval_20260909"
for setting in ("cot_aime24", "cot_aime25", "tir_aime24", "tir_aime25"):
    accs, boxeds, halves = [], [], []
    for run in (1, 2, 3):
        d = json.load(open(f"{base}/{setting}/run{run}/summary.json"))
        accs.append(d["answer_accuracy"])
        boxeds.append(d["is_boxed_ratio"])
        halves.append(d["score_with_halving"])
    mean = lambda xs: sum(xs) / len(xs)
    rng = lambda xs: max(xs) - min(xs)
    runs = "[" + ", ".join("%.3f" % a for a in accs) + "]"
    print(
        f"{setting}: acc runs={runs} mean={mean(accs):.3f} range={rng(accs):.3f} | "
        f"boxed mean={mean(boxeds):.3f} | half-score mean={mean(halves):.3f}"
    )
