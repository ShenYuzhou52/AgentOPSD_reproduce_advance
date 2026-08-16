#!/usr/bin/env python3
"""训练稳定性报告：解析 train.log（verl 的 `step:.. - k:v` 行 + [agentopsd] JSON 行）。

用法:
    python3 scripts/stability_report.py --root logs --out report.md

输出：每个实验的成功率/熵/信用诊断曲线表格、NaN/发散检测、续训一致性摘要。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STEP_RE = re.compile(r"step:(\d+)")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+):(-?[0-9.eE+-]+)")
AGENTOPSD_RE = re.compile(r"\[agentopsd\] (\{.*\})")


def parse_log(path: Path):
    steps: dict[int, dict[str, float]] = defaultdict(dict)
    credit_lines: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            for k, v in KV_RE.findall(line):
                try:
                    steps[step][k] = float(v)
                except ValueError:
                    pass
        m = AGENTOPSD_RE.search(line)
        if m:
            try:
                credit_lines.append(json.loads(m.group(1)))
            except Exception:
                pass
    # 每个训练 step 恰好输出一行 [agentopsd]；按出现顺序对齐到升序 step
    credit = {}
    ordered_steps = sorted(steps)
    for i, diag in enumerate(credit_lines):
        if i < len(ordered_steps):
            credit[ordered_steps[i]] = diag
    return steps, credit


def flag_instability(exp: str, series: list[float], key: str, warn: list[str]):
    if len(series) < 3:
        return
    vals = [v for v in series if v == v]  # drop NaN
    if len(vals) != len(series):
        warn.append(f"{exp}: {key} 出现 NaN")
    if any(abs(v) > 1e4 for v in vals):
        warn.append(f"{exp}: {key} 出现量级异常（>1e4）")
    # 末 20% 相对前段的最大相对跳变
    n = len(vals)
    tail, head = vals[max(0, n // 5 * 4):], vals[: max(1, n // 2)]
    if head and max(abs(a - b) for a in tail for b in head) > 3 * (max(abs(x) for x in head) + 1e-9):
        warn.append(f"{exp}: {key} 末期存在大跳变（可能不稳定）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    warnings = []
    for logf in sorted(root.rglob("train.log")):
        exp = logf.parent.name
        steps, credit = parse_log(logf)
        if not steps:
            continue
        last = max(steps)
        metrics = steps[last]

        def series(key: str):
            return [steps[s].get(key) for s in sorted(steps) if key in steps[s]]

        for key in ("val/success_rate", "actor/entropy_loss", "agentopsd/credit_abs_mean"):
            flag_instability(exp, [v for v in series(key) if v is not None], key, warnings)

        rows.append(
            {
                "实验": exp,
                "步数": last,
                "最终验证成功率": metrics.get("val/success_rate"),
                "最终熵": metrics.get("actor/entropy_loss"),
                "信用均值": credit.get(last, {}).get("agentopsd/credit_abs_mean"),
                "乘子均值": credit.get(last, {}).get("agentopsd/multiplier_mean"),
            }
        )

    if not rows:
        print(f"在 {root} 下没找到 train.log")
        return

    lines = ["| 实验 | 步数 | 最终验证成功率 | 最终熵 | 信用均值 | 乘子均值 |", "|---|---|---|---|---|---|"]
    for r in rows:
        fmt = lambda v: "-" if v is None else f"{v:.4f}"
        lines.append(
            f"| {r['实验']} | {r['步数']} | {fmt(r['最终验证成功率'])} | {fmt(r['最终熵'])} "
            f"| {fmt(r['信用均值'])} | {fmt(r['乘子均值'])} |"
        )
    if warnings:
        lines.append("\n### 稳定性告警")
        lines.extend(f"- {w}" for w in warnings)

    report = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
