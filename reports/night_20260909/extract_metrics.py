import json, re

R = "/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907"
runs = {"GRPO": "grpo/grpo_formal200_s42", "AgentOPSD": "agentopsd/agentopsd_formal200_s42", "OPSD": "opsd/opsd_formal200_s42"}
for name, rel in runs.items():
    print("##### " + name)
    with open(R + "/" + rel + "/train.log", errors="ignore") as f:
        lines = f.readlines()
    joined = []
    i = 0
    while i < len(lines):
        if re.search(r"step:\d+ -", lines[i]):
            joined.append(" ".join(x.strip() for x in lines[i:i + 4]))
            i += 4
        else:
            i += 1
    vals = []
    for chunk in joined:
        m = re.search(r"step:(\d+) - .*?val-core/[^:]+:np\.float64\(([\d.]+)\)", chunk)
        if m:
            vals.append((int(m.group(1)), float(m.group(2))))
    seen = {}
    for s, v in vals:
        seen.setdefault(s, v)
    print("VAL|" + ";".join("%d:%.3f" % (s, seen[s]) for s in sorted(seen)))
