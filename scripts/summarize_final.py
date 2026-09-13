import json
import statistics as st
R="/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910/final_eval_20260912"
print(f"{'rep':>3} | {'AIME24':>7} | {'AIME25':>7} | {'val100':>7}")
data={"aime24":[],"aime25":[],"val100":[]}
for r in (1,2,3,4):
    row=[]
    for ds in ("aime24","aime25","val100"):
        recs=[json.loads(x) for x in open(f"{R}/rep{r}/{ds}.jsonl")]
        acc=sum(float(x["answer_accuracy"]) for x in recs)/len(recs)
        boxed=sum(float(x["is_boxed_ratio"]) for x in recs)/len(recs)
        data[ds].append((acc,boxed))
        row.append(f"{acc:.3f}({len(recs)})")
    print(f"{r:>3} | {row[0]:>7} | {row[1]:>7} | {row[2]:>7}")
print()
for ds in data:
    accs=[a for a,_ in data[ds]]; boxeds=[b for _,b in data[ds]]
    m=st.mean(accs); s=st.stdev(accs) if len(accs)>1 else 0
    print(f"{ds}: {m:.1%} ± {s:.1%} (n=4x{len(recs)}), boxed {st.mean(boxeds):.1%}, range [{min(accs):.1%},{max(accs):.1%}]")
