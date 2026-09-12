import json, statistics as st, glob, os
R="/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910/final_eval_20260912"
def stats(path):
    recs=[json.loads(x) for x in open(path)]
    n=len(recs)
    acc=sum(float(r["answer_accuracy"]) for r in recs)/n
    boxed=sum(float(r["is_boxed_ratio"]) for r in recs)/n
    score=sum(float(r["score"]) for r in recs)/n
    ov=sum(float(r.get("overlong",0)) for r in recs)/n
    turns=st.mean(float(r.get("turns",0)) for r in recs)
    return dict(n=n,acc=acc,boxed=boxed,score=score,overlong=ov,turns=turns)

models={}
for rep in (1,2,3,4):
    for ds in ("aime24","aime25"):
        models.setdefault(f"step100_rep{rep}",{})[ds]=stats(f"{R}/rep{rep}/{ds}.jsonl")
    models[f"step100_rep{rep}"]["val100"]=stats(f"{R}/rep{rep}/val100.jsonl")
for name,d in (("baseline4b",f"{R}/baseline4b_once"),("ckpt40_nolenpen",f"{R}/ckpt40_nolenpen_once")):
    models[name]={ds:stats(f"{d}/{ds}.jsonl") for ds in ("aime24","aime25","val100")}

def fmt(m):
    a=m["aime24"]; b=m["aime25"]; v=m["val100"]
    return (f"acc {a['acc']:.3f}/{b['acc']:.3f}/{v['acc']:.3f}  "
            f"score {a['score']:.3f}/{b['score']:.3f}/{v['score']:.3f}  "
            f"boxed {a['boxed']:.2f}/{b['boxed']:.2f}/{v['boxed']:.2f}  "
            f"ovl {a['overlong']:.2f}/{b['overlong']:.2f}/{v['overlong']:.2f}  "
            f"turns {a['turns']:.1f}/{b['turns']:.1f}/{v['turns']:.1f}")
print("=== single-run models (aime24/aime25/val100) ===")
for name in ("baseline4b","ckpt40_nolenpen"):
    print(f"{name}:\n  {fmt(models[name])}")
print("\n=== step100 reps ===")
for r in (1,2,3,4):
    print(f"rep{r}: {fmt(models[f'step100_rep{r}'])}")
print("\n=== step100 mean over 4 reps (per dataset) ===")
for ds in ("aime24","aime25","val100"):
    accs=[models[f"step100_rep{r}"][ds]["acc"] for r in (1,2,3,4)]
    scs=[models[f"step100_rep{r}"][ds]["score"] for r in (1,2,3,4)]
    bx=[models[f"step100_rep{r}"][ds]["boxed"] for r in (1,2,3,4)]
    ov=[models[f"step100_rep{r}"][ds]["overlong"] for r in (1,2,3,4)]
    print(f"{ds}: acc {st.mean(accs):.3f}±{st.stdev(accs):.3f}  score {st.mean(scs):.3f}±{st.stdev(scs):.3f}  boxed {st.mean(bx):.3f}  overlong {st.mean(ov):.3f}")
