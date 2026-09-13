"""Plot key training curves for both GRPO arms from exported wandb history."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)

data = json.load(open(HERE / "wandb_history.json"))

def series(arm, col, offset):
    cols = data[arm]["columns"]
    si = cols.index("_step")
    j = cols.index(col)
    xs, ys = [], []
    for row in data[arm]["rows"]:
        s, v = row[si], row[j]
        if v is not None:
            xs.append(s + offset)
            ys.append(v)
    return xs, ys

NOPEN_OFF, LEN_OFF = 0, 40
C_N, C_L = "#d62728", "#1f77b4"
LABEL = {"nopen": "no-penalty arm", "lenpen": "length-penalty arm (λ=0.25)"}

def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

# 1. training reward
fig, ax = plt.subplots(figsize=(8, 4.5))
for arm, off, c in (("nopen", NOPEN_OFF, C_N), ("lenpen", LEN_OFF, C_L)):
    xs, ys = series(arm, "critic/score/mean", off)
    ax.plot(xs, ys, color=c, alpha=0.35, lw=0.8)
    # rolling mean(5)
    roll = [sum(ys[max(0,i-4):i+1])/len(ys[max(0,i-4):i+1]) for i in range(len(ys))]
    ax.plot(xs, roll, color=c, lw=2, label=LABEL[arm])
ax.axhline(0.425, color="gray", ls="--", lw=1, label="dataset baseline 0.425")
ax.annotate("peak val: step40", xy=(40, 0.73), fontsize=8, color=C_N)
ax.annotate("peak val: step100", xy=(100, 0.82), fontsize=8, color=C_L)
ax.set_xlabel("training step")
ax.set_ylim(0.3, 1.0)
style(ax, "Training reward (temp=1.0, per-step + MA5)", "critic/score/mean")
fig.tight_layout(); fig.savefig(PLOTS / "1_train_reward.png", dpi=150)

# 2. response length + overlong
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for arm, off, c in (("nopen", NOPEN_OFF, C_N), ("lenpen", LEN_OFF, C_L)):
    xs, ys = series(arm, "response_length/mean", off)
    axes[0].plot(xs, ys, color=c, lw=1.5, label=LABEL[arm])
    xs, ys = series(arm, "response_length/clip_ratio", off)
    axes[1].plot(xs, ys, color=c, lw=1.2, label=LABEL[arm])
axes[0].axhline(16384, color="gray", ls="--", lw=1, label="penalty free quota 16k")
axes[0].set_ylabel("response tokens (mean)")
style(axes[0], "Response length", "tokens")
axes[1].set_ylabel("overlong ratio")
axes[1].set_xlabel("training step")
style(axes[1], "Episode overlong ratio (hit 32k budget)", "ratio")
fig.tight_layout(); fig.savefig(PLOTS / "2_length_overlong.png", dpi=150)

# 3. KL & entropy
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for arm, off, c in (("nopen", NOPEN_OFF, C_N), ("lenpen", LEN_OFF, C_L)):
    xs, ys = series(arm, "actor/kl_loss", off)
    axes[0].plot(xs, ys, color=c, lw=1.5, label=LABEL[arm])
    xs, ys = series(arm, "actor/entropy", off)
    axes[1].plot(xs, ys, color=c, lw=1.2, label=LABEL[arm])
axes[0].set_ylabel("KL loss (vs base)")
style(axes[0], "KL to base policy (coef 0.01)", "kl")
axes[1].set_ylabel("entropy")
axes[1].set_xlabel("training step")
style(axes[1], "Policy entropy", "entropy")
fig.tight_layout(); fig.savefig(PLOTS / "3_kl_entropy.png", dpi=150)

# 4. validation three sources
VC = [("val-core/deepmath_val100/reward/mean@1", "val100 (DeepMath mix)"),
      ("val-core/aime24/reward/mean@1", "AIME24"),
      ("val-core/aime25/reward/mean@1", "AIME25")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (col, title) in zip(axes, VC):
    for arm, off, c in (("nopen", NOPEN_OFF, C_N), ("lenpen", LEN_OFF, C_L)):
        xs, ys = series(arm, col, off)
        ax.plot(xs, ys, marker="o", ms=4, color=c, lw=1.5, label=LABEL[arm])
    ax.axhline(0.45, color="gray", ls=":", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("reward @1 (greedy)")
axes[0].legend(fontsize=8)
fig.suptitle("Validation reward every 20 steps (gray dotted: dataset baseline 0.45)", fontsize=11)
fig.tight_layout(); fig.savefig(PLOTS / "4_validation_three_sources.png", dpi=150)
print("saved:", [p.name for p in sorted(PLOTS.glob("*.png"))])
