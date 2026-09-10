"""Orchestrator for the 2026-09-10 formal GRPO run on the DeepMath mix.

Phases (sequential, crash-resumable via --start-phase):
  pilots   wait for the six baseline pilot evals and gate on overlong <= 0.10
  build    run build_deepmath_mix.py and gate the achieved mix reward
  smoke    2-step smoke with the exact formal config (console logging only)
           exercising train metrics + the 3-file validation + dump path
  formal   wait for the wandb credential, then launch the 200-step run with
           wandb on and the overlong kill-switch watcher attached

All state lands in <root>/orchestrator_state.json; every phase prints its
decision inputs.  Nothing is auto-deleted; a failed gate exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910")
PILOT_ROOT = Path("/data2/ssd/yixinshen/experiments/qwen35-simpletir/dataset_select_20260910")
DATASET_DIR = Path("/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepmath")
OVERLAY = Path("/data2/ssd/yixinshen/AgentOPSD-tir")
PYBIN = "/data2/ssd/yixinshen/benchmarks/verl-qwen35-base/.venv/bin/python"
BUDGET = 32768
MAX_TURNS = 15
MAX_MODEL_LEN = 45056
SMOKE_NAME = "grpo_deepmath_think32k_t15_s42_smoke2"
FORMAL_NAME = "grpo_deepmath_think32k_t15_s42_formal200"
Gpus = "4,5,6,7"

state_path = ROOT / "orchestrator_state.json"


def write_state(stage: str, **extra) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(), **extra}
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(state_path)


def parse_steps(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: dict[int, dict] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if "response_length/clip_ratio:" not in line or "critic/score/mean:" not in line:
            continue
        step_m = re.search(r"step:(\d+)", line)
        score_m = re.search(r"critic/score/mean:([-+0-9.eE]+)", line)
        clip_m = re.search(r"response_length/clip_ratio:([-+0-9.eE]+)", line)
        abort_m = re.search(r"response/aborted_ratio:([-+0-9.eE]+)", line)
        ep_m = re.search(r"simpletir/episode_overlong_ratio:([-+0-9.eE]+)", line)
        if step_m and score_m and clip_m:
            rows[int(step_m.group(1))] = {
                "step": int(step_m.group(1)),
                "score_mean": float(score_m.group(1)),
                "clip_ratio": float(clip_m.group(1)),
                "aborted_ratio": float(abort_m.group(1)) if abort_m else None,
                "episode_overlong": float(ep_m.group(1)) if ep_m else None,
            }
    return [rows[k] for k in sorted(rows)]


def wandb_ready() -> bool:
    code = (
        "import os,sys\n"
        "key=os.environ.get('WANDB_API_KEY')\n"
        "if not key:\n"
        "    try:\n"
        "        import wandb\n"
        "        key=wandb.api.api_key\n"
        "    except Exception:\n"
        "        key=None\n"
        "sys.exit(0 if key else 1)\n"
    )
    return subprocess.call([PYBIN, "-c", code], env=os.environ.copy()) == 0


def run_env(experiment: str, steps: int, use_wandb: bool, test_freq: int, save_freq: int, max_ckpts: int, resume: str, val_before: bool) -> dict:
    manifest = json.loads((DATASET_DIR / "build_manifest.json").read_text())
    val_files = ":".join([manifest["val_file"], manifest["aime_files"]["aime24"], manifest["aime_files"]["aime25"]])
    env = os.environ.copy()
    env.update({
        "METHOD": "grpo",
        "TRAIN_FILE": manifest["train_file"],
        "VAL_FILE": val_files,
        "RUN_ROOT": str(ROOT),
        "EXPERIMENT": experiment,
        "TRAIN_STEPS": str(steps),
        "TRAIN_PROMPTS": "16",
        "ROLLOUT_N": "8",
        "MAX_TURNS": str(MAX_TURNS),
        "MAX_PROMPT_LENGTH": "4096",
        "MAX_RESPONSE_LENGTH": str(BUDGET),
        "MAX_EPISODE_RESPONSE_TOKENS": str(BUDGET),
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "ACTOR_MINI_BATCH": "8",
        "GPU_MEMORY_UTILIZATION": "0.40",
        "N_GPUS": "4",
        "TEST_FREQ": str(test_freq),
        "SAVE_FREQ": str(save_freq),
        "MAX_CKPTS": str(max_ckpts),
        "RESUME_MODE": resume,
        "VAL_BEFORE_TRAIN": "true" if val_before else "false",
        "VAL_ROLLOUT_N": "1",
        "ENABLE_THINKING": "true",
        "DATALOADER_WORKERS": "0",
        "CUDA_VISIBLE_DEVICES": Gpus,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "USE_WANDB": "1" if use_wandb else "0",
        "LOG_VAL_GENERATIONS": "24",
    })
    return env


def launch(env: dict, log_path: Path) -> subprocess.Popen:
    log = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        ["bash", str(OVERLAY / "scripts/run_simpletir_qwen35_4b.sh")],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def phase_pilots() -> None:
    names = ["d5_0_b32k", "d5_5_b32k", "d6_0_b32k", "d6_5_b32k", "d5_5_b24k", "dsr_b32k"]
    deadline = time.time() + 4 * 3600
    while True:
        done = {n: (PILOT_ROOT / f"eval_{n}" / "summary.json").exists() for n in names}
        if all(done.values()):
            break
        if time.time() > deadline:
            write_state("pilots_timeout", pilots=done)
            raise SystemExit(2)
        write_state("waiting_pilots", pilots=done)
        time.sleep(60)
    rewards, overlongs = {}, {}
    for n, d in (("d5_0_b32k", 5.0), ("d5_5_b32k", 5.5), ("d6_0_b32k", 6.0), ("d6_5_b32k", 6.5)):
        data = json.loads((PILOT_ROOT / f"eval_{n}" / "summary.json").read_text())
        rewards[d] = float(data["score_with_halving"])
        overlongs[d] = float(data.get("episode_overlong_ratio", -1.0))
    write_state("pilots_done", rewards=rewards, overlongs=overlongs)
    ref = overlongs[5.5]
    if ref > 0.10:
        write_state("pilots_rejected_overlong", rewards=rewards, overlongs=overlongs,
                    reason=f"d5.5 overlong {ref:.3f} > 0.10 even at 32768 total budget")
        raise SystemExit(3)


def phase_build() -> None:
    write_state("building")
    log = (ROOT / "build.log").open("wb")
    rc = subprocess.call([PYBIN, str(OVERLAY / "scripts/build_deepmath_mix.py")], stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        write_state("build_failed", returncode=rc)
        raise SystemExit(rc)
    manifest = json.loads((DATASET_DIR / "build_manifest.json").read_text())
    expected = manifest["expected_mix_reward"]
    if not (0.40 <= expected <= 0.45):
        write_state("build_rejected_reward", expected=expected)
        raise SystemExit(4)
    write_state("build_done", manifest=manifest)


def phase_smoke() -> None:
    run_dir = ROOT / SMOKE_NAME
    if run_dir.exists() and any(run_dir.iterdir()):
        write_state("smoke_dir_not_empty", run_dir=str(run_dir))
        raise SystemExit(5)
    env = run_env(SMOKE_NAME, steps=2, use_wandb=False, test_freq=2, save_freq=1,
                  max_ckpts=2, resume="disable", val_before=False)
    (ROOT / "smoke_config.json").write_text(json.dumps({k: env[k] for k in sorted(env) if not k.startswith("_")}, indent=2))
    proc = launch(env, ROOT / "smoke_launcher.log")
    (ROOT / "smoke_pid").write_text(str(proc.pid))
    write_state("smoke_running", pid=proc.pid)
    rc = proc.wait()
    steps = parse_steps(run_dir / "train.log")
    dumps = sorted(p.name for p in (run_dir / "val_generations").glob("*.jsonl")) if (run_dir / "val_generations").is_dir() else []
    if rc != 0:
        write_state("smoke_failed", returncode=rc, steps=steps)
        raise SystemExit(rc)
    if len(steps) < 2:
        write_state("smoke_rejected", reason="missing two training metric rows", steps=steps)
        raise SystemExit(6)
    initial = steps[0]["score_mean"]
    worst_clip = max(s["clip_ratio"] for s in steps)
    worst_ep = max((s["episode_overlong"] or 0.0) for s in steps)
    reasons = []
    if not (0.30 <= initial <= 0.55):
        reasons.append(f"initial reward {initial:.3f} outside [0.30,0.55]")
    if worst_clip >= 0.10:
        reasons.append(f"clip ratio {worst_clip:.3f} >= 0.10")
    if worst_ep >= 0.10:
        reasons.append(f"episode overlong {worst_ep:.3f} >= 0.10")
    if not dumps:
        reasons.append("no validation generation dump produced")
    if reasons:
        write_state("smoke_rejected", reasons=reasons, steps=steps, val_dumps=dumps)
        raise SystemExit(7)
    write_state("smoke_passed", steps=steps, val_dumps=dumps)


def phase_formal() -> None:
    deadline = time.time() + 8 * 3600
    while not wandb_ready():
        if time.time() > deadline:
            write_state("formal_aborted_no_wandb_key",
                        hint="deploy the key then rerun with --start-phase formal")
            raise SystemExit(8)
        write_state("waiting_wandb_key", hint="export WANDB_API_KEY or run wandb login as yixinshen")
        time.sleep(60)

    run_dir = ROOT / FORMAL_NAME
    if run_dir.exists() and any(run_dir.iterdir()):
        write_state("formal_dir_not_empty", run_dir=str(run_dir))
        raise SystemExit(9)
    env = run_env(FORMAL_NAME, steps=200, use_wandb=True, test_freq=20, save_freq=20,
                  max_ckpts=3, resume="auto", val_before=True)
    (ROOT / "formal_config.json").write_text(json.dumps({k: env[k] for k in sorted(env) if not k.startswith("_")}, indent=2))
    proc = launch(env, ROOT / "formal_launcher.log")
    (ROOT / "formal_pid").write_text(str(proc.pid))
    monitor_log = (ROOT / "monitor.log").open("ab", buffering=0)
    mon = subprocess.Popen(
        [PYBIN, str(OVERLAY / "scripts/monitor_overlong.py"),
         "--run-dir", str(run_dir), "--pid", str(proc.pid),
         "--sustained", "0.10", "--sustained-steps", "3", "--single", "0.15", "--interval", "60"],
        stdin=subprocess.DEVNULL, stdout=monitor_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (ROOT / "monitor_pid").write_text(str(mon.pid))
    write_state("formal_running", pid=proc.pid, monitor_pid=mon.pid, run_dir=str(run_dir),
                wandb_project="qwen35_simpletir", wandb_run=FORMAL_NAME)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-phase", choices=["pilots", "build", "smoke", "formal"], default="pilots")
    args = parser.parse_args()
    phases = [("pilots", phase_pilots), ("build", phase_build), ("smoke", phase_smoke), ("formal", phase_formal)]
    index = [p for p, _ in phases].index(args.start_phase)
    for name, fn in phases[index:]:
        fn()
    write_state("orchestrator_complete")


if __name__ == "__main__":
    main()
