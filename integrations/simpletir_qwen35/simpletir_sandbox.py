"""Local, unprivileged Python sandbox for SimpleTIR rollouts.

执行模型生成的 Python 的唯一入口。与上游的 HTTP 沙箱不同，这里用
systemd-run（用户级瞬时服务，施加 MemoryMax/TasksMax/CPUQuota 等资源上限）
+ bubblewrap（独立 PID/net/IPC 命名空间、清空环境、只读绑定 /usr 与科学
计算 venv、可写区仅一个 tmpfs /tmp 和空 /work）在本地无特权执行：

    模型代码 → systemd-run --user（资源限额）
             → timeout（硬超时）
             → bwrap（文件系统/网络隔离）
             → /opt/sb_venv/bin/python3 -I -（隔离模式解释器）

两层防线互补：systemd 管"用了多少"，bwrap 管"能看见什么"。历史上这里
出过两个真实故障（都已修并有回归测试）：系统 Python 缺 sympy/numpy 导致
工具奖励永远拿不到；OpenBLAS 按宿主 112 核开线程被 TasksMax 杀掉导致
numpy 段错误。
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    """一次沙箱执行的结构化结果。

    timed_out / rejected 与普通非零退出严格区分：前者是模型行为（写死循环），
    后者是宿主问题（配置错误），混在一起会让"沙箱坏了"被误读成"模型不行"。
    """

    stdout: str
    stderr: str
    returncode: int | None
    elapsed_seconds: float
    timed_out: bool
    rejected: bool = False

    @property
    def ok(self) -> bool:
        """只有"未被拒 + 未超时 + 退出码 0"才算成功；退出码非零通常意味着
        模型代码抛了异常（如 ImportError/SyntaxError），是正常的学生错误。"""
        return not self.rejected and not self.timed_out and self.returncode == 0


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


# 只保留请求用户 systemd 服务所需的最小环境变量。
def _systemd_environment() -> dict[str, str]:
    """Keep D-Bus routing outside bwrap; never pass broad host env to the program."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


# 指定服务器上的科学计算虚拟环境；挂载时保持只读。
DEFAULT_SANDBOX_VENV = "/data2/ssd/yixinshen/sb_venv"
# Mount point inside the namespace; a neutral path keeps the host data disk
# invisible to model programs while pyvenv.cfg still resolves next to the
# interpreter.
SANDBOX_VENV_MOUNT = "/opt/sb_venv"


# 确认虚拟环境解释器存在，缺失时回退到系统 Python。
def _sandbox_venv() -> Path | None:
    """Resolve the read-only scientific-compute venv for model programs.

    The system interpreter under bwrap has only the standard library, so any
    model program importing sympy/numpy/scipy failed and could never earn the
    SimpleTIR tool reward.  The venv is bound read-only into the same namespace;
    unset or missing paths fall back to the bare system interpreter, which unit
    tests rely on when describing historical behaviour.
    """
    venv = Path(os.environ.get("SIMPLETIR_SANDBOX_VENV", DEFAULT_SANDBOX_VENV))
    return venv if (venv / "bin" / "python3").exists() else None


def _bwrap_command() -> list[str]:
    # 收集需要只读绑定进 bubblewrap 命名空间的运行时目录。
    binds: list[str] = []
    for path in ("/usr", "/lib", "/lib64"):
        if Path(path).exists():
            binds.extend(["--ro-bind", path, path])
    venv = _sandbox_venv()
    interpreter = "/usr/bin/python3"
    # 将包含 numpy、sympy 等依赖的虚拟环境以只读方式暴露给代码。
    if venv is not None:
        binds.extend(["--ro-bind", str(venv), SANDBOX_VENV_MOUNT])
        interpreter = f"{SANDBOX_VENV_MOUNT}/bin/python3"
    # 构造清空环境、隔离命名空间并限制可写路径的 bwrap 命令。
    return [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
        # OpenBLAS sizes its pool from sched_getaffinity, which still sees every
        # host CPU inside the cgroup; with TasksMax=32 the extra threads fail
        # to spawn and numpy segfaults.  CPUQuota is one core anyway, so a
        # single BLAS thread is the only setting that can use the budget.
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--setenv",
        "MKL_NUM_THREADS",
        "1",
        "--setenv",
        "NUMEXPR_NUM_THREADS",
        "1",
        "--setenv",
        "HOME",
        "/nonexistent",
        *binds,
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--chdir",
        "/work",
        interpreter,
        "-I",
        "-B",
        "-",
    ]


def run_python(
    code: str,
    *,
    timeout_seconds: float = 5.0,
    memory_mb: int = 512,
    tasks_max: int = 32,
    output_limit: int = 512,
    code_limit: int = 64 * 1024,
) -> SandboxResult:
    """Execute model-generated Python in the local sandbox.

    A timeout never raises into a rollout worker.  A non-timeout setup error is
    kept distinct from a timeout so diagnostics cannot mislabel host failures
    as a model behaviour.  ``output_limit`` limits only data retained by the
    trusted rollout worker; the student-visible observation is bounded again
    by ``format_observation``.  Callers that compute a terminal reward should
    therefore request a modestly larger, reward-only limit than the 512-char
    observation limit.
    """
    # 先拒绝超长或非法输入，避免把异常代码交给沙箱进程。
    if not isinstance(code, str) or len(code.encode("utf-8", errors="ignore")) > code_limit:
        return SandboxResult(
            stdout="",
            stderr="program rejected: input exceeds sandbox limit",
            returncode=None,
            elapsed_seconds=0.0,
            timed_out=False,
            rejected=True,
        )
    if os.name != "posix":
        return SandboxResult("", "sandbox requires Linux", None, 0.0, False, True)

    # 将超时下限固定为正值，保证 timeout 命令参数合法。
    timeout_seconds = max(float(timeout_seconds), 0.1)
    # 用 systemd-run 施加内存、进程数、CPU 和文件大小限制。
    cmd = [
        "systemd-run",
        "--user",
        "--quiet",
        "--collect",
        "--pipe",
        "-p",
        f"TasksMax={int(tasks_max)}",
        "-p",
        f"MemoryMax={int(memory_mb)}M",
        "-p",
        "CPUQuota=100%",
        "-p",
        "LimitNOFILE=64",
        "-p",
        "LimitFSIZE=1M",
        "-p",
        "LimitCORE=0",
        "timeout",
        "--signal=KILL",
        "--kill-after=1s",
        f"{timeout_seconds}s",
        *_bwrap_command(),
    ]
    started = time.monotonic()
    try:
        # 同步等待隔离程序退出，并捕获 stdout 与 stderr。
        completed = subprocess.run(
            cmd,
            input=code,
            text=True,
            capture_output=True,
            timeout=timeout_seconds + 3.0,
            check=False,
            env=_systemd_environment(),
        )
        elapsed = time.monotonic() - started
    # 宿主等待超时时返回结构化结果，不让异常中断 rollout worker。
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        return SandboxResult(
            stdout=_truncate((exc.stdout or ""), output_limit),
            stderr=_truncate((exc.stderr or ""), output_limit),
            returncode=None,
            elapsed_seconds=elapsed,
            timed_out=True,
        )

    # ``systemd-run --pipe`` reports 255 when the inner GNU timeout kills a
    # transient unit.  Requiring near-budget elapsed time prevents immediate
    # systemd/bwrap configuration errors from being misclassified as a timeout.
    # 结合退出码和耗时区分真实超时与沙箱配置错误。
    timed_out = completed.returncode in {124, 137} or (
        completed.returncode == 255 and elapsed >= timeout_seconds * 0.9
    )
    return SandboxResult(
        stdout=_truncate(completed.stdout or "", output_limit),
        stderr=_truncate(completed.stderr or "", output_limit),
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
        timed_out=timed_out,
    )
