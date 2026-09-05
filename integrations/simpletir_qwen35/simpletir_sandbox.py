"""Local, unprivileged Python sandbox for SimpleTIR rollouts.

This deliberately does not use SimpleTIR's default HTTP sandbox endpoint.  A
model program is launched in a fresh bubblewrap namespace underneath a scoped
user systemd service; its only writable files are a new tmpfs and an empty work
directory.  The host process keeps just enough D-Bus environment to request the
user service.  ``bwrap --clearenv`` ensures none of that environment reaches
the untrusted code.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int | None
    elapsed_seconds: float
    timed_out: bool
    rejected: bool = False

    @property
    def ok(self) -> bool:
        return not self.rejected and not self.timed_out and self.returncode == 0


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def _systemd_environment() -> dict[str, str]:
    """Keep D-Bus routing outside bwrap; never pass broad host env to the program."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _bwrap_command() -> list[str]:
    binds: list[str] = []
    for path in ("/usr", "/lib", "/lib64"):
        if Path(path).exists():
            binds.extend(["--ro-bind", path, path])
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
        "/usr/bin/python3",
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

    timeout_seconds = max(float(timeout_seconds), 0.1)
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
