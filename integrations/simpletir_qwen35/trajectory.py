"""Pure helpers for SimpleTIR's fenced-Python trajectory semantics.

No helper in this module formats a gold answer into a student-visible message.
The only function that accepts ``ground_truth`` is the terminal reward scorer.
"""

from __future__ import annotations

import ast
import re
import threading
from dataclasses import dataclass
from typing import Any


_PYTHON_FENCE = re.compile(r"```(?:py|python)?\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class TurnParse:
    """Non-secret parsing metadata for one model turn."""

    code: str | None
    has_boxed_answer: bool

    @property
    def is_void(self) -> bool:
        return self.code is None and not self.has_boxed_answer


def extract_python_fence(text: str) -> str | None:
    """Return the first fenced Python program, matching SimpleTIR's action parser."""
    match = _PYTHON_FENCE.search(text or "")
    return match.group(1).strip() if match else None


def extract_last_boxed(text: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` expression without regex nesting limits."""
    start = -1
    marker = r"\boxed{"
    pos = 0
    while True:
        candidate = (text or "").find(marker, pos)
        if candidate < 0:
            break
        start = candidate
        pos = candidate + len(marker)
    if start < 0:
        return None

    depth = 0
    for index in range(start + len(marker) - 1, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_turn(text: str) -> TurnParse:
    return TurnParse(code=extract_python_fence(text), has_boxed_answer=extract_last_boxed(text) is not None)


def requires_sandbox(parsed: TurnParse) -> bool:
    """Whether SimpleTIR must execute this turn's fenced code.

    Upstream marks a direct ``\\boxed{}`` answer terminal, but still executes
    a code action that occurs in the same model turn.  Keeping this rule in a
    pure helper makes the terminal/action ordering regression-testable.
    """
    return parsed.code is not None


def is_only_final_answer(code: str) -> bool:
    """Match SimpleTIR's half-credit check for a non-substantive tool call."""
    try:
        stmts = ast.parse(code).body
    except (SyntaxError, ValueError, TypeError):
        return False
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(getattr(stmts[0], "value", None), ast.Constant):
        if isinstance(stmts[0].value.value, str):
            stmts = stmts[1:]
    if len(stmts) != 1 or not isinstance(stmts[0], ast.Expr):
        return False
    call = stmts[0].value
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "final_answer"


def with_final_answer_helper(code: str) -> str:
    """Add SimpleTIR's helper so generated code can print a canonical answer."""
    helper = 'def final_answer(result):\n    print("\\\\boxed{" + str(result) + "}")\n\n'
    return helper + code


def format_observation(*, stdout: str, stderr: str, timed_out: bool, limit: int = 512) -> str:
    """Return the bounded, student-visible result of a model-generated program."""
    if timed_out:
        body = "interpreter timeout"
    elif stderr:
        body = "\n".join(stderr.splitlines()[-1:])
    else:
        body = stdout
    body = body[-limit:]
    return f"\nCode execution result: {body}\n"


def _canonical_boxed(value: Any) -> str:
    text = str(value).strip()
    return text if text.startswith(r"\boxed{") else rf"\boxed{{{text}}}"


def _in_worker_thread() -> bool:
    """Whether signal-based timeouts are unusable in the current thread.

    Verl runs agent-loop coroutines on worker threads.  math_verify's default
    timeout installs a ``SIGALRM`` handler, which raises ``ValueError`` off the
    main thread; that exception was swallowed by the scorer's guard and turned
    every correct answer into a silent zero reward.  math_verify's documented
    threaded mode (``parsing_timeout=None`` / ``timeout_seconds=None``) skips
    the signal path; the caller-side watchdog in the agent loop bounds runtime.
    """
    return threading.current_thread() is not threading.main_thread()


def _parse_thread_safe(text: str) -> list:
    from math_verify import parse

    if _in_worker_thread():
        return parse(text, parsing_timeout=None)
    return parse(text)


def _verify_thread_safe(gold: list, target: list) -> bool:
    from math_verify import verify

    if _in_worker_thread():
        return bool(verify(gold, target, 6, 15, True, timeout_seconds=None))
    return bool(verify(gold, target, 6, 15, True))


def _python_style_fallback(gold_extractions: list, target_extractions: list) -> bool:
    """Compare still-string predictions symbolically after math_verify fails.

    ``final_answer(str(result))`` prints Python syntax (``5*x**2/2``); math_verify's
    latex-first parser keeps such content as an unparsed string, so an exactly
    equivalent answer scored zero.  Mirror the grader the upstream recipe uses
    (``sympy_expr_eq`` with the same rounding arguments as ``verify``) after
    ``sympify``-ing only the string extractions, never rewriting the gold side.
    Sympified symbols carry no assumptions while math_verify declares them
    real, so each symbol is aligned to the gold symbol of the same name before
    comparing; without that alignment structurally identical forms stay unequal.
    """
    from sympy import Basic, MatrixBase, Symbol, sympify

    gold_exprs = [value for value in gold_extractions if isinstance(value, (Basic, MatrixBase))]
    if not gold_exprs:
        return False
    gold_symbols = {symbol.name: symbol for expr in gold_exprs for symbol in expr.free_symbols}

    def _eq(gold_value, target_value) -> bool:
        if _in_worker_thread():
            from math_verify.grader import sympy_expr_eq

            try:
                return bool(sympy_expr_eq(gold_value, target_value, 6, 15, True))
            except Exception:
                return False

        from math_verify.grader import sympy_expr_eq
        from math_verify.utils import timeout

        @timeout(5)
        def _compare() -> bool:
            return bool(sympy_expr_eq(gold_value, target_value, 6, 15, True))

        return _compare()

    for target in target_extractions:
        if not isinstance(target, str):
            continue
        try:
            target_expr = sympify(target)
            target_expr = target_expr.subs(
                {symbol: gold_symbols[symbol.name] for symbol in target_expr.free_symbols if symbol.name in gold_symbols}
            )
        except Exception:
            continue
        for gold_value in gold_exprs:
            try:
                if _eq(gold_value, target_expr):
                    return True
            except Exception:
                continue
    return False


def score_simpletir_math(solution_text: str, ground_truth: Any, *, substantive_tool_use: bool) -> dict[str, float]:
    """Compute SimpleTIR-style binary math reward without emitting secret text.

    The current upstream run uses ``math_verify`` for symbolic equivalence.  A
    missing parser or malformed expression is a normal zero reward, not a log
    event containing the target answer.
    """
    boxed_prediction = extract_last_boxed(solution_text)
    boxed_gold = _canonical_boxed(ground_truth)
    correct = False
    if boxed_prediction:
        try:
            from math_verify import parse, verify
        except ImportError as exc:
            raise RuntimeError("SimpleTIR math reward requires the math-verify package") from exc

        try:
            target = _parse_thread_safe(boxed_prediction)
            gold = _parse_thread_safe(boxed_gold)
            correct = _verify_thread_safe(gold, target) or _python_style_fallback(gold, target)
        except Exception:
            correct = False

    raw_score = float(correct)
    reward = raw_score if substantive_tool_use else raw_score * 0.5
    return {
        "score": reward,
        "answer_accuracy": raw_score,
        "is_boxed_ratio": float(boxed_prediction is not None),
        "substantive_tool_use": float(substantive_tool_use),
    }
