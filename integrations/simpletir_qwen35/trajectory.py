"""Pure helpers for SimpleTIR's fenced-Python trajectory semantics.

No helper in this module formats a gold answer into a student-visible message.
The only function that accepts ``ground_truth`` is the terminal reward scorer.
"""

from __future__ import annotations

import ast
import re
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
            target = parse(boxed_prediction)
            gold = parse(boxed_gold)
            correct = bool(verify(gold, target, 6, 15, True))
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
