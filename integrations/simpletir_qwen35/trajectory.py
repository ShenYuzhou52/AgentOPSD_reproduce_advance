"""Pure helpers for SimpleTIR's fenced-Python trajectory semantics.

本文件全是无状态纯函数，agent loop 与离线评测脚本（scripts/eval_baseline.py）
共用，保证训练与 baseline 的解析/打分语义完全一致。职责分三块：

- 动作解析：extract_python_fence / parse_turn / requires_sandbox 决定每轮
  "执行代码 / 终止 / 继续"；
- 观察构造：with_final_answer_helper / format_observation 生成学生可见文本；
- 终局打分：score_simpletir_math（唯一接触 ground_truth 的函数）。

安全边界：任何函数都不得把金答案格式化进学生可见消息；除打分器外无人
接收 ground_truth。
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
    """Non-secret parsing metadata for one model turn.

    is_void（既无代码又无 boxed）在 1024 预算时代是主要失败模式：回合在
    代码围栏中间被截断 → 围栏不闭合 → 解析不出代码 → 判为 void 并终止
    整个 episode。void_turn_ratio 指标因此直接反映截断级联的严重程度。
    """

    code: str | None
    has_boxed_answer: bool

    @property
    def is_void(self) -> bool:
        return self.code is None and not self.has_boxed_answer


# 从助手文本中抽取第一个 Python 代码围栏，作为可执行动作。
def extract_python_fence(text: str) -> str | None:
    """Return the first fenced Python program, matching SimpleTIR's action parser.

    取"第一个"围栏与上游动作解析器一致：一轮只执行一个动作，后面的围栏
    属于模型预写的后续计划，不执行。
    """
    match = _PYTHON_FENCE.search(text or "")
    return match.group(1).strip() if match else None


# 提取最后一个 LaTex boxed 表达式，作为候选数学答案。
def extract_last_boxed(text: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` expression without regex nesting limits.

    用手写括号配平而不是正则：\\boxed{\\frac{1}{2}} 这类嵌套花括号会让
    固定层数的正则失效。取"最后一个"是因为多轮轨迹里前面的 boxed 是中间
    结果，最终答案总在末尾。
    """
    start = -1
    marker = r"\boxed{"
    pos = 0
    while True:
        # 先扫完整个文本记住最后一次出现的位置，再从那里开始配平。
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


# 同时解析代码和 boxed answer，供循环判断下一步动作。
def parse_turn(text: str) -> TurnParse:
    return TurnParse(code=extract_python_fence(text), has_boxed_answer=extract_last_boxed(text) is not None)


# 只有含代码且并非纯最终答案的回合才进入隔离执行环境。
def requires_sandbox(parsed: TurnParse) -> bool:
    """Whether SimpleTIR must execute this turn's fenced code.

    Upstream marks a direct ``\\boxed{}`` answer terminal, but still executes
    a code action that occurs in the same model turn.  Keeping this rule in a
    pure helper makes the terminal/action ordering regression-testable.
    """
    return parsed.code is not None


def is_only_final_answer(code: str) -> bool:
    """Match SimpleTIR's half-credit check for a non-substantive tool call.

    判定"整段代码只是调 final_answer(x) 交答案"：先跳过一条纯字符串
    表达式（模型常写的文档字符串/说明），剩余必须恰好一条语句且是对
    final_answer 的直接调用。这种回合执行成功了也不算实质工具使用，
    正确时只拿半分——防止模型学会"不计算、直接背答案"的捷径。
    """
    try:
        stmts = ast.parse(code).body
    except (SyntaxError, ValueError, TypeError):
        # 语法错误的代码连 final_answer 都构不成，交由沙箱去报错。
        return False
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(getattr(stmts[0], "value", None), ast.Constant):
        if isinstance(stmts[0].value.value, str):
            stmts = stmts[1:]
    if len(stmts) != 1 or not isinstance(stmts[0], ast.Expr):
        return False
    call = stmts[0].value
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "final_answer"


# 为模型代码补充 final_answer 辅助函数，使标准答案输出可被捕获。
def with_final_answer_helper(code: str) -> str:
    """Add SimpleTIR's helper so generated code can print a canonical answer.

    helper 源码里的 "\\\\boxed{" 是四层转义的终点：本函数字符串含两个反斜杠，
    被沙箱里的 print 执行后输出单个反斜杠的 \\boxed{...}——与
    extract_last_boxed 的搜索标记一致（此处是历史上最易写错的一行）。
    """
    helper = 'def final_answer(result):\n    print("\\\\boxed{" + str(result) + "}")\n\n'
    return helper + code


# 把执行结果裁剪并格式化为学生模型下一轮可见的 observation。
def format_observation(*, stdout: str, stderr: str, timed_out: bool, limit: int = 512) -> str:
    """Return the bounded, student-visible result of a model-generated program.

    stderr 只保留最后一行：异常 traceback 的关键信息（错误类型与消息）在
    末尾，整段 traceback 对模型没有额外价值却挤占 512 字符预算。超时则只
    报 "interpreter timeout"，不给可能不完整的输出造成"算出来了"的假象。
    """
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


# 在工作线程中调用数学解析器，避免阻塞 rollout 事件循环。
def _parse_thread_safe(text: str) -> list:
    from math_verify import parse

    if _in_worker_thread():
        return parse(text, parsing_timeout=None)
    return parse(text)


# 在线程中验证候选答案与标准答案的数学等价性。
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


# 按 SimpleTIR 数学规则计算正确性与有效工具使用组成的奖励。
def score_simpletir_math(solution_text: str, ground_truth: Any, *, substantive_tool_use: bool) -> dict[str, float]:
    """Compute SimpleTIR-style binary math reward without emitting secret text.

    输入的 solution_text 是完整 episode 文本（各轮 assistant 文本 + 观察，
    对齐上游 hf_math_verify 的全文提取语义）。返回四个标量：

    - score：最终奖励 = answer_accuracy × (实质工具使用 ? 1 : 0.5)；
    - answer_accuracy：二值正确性，AgentOPSD 的 B0 与一致性校验依赖它；
    - is_boxed_ratio / substantive_tool_use：诊断比率，进 rollout 监控。

    解析失败或表达式非法就是普通的 0 分；绝不能把金答案写进日志来"帮助
    排查"（泄漏边界）。异常统一吞掉——打分器崩了会杀死整个 rollout worker。
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
