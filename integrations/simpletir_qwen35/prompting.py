"""Reference-compatible, student-visible prompt construction for SimpleTIR.

唯一职责：把上游 SimpleTIR 的工具使用契约前缀拼到题目消息前。前缀逐字
复制自上游配置（含围栏代码格式、final_answer() 用法、\\boxed{} 答案格式
的完整说明），任何措辞改动都可能改变模型行为、破坏与上游的可比性。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Copied from SimpleTIR's ``recipe/simpletir/config/simpletir_trainer.yaml``.
# Its custom RLCustomPromptDataset prepends this text to every dataset message;
# modern Verl's RLHFDataset deliberately has no ``data.prompt`` mechanism.
SIMPLETIR_PROMPT_PREFIX = r'''Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (after "Code execution result: ") is returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.

Code Format:
Each code snippet is wrapped between ```. You need to use `print()` to output intermediate results.

Answer Format:
You can use the `final_answer()` function in the code to return your final answer. For example, to answer the User Question: What is the result of the 5 + 3 + 1294.678?, you can write:
```py
answer = 5 + 3 + 1294.678
final_answer(answer)
```

You can also use \boxed to return your answer. The last part of your response should be:
\boxed{'The final answer goes here.'}

User Question:
'''


def with_simpletir_prompt(raw_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the reference prefix exactly once, without mutating dataset rows.

    deepcopy 保证不污染数据集缓存里的原始行；幂等检查防止上层把同一条
    prompt 重复包装两遍。前缀拼到每条消息的 content 前沿——与上游
    RLCustomPromptDataset 的行为一致（实际数据集每行只有一条 user 消息，
    后续观察消息由 agent loop 追加、不经此函数）。
    """
    messages = deepcopy(raw_prompt)
    if not messages or not all(isinstance(message, dict) for message in messages):
        raise TypeError("SimpleTIR requires a non-empty chat-message list")
    if any(SIMPLETIR_PROMPT_PREFIX in str(message.get("content", "")) for message in messages):
        return messages
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("SimpleTIR chat message content must be text")
        message["content"] = SIMPLETIR_PROMPT_PREFIX + content
    return messages
