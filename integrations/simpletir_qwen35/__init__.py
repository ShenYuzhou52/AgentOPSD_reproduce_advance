"""Qwen3.5 + SimpleTIR integration overlay.

本包是"overlay 而非 fork"的关键：所有实验代码放在钉版 Verl 之外，通过
``agent_loop_config_path`` / ``agent_loop_manager_class`` 等标准扩展点接入。
这样基座仓库可以保持原样升级，本地每一处改动都可审计。

从 trajectory 导出的两个函数是给外部脚本（baseline 评测等）复用的稳定
入口；其余模块一律从完整路径导入，避免包级 import 拖起 verl 依赖。
"""

from .trajectory import extract_python_fence, score_simpletir_math

__all__ = ["extract_python_fence", "score_simpletir_math"]
