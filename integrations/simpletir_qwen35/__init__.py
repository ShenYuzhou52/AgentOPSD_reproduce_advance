"""Qwen3.5 + SimpleTIR integration overlay.

This package is intentionally kept outside the vendored trainer.  It is loaded
through Verl's ``agent_loop_config_path`` so that the experiment can pin the
upstream trainer revision and audit every local change.
"""

from .trajectory import extract_python_fence, score_simpletir_math

__all__ = ["extract_python_fence", "score_simpletir_math"]
