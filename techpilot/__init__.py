"""TechPilot: a repo-aware coding agent workspace."""

__version__ = "0.1.0"

from .runtime.contracts import (
    RuntimeResultStatus,
    TaskRuntimeIdentity,
    TaskRuntimePaths,
    TaskRuntimeResult,
)

__all__ = [
    "RuntimeResultStatus",
    "TaskRuntimeIdentity",
    "TaskRuntimePaths",
    "TaskRuntimeResult",
    "__version__",
]
