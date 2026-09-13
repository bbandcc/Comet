"""工具调用与结果的纯数据契约。"""

from dataclasses import dataclass, field
from typing import Any, Literal

TOOL_READ_ONLY_METADATA_KEY = "comet.read_only"
TOOL_CACHEABLE_METADATA_KEY = "comet.cacheable"

ToolStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """一次逻辑工具调用；参数保留原工具的完整结构。"""

    call_id: str
    tool_key: str
    validated_args: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "validated_args", dict(self.validated_args))


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """工具执行后的规范化结果。"""

    status: ToolStatus
    content: str
    error_code: str | None = None
    retryable: bool = False
    artifact_ref: str | None = None
    attempt: int = 1
    latency_ms: int = 0
    cached: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


class ToolExecutionError(Exception):
    """工具可预期的业务失败，由共享执行器转换为 error outcome。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool = False,
        artifact_ref: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
        self.artifact_ref = artifact_ref


class ToolCallValidationError(ValueError):
    """工具名或参数未通过共享校验；调用方可把错误回灌给模型纠正。"""

    def __init__(self, message: str, *, error_code: str = "tool_validation_error") -> None:
        super().__init__(message)
        self.error_code = error_code


__all__ = [
    "TOOL_CACHEABLE_METADATA_KEY",
    "TOOL_READ_ONLY_METADATA_KEY",
    "ToolCall",
    "ToolCallValidationError",
    "ToolExecutionError",
    "ToolOutcome",
    "ToolStatus",
]
