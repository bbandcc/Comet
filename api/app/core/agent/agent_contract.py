"""Agent 单次运行的预算与终态契约。"""

from dataclasses import dataclass
from typing import Literal

from app.core.agent.tool_contract import ToolOutcome

AgentStatus = Literal["completed", "failed", "cancelled", "budget_exhausted"]

DEFAULT_MAX_MODEL_REQUESTS = 5
DEFAULT_MAX_PARAMETER_CORRECTIONS = 2
DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0
DEFAULT_TOTAL_DEADLINE_SECONDS = 120.0
MAX_TOOL_RESULT_PREVIEW = 600


def _event_args(raw_args: object) -> object:
    return dict(raw_args) if isinstance(raw_args, dict) else raw_args


def tool_start_event(
    call_id: str,
    tool_key: str,
    args: object,
    *,
    args_validated: bool,
) -> dict:
    """序列化工具开始事件；非法输入保留原值但不会伪称 validated args。"""
    return {
        "type": "tool_start",
        "call_id": call_id,
        "tool": tool_key,
        "args": _event_args(args),
        "query": args.get("query", "") if isinstance(args, dict) else "",
        "args_validated": args_validated,
    }


def tool_result_event(
    call_id: str,
    tool_key: str,
    args: object,
    outcome: ToolOutcome,
    *,
    args_validated: bool,
) -> dict:
    """以与 ToolOutcome 一致的状态序列化工具结果事件。"""
    content = outcome.content
    if len(content) > MAX_TOOL_RESULT_PREVIEW:
        content = content[:MAX_TOOL_RESULT_PREVIEW].rstrip() + "..."
    return {
        "type": "tool_result",
        "call_id": call_id,
        "tool": tool_key,
        "args": _event_args(args),
        "query": args.get("query", "") if isinstance(args, dict) else "",
        "args_validated": args_validated,
        "status": outcome.status,
        "text": content,
        "stats": outcome.stats,
        "latency_ms": outcome.latency_ms,
        "cached": outcome.cached,
        "error_code": outcome.error_code,
        "retryable": outcome.retryable,
        "artifact_ref": outcome.artifact_ref,
        "attempt": outcome.attempt,
    }


@dataclass(frozen=True, slots=True)
class AgentRunLimits:
    """一轮 Agent 的四类独立上限；不包含自动工具重试。"""

    max_model_requests: int = DEFAULT_MAX_MODEL_REQUESTS
    max_parameter_corrections: int = DEFAULT_MAX_PARAMETER_CORRECTIONS
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS
    total_deadline_seconds: float = DEFAULT_TOTAL_DEADLINE_SECONDS

    def __post_init__(self) -> None:
        if self.max_model_requests < 1:
            raise ValueError("max_model_requests 必须大于 0")
        if self.max_parameter_corrections < 1:
            raise ValueError("max_parameter_corrections 必须大于 0")
        if self.tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds 必须大于 0")
        if self.total_deadline_seconds <= 0:
            raise ValueError("total_deadline_seconds 必须大于 0")


@dataclass(frozen=True, slots=True)
class AgentTerminal:
    """Agent 终态；final_answer 与失败时可保留的 partial_answer 明确分离。"""

    status: AgentStatus
    stop_reason: str
    final_answer: str = ""
    partial_answer: str = ""

    def as_event(self) -> dict:
        return {
            "type": "final",
            "status": self.status,
            "stop_reason": self.stop_reason,
            "text": self.final_answer,
            "partial_answer": self.partial_answer,
        }


__all__ = [
    "AgentRunLimits",
    "AgentStatus",
    "AgentTerminal",
    "DEFAULT_MAX_MODEL_REQUESTS",
    "DEFAULT_MAX_PARAMETER_CORRECTIONS",
    "DEFAULT_TOOL_TIMEOUT_SECONDS",
    "DEFAULT_TOTAL_DEADLINE_SECONDS",
    "tool_result_event",
    "tool_start_event",
]
