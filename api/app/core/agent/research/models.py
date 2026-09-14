"""深度研究的中间数据模型（阶段间用明确结构传递，不靠隐式约定）。"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# 来源类型
SOURCE_WEB = "web"  # 联网搜索 + 抓正文
SOURCE_KB = "kb"  # 用户知识库
SOURCE_MCP = "mcp"  # MCP 工具产出


@dataclass
class PlanSection:
    """报告提纲中的一个章节。"""

    heading: str  # 章节标题
    points: str = ""  # 该章节要写的要点/角度（指导写作，不直接展示）
    sub_questions: list[str] = field(default_factory=list)  # 多视角子问题（v2）


@dataclass
class ResearchPlan:
    """规划阶段产出：报告标题 + 章节提纲 + 多角度子查询。"""

    title: str
    sections: list[PlanSection] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)  # 扁平化的检索子查询（供检索复用）


@dataclass
class Learning:
    """逐源提炼出的一条「要点」（v2 核心）：干净事实 + 绑定来源号。

    引用对齐提前到提炼阶段——写作时直接用带号要点，[来源N] 天然正确。
    """

    text: str  # 提炼后的干净要点（事实/数据/观点）
    source_index: int  # 绑定的来源号（对应 Source.index）
    date_hint: str = ""  # 提炼时识别到的时效（如 "2026-03"），无则空
    relevance: float = 0.5  # 与研究主题相关度 0~1，低于阈值丢弃


@dataclass
class CuratedSection:
    """大纲整理阶段产出：一个章节 + 核心论点 + 分配给它的要点编号。"""

    heading: str
    thesis: str = ""  # 本节核心论点（一句话，指导写作）
    learning_ids: list[int] = field(default_factory=list)  # 分配的 Learning 全局编号（1 起）


@dataclass
class Source:
    """一条可追溯的证据来源；index 仅是本次报告的展示编号。"""

    index: int  # 引用号（从 1 起，全文统一）
    type: str  # SOURCE_WEB / SOURCE_KB / SOURCE_MCP
    title: str
    content: str  # 完整保留的正文/检索片段/工具结果
    url: str | None = None  # web 源有 url；kb/mcp 可空
    stable_id: str = ""
    origin_ref: str = ""
    content_hash: str = ""
    content_ref: str = ""
    fetch_status: str = "full"
    truncated: bool = False
    retrieved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    call_id: str | None = None
    artifact_ref: str | None = None
    locations: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.content = (self.content or "").strip()
        if not self.origin_ref:
            self.origin_ref = self._default_origin_ref()
        if not self.stable_id:
            identity = f"{self.type}\0{self.origin_ref}"
            self.stable_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not self.content_ref:
            self.content_ref = f"evidence:{self.stable_id}:{self.content_hash}"

    @staticmethod
    def normalize_web_url(url: str) -> str:
        """规范化 URL，同时保留可表达内容版本的 query。"""
        parts = urlsplit((url or "").strip())
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        port = parts.port
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        path = parts.path or "/"
        if path != "/":
            path = path.rstrip("/")
        query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
        return urlunsplit((scheme, host, path, query, ""))

    def _default_origin_ref(self) -> str:
        if self.type == SOURCE_WEB and self.url:
            return self.normalize_web_url(self.url)
        location = json.dumps(
            self.locations, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return f"{self.type}:{self.title}:{location}"

    @staticmethod
    def mcp_origin_ref(tool_key: str, validated_args: dict) -> str:
        canonical = json.dumps(
            validated_args,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        args_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"mcp:{tool_key}:args_sha256:{args_hash}"

    def merge_evidence(self, other: "Source") -> None:
        """同一 origin 保留一个 citation；KB 合并块，其余来源采用最新内容版本。"""
        if self.stable_id != other.stable_id:
            raise ValueError("只能合并相同 stable_id 的证据")
        accepted_other_content = True
        if self.type == SOURCE_KB:
            parts = [part for part in (self.content, other.content) if part]
            unique_parts = list(dict.fromkeys(parts))
            self.content = "\n\n".join(unique_parts)
        elif (
            self.type == SOURCE_WEB and self.fetch_status == "full" and other.fetch_status != "full"
        ):
            # 抓取失败后的摘要 fallback 不能覆盖已取得的正文版本。
            accepted_other_content = False
        elif other.content:
            self.content = other.content
            self.fetch_status = other.fetch_status
        for location in other.locations:
            if location not in self.locations:
                self.locations.append(location)
        self.call_id = other.call_id or self.call_id
        self.artifact_ref = other.artifact_ref or self.artifact_ref
        if accepted_other_content:
            self.retrieved_at = other.retrieved_at
            self.truncated = self.truncated or other.truncated
        self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        self.content_ref = f"evidence:{self.stable_id}:{self.content_hash}"

    def as_brief(self) -> dict:
        return {
            "index": self.index,
            "type": self.type,
            "title": self.title,
            "url": self.url,
        }

    def as_persisted_evidence(self) -> dict:
        """内部持久化版本；公共 SSE/API 不得直接返回。"""
        return {
            **self.as_brief(),
            "stable_id": self.stable_id,
            "origin_ref": self.origin_ref,
            "content": self.content,
            "content_hash": self.content_hash,
            "content_ref": self.content_ref,
            "fetch_status": self.fetch_status,
            "truncated": self.truncated,
            "retrieved_at": self.retrieved_at,
            "call_id": self.call_id,
            "artifact_ref": self.artifact_ref,
            "locations": self.locations,
        }

    def as_verifier_evidence(self) -> dict:
        return {
            **self.as_brief(),
            "stable_id": self.stable_id,
            "origin_ref": self.origin_ref,
            "content_ref": self.content_ref,
            "content_hash": self.content_hash,
            "content": self.content,
            "fetch_status": self.fetch_status,
            "truncated": self.truncated,
        }

    def cite_label(self) -> str:
        """参考来源区的一行展示文本。"""
        if self.url:
            return f"[{self.title or self.url}]({self.url})"
        prefix = {SOURCE_KB: "知识库", SOURCE_MCP: "工具"}.get(self.type, "")
        name = self.title or "未命名来源"
        return f"{name}（{prefix}）" if prefix else name


@dataclass
class WrittenSection:
    """写作阶段产出：一个已写好的章节。"""

    heading: str
    content: str


@dataclass
class ResearchResult:
    """整篇研究最终产物（落库用）。"""

    title: str
    markdown: str
    sources: list[dict]  # 序列化后的来源列表
