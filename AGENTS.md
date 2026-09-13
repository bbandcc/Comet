# Comet 协作开发规范

## 适用范围

- 本文件约束整个仓库；子目录中的 `AGENTS.md` 可补充更具体的规则。
- 目标是在不破坏现有产品能力的前提下，逐步提高执行可靠性、证据完整度与可恢复性。
- 每轮只完成一个可独立测试、提交、回退的切片。禁止把多个优化阶段一次性混改。

## 固定优化路线

1. S1 任务锁所有权：owner token、原子释放、锁异常显式失败。
2. S2 工具执行契约：统一结果、参数校验、缓存语义、停止原因与 MCP 生命周期。
3. S3 审稿与证据：严格状态、完整证据、断言级修复。
4. S4 可恢复执行：持久化 occurrence、派发状态与阶段 checkpoint。
5. S5 记忆与上下文：缓存修订、时态检索与统一 token 预算。
6. S6 RAG：父块去重、批量读取、降级策略和可对照的融合实验。

顺序可因新证据调整，但必须先说明依赖、风险和验收方式。尚未通过测试的设计不得写成已实现能力。

## 每个切片的工作顺序

1. 先读取目标文件、调用链、现有测试和相关配置，确认真实行为。
2. 优先查找项目已有实现、固定依赖版本源码和官方一手资料；只借用必要机制，不复制完整框架。
3. 写清不变量、失败时序、改动边界和非目标；复杂工作维护 `task_plan.md` 与证据笔记。
4. 先建立能确定复现问题的回归测试，再做满足契约的最小实现。
5. 先跑定向测试，再跑相关测试集、静态检查和必要的集成测试。
6. 审查完整 diff，确认没有调试输出、密钥、生成物和无关改动后再提交。

遇到测试失败必须记录原因。不得通过删除断言、扩大异常吞噬或降低正确性门槛让测试变绿。

## 架构与代码约束

- 后端保持 `Controller → Service → Repository → Model/DB` 单向依赖；跨模块基础设施放在 `core/`，Celery task 只负责编排。
- 前端按页面、组件、API 与 store 分责；协议字段变更必须同步类型和消费者。
- 优先复用成熟依赖提供的深层机制；项目封装只暴露业务需要的窄接口。
- 避免超大文件、万能工具类、跨层调用和重复状态机。新模块应职责单一，数据结构精简且类型明确。
- Python 新代码必须有类型标注；TypeScript 不引入无边界的 `any`。
- 正确性关键的基础设施失败必须显式表达。可降级副作用不得掩盖主流程失败。
- 日志记录业务 ID、attempt、状态与异常类型；不得输出 API Key、owner token 或用户敏感正文。
- 不顺手重构无关代码，不新增未被当前切片证明必要的框架或抽象。

## 测试与验证

后端默认从 `api/` 执行：

```powershell
uv run python -m unittest discover -s tests -v
uv run ruff check app tests
```

前端改动必须从 `web/` 使用 `package.json` 中已有脚本验证类型、构建和相关交互。

涉及并发、重试、超时、缓存或状态迁移时，测试必须控制时序并断言后置状态，不能只靠大量随机重复。依赖 Redis、数据库或浏览器的测试应使用隔离命名空间和独立测试数据，不操作开发者现有数据。

## 当前切片：S2b 参数、能力范围与停止原因

- FC 与 ReAct 必须经同一 schema 校验后才能构造 `ToolCall.validated_args` 并执行工具；校验失败
  不执行，可在有界次数内回灌模型纠正。ReAct 支持 JSON object，并仅为单 `query` 工具保留
  纯文本兼容。
- builtin 与 MCP 合并后的最终工具集合必须统一经过 capability filtering；非空 Skill 白名单当前
  只能授权明确列出的 builtin，因此保守拒绝并跳过加载 MCP。无 Skill 时保持用户原有启停配置。
- Agent 终态固定为 `completed / failed / cancelled / budget_exhausted`，并携带
  `stop_reason`；失败或预算耗尽可携带 `partial_answer`，ReAct 的 Thought/Action 协议文本不得成为
  final 或 partial。单聊和群聊前端必须按终态收尾，不能把非 completed 显示为成功。
- 独立限制单次工具 timeout、参数纠错次数、模型请求次数和总 deadline；时间判断使用 monotonic
  clock。无工具流式模型同样受 deadline 约束，但下游 SSE/Redis 消费时间不计入模型等待预算。
  S2a 的 `retryable` 只作元数据，不新增自动重试，写工具 timeout 后不得自动重试。
- 单个 `ToolOutcome=error` 不等于 Agent failed；只要预算仍允许，就把明确错误回灌模型继续。
- `tool_schema_invalid` 属于不可纠正的基础设施错误，必须直接 failed，不消耗参数纠错预算。
- 单聊与群聊只最小同步终态、停止原因和 partial 持久化字段，不大改现有 SSE 事件协议。
- 继续复用 S2a `ToolExecutor` 与既有 FC/ReAct 路由，不重设计 outcome/cache，不增加 parallel
  tool calls、数据库表、Agent framework 或无关重构。
- 本切片不修改 MCP `AsyncExitStack` / session ownership 或其他 lifecycle 行为（留给 S2c），
  不处理 Research/Verifier/evidence（S3）、occurrence/checkpoint（S4）、Memory（S5）或 RAG（S6）。

## Git 规则

- 尊重工作区已有改动；只暂存当前切片相关文件。
- 每次提交必须功能完整、业务可用、测试通过，禁止提交半成品或故意失败的测试。
- 使用 Conventional Commits，标题说明用中文，例如：`fix(runtime): 修复定时研究任务锁的所有权校验`。
- 不提交 `.env`、凭据、测试缓存、构建产物或本地研究快照；除非用户明确要求，不执行 push。
- Windows 环境统一使用 PowerShell 7（`pwsh.exe`）。
