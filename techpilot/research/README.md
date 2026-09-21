# M1：固定本地资料到可校验报告

本模块实现 M1 最小离线闭环；报告可读性修订已实现，仍待用户审阅验收。
最初的一句话资料只够验证引用链，不能据此宣称完整报告质量已达标。
当前演示使用三个官方 GitHub 固定提交文档的中文整理资料卡，统一比较用例写法、
测试发现、运行入口；保留短原文对照，明确非官方译文。
建议受示例中的写法偏好限定，不是完整功能排名或模型调研能力证明。
资料收集是开发准备工作；运行时不访问 GitHub、
搜索服务或真实模型，普通 Chat 配置与工具集不变。

## 职责与事实来源

- `contracts.py`：不可变调研任务、来源快照、行范围证据、结构化报告、纯函数引用校验。
- `rendering.py`：问题与约束、结论摘要、同维度表格、未知事项及引用附录。
  短引用按正文可见顺序编号，重复证据复用编号；读者编号不替代稳定证据 ID。
- `workflow.py`：任务记录、快照存储、指定版本的大小写不敏感逐行字面查询、报告交付。
  业务事实存于 `store/task.json` 和 `store/snapshots/<snapshot_id>.json`。
- `runtime_files.py`：固定 Provider 响应驱动真实 Runtime 的 read/write 工具，沿用
  RepositoryToolExecutor、Permission、Trusted Diff、Session 及工具状态检查。
- `host_files.py`、`tools.py`：宿主限定资料、store 与报告目录；提供导入、证据查询、
  按行补读和报告提交四个业务工具，不在工具内部创建 Provider 或嵌套 Runtime。
- `runtime.py`：`build_research_runtime(...)` 显式装配独立 Research Runtime。证据读取自动
  允许，导入快照与提交报告必须由宿主确认；普通 Chat 默认工具集不变。
- `scripts/run_m1_demo.py`：显式离线入口，仅预批准新演示目录内 store/reports 的新文件
  写入；不会把该 prompt 安装到普通 Chat，不增加权限，不改变 C3/C4/C5 或 M0 评测。

快照摘要是传入正文 UTF-8 的 SHA-256，ID 绑定来源与摘要，不随导入时间变化。
契约保留传入正文；文件导入器把 ReadFileTool 编号视图规范化为 LF 加末尾换行后建
快照，因此不是上游原始字节摘要。固定提交和摘录方式见 `tests/fixtures/research/SOURCES.md`。
导入器自动按 2,000 行读取完整来源；每页绑定同一源文件 SHA-256，后续页变化即拒绝。
空文件、UTF-8 解码替换、页事实不连续、超过 20,000 行或 2 MB 的来源都拒绝导入，
不生成有效快照。

同来源同内容复用已存快照，变化产生新快照，旧版本保留。重新打开 store 可读取旧
快照，但不恢复业务执行进度、授权或副作用（跨会话编排属于 M2）。证据位置为
1-based inclusive 行范围；报告正文用 `[1]` 等短引用，附录显示本地快照摘录、
行号和可点击来源。资料卡摘录不是上游官方原文。
比较单元格显式绑定 candidate 与维度；拒绝缺失、重复或不属于任务的候选项。
缺信息必须保留未知单元格，不能通过删除候选项制造完整比较的假象。
未提供的信息显示“未知/证据不足”。来源文本只作数据，不能生成工具指令。

## 成功与失败边界

`ready_for_delivery` 只是校验结果。报告只写任务声明的产物目录，已有文件不覆盖。
交付依次执行：实际快照校验 → 受控写入 Markdown 并回读 → 写入 JSON 校验记录并回读。
每次受控操作都要求明确 completed、成功 Runtime 结果和 Session 持久化检查。
全部通过才返回 `ReportDelivery`，演示才打印 `Delivered`。

JSON 保存正文摘要、快照、结论、证据和验证结果，状态为 `validated`，而不是
`delivered`：写入可能已生效，随后的 Runtime/Session 检查仍可能失败。文件存在或
校验记录存在不等于交付成功。失败可留诊断产物，不自动删除、重试；检查后换新产物名。
当前是单 writer，多文件写入不是事务，不提供并发安全存储。

报告校验记录升级为 schema_version 2，保存标题/问题/约束、candidate、references
（读者编号→完整 evidence_id）及 snapshot_files（版本→相对快照路径）。完整 ID/摘要
不再占据正文。快照 schema 仍为 1，旧报告和记录不重写；当前没有报告恢复读取器，
后续实现时须显式区分 v1/v2，不能把显示编号当作业务主键。

校验器检查引用、重复 ID、来源/版本、实际使用清单、行范围、原文和路径。
它不证明原文在语义上支持结论，也不证明远端真实性；哈希不是数字签名。
来源和结论转义为 Markdown 文本。路径检查不是 OS 沙箱，仍依赖 Runtime 写入时
检查与宿主文件系统信任边界。

## 演示与验证

仓库根目录执行，每次默认使用新的 `.tmp/m1-demo-*` 子目录：

```powershell
.\.venv\Scripts\python.exe scripts/run_m1_demo.py
.\.venv\Scripts\python.exe -m pytest -q tests/techpilot/test_research_contracts.py tests/techpilot/test_research_workflow.py tests/techpilot/test_research_rendering.py
.\.venv\Scripts\ruff.exe check techpilot tests scripts
```

输出包含 store、sessions、reports，报告与 JSON 校验记录的绝对路径打印到终端。
三个 M1 测试文件已纳入 CI；兼容性组包含 `test_tool_boundaries.py`。
2026-09-20 可读性修订：M1 测试 48 passed、1 skipped，覆盖矩阵完整性、短引用导航、
编号映射、转义及原有失败边界。全部 36 个测试文件按六组回归：550 passed、3 skipped，
退出码均为 0；Ruff、compileall、wheel 构建通过。报告摘要、9 个编号映射及快照路径已核对。
符号链接测试受系统权限限制。此前单进程全量的 KeyboardInterrupt 根因仍未定位。
## 下一阶段方向（2026-09-20 已确认，尚未实施）

M1 保留为基础设施与确定性回归；下一切片为 M1.5“面向真实代码库的技术决策验证”，
不默认先扩建 M2 Role/Skills。首题是 research 业务任务、快照和报告版本继续 JSON 文件
还是引入 SQLite；这是调查问题，不是已决定迁移，也不涉及 Session/Run 存储替换。

先确认实际约束与验收规则，使用完整版本化本地资料；后续获实施与运行授权后由真实
模型决定调查步骤，必要时在独立目录做可复现实验，形成有前提、取舍、代价和未知项的决策。
区别事实/推断/建议和支持/反驳/背景证据；引用正确不等于结论正确，关键判断需实验或人工复核。
固定响应演示继续保留，真实模型质量与 Runtime 契约结果分开报告，以实际失败决定
M2/M3/M4 优先级。完整工作卡见 `docs/M1_5_真实技术决策验证_2026-09-20.md`（被 Git 忽略）。

本轮仅更新文档，未启动开发、模型调用或产品数据迁移；模型、资料发送范围和预算仍待确认。
M0 冻结评测与现有安全合同不变，不自动暂存、提交、推送或同步 Notion。
该方向调整的个人知识库影响为需要更新，主题为技术决策 Agent 与业务验收；
原因是下一阶段职责和验收方式改变，证据为上述工作卡及本节；收件箱待记录。

## 工具设计补充（2026-09-21，Bash 首版已实现）

对照用户提供的两个 Claude Code 第三方重建仓库及本地代码后，完整设计记录在
`docs/工具设计_Chat与M1_5_2026-09-20.md`（Git 忽略）；两仓库四个核对文件内容相同，
不作为两套独立产品证据，也不视为官方实现保证。

- 当前已有原生只读工具并行及显式 ToolResult 状态；Chat 的严格白名单 Git 查询也可并行，其余 Bash 仍独占，并固定以仓库根执行。
- 目标是统一校验后的调用事实、权限与实际执行目标；按单次命令判断 Bash 并发，而不是按工具名或权限白名单判断。
- 动态分类须对应实际 Shell、参数、资源和 cwd；未知命令、测试/构建保持独占，SAFE 不绕过权限。
- 已实现 `PreparedToolCall`：单工具与多工具轮次共享一次准备得到的原始参数、规范化参数和调度事实；外来或参数不一致对象会重新准备，批准仍只在执行时检查。
- 先保留默认 cwd 语义；若以后支持会话 cwd，宿主拥有状态，屏障后重新分析剩余调用，不依赖 thread-local 或 cd 文本推断。
- 已实现 Bash 结构化结果：`ToolResultFacts` 经 `result_facts` 事件和 Session 保存退出码、cwd、
  stdout/stderr、超时与截断事实；模型仍只收到兼容的结果文本。当前输出被截断时不保存不完整
  流，明确标记 `truncated=true` 且没有 `output_artifact`，直到受控产物服务落地。
- 已实现完整来源受控分页：`read_file` 的 `result_facts` 保存源 SHA-256、页范围、总行数、
  下一页与 UTF-8 状态；分页导入会回传摘要校验版本，不能将截断或混合版本内容冒充完整快照。
- 已实现可宿主注入的最小 research 工具集：导入批准资料、分页查询证据、按行补读上下文、
  校验已发出证据后提交新报告。工具不创建 Provider、不嵌套 Runtime，普通 Chat 默认不启用。
- 拟议最小工具为来源导入、证据查询、上下文读取、报告提交；导入可先由宿主预处理，其余供真实模型自主调用。
- 生产工具复用受控 IO 边界，不在内部嵌套 FixedToolProvider/Runtime；演示适配器继续保留做回归。
- 下一步先完成完整资料与最小 research 接入的实施卡；Bash 并发的首个窄切片已完成：无 Shell 元字符的受限 `git status`/`git diff`、`git rev-parse --show-toplevel`、`git branch --show-current` 可并行。测试、构建、`rg`、路径或修订参数和未知命令仍独占，不阻塞首个真实任务。

本补充已实现 Bash 并发窄切片、Bash 结构化结果、完整资料分页和可显式装配的 research 接口；
受控完整输出产物或会话 cwd 尚未实现。2026-09-21 分页/研究工作流回归为 124 passed、2 skipped；
Ruff 与无字节码导入检查通过。个人知识库影响：需要更新；相关主题为
工具调度与权限、Bash/cwd、结构化结果和业务完成边界；证据见设计稿及现有
`engine/tool_execution.py`、`chat/executor.py`、`engine/tool_results.py`；收件箱待记录。

以下是 M1 实现已有的知识库影响记录：

个人知识库影响：需要更新。
相关主题：版本快照与引用、报告领域模型、读者表示与机器校验边界。
变化：比较单元格增加候选方案归属；短引用与稳定证据 ID 分离；校验记录升级 v2。
原因：不仅是排版，改变了报告结构和持久化表示，后续恢复需辨别版本。
代码与测试证据：contracts.py、rendering.py、workflow.py 及三个 research 测试文件。
收件箱状态：待记录（本次未向 Notion 写入）。
