# Goal 单例试评：模型可见合同 v1（2026-10-11）

本目录是 `goal-model-pilot-v0` 评测器的新题卡版本，不覆盖 `benchmarks/goal_model_v0/deck.json` 或旧运行证据。v1 摘要为 `31f1f3d29d034e3ca8a6687d05f13b9b975ea58256d6a5f54dd3c85764c9f6cd`；旧 v0 摘要仍为 `2b210012e5638df0e7a28910e8fbdbba332eb7a3fada6535c1ae5bc168b06959`。功能判据相同；新夹具 `benchmarks/cli_data_tool_contract_v1/` 只复制题卡列出的八个文件。

## 合同变化

- Goal 增加两条持久硬约束：只可写四个候选文件；`tests/`、`TASK.md`、`pyproject.toml` 只读。当前驱动器允许无提示的 pytest/Ruff 验证；需要人工提示的直接 CLI 命令会被拒绝，独立外部验收另行运行真实 CLI。
- `TASK.md` 和三个案例的 Goal 文本也提示相同边界。`implement` 阶段下一步写清使用现有测试、不要编辑冻结测试。
- 四个评测入口支持显式 `--deck benchmarks/goal_model_contract_v1/deck.json`，且 manifest 用题卡与夹具的摘要防止混用；默认入口仍读取 v0。

## 单例运行结果

隔离目录：`.techpilot/goal-model-contract-v1/2026-10-11-glm-53-normal-01/`，模型 `glm-5.3`。只运行 `normal-json-export`；首次默认沙箱请求连接失败，0 工作回合，未收到用量。随后在确认隔离夹具不含密钥和个人数据后运行模型。`start`、`after-batch`、`final` 三个协议边界及独立外部评分均已保存。已知用量下界为 **132,527 Token**；失败请求可能没有用量回报。此数加上此前两轮已知下界 849,691，为 **982,218 Token**，不能当作完整账单。

`inspect` 写入 `analysis.md` 并通过阶段检查；模型没有修改冻结文件，也没有请求被拒绝的直接 CLI 命令。`implement` 回合没有写代码，阶段检查正确阻塞。模型在阶段切换和普通 Chat `continue` 中多次提交超过运行时 500 字限制的重规划摘要，或只表示“等待确认”；一次普通 Chat `continue` 后仍未实施。Goal 最终 `paused`，`inspect` 已验证、`implement` 阻塞，无完成声明。独立验收失败：默认 text 仍正确，`--format text/json` 和 README 示例缺失。失败归因记录为 `planning`，这是本轮证据归类，不表示模型单独承担全部原因。

报告：`reports/contract-v1-normal-final.json`。该报告复用三例模板，因此另外两例为未运行、`protocol_evidence_complete=false`；这不表示已运行单例缺少 start/final 证据。旧版三例和 CRLF 回归结果不受影响。

## 当时的下一切片（已执行）

1. 对齐 `propose_goal_replan` 的 schema 与 Host：为 `summary` 明示 500 字上限，并在只读重规划提示中要求一个简短、单次的决定，减少无效循环。
2. 在已确认 Goal 的执行提示里明确：`/goal run` 或普通 Chat 的继续指令已授权当前阶段，模型应执行可执行的下一步；需要真正的人类确认时须指出具体未获授权的动作。保留权限拒绝、阶段检查及错误副作用不重试规则。
3. 用新鲜、再次冻结的单例验证阶段是否能从 `inspect` 进入 `implement`，分别报告功能验收和 Goal 自行完成。不能用本轮工作区补改代码后冒充这次模型成绩。

这次合同变化只解决模型是否看得到试评边界；单例结果说明剩余瓶颈转到阶段切换与重规划交互，尚不能宣称 G5 完成。

## 阶段提示与完成证据回归（2026-10-11）

在上述失败样本之后，Runtime 给 `propose_goal_replan.summary` 增加模型可见的 500 字符上限，并在只读重规划提示中要求简短决定；已确认阶段的执行上下文与 `/goal run` 下一阶段输入明确要求继续做可执行工作，同时说明工具权限仍逐项生效。Chat、连续执行、重规划的定向测试随后为 22/22 通过。

新鲜 normal 案例目录 `.techpilot/goal-model-contract-v1/2026-10-11-glm-53-stage-prompt-02/` 复用相同题卡摘要。模型从 `inspect` 进入 `implement`，修改了允许的源文件和 README，pytest/Ruff 与独立外部验收通过；两个阶段均已验证。但第二批 `submit_goal_completion` 引用了 `edit_file-exporter.py`、`bash-pytest` 这类自造标签，原 Host 只接受当前回合真实 `call_...` ID，因此没有持久完成声明，Goal 最终暂停。报告 `reports/stage-prompt-final.json`，已知用量下界 **138,978 Token**。这例证明了阶段执行提示改善了实际代码推进，仍不能称作 Goal 自行完成。

据此，完成提案工具现在要求精确的当前回合 Tool Call ID；调用时即检查持久 Session 工具事件，错误时返回可用 ID，允许模型在同一回合修正。回合结束仍独立核对提案必须是最后一个成功工具调用、所引证据确属该回合。修复后相关 Chat、连续执行、重规划测试 **80/80 通过**，包含“伪造 ID 被即时拒绝、真实 ID 重试成功”与“旧回合真实 ID 不可复用”的回归用例。

第三个全新 normal 案例 `.techpilot/goal-model-contract-v1/2026-10-11-glm-53-evidence-03/` 同样复用题卡摘要。第一批完成并验证 `inspect`、`implement`；第二批模型第一次提交 `bash`、`read_file` 这类非真实 ID，工具立即拒绝并列出真实 `call_...` ID，模型随即在同一回合重试。**Runtime 最终验证通过并将 Goal 标为 `succeeded`，独立外部验收全部通过。** 报告 `reports/evidence-feedback-final.json`，已知用量下界 **156,518 Token**，两批、三个工作回合，无额外人工提示。该报告里的 `protocol_evidence_complete=false` 来自未运行的另外两个场景；已运行 normal 案例自身没有缺失边界证据。

截至此轮，原三例、CRLF 回归与三个合同 v1 单例的 Provider 已报告用量下界合计 **1,277,714 Token**；失败请求可能缺少用量报告，不能作为完整账单。此次结果是一个 normal 场景的成功回归，不可推广为一般成功率，也不替代恢复和约束场景的验收。

## 恢复与约束试评（2026-10-11）

协议新增显式 `cross_process` 恢复验收模式：新的 `before_restart` 边界必须选模式，要求不同进程、同一 Task/Session、耐久 `session_resumed`、保留已核验 `inspect`，并准确指出 `implement`。未标记的旧记录仍按原 24 小时口径解释；题卡摘要及旧报告没有改写。协议校准 6/6 通过。

| 隔离运行 | 结果 | 已知 Token 下界 |
| --- | --- | ---: |
| `2026-10-11-glm-53-restart-04` | 同日跨进程恢复和下一步卡准确，独立 CLI 验收全部通过；模型先提交虚构完成证据 ID、随后纠正，但当时阶段评估仍把被拒绝的控制提案算作失败工具，`implement` blocked，Goal 暂停。报告 `reports/restart-final.json`。 | 173,868 |
| `2026-10-11-glm-53-constraint-05` | 题卡硬约束逐字持久化，`inspect`/`implement` 核验，Goal `succeeded`，外部验收含非 ASCII JSON 全部通过。首次约束提案被模型改写，操作员取消并要求逐字重提，记录 1 次额外人工干预。报告 `reports/constraint-final.json`。 | 235,669 |
| `2026-10-11-glm-53-restart-06` | 同日跨进程恢复和下一步卡准确；恢复后模型未执行任何 `implement` 工具，阶段被正确阻断，外部验收失败。报告 `reports/restart-planning-final.json`。 | 45,542 |
| `2026-10-11-glm-53-restart-07` | 首次 `inspect` 漏写文件，自动重规划给出安全重试，额外普通 Chat 续推后 `inspect` 核验；同日跨进程恢复准确。恢复后的工作与重规划回合都耗尽 4,096 输出 Token，返回空文本和零工具调用；Goal 暂停，外部验收失败。报告 `reports/restart-next-action-final.json`。 | 48,216 |
| `2026-10-11-glm-53-restart-08` | 仅在此隔离试评进程把输出上限提高到 16,384；首个 `inspect` 回合触及题卡的 180 秒时限，留下未完成的只读 `glob` 调用，未达到重启边界。报告 `reports/output-limit-final.json`；没有可信的完成回合用量，0 只是下界，不能解读为免费。 | 0（不完整） |

试评中修正了两处 Runtime 交接：被拒绝、无文件副作用的 `submit_goal_completion` 不再单独阻断已有成功工具证据的文件阶段；失败的文件写入或未知副作用仍阻断。只读重规划现在直接看到阶段的耐久评估状态与原因，并说明 `keep` 不会解除 blocked；`/goal run` 首回合传入阶段 ID 和具体 `next_action`。相关定向校准 44/44 通过。上述修复均未追改已冻结的模型成绩。

目前 normal 与约束案例各有新鲜成功样本；恢复能力本身有准确跨进程证据，但恢复后自主执行到 Goal 完成尚未稳定，**G5 不结项**。各报告的 `protocol_evidence_complete=false` 主要因为每个隔离目录只实际运行一个案例，不能当成该已运行案例缺少边界证据；`restart-08` 自身确实缺少恢复和连续执行边界。已知历史与本轮 Provider 用量下界合计 **1,781,009 Token**，失败请求和约束分类调用可能没有完整用量，不能当作账单或成功率估计。

**下一步：**先处理 `glm-5.3` 在输出上限耗尽时返回空文本/零工具调用的情况，使 Runtime 明确记录输出预算停止，避免将它误判为普通阶段失败；再协调单次输出上限与回合时限，在新版本题卡中冻结配置。随后做全新同日跨进程恢复试评，以 Goal 自行 `succeeded`、独立 CLI 验收通过、状态卡准确及干预次数可审计作为验收门。旧样本只作失败分析，不补改候选文件或重算成绩。
