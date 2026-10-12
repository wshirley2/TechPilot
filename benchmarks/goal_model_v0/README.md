# Goal model pilot v0

G5 的第一组真实模型任务合同。三个案例共用 `benchmarks/cli_data_tool` 的冻结初始文件，分别观察正常连续执行、第一阶段后跨进程恢复，以及第一阶段后新增已确认约束。共用代码基底有利于比较运行方式，但这只是三例试运行，不能据此估计一般任务成功率。

`deck.json` 固定 Goal、两阶段计划、最终 Runtime 验收、允许修改的路径、干预点和单回合限制，外部功能判据也计入题卡摘要。第一阶段的 `analysis.md` 与第二阶段的 `--format` 文件检查只证明阶段产物存在；独立验收还必须检查默认 text、显式 text、JSON 结构、不支持格式的失败状态和 README 示例。约束案例还必须检查非 ASCII 字符直接输出、未增加第三方依赖。模型不得修改作为外部判据的 `tests/` 文件。

准备入口不会调用模型：

```powershell
python -m techpilot.evaluation.goal_model_v0 --inspect
python -m techpilot.evaluation.goal_model_v0 --prepare .techpilot/goal-model-pilot-v0/<run-id> --model <model-id> --max-cost-usd <budget>
```

`--prepare` 要求新目录、明确模型和正的费用或 Token 总预算（分别使用 `--max-cost-usd`、`--max-total-tokens`），并生成类型为 `preparation` 的 manifest、每例隔离工作区及 case contract；准备入口不会覆盖已有目录。manifest 记录准备时解析到的 provider、温度、输出与上下文上限，以及自定义 Base URL 是否存在和其 SHA-256 摘要，不保存 Key 或完整 URL。`case_set_digest` 同时覆盖题卡与原始文件字节。预算是试运行管理上限；当前 Runtime 的限额按回合生效，不能称为整个 Goal 的费用硬上限。准备完成不代表模型已运行或题目通过。

当前 Chat 在确认 setup 后会立即执行首个工作回合，因此试评使用 `--setup-case` 将操作员已批准的冻结题卡直接写为 Goal setup，不调用模型。这保证三例有相同的两阶段计划、最终验收条件，并能在第一笔模型工作前留下 `start` 证据；这组三例评估**给定计划后的执行与恢复**，不把计划生成能力计入模型成绩。每例只能初始化一次：

```powershell
python -m techpilot.evaluation.goal_model_v0 --setup-case .techpilot/goal-model-pilot-v0/<run-id> --case-id normal-json-export
```

入口返回 Task ID 和 Session ID。Goal 在案例工作区运行时，将 Session 目录指定为该案例目录下的 `sessions/`。有 Task ID 后，可在每个阶段边界只读采集一份快照：

```powershell
python -m techpilot.evaluation.goal_model_v0 --collect .techpilot/goal-model-pilot-v0/<run-id> --case-id restart-json-export --task-id <goal-task-id> --label before-restart
```

快照写入该案例的 `observations/<label>.json`，同名文件不会被覆盖。它记录阶段、Checkpoint、停止原因、请求与完成的工具调用、恢复次数、压缩次数和可观测用量。`completed_turn_tokens_lower_bound` 只统计 Session 能回放的完成回合，不能当作完整费用。

候选代码写好后，独立运行外部功能验收：

```powershell
python -m techpilot.evaluation.goal_model_oracle .techpilot/goal-model-pilot-v0/<run-id> --case-id normal-json-export --label final
```

验收器从候选工作区读取一份文件快照，每条 CLI 命令都在单独的临时副本上运行，最多等待 10 秒。它关闭 Python 的用户环境与第三方包加载，核对冻结文件、允许的文件范围、README 示例、默认与显式 text 输出、JSON 的 `items/count`、非法格式退出码；约束案例额外核对 Unicode 原样输出。结果单独写入 `assessments/<label>.json`，同名记录不能覆盖。它不会调用模型，也不会把外部验收结果写成 Goal 的自评结论。临时副本保护原案例文件，但不是任意候选代码的安全沙箱；只应对可信的本地试评代码运行。失败归因仍需结合 Task/Session 快照填写。

2026-10-11 的本地校准用例 `tests/techpilot/test_goal_model_oracle.py` 为 7/7 通过：原始样例失败、三例人工正确样例通过、转义 Unicode 在约束案例失败、冻结文件与允许范围违规失败、同名记录不能覆盖，且评分前后候选工作区文件相同。这只是验收器校准，不是模型任务成功率。

执行协议在模型调用前冻结：三例各一次、每例最多两个 `/goal run` 批次与八个工作回合；每批次前检查最近记录的全套用量。恢复案例在 `inspect` 已验证后保存 Session 与 Task 状态，停止进程，再从原 Session 恢复，先核对状态卡再继续。约束案例在同一边界记录并确认新增约束，然后继续。外部验收与 Task 自评分别报告，失败归因为证据、规划、工具、权限、恢复、预算或模型执行之一。准备产物不能当作模型结果。

## 执行记录入口

`goal_model_protocol` 只读已有快照和外部验收，向 `protocol/0001.json` 等新文件追加人工边界记录；每条记录绑定证据摘要，报告会重新核对。它不会启动模型、修改 Goal 状态或自动批准下一批。

在无模型初始化已完成、第一笔工作回合前，先采集 `start` 快照并绑定 Task/Session：

```powershell
python -m techpilot.evaluation.goal_model_v0 --collect .techpilot/goal-model-pilot-v0/<run-id> --case-id normal-json-export --task-id <goal-task-id> --label start
python -m techpilot.evaluation.goal_model_protocol --record .techpilot/goal-model-pilot-v0/<run-id> --case-id normal-json-export --phase start --observation-label start --reported-suite-cost-usd <current-suite-cost>
python -m techpilot.evaluation.goal_model_protocol --budget-review .techpilot/goal-model-pilot-v0/<run-id> --case-id normal-json-export
```

若准备时采用 Token 预算，将记录命令中的 `--reported-suite-cost-usd` 换成 `--reported-suite-tokens <known-cumulative-tokens>`；失败请求没有用量回报时，这个数只能作为已知下界，并应在 `--operator-note` 说明。

试评可用 `python -u -m techpilot.evaluation.goal_model_driver <prepared-root> --case-id <case-id>` 打开同一产品 ChatSession；驱动器固定题卡的每回合限额，将命令逐行从标准输入读取，只自动批准允许路径内带可信 Diff 的文件写入，其他需要确认的操作拒绝。进入 Chat 后执行 `/goal resume <task-id>` 接手已确认的 Goal，再用 `/goal run` 推进。也可手工用 `python -m techpilot.cli chat <case>/workspace --model <model-id> --sessions-dir <case>/sessions --resume <session-id> --max-provider-calls 8 --max-tool-rounds 8 --max-turn-seconds 180`；手工路径由终端提示逐项审批。Session 的这次初始接手不计作恢复案例的中途重启；恢复案例仍须在 `inspect` 已核验后另行退出并跨进程恢复。不要使用旧版 `/goal plan load`、`/goal criterion` 或 `/goal start` 命令。

`--budget-review` 只会返回 `stop` 或 `manual_review_required`。Goal 不在 `running` 状态、当前阶段已阻塞、存在恢复阻塞或已达到批次、回合、已报告预算上限时返回 `stop`；后者表示尚未触及这些停止条件，仍须人工判断剩余预算能否承担下一批。阶段阻塞时仍可由操作员按产品规则明确尝试普通 Chat 单回合。预算复核不是费用硬上限或模型调用许可。每次 `/goal run` 后采集新快照，以 `--phase after_batch` 绑定，并记录**包括 setup 与普通 Chat 在内**的全套累计费用或 Token；最终边界也必须提供最新总量。协议以 `start` 之后新增的 Task Checkpoint 计算八个工作回合，超限仍保留记录并在报告中标记。

恢复案例在 `inspect` 阶段核验后先采集并记录 `before_restart`，填入当前进程实例标签；停止进程，隔天重新打开原 Session 与 Goal，先查看状态卡，再采集并记录 `after_restart`，填入不同进程实例标签、状态卡下一步原文及 `--next-step-accurate true/false`。协议会核对 Session 的持久恢复事件、原阶段证据和下一阶段。约束案例相应记录 `before_constraint` 与 `after_constraint`；后者必须在 Task 中看到题卡的约束原文。所有阶段标记各用新的快照标签。

候选工作区完成后运行独立验收，采集最终快照，再记 `final`，引用验收标签；外部验收失败时还须用 `--failure-attribution` 记录原因。最后生成不可覆盖的报告：

```powershell
python -m techpilot.evaluation.goal_model_protocol --report .techpilot/goal-model-pilot-v0/<run-id> --label draft
```

报告分别呈现 Goal 完成声明、外部验收、阶段数、重复工具调用、计划内干预、额外人工介入、用量和恢复后下一步。`protocol_evidence_complete` 只表示本试评的证据齐全且已报告预算未超；它不表示三例都成功，更不能推广为一般任务成功率。执行记录入口已用于三例 `glm-5.3` 试跑，并通过 5 项独立定向校准：快照绑定与篡改拒绝、Token 上限停止、恢复事件与 24 小时间隔、外部验收与候选文件绑定。正式报告及协议调整见 [G5 阶段记录](PILOT-2026-10-11.md)。

## G5 后续顺序与完成条件

1. **执行记录协议（已开发并完成 5 项定向校准）：** 固定模型版本、试运行总预算、每例 Task/Session ID、阶段边界、人工介入、每批次用量与停止原因；让三例记录可重放，并在累计预算不足时停止下一批次。当前 Goal 没有完整费用硬上限，记录器不能冒充硬限额。
2. **真实模型三例：** 在明确预算后各运行一次；正常案例观察两阶段连续推进，恢复案例在 `inspect` 后停止进程并跨进程恢复，约束案例在相同边界确认新约束后继续。至少保留一次隔天恢复的状态卡与下一步核对证据。
3. **独立报告：** 汇总 Task 自评、外部验收、阶段通过率、错误完成声明、重复工具调用、人工介入、Token/费用、耗时与压缩/重启后下一步是否准确；逐例归因失败并提出下一切片。

G5 只有在三例的题卡摘要、执行记录、外部验收、预算核对和失败归因都能复核，且隔天恢复演示有证据后才算完成。缺少预算或真实运行结果时只能标为进行中。

**本组实际执行与原协议的差异（2026-10-11）：** 用户指出跨进程恢复不需要隔天；试评随即改为同日从原 Task/Session 恢复。三例数据采集、外部验收、预算核对及失败归因均已完成，同日恢复的状态卡准确，但原协议自行设定的 24 小时观察没有做。`reports/final-same-day.json` 的 `protocol_evidence_complete=false` 因此是正确结果，不能把本组称为按原协议完整验收。三例外部验收为 2 通过、1 失败；三例 Goal 都未自行完成。已知 Provider 用量下界 655,735 Token。恢复案例暴露的纯 CRLF 文件与 LF 编辑请求不匹配问题已在后续代码中修复，并在[独立单例模型回归](CRLF-REGRESSION-2026-10-11.md)中观察到 README 编辑与外部验收通过；该修复不改变本组冻结结果。
