# Goal 模型试评合同 v2（2026-10-11）

本题卡从 [合同 v1](../goal_model_contract_v1/README.md) 派生，保留相同的合成 CLI 夹具、Goal 文本、两阶段计划、可写范围、硬约束和独立功能验收；仅把单回合时限从 180 秒改为 300 秒。题卡摘要为 `cac964d5d856569452c20a24be320ea2fb74d2ecc84c77ef0affe276cb879a98`。旧题卡和报告保持原貌。

新试评需单独冻结模型输出上限为 8,192 Token（通过仅在准备、设置和驱动进程内生效的 `TECHPILOT_MAX_TOKENS`），与 v1 的 4,096 或曾单独尝试的 16,384 不混算。预期验证 Runtime 能将空输出且达到上限的回复记录为 `output_tokens` 停止，并在更协调的输出/时长配置下完成同日跨进程恢复。先跑恢复案例；只有它通过 Goal 自行完成与独立 CLI 验收后，再决定是否重跑 normal、约束案例以形成同配置三例结论。

## qwen3.8-27b 恢复案例阶段结果

当前项目 `.env` 已改为 `qwen3.8-27b`；每次试跑仍以创建时 `manifest.json` 中的模型 ID 为准，旧 `glm-5.3` 样本不随 `.env` 改动。`2026-10-11-qwen-38-27b-restart-01` 因连续 `/goal run` 在记录重启边界前进入实施阶段，仅作探索证据；`restart-02` 遇到系统 Python 缺少 pytest 和模型提交需要确认的管道命令，也不计为恢复通过。两份样本均保留原始记录。

`2026-10-11-qwen-38-27b-restart-03` 从准备开始使用项目 `.venv`，以普通聊天的精确「继续」执行单个工作回合，在 `inspect` verified 且实施文件与夹具完全一致时退出进程；新进程恢复同一 Task/Session，状态卡准确指向 `implement`。随后模型完成代码和 README 修改，分别执行 pytest（3 passed）和 Ruff（通过），Goal 自行 `succeeded`，独立 CLI 验收全部通过。最终报告为 `.techpilot/goal-model-contract-v2/2026-10-11-qwen-38-27b-restart-03/reports/final-single-turn-evidence.json`，该恢复案例的 `missing_evidence` 为空，已报告用量下界 44,532 Token。失败或中断请求可能没有用量回报，此数字不是账单。模型曾提交一条被拒绝的链式验证命令，之后自行改用两条单独命令；记录未抹去该拒绝。

协议现在区分连续 `/goal run` 批次与恢复后普通聊天单回合：恢复案例只有在 `after_restart` 后确有新增持久工作 checkpoint 时才接受后者，报告标记 `execution_evidence_mode=post_restart_work_turn`。normal 与 constraint 案例仍要求连续批次证据。整套三例 `protocol_evidence_complete` 目前仍为 `false`，下一步是在同一模型和题卡配置下补跑 normal、constraint，再生成完整报告。
