# G5 后续：CRLF 编辑修复的单例模型回归（2026-10-11）

## 边界

- 本记录是原三例试评之后的新一轮**单例回归**，不覆盖 `2026-10-11-glm-53-03` 的冻结结果。
- 新目录：`.techpilot/goal-model-pilot-v0/2026-10-11-glm-53-crlf-regression-01/`；只运行 `normal-json-export`。题卡摘要仍为 `2b210012e5638df0e7a28910e8fbdbba332eb7a3fada6535c1ae5bc168b06959`，模型仍为 `glm-5.3`。
- 本轮 300,000 Token 是管理上限；模型进程启动时把项目 `.venv/Scripts` 放在 PATH 前面，使 `python -m pytest` 使用已有的测试环境。原三例并未使用这一环境设置，因此回归同时移除了先前缺少 pytest 的环境干扰。

## 观察结果

1. 连续执行两回合：`inspect` 验证通过，模型基线 `python -m pytest -q` 为 3/3；`implement` 修改了 exporter、CLI 和 README。**两次 README `edit_file` 均成功**，没有再出现 CRLF/LF 的 `old_string not found`。
2. 模型运行 `python -m pytest -q` 与 Ruff 成功；请求 `python -m cli_data_tool.cli export` 时，试评驱动器对需要提示的 `bash` 命令保持拒绝。该回合的阶段检查因失败工具阻塞，连续执行停止。
3. 随后 1 次不带新提示的普通 Chat `continue` 中，模型尝试修改冻结的 `tests/test_export.py`，试评权限正确拒绝；阶段继续阻塞。模型的文字判断认为拒绝不妨碍完成，但 Runtime 没有采用这一自评。
4. 独立外部验收 **全部通过**：README 示例、默认/显式 text、JSON 结构、非法格式退出、冻结文件和允许路径均符合题卡。Goal 最终暂停，`inspect` 已验证、`implement` 阻塞，未自行完成。

本轮 Provider 已报告 **193,956 Token**；加上原三例已知下界 655,735，当前累计已知下界为 **849,691 Token**。失败请求可能缺少用量回报，不能把此数当完整账单。证据位于新目录的 `normal-json-export/assessments/final.json`、`protocol/0001.json` 至 `0003.json` 和 `reports/focused-regression-final.json`。报告为复用三例模板，另外两例显示未运行、`protocol_evidence_complete=false`，不表示本单例缺少外部验收。

## 下一切片

先做**评测合同与模型可见说明的一致性**：明确告诉模型 `tests/` 是只读的外部判据，且当前驱动器不会自动批准需要提示的直接 CLI 命令；允许的验证方式应写清。新合同要另立版本或目录，不更改已冻结三例及本轮回归。保持权限拒绝和 Runtime 的保守阻塞行为，先在合同清晰的新鲜案例上观察 Goal 是否能自行完成；若仍因与任务结果无关的失败工具受阻，再设计有证据、可复核的阶段恢复规则。
