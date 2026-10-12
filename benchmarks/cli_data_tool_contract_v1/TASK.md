# Feature Task: JSON export

为 `export` 命令增加 `--format` 参数，支持 `text` 和 `json`。默认 text 输出保持不变；非法格式需报错并返回失败状态；README 补充 JSON 示例。

## 试评工作边界

- 只允许修改 `analysis.md`、`README.md`、`src/cli_data_tool/cli.py`、`src/cli_data_tool/exporter.py`。
- `tests/`、`TASK.md`、`pyproject.toml` 是只读输入；可以阅读和运行已有测试，不得编辑、替换或新增测试文件。
- 当前试评驱动器可让 `python -m pytest -q` 与 Ruff 等无需人工提示的只读验证运行；需要人工提示的直接 CLI 命令不会获批。请使用现有测试与代码检查验证，并说明无法直接运行的命令。独立外部验收会运行真实 CLI 功能检查。
- 不要把一次被拒绝的权限请求解释成已执行成功；按 Goal 的实际阶段状态继续或报告阻塞。
