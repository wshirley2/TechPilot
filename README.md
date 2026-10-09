# TechPilot

TechPilot 是一个面向本地开发工作流的终端 Agent。

它提供交互式 TUI、工具调用、权限控制、会话管理和上下文处理等基础能力，内置 Coding Agent 作为示例。你可以直接使用它完成开发任务，也可以基于这些能力继续设计和运行自己的 Agent 工作流。

## TechPilot 包含以下：

- Runtime 负责调用工具、维护状态和处理执行过程；
- Session 与 Context 负责保存任务过程中的会话和必要信息；
- 通过权限和用户确认机制控制任务执行边界；
- 通过组合 Role、Skill 与工具，可以构建自己的 Agent 工作流。


## 快速开始

需要 Python 3.10 或更高版本。下面以 PowerShell 为例：

### 从源码安装与开发（当前推荐）

```powershell
git clone https://github.com/wshirley2/TechPilot.git
cd TechPilot

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

首次安装后，直接运行主入口会进入一次性的配置向导：

```powershell
techpilot
```

向导保存 Provider、Base URL、Model 和 API Key 到当前用户目录：Windows 为
`%APPDATA%\TechPilot\config.json`，Linux/macOS 为
`~/.config/techpilot/config.json`。API Key 不会打印、提交或写入安装包。之后
`techpilot` 是默认的 TUI 入口；非交互终端会自动回退到普通文本 Chat。

也可以随时显式初始化或修复配置：

```powershell
techpilot init
techpilot reconfigure
```

项目根目录的 `.env` 仍可作为团队共享的最低优先级默认值。如果使用 DeepSeek：

```dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.deepseek.com
TECHPILOT_MODEL=deepseek-v4-flash
```

配置优先级固定为：`CLI 参数 > 环境变量 > 用户配置 > 项目 .env`。这里的模型名会原样显示在 Chat 和 `/status` 中。当前 DeepSeek 配置可使用
`deepseek-v4-flash` 或 `deepseek-v4-pro`；也可以通过 `--api-key`、`--base-url`
和 `-m` 临时覆盖这些配置。

安装成功后，直接把本地仓库路径交给它：

```powershell
techpilot D:\path\to\your-project
```

首次打开仓库会先确认目录信任；TUI 的快捷键和 Chat 命令可以在运行后输入 `/help` 查看。

## 使用方式

进入 Chat 后，可以先从只读问题开始：

```text
请阅读这个项目，告诉我它解决什么问题、入口在哪里、如何运行测试。不要修改文件。
```

TUI 会展示流式回答、工具调用、Session 状态和待确认操作。写入文件前会先展示 Diff；输入 `1` 允许一次，输入 `2` 在当前会话内允许同类操作，直接回车拒绝。输入 `/help` 查看 Chat 内命令，常用的是 `/status`、`/resume <session-id>` 和 `/exit`。

需要跨多个回合持续推进时，在 Chat 中用 `/goal <目标>` 建立 Goal，先通过 `/goal criterion` 记录可验证的完成条件，再用 `/goal start` 开始。Goal 与普通 Chat 共用工具、权限和 Session；旧的 Plan、Workspace 和 Managed Run 命令已退场。

Session 默认保存在目标仓库的 `.techpilot/sessions/` 下；可以用 `--resume` 恢复，也可以用 `--sessions-dir` 指定其他目录。恢复只重建对话，不会重新执行历史工具调用。

## 安全边界

读取、搜索和常规测试通常可以直接执行；写文件、安装依赖、网络访问和命令执行会经过权限确认。删除、仓库外写入和破坏性 Git 操作会被阻止。TechPilot 提供的是交互和执行策略，不是系统级沙箱。

更多参数可通过 `techpilot --help` 或各子命令的 `--help` 查看。

## 来源与许可

项目从最小 AgentLoop 起步，当前运行内核已收敛到 `techpilot.engine`，唯一公开命令和发行包均为 TechPilot。许可信息见 [LICENSE](LICENSE)。
