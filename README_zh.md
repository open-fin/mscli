[English](README.md) | 中文

# mscli

`mscli` 是一个用于构建和探索 AI 基础设施 agent 工作流的实验性 CLI harness。

它用于测试 agentic CLI 背后的运行时组件，包括模型提供方接入、工具执行、权限管理、上下文管理、会话、MCP server、skills、记忆和终端交互。

`mscli` 不是 MindSpore 官方产品，也不是面向某个特定模型的专用 agent。它是一个持续演进的工程实验项目，接口和行为可能发生变化。

## 可探索内容

- 交互式终端 UI 与无头任务执行
- OpenAI Chat Completions、OpenAI Responses 和 Anthropic 兼容提供方
- 带上下文压缩的 ReAct 风格模型/工具循环
- 受权限和路径策略约束的文件系统与 shell 工具
- 持久化会话、恢复、回放和文件型记忆
- MCP server 发现与工具执行
- 内置和动态加载的 skills
- 大型工具结果的 artifact 与调试工作流

当前运行流程和包边界请参阅[架构文档](docs/arch.md)。

## 项目状态

当前最新 tag 为 `v0.1.5`。版本历史请参阅 [changelog.md](changelog.md)。

请将 `mscli` 视为实验性软件：

- 批准命令前检查工具权限。
- `exec`、`fix` 和 `diagnose` 会启用无人值守执行，请谨慎使用。
- 如果依赖尚未稳定的接口，请固定版本。

## 安装

### 使用脚本安装

```bash
curl -fsSL https://api.gitcode.com/api/v5/repos/mindspore/mscli/raw/scripts/install.sh?ref=main | bash
```

### 从源码构建

需要 Go 1.24.2 或更高版本。

```bash
git clone https://github.com/open-fin/mscli.git
cd mscli
go build -o mscli ./cmd/mscli
./mscli --help
```

## 快速开始

### 使用 Kimi Code Plan

```bash
export MSCLI_PROVIDER=anthropic
export MSCLI_BASE_URL=https://api.kimi.com/coding/
export MSCLI_API_KEY=<your Kimi Code API key>
export MSCLI_MODEL=kimi-k2.6

mscli
```

### 使用 DeepSeek V4 Pro

```bash
export MSCLI_PROVIDER=anthropic
export MSCLI_BASE_URL=https://api.deepseek.com/anthropic
export MSCLI_API_KEY=<your DeepSeek API key>
export MSCLI_MODEL=deepseek-v4-pro

mscli
```

### 使用其他兼容提供方

```bash
export MSCLI_PROVIDER=<openai-completion|openai-responses|anthropic>
export MSCLI_BASE_URL=<provider-base-url>
export MSCLI_API_KEY=<your-api-key>
export MSCLI_MODEL=<model-name>

mscli
```

## 使用方式

启动交互式终端界面：

```bash
mscli
```

以无头模式执行自由文本任务：

```bash
mscli exec "inspect this repository"
```

常用命令：

```text
mscli resume [sess_xxx]
mscli replay [sess_xxx|trajectory.json|trajectory.jsonl]
mscli diagnose <problem>
mscli fix <problem>
mscli exec <task>
```

运行 `mscli --help` 或对子命令使用 `--help`，可查看当前命令界面。

## Skills

`mscli` 提供 skill 加载机制，当前会在构建时内置一组面向基础设施场景的 skills。

[ms-skills](https://github.com/open-fin/ms-skills) 是一个独立仓库，包含 AI 基础设施相关的 skill 示例和可复用模式。

两个仓库可以独立使用。它们是相关的实验项目，而不是同一个产品的组成部分。

## 文档

- [架构](docs/arch.md)
- [MCP](docs/mcp.md)
- [工具系统](docs/tool-system.md)
- [贡献者指南](docs/agent-contributor-guide.md)
- [变更记录](changelog.md)

## 贡献

代码风格、依赖规则和验证规范请参阅[贡献者指南](docs/agent-contributor-guide.md)。

## 许可证

Apache-2.0
