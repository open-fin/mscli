[English](README.md) | 中文

# MindSpore Model Agent

MindSpore Model Agent 是一个面向 MindSpore 生态的、聚焦模型训练场景的 AI agent solution。它面向模型训练周边高频工程工作而设计，适用于那些仅靠通用代码生成还不够、还需要训练领域专项能力支持的场景。

它由两个紧密相关的部分组成：

- [`mindspore-skills`](https://gitcode.com/mindspore/mindspore-skills)：面向模型训练与调试任务的领域能力层，提供可复用的技能，包括 readiness 检查、failure diagnosis、accuracy analysis、performance analysis、model migration、algorithm adaptation 和 operator implementation。这些 skills 不仅可用于 MindSpore Model Agent，也可以与 Claude Code、OpenCode、Codex 等其他 agentic CLI 环境配合使用。
- `mscli`：MindSpore Model Agent 的官方 CLI。它与相关 skills 有更好的集成，并针对模型训练场景进行了优化，提供更统一的端到端训练任务交互体验。

## 最新版本

最新版本：`MindSpore Model Agent v0.1.5`。更新历史请参见 [changelog.md](changelog.md)。

亮点包括：

- `[MCP]` 新增端到端 MCP 支持，包括项目审批、stdio / HTTP 传输、工具发现、生命周期控制与断线重连。
- `[记忆]` 新增基于 `MEMORY.md` 和项目 `MSCLI.md` 的文件型自动记忆，以及 `/init` 和运行时推理强度调节。
- `[CLI]` 新增无头 `exec` 工作流和会话恢复提示，支持脚本与 CI 使用场景。
- `[工具]` 新增外部路径策略、能力元数据、标准化结果状态、生命周期事件与大结果 artifact 持久化。
- `[skills]` 同步最新内置 MindSpore skills，并通过独立的 ACLNN 和自定义算子 builder 扩展算子开发能力。
- `[安全]` 加固权限提示、shell 命令过滤、安全命令分类与更新二进制完整性校验。

## MindSpore CLI

MindSpore CLI 是 MindSpore Model Agent 的官方端到端交互入口。它面向训练任务工作流提供统一的 CLI 体验，并与方案背后的相关 skills 做更紧密的集成。

## Installation

### 脚本安装

```bash
curl -fsSL https://api.gitcode.com/api/v5/repos/mindspore/mscli/raw/scripts/install.sh?ref=main | bash
```

### 从源码构建

需要 Go 1.24.2+：

```bash
git clone https://gitcode.com/mindspore/mscli.git
cd mscli
go build -o mscli ./cmd/mscli
./mscli
```

## Quick Start

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

## Documentation

- [架构](docs/arch.md)
- [贡献者指南](docs/agent-contributor-guide.md)

## Contributing

请参阅 [贡献者指南](docs/agent-contributor-guide.md) 了解代码风格、依赖规则和测试规范。
