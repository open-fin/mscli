English | [中文](README_zh.md)

# MindSpore Model Agent

MindSpore Model Agent is a training-focused AI agent solution for the MindSpore ecosystem. It is designed for the high-frequency engineering work around model training, where users need more than general code generation and need help with domain-specific training tasks.

It is built on two closely related parts:

- [`mindspore-skills`](https://gitcode.com/mindspore/mindspore-skills): the domain capability layer for model training and debugging tasks. It provides reusable skills for readiness checking, failure diagnosis, accuracy analysis, performance analysis, model migration, algorithm adaptation, and operator implementation. These skills can work not only with MindSpore Model Agent, but also with other agentic CLI environments such as Claude Code, OpenCode, and Codex.
- `mscli`: the official CLI of MindSpore Model Agent. It provides better integration with related skills and is optimized for model training use cases, offering a more unified end-to-end experience for training-oriented workflows.

## Latest Version

Latest version: `MindSpore Model Agent v0.1.4`. See [changelog.md](changelog.md) for update history.

Highlights:

- `[skills]` Added baseline analysis support for Ascend A2 training runtime failures, accuracy drift, and performance bottlenecks.
- `[skills]` Added Hugging Face Transformers model migration support and integrated `mhc` / `attn-residual` into the Qwen3 skill template.
- `[skills]` Integrated `openjiuwen claw`, with precision-location examples and deployment guidance.
- `[cli]` Improved live task progress feedback, including during hidden tool-call assembly.
- `[cli]` Added diff view for edit-style tool results and improved transcript layout and readability.
- `[cli]` Fixed shell interrupt handling in truncated streaming output scenarios, unified bug / issue data structures, and fixed GitCode-incompatible install examples.

## MindSpore CLI

MindSpore CLI is the official end-to-end interface of MindSpore Model Agent. It is designed to provide a unified CLI experience for training-oriented workflows, with tighter integration with the related skills behind the solution.

## Installation

### Install from script

```bash
curl -fsSL https://api.gitcode.com/api/v5/repos/mindspore/mscli/raw/scripts/install.sh?ref=main | bash
```

### Build from source

Go 1.24.2+:

```bash
git clone https://gitcode.com/mindspore/mscli.git
cd mscli
go build -o mscli ./cmd/mscli
./mscli
```

## Quick Start

### Use Kimi Code Plan

```bash
export MSCLI_PROVIDER=anthropic
export MSCLI_BASE_URL=https://api.kimi.com/coding/
export MSCLI_API_KEY=<your Kimi Code API key>
export MSCLI_MODEL=kimi-k2.6

mscli
```

### Use DeepSeek V4 Pro

```bash
export MSCLI_PROVIDER=anthropic
export MSCLI_BASE_URL=https://api.deepseek.com/anthropic
export MSCLI_API_KEY=<your DeepSeek API key>
export MSCLI_MODEL=deepseek-v4-pro

mscli
```

## Documentation

- [Architecture](docs/arch.md)
- [Contributor Guide](docs/agent-contributor-guide.md)

## Contributing

See the [Contributor Guide](docs/agent-contributor-guide.md) for code style, dependency rules, and testing conventions.
