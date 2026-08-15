[English](README.md) | [中文](README_zh.md)

# mscli

`mscli` is an experimental CLI harness for building and exploring agent workflows in AI infrastructure.

It provides a place to test the runtime components behind an agentic CLI, including model-provider integration, tool execution, permissions, context management, sessions, MCP servers, skills, memory, and terminal interaction.

`mscli` is not an official MindSpore product or a model-specific agent. It is an evolving engineering experiment, so interfaces and behavior may change.

## What You Can Explore

- interactive terminal UI and headless execution
- OpenAI Chat Completions, OpenAI Responses, and Anthropic-compatible providers
- a ReAct-style model/tool loop with context compaction
- filesystem and shell tools with permission and path policies
- persistent sessions, resume, replay, and file-backed memory
- MCP server discovery and tool execution
- embedded and dynamically loaded skills
- artifacts for large tool results and debugging workflows

See [Architecture](docs/arch.md) for the current runtime flow and package boundaries.

## Status

The latest tagged version is `v0.1.5`. See [changelog.md](changelog.md) for the release history.

Treat `mscli` as experimental software:

- Review tool permissions before approving commands.
- Use headless commands carefully because `exec`, `fix`, and `diagnose` enable unattended execution.
- Pin a version if you depend on an interface that is not yet stable.

## Installation

### Install from Script

```bash
curl -fsSL https://api.gitcode.com/api/v5/repos/mindspore/mscli/raw/scripts/install.sh?ref=main | bash
```

### Build from Source

Go 1.24.2 or newer is required.

```bash
git clone https://github.com/open-fin/mscli.git
cd mscli
go build -o mscli ./cmd/mscli
./mscli --help
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

### Use Another Compatible Provider

```bash
export MSCLI_PROVIDER=<openai-completion|openai-responses|anthropic>
export MSCLI_BASE_URL=<provider-base-url>
export MSCLI_API_KEY=<your-api-key>
export MSCLI_MODEL=<model-name>

mscli
```

## Usage

Start the interactive terminal interface:

```bash
mscli
```

Run a free-form task headlessly:

```bash
mscli exec "inspect this repository"
```

Useful commands include:

```text
mscli resume [sess_xxx]
mscli replay [sess_xxx|trajectory.json|trajectory.jsonl]
mscli diagnose <problem>
mscli fix <problem>
mscli exec <task>
```

Run `mscli --help` or use `--help` on a subcommand for the current command surface.

## Skills

`mscli` includes a skill-loading mechanism and currently embeds a set of infrastructure-oriented skills at build time.

[ms-skills](https://github.com/open-fin/ms-skills) is a separate repository containing skill examples and reusable patterns for AI-infrastructure work.

The two repositories can be used independently. They are related experiments, not parts of a single product.

## Documentation

- [Architecture](docs/arch.md)
- [MCP](docs/mcp.md)
- [Tool system](docs/tool-system.md)
- [Contributor guide](docs/agent-contributor-guide.md)
- [Changelog](changelog.md)

## Contributing

See the [Contributor Guide](docs/agent-contributor-guide.md) for code style, dependency rules, and validation conventions.

## License

Apache-2.0
