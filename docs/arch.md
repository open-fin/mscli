# MindSpore CLI Architecture

## Two-Repo Model

```text
mscli (this repo)    runtime — TUI, agent loop, tool registry
mindspore-skills             instructions — SKILL.md + skill.yaml per skill
```

- `mscli` loads skills from `mindspore-skills` at build time (embedded in binary)
- Skills are portable across CLIs (Claude Code, OpenCode, Gemini CLI, Codex)
- `mscli` is the official end-to-end entrypoint
- `mindspore-skills` is the reusable capability layer

## Top-Level Shape

```text
mscli/
  cmd/mscli/              process entrypoint
  internal/
    app/                   composition root, startup, commands, UI bridging
    factory/               local Factory card, pack, compiler, and runtime logic
    train/                 train request and target types
    update/                binary update checker
    workspacefile/         workspace path validation
    version/               build and release version metadata
  agent/
    context/               context window management and compaction
    loop/                  ReAct-style execution engine (the core runtime)
    memory/                memory store, retrieval, and policy
    session/               session state and persistence
  workflow/
    train/                 train lane controller, setup, run, demo backend
  integrations/
    domain/                domain constants and shared taxonomy
    llm/                   unified provider manager (openai-completion/openai-responses/anthropic)
    skills/                skill listing, loading, and metadata (embedded at build time)
  permission/              permission policy, types, store, safe command allowlist
  report/                  report data structures and helpers
  runtime/
    artifacts/             state-root artifact writer for large tool results
    mcp/                   MCP config, approvals, stdio JSON-RPC, and server lifecycle
    shell/                 stateful shell command runner
    probes/                local and target readiness probes
  tools/
    fs/                    read, grep, glob, edit, write tools
    mcp/                   MCP tool adapter into the local tools.Tool interface
    shell/                 shell tool wrapper
    skills/                skill loading tool
  ui/                      Bubble Tea app, shared model, panels, slash commands
  configs/                 config loading, state, shared config types
  scripts/                 install, release, and docs lint scripts
  docs/                    architecture, contributor, and factory docs
```

## Primary Runtime Flow

```text
cmd/mscli
  -> internal/app.Run(...)
  -> internal/app.Wire(...)
  -> ui.New(...)

user input
  -> internal/app.processInput(...)
  -> slash command (/diagnose, /fix, /model, ...)
     or free text -> runTask(...)

runTask:
  -> compose effective conversation context:
       EngineConfig.SystemPrompt
       + skill summaries (from integrations/skills)
       + any skill content preloaded by /skill
  -> agent/loop.Engine.RunWithContext(task)
  -> tools.Registry
  -> tools/fs, tools/shell, tools/skills, or tools/mcp
  -> runtime/shell.Runner, runtime/mcp.Manager, or runtime/artifacts.Store
  -> loop.Event stream -> model.Event -> ui
```

No orchestrator, no planner, no adapter layer. The app calls the engine
directly. The LLM plans inline within the agent loop.

MCP initialization happens during app wiring, before the TUI starts:

```text
internal/app init
  -> runtime/mcp resolve user/project/local configs and approvals
  -> runtime/mcp connect approved stdio servers
  -> runtime/mcp tools/list
  -> tools/mcp wrap discovered tools
  -> tools.Registry
```

The MCP MVP supports stdio `tools/list` and `tools/call`. Project MCP configs
must be approved before their server process starts; rejected or pending project
servers are skipped and surfaced as warnings.

### Skill activation

Skills are embedded in the binary at build time from the `mindspore-skills` repo.
The `scripts/update-skills.sh` script pulls the latest skills before each release build.

Skill activation is session-visible. `/skill <name>` and `/<name>` preload the
skill by injecting a synthetic `load_skill` tool call/result into conversation
history.

```text
/skill failure-agent
  -> app loads failure-agent SKILL.md from integrations/skills/builtin
  -> app injects synthetic load_skill tool call/result into context
  -> app submits a default start request if the user did not provide one
  -> engine runs with base prompt + existing conversation context
```

Free text uses the base system prompt which includes skill summaries
(name + one-line description). For reliable skill activation, use explicit commands.

## Package Responsibilities

- **`internal/app/`**
  Loads config, wires dependencies, starts the TUI, handles slash commands,
  dispatches tasks to the engine, and converts `loop.Event` to `ui/model.Event`.

- **`agent/loop/`**
  The core runtime. Runs the LLM/tool loop: tool calling, permission checks,
  context updates. Composes effective system prompt per task.

- **`agent/session/`**
  Owns session state, trajectory persistence, and resume reconstruction.

- **`integrations/skills/`**
  Lists available skills, loads one skill fully on demand (`SKILL.md` + metadata).
  Skills are embedded in `integrations/skills/builtin/` at build time.

- **`integrations/llm/`**
  Unified LLM provider interface. Supports OpenAI Chat Completions,
  OpenAI Responses, and Anthropic Messages protocols.

- **`tools/`**
  LLM-callable tool surfaces (filesystem, shell, skill loading). Stateless tool definitions.

- **`tools/mcp/`**
  Adapts discovered MCP tools into `tools.Tool` implementations. It preserves
  original server/tool names for execution while exposing registry names like
  `mcp__server__tool`.

- **`runtime/mcp/`**
  Owns MCP config discovery, project approval filtering, precedence merge,
  stdio JSON-RPC clients, server process lifecycle, tool listing, tool calls,
  stderr diagnostics, cancellation cleanup, and reconnect behavior.

- **`runtime/artifacts/`**
  Owns state-root artifact path construction and filename sanitization for large
  tool results. Current artifacts are scoped under
  `~/.mscli/projects/<workspace-key>/tool-results/`.

- **`runtime/shell/`**
  Stateful command runner with workspace, timeout, and safety checks.

- **`permission/`**
  Permission decisions and persistence. Safe command allowlist for auto-approving
  read-only commands (ls, cat, grep, git, etc.).

- **`ui/`**
  Bubble Tea inline-mode interface. Consumes events, renders panels.
  Not imported by lower layers.

- **`configs/`**
  Shared configuration types and loaders. Reads `MSCLI_*` environment variables.

## Dependency Boundaries

```text
cmd/mscli -> internal/app
internal/app -> agent, workflow, ui, configs, integrations, tools, permission, internal/factory, internal/train
agent -> integrations, permission, configs
workflow -> internal/train, runtime/probes
tools -> runtime, integrations
runtime -> configs
ui -> configs
```

Constraints:

- `cmd/mscli/` stays thin.
- `internal/app/` is the wiring layer, not a reusable core package.
- `agent/` must not depend on `ui/` or `runtime/` directly.
- `tools/` may call `runtime/`, but `runtime/` must not call `tools/`.
- `configs/` is shared configuration, not a home for application logic.

When docs and code disagree, follow the code and update the docs.
