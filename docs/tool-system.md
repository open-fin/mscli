# Tool System

This document describes the current tool runtime and the incremental contracts
used to make tool execution auditable, replayable, and extensible.

## Current Runtime Flow

The runtime path is intentionally direct:

```text
internal/app
  -> agent/loop
  -> tools.Registry
  -> tools/fs, tools/shell, tools/skills, tools/mcp
  -> runtime/shell, runtime/mcp, or runtime/artifacts
```

`internal/app` wires configuration, permissions, sessions, MCP startup, and UI
event conversion. It creates the `tools.Registry`, registers built-in tools, and
registers approved MCP tools during startup.

`agent/loop` owns the ReAct execution loop. It sends LLM requests, records
assistant tool calls, checks permission before emitting tool start events,
executes registered tools, appends tool results back into context, and emits
loop events consumed by the app.

`tools` owns the LLM-callable tool interface and registry. Built-in tool
packages adapt filesystem, shell, skill loading, and MCP execution into the
same `tools.Tool` interface.

`runtime` owns lower-level execution systems. `runtime/shell` runs stateful shell
commands inside the workspace. `runtime/mcp` owns MCP configuration, approval
filtering, server lifecycle, tool listing, and tool calls. `runtime/artifacts`
owns state-root paths for large tool-result artifacts.

## Current Tools

- `read`, `grep`, and `glob` read workspace files.
- `edit` and `write` mutate workspace files.
- `shell` executes workspace shell commands through `runtime/shell`.
- `load_skill` loads detailed skill instructions through `integrations/skills`.
- `mcp__<server>__<tool>` tools are discovered from approved MCP servers and
  adapted by `tools/mcp`.

## Permission Path

The loop checks permission before emitting `ToolCallStart`. This keeps the UI
from showing a tool as running while a permission prompt is still pending or
after a request is denied. Denied permission is represented on the tool
result/error path.

## MCP Path

During app startup, `internal/app` resolves MCP configuration, prompts for
project server approval when needed, starts approved servers through
`runtime/mcp`, lists tools, normalizes names, and registers MCP adapters in the
tool registry.

MCP adapters preserve the exposed registry name, original server name, and
original tool name. Large MCP results are persisted through `runtime/artifacts`
and replaced with a model-facing notice.

## Runtime Contracts

Every registered built-in and MCP tool exposes capability metadata through the
optional `tools.CapabilityProvider` interface. Unknown tools use conservative
defaults: unknown kind, mutating, long-running, and risk `unknown`.

Tool results carry standard metadata in `tools.Result.Meta`. Standard keys
include status, duration, source, exit code, truncation, artifact paths, content
type, and MCP server/tool identity. Metadata is additive: tool-specific keys such
as edit diffs must be preserved.

Tool result metadata is persisted with session trajectory records and replayed
into UI events. Start events are reconstructed from existing tool-call records in
this cycle; tool-call/start metadata is not expanded here.

Large tool outputs use the shared artifact store under:

```text
~/.mscli/projects/<workspace-key>/tool-results/
```

Artifact metadata includes absolute path, state-relative path, byte size,
content type, and truncation state.

Lifecycle events keep the current external event names for UI compatibility.
Terminal state is represented as result metadata:

```text
completed | failed | interrupted | declined
```

Started and streaming output remain represented by their event types.

## Session Replay

Tool result metadata is written to `trajectory.jsonl` on `tool_result` records.
Replay copies that metadata back to UI events. Older records without `meta`
continue to load and replay. Known numeric metadata keys are normalized after
load so `duration_ms` and `bytes` are usable as `int64`, while `exit_code` is
usable as `int`.

## Package Ownership

- `tools` defines tool-facing metadata helpers and registry capability listing.
- `agent/loop` centrally adds default duration and terminal status metadata.
- `agent/session` persists and replays tool result metadata.
- `runtime/artifacts` owns state-root artifact paths and filename sanitization.
- `internal/app` wires artifact storage and preserves metadata while converting
  loop events to UI events.
- `ui` consumes existing event types and optional metadata without owning runtime
  policy.

## Staged Roadmap

1. Document the current and target runtime contracts.
2. Add capability metadata for built-in and MCP tools.
3. Standardize result metadata and shell truncation fields.
4. Persist tool result metadata in sessions and replay.
5. Add a shared artifact store and route large MCP results through it.
6. Normalize lifecycle metadata while preserving UI compatibility.
7. Update architecture and MCP docs after verification.
