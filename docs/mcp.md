# MCP Support

`mscli` supports Model Context Protocol (MCP) stdio servers and Streamable HTTP
servers. MCP tools are exposed as normal agent tools with names like
`mcp__server__tool`.

## Supported Scope

Supported now:

- stdio MCP servers
- Streamable HTTP servers using `type: "http"`
- `tools/list`
- `tools/call`

Not supported yet:

- legacy HTTP+SSE / SSE transport
- WebSocket transports
- OAuth
- MCP prompts
- MCP resources
- dynamic `tools/list_changed` refresh
- multimodal MCP results

Unsupported transports are skipped with a warning.

## Config Files

`mscli` reads MCP config from three locations:

```text
~/.mscli/mcp.json
<workspace>/.mscli/mcp.json
<workspace>/.mscli/mcp.local.json
```

Precedence is:

```text
user < approved project < local
```

Example:

```json
{
  "mcpServers": {
    "puppeteer": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
      "env": {}
    }
  }
}
```

Missing `type` is treated as `stdio`.

Streamable HTTP servers use the MCP endpoint URL and `type: "http"`:

```json
{
  "mcpServers": {
    "remote": {
      "type": "http",
      "url": "https://example.com/mcp"
    }
  }
}
```

`mscli` sends JSON-RPC requests with an `Accept` header for both
`application/json` and `text/event-stream`, and handles either response format
from the Streamable HTTP endpoint. The older HTTP+SSE transport that used a
separate SSE endpoint is still unsupported.

## Project Approval

Project MCP config can start arbitrary local commands, so servers from
`<workspace>/.mscli/mcp.json` require approval before `mscli` starts them.

Approval records are stored globally:

```text
~/.mscli/mcp_approvals.json
```

Each approval is bound to:

- absolute workspace path
- server name
- hash of the explicit server config

Changing `type`, `command`, `args`, `env`, or future explicit transport fields
changes the hash and requires a new decision.

Rejected servers are skipped and suppress future prompts until the config hash
changes or the approval record is reset.

## Reset Approvals

To reset one project MCP decision, open:

```text
~/.mscli/mcp_approvals.json
```

Remove the matching entry for the workspace, server, and hash.

To reset all MCP project approvals, delete `~/.mscli/mcp_approvals.json`.

Deleting approval records does not delete MCP configs and does not approve any
server by itself. The next `mscli` startup will prompt again for pending project
servers.

## Tool Permissions

Server startup approval is separate from tool-call permission.

MCP tool calls use the existing permission system. Use fully qualified tool
names in permission rules, for example:

```text
mcp__github__*
mcp__puppeteer__puppeteer_navigate
```

## Large Results

MCP tool results larger than the model-facing limit are written as tool-result
artifacts and replaced with a short notice in the tool output. Artifacts are
stored under:

```text
~/.mscli/projects/<workspace-key>/tool-results/
```

The tool result metadata includes:

- `artifact_path`: absolute local path
- `artifact_relative_path`: path relative to `~/.mscli`
- `bytes`: full artifact size
- `content_type`: `text/plain` or `application/json`
- `truncated`: `true` when the model-facing output was replaced by a notice

MCP result metadata also includes `source=mcp`, the MCP `server`, the original
MCP `tool`, and terminal `status`.
