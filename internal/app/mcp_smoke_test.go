package app

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestSmokeMCPRealStdioEchoServerProjectApprovalAndCall(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)

	serverPath := filepath.Join(t.TempDir(), "echo_mcp_server.py")
	if err := os.WriteFile(serverPath, []byte(echoMCPServerPython), 0755); err != nil {
		t.Fatalf("write echo server: %v", err)
	}

	configPath := filepath.Join(workDir, ".mscli", "mcp.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0755); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	config := map[string]any{
		"mcpServers": map[string]any{
			"echo": map[string]any{
				"type":    "stdio",
				"command": "python3",
				"args":    []string{serverPath},
				"env":     map[string]string{"ECHO_PREFIX": "echo"},
			},
		},
	}
	configData, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, configData, 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	registry := tools.NewRegistry()
	eventCh := make(chan model.Event, 32)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	manager, err := initMCPTools(ctx, registry, workDir, time.Second, eventCh, fixedMCPPrompter{decision: runtimemcp.DecisionApproved})
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	defer manager.Close(context.Background())

	store := runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home))
	resolved, err := runtimemcp.ResolveConfig(ctx, runtimemcp.ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workDir,
		ApprovalStore: store,
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	if got, want := len(resolved.Servers), 1; got != want {
		t.Fatalf("resolved servers = %d, want %d; pending=%d rejected=%d warnings=%v", got, want, len(resolved.Pending), len(resolved.Rejected), resolved.Warnings)
	}
	if got, want := resolved.Servers[0].Name, "echo"; got != want {
		t.Fatalf("resolved server = %q, want %q", got, want)
	}

	tool, ok := registry.Get("mcp__echo__echo")
	if !ok {
		t.Fatalf("registry missing mcp__echo__echo; names=%v", registry.Names())
	}
	result, err := tool.Execute(ctx, json.RawMessage(`{"text":"hello"}`))
	if err != nil {
		t.Fatalf("tool.Execute() err = %v", err)
	}
	if result.Error != nil {
		t.Fatalf("tool.Execute() result error = %v", result.Error)
	}
	if !strings.Contains(result.Content, "echo: hello") {
		t.Fatalf("tool result content = %q, want echo response", result.Content)
	}
}

const echoMCPServerPython = `#!/usr/bin/env python3
import json
import os
import sys

def send(req_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    req_id = req.get("id")
    if method == "initialize":
        send(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "echo", "version": "smoke"}
        })
    elif method == "tools/list":
        send(req_id, {
            "tools": [{
                "name": "echo",
                "description": "Echo input text",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Text to echo"}},
                    "required": ["text"]
                }
            }]
        })
    elif method == "tools/call":
        params = req.get("params") or {}
        args = params.get("arguments") or {}
        text = args.get("text", "")
        prefix = os.environ.get("ECHO_PREFIX", "echo")
        send(req_id, {
            "content": [{"type": "text", "text": prefix + ": " + text}],
            "structuredContent": {"echoed": text},
            "isError": False
        })
    else:
        send(req_id, error={"code": -32601, "message": "method not found"})
`
