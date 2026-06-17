package mcp

import (
	"testing"

	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
)

func TestMCPToolCapabilities(t *testing.T) {
	tool := NewTool(runtimemcp.ToolDefinition{
		Name:             "mcp__srv__lookup",
		ServerName:       "srv",
		OriginalToolName: "lookup",
	}, nil)

	got := tools.CapabilitiesForTool(tool)
	if got.Kind != tools.KindMCP {
		t.Fatalf("Kind = %q, want %q", got.Kind, tools.KindMCP)
	}
	if got.ReadOnly {
		t.Fatal("ReadOnly = true, want false for unknown MCP tool")
	}
	if !got.MutatesWorkspace {
		t.Fatal("MutatesWorkspace = false, want conservative true")
	}
	if !got.NetworkAccess {
		t.Fatal("NetworkAccess = false, want true")
	}
	if got.Risk != "unknown" {
		t.Fatalf("Risk = %q, want unknown", got.Risk)
	}
}
