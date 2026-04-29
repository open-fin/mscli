package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
)

func TestToolUsesQualifiedNameAndOriginalCallTarget(t *testing.T) {
	caller := &fakeCaller{result: &runtimemcp.CallResult{
		Content: []runtimemcp.ContentBlock{{Type: "text", Text: "ok"}},
	}}
	def := runtimemcp.ToolDefinition{
		ServerName:       "GitHub MCP",
		OriginalToolName: "search.code",
		Name:             "mcp__GitHub_MCP__search_code",
		Description:      "search code",
		InputSchema:      map[string]any{"type": "object"},
	}
	tool := NewTool(def, caller)
	if got, want := tool.Name(), "mcp__GitHub_MCP__search_code"; got != want {
		t.Fatalf("Name() = %q, want %q", got, want)
	}
	_, err := tool.Execute(context.Background(), json.RawMessage(`{"query":"x"}`))
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	if caller.calledServer != "GitHub MCP" || caller.calledTool != "search.code" {
		t.Fatalf("called %s/%s, want original names", caller.calledServer, caller.calledTool)
	}
	if string(caller.calledArgs) != `{"query":"x"}` {
		t.Fatalf("called args = %s", caller.calledArgs)
	}
}

func TestToolFormatsTextStructuredAndUnsupportedContent(t *testing.T) {
	tool := NewTool(runtimemcp.ToolDefinition{
		ServerName:       "server",
		OriginalToolName: "tool",
	}, &fakeCaller{result: &runtimemcp.CallResult{
		Content: []runtimemcp.ContentBlock{
			{Type: "text", Text: "hello"},
			{Type: "image", Data: map[string]any{"type": "image"}},
		},
		StructuredContent: map[string]any{"ok": true},
	}})
	result, err := tool.Execute(context.Background(), nil)
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	for _, want := range []string{"hello", "Structured content:", `"ok": true`, "Unsupported MCP content block type: image"} {
		if !strings.Contains(result.Content, want) {
			t.Fatalf("content missing %q:\n%s", want, result.Content)
		}
	}
}

func TestToolReturnsErrorResultForMCPError(t *testing.T) {
	tool := NewTool(runtimemcp.ToolDefinition{
		ServerName:       "server",
		OriginalToolName: "tool",
	}, &fakeCaller{result: &runtimemcp.CallResult{
		IsError: true,
		Content: []runtimemcp.ContentBlock{
			{Type: "text", Text: "failed"},
		},
	}})
	result, err := tool.Execute(context.Background(), nil)
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	if result.Error == nil {
		t.Fatal("result.Error = nil, want MCP error")
	}
	if !strings.Contains(result.Error.Error(), "failed") {
		t.Fatalf("result.Error = %v", result.Error)
	}
}

func TestToolSetsStandardMetadataForMCPResultPaths(t *testing.T) {
	tests := []struct {
		name        string
		caller      Caller
		wantStatus  string
		wantContent string
	}{
		{
			name:        "missing caller",
			caller:      nil,
			wantStatus:  tools.StatusFailed,
			wantContent: tools.ContentTypeText,
		},
		{
			name:        "call error",
			caller:      &fakeCaller{err: errors.New("boom")},
			wantStatus:  tools.StatusFailed,
			wantContent: tools.ContentTypeText,
		},
		{
			name:        "nil result",
			caller:      &fakeCaller{},
			wantStatus:  tools.StatusFailed,
			wantContent: tools.ContentTypeText,
		},
		{
			name: "call result error",
			caller: &fakeCaller{result: &runtimemcp.CallResult{
				IsError: true,
				Content: []runtimemcp.ContentBlock{{Type: "text", Text: "failed"}},
			}},
			wantStatus:  tools.StatusFailed,
			wantContent: tools.ContentTypeText,
		},
		{
			name: "structured success",
			caller: &fakeCaller{result: &runtimemcp.CallResult{
				StructuredContent: map[string]any{"ok": true},
			}},
			wantStatus:  tools.StatusCompleted,
			wantContent: tools.ContentTypeJSON,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := NewTool(runtimemcp.ToolDefinition{
				ServerName:       "server",
				OriginalToolName: "tool",
			}, tt.caller)
			result, err := tool.Execute(context.Background(), nil)
			if err != nil {
				t.Fatalf("Execute() err = %v", err)
			}
			for key, want := range map[string]any{
				tools.MetaSource:      tools.SourceMCP,
				tools.MetaServer:      "server",
				tools.MetaTool:        "tool",
				tools.MetaStatus:      tt.wantStatus,
				tools.MetaContentType: tt.wantContent,
			} {
				if got := result.Meta[key]; got != want {
					t.Fatalf("Meta[%s] = %#v, want %#v (meta %#v)", key, got, want, result.Meta)
				}
			}
		})
	}
}

func TestToolPersistsLargeResults(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)
	large := strings.Repeat("x", maxResultContent+100)
	tool := NewTool(runtimemcp.ToolDefinition{
		ServerName:       "server",
		OriginalToolName: "tool",
	}, &fakeCaller{result: &runtimemcp.CallResult{
		Content: []runtimemcp.ContentBlock{
			{Type: "text", Text: large},
		},
	}})
	result, err := tool.Execute(context.Background(), nil)
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	if strings.Contains(result.Content, large[:1024]) {
		t.Fatalf("content includes large inline result")
	}
	if !strings.Contains(result.Content, "exceeds maximum allowed tokens") || !strings.Contains(result.Content, "Output has been saved to ") {
		t.Fatalf("content missing persistence notice:\n%s", result.Content)
	}
	path := persistedPathFromNotice(t, result.Content)
	if !strings.HasPrefix(path, filepath.Join(home, ".mscli", "projects")) {
		t.Fatalf("persist path = %q, want under mscli projects", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read persisted result: %v", err)
	}
	if string(data) != large {
		t.Fatalf("persisted content len=%d, want %d", len(data), len(large))
	}
	if got := result.Meta[tools.MetaContentType]; got != tools.ContentTypeText {
		t.Fatalf("content type = %#v, want text/plain", got)
	}
}

func persistedPathFromNotice(t *testing.T, notice string) string {
	t.Helper()
	prefix := "Output has been saved to "
	start := strings.Index(notice, prefix)
	if start < 0 {
		t.Fatalf("notice missing path: %s", notice)
	}
	start += len(prefix)
	end := strings.Index(notice[start:], ".\n")
	if end < 0 {
		t.Fatalf("notice missing path terminator: %s", notice)
	}
	return notice[start : start+end]
}

type fakeCaller struct {
	calledServer string
	calledTool   string
	calledArgs   json.RawMessage
	result       *runtimemcp.CallResult
	err          error
}

func (f *fakeCaller) CallTool(ctx context.Context, serverName, toolName string, args json.RawMessage) (*runtimemcp.CallResult, error) {
	f.calledServer = serverName
	f.calledTool = toolName
	f.calledArgs = append(json.RawMessage(nil), args...)
	return f.result, f.err
}
