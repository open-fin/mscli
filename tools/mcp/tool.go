package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"gitcode.com/mindspore/mscli/integrations/llm"
	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
)

const maxResultContent = 64 * 1024

var safePersistNamePattern = regexp.MustCompile(`[^A-Za-z0-9_-]+`)

// Caller is the runtime dependency used to execute MCP tools.
type Caller interface {
	CallTool(ctx context.Context, serverName, toolName string, args json.RawMessage) (*runtimemcp.CallResult, error)
}

// Tool adapts a discovered MCP tool to the local tools.Tool interface.
type Tool struct {
	def    runtimemcp.ToolDefinition
	caller Caller
}

// NewTool creates an MCP tool adapter.
func NewTool(def runtimemcp.ToolDefinition, caller Caller) *Tool {
	if strings.TrimSpace(def.Name) == "" {
		def.Name = runtimemcp.BuildToolName(def.ServerName, def.OriginalToolName)
	}
	return &Tool{def: def, caller: caller}
}

// WrapTools creates tool adapters for discovered MCP tool definitions.
func WrapTools(defs []runtimemcp.ToolDefinition, caller Caller) []tools.Tool {
	out := make([]tools.Tool, 0, len(defs))
	for _, def := range defs {
		out = append(out, NewTool(def, caller))
	}
	return out
}

func (t *Tool) Name() string {
	return t.def.Name
}

func (t *Tool) MCPServerName() string {
	return t.def.ServerName
}

func (t *Tool) Capabilities() tools.Capabilities {
	return tools.Capabilities{
		Kind:             tools.KindMCP,
		MutatesWorkspace: true,
		NetworkAccess:    true,
		LongRunning:      true,
		ResultTypes:      []string{tools.ResultTypeText, tools.ResultTypeJSON, tools.ResultTypeArtifact},
		Risk:             "unknown",
	}
}

func (t *Tool) Description() string {
	description := strings.TrimSpace(t.def.Description)
	prefix := fmt.Sprintf("MCP tool %s/%s", t.def.ServerName, t.def.OriginalToolName)
	if description == "" {
		return prefix
	}
	return prefix + ": " + description
}

func (t *Tool) Schema() llm.ToolSchema {
	return ConvertSchema(t.def.InputSchema)
}

func (t *Tool) Execute(ctx context.Context, raw json.RawMessage) (*tools.Result, error) {
	if t.caller == nil {
		return tools.ErrorResultf("mcp tool %s has no caller", t.Name()), nil
	}
	result, err := t.caller.CallTool(ctx, t.def.ServerName, t.def.OriginalToolName, raw)
	if err != nil {
		return tools.ErrorResult(err), nil
	}
	if result == nil {
		return tools.ErrorResultf("mcp tool %s returned no result", t.Name()), nil
	}
	content := formatCallResult(result)
	if result.IsError {
		return tools.ErrorResultf("mcp tool %s/%s failed: %s", t.def.ServerName, t.def.OriginalToolName, content), nil
	}
	if len(content) > maxResultContent {
		content = persistLargeResultNotice(content, t.def.ServerName, t.def.OriginalToolName, result)
	}
	return tools.StringResultWithSummary(content, fmt.Sprintf("mcp %s/%s", t.def.ServerName, t.def.OriginalToolName)), nil
}

func formatCallResult(result *runtimemcp.CallResult) string {
	if result == nil {
		return "MCP tool returned no content"
	}
	parts := make([]string, 0, len(result.Content)+1)
	for _, block := range result.Content {
		switch block.Type {
		case "text":
			if strings.TrimSpace(block.Text) != "" {
				parts = append(parts, strings.TrimSpace(block.Text))
			}
		case "":
			parts = append(parts, "Unsupported MCP content block type: unknown")
		default:
			parts = append(parts, "Unsupported MCP content block type: "+block.Type)
		}
	}
	if result.StructuredContent != nil {
		if data, err := json.MarshalIndent(result.StructuredContent, "", "  "); err == nil {
			parts = append(parts, "Structured content:\n"+string(data))
		}
	}
	content := strings.TrimSpace(strings.Join(parts, "\n\n"))
	if content == "" {
		content = "MCP tool returned no content"
	}
	return content
}

func persistLargeResultNotice(content, serverName, toolName string, result *runtimemcp.CallResult) string {
	path, err := persistLargeResult(content, serverName, toolName)
	if err != nil {
		return fmt.Sprintf("Error: result (%s characters) exceeds maximum allowed tokens. Failed to save output to file: %v. If this MCP server provides pagination or filtering tools, use them to retrieve specific portions of the data.", formatInt(len(content)), err)
	}
	return fmt.Sprintf("Error: result (%s characters) exceeds maximum allowed tokens. Output has been saved to %s.\nFormat: %s\nIf this MCP server provides pagination or filtering tools, use them to retrieve specific portions of the data.", formatInt(len(content)), path, resultFormatDescription(result))
}

func persistLargeResult(content, serverName, toolName string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	workspace, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("resolve workspace: %w", err)
	}
	workspaceAbs, err := filepath.Abs(workspace)
	if err != nil {
		return "", fmt.Errorf("resolve workspace path: %w", err)
	}
	dir := filepath.Join(home, ".mscli", "projects", workspaceKey(workspaceAbs), "tool-results")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create tool-results directory: %w", err)
	}
	name := fmt.Sprintf("mcp-%s-%s-%d.txt", safePersistName(serverName), safePersistName(toolName), time.Now().UnixMilli())
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("write tool result: %w", err)
	}
	return path, nil
}

func workspaceKey(workspace string) string {
	key := strings.ReplaceAll(workspace, string(filepath.Separator), "-")
	key = strings.ReplaceAll(key, ":", "-")
	key = strings.Trim(key, "-")
	if key == "" {
		return "workspace"
	}
	return key
}

func safePersistName(name string) string {
	name = safePersistNamePattern.ReplaceAllString(strings.TrimSpace(name), "_")
	name = strings.Trim(name, "_")
	if name == "" {
		return "tool"
	}
	return name
}

func resultFormatDescription(result *runtimemcp.CallResult) string {
	if result != nil && result.StructuredContent != nil {
		return "JSON"
	}
	return "text"
}

func formatInt(n int) string {
	raw := fmt.Sprintf("%d", n)
	if len(raw) <= 3 {
		return raw
	}
	var parts []string
	for len(raw) > 3 {
		parts = append([]string{raw[len(raw)-3:]}, parts...)
		raw = raw[:len(raw)-3]
	}
	parts = append([]string{raw}, parts...)
	return strings.Join(parts, ",")
}
