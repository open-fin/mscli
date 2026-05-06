package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/runtime/artifacts"
	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
)

const maxResultContent = 64 * 1024

// Caller is the runtime dependency used to execute MCP tools.
type Caller interface {
	CallTool(ctx context.Context, serverName, toolName string, args json.RawMessage) (*runtimemcp.CallResult, error)
}

type ArtifactWriteRequest = artifacts.WriteRequest
type Artifact = artifacts.Artifact

type ArtifactStore interface {
	Write(ArtifactWriteRequest) (Artifact, error)
}

// Tool adapts a discovered MCP tool to the local tools.Tool interface.
type Tool struct {
	def           runtimemcp.ToolDefinition
	caller        Caller
	artifactStore ArtifactStore
}

// NewTool creates an MCP tool adapter.
func NewTool(def runtimemcp.ToolDefinition, caller Caller) *Tool {
	return NewToolWithArtifactStore(def, caller, nil)
}

func NewToolWithArtifactStore(def runtimemcp.ToolDefinition, caller Caller, store ArtifactStore) *Tool {
	if strings.TrimSpace(def.Name) == "" {
		def.Name = runtimemcp.BuildToolName(def.ServerName, def.OriginalToolName)
	}
	return &Tool{def: def, caller: caller, artifactStore: store}
}

// WrapTools creates tool adapters for discovered MCP tool definitions.
func WrapTools(defs []runtimemcp.ToolDefinition, caller Caller) []tools.Tool {
	return WrapToolsWithArtifactStore(defs, caller, nil)
}

func WrapToolsWithArtifactStore(defs []runtimemcp.ToolDefinition, caller Caller, store ArtifactStore) []tools.Tool {
	out := make([]tools.Tool, 0, len(defs))
	for _, def := range defs {
		out = append(out, NewToolWithArtifactStore(def, caller, store))
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
		return t.withMeta(tools.ErrorResultf("mcp tool %s has no caller", t.Name()), tools.StatusFailed, tools.ContentTypeText), nil
	}
	result, err := t.caller.CallTool(ctx, t.def.ServerName, t.def.OriginalToolName, raw)
	if err != nil {
		return t.withMeta(tools.ErrorResult(err), tools.StatusFailed, tools.ContentTypeText), nil
	}
	if result == nil {
		return t.withMeta(tools.ErrorResultf("mcp tool %s returned no result", t.Name()), tools.StatusFailed, tools.ContentTypeText), nil
	}
	content := formatCallResult(result)
	if result.IsError {
		return t.withMeta(tools.ErrorResultf("mcp tool %s/%s failed: %s", t.def.ServerName, t.def.OriginalToolName, content), tools.StatusFailed, tools.ContentTypeText), nil
	}
	contentType := mcpContentType(result, false)
	if len(content) > maxResultContent {
		var artifact *Artifact
		content, artifact = t.persistLargeResultNotice(content, result)
		contentType = mcpContentType(result, true)
		if artifact != nil {
			out := t.withMeta(tools.StringResultWithSummary(content, fmt.Sprintf("mcp %s/%s", t.def.ServerName, t.def.OriginalToolName)), tools.StatusCompleted, artifact.ContentType)
			tools.SetResultArtifact(out, artifact.Path, artifact.RelativePath)
			tools.SetResultBytes(out, artifact.Size)
			tools.SetResultTruncated(out, true)
			return out, nil
		}
		out := t.withMeta(tools.StringResultWithSummary(content, fmt.Sprintf("mcp %s/%s", t.def.ServerName, t.def.OriginalToolName)), tools.StatusFailed, contentType)
		out.Error = fmt.Errorf("%s", content)
		tools.SetResultTruncated(out, true)
		return out, nil
	}
	return t.withMeta(tools.StringResultWithSummary(content, fmt.Sprintf("mcp %s/%s", t.def.ServerName, t.def.OriginalToolName)), tools.StatusCompleted, contentType), nil
}

func (t *Tool) withMeta(result *tools.Result, status, contentType string) *tools.Result {
	tools.SetResultSource(result, tools.SourceMCP)
	tools.SetResultServer(result, t.def.ServerName)
	tools.SetResultTool(result, t.def.OriginalToolName)
	tools.SetResultStatus(result, status)
	tools.SetResultContentType(result, contentType)
	return result
}

func mcpContentType(result *runtimemcp.CallResult, artifact bool) string {
	if artifact {
		return ContentTypeForMCPArtifact(result)
	}
	if result != nil && result.StructuredContent != nil && len(result.Content) == 0 {
		return tools.ContentTypeJSON
	}
	return tools.ContentTypeText
}

func ContentTypeForMCPArtifact(result *runtimemcp.CallResult) string {
	if result != nil && result.StructuredContent != nil && len(result.Content) == 0 {
		return tools.ContentTypeJSON
	}
	return tools.ContentTypeText
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

func (t *Tool) persistLargeResultNotice(content string, result *runtimemcp.CallResult) (string, *Artifact) {
	if t.artifactStore == nil {
		return fmt.Sprintf("Error: result (%s characters) exceeds maximum allowed tokens. No artifact store is configured. If this MCP server provides pagination or filtering tools, use them to retrieve specific portions of the data.", formatInt(len(content))), nil
	}
	contentType := ContentTypeForMCPArtifact(result)
	artifact, err := t.artifactStore.Write(ArtifactWriteRequest{
		Prefix:      "mcp",
		Name:        fmt.Sprintf("%s-%s", t.def.ServerName, t.def.OriginalToolName),
		ContentType: contentType,
		Data:        []byte(content),
	})
	if err != nil {
		return fmt.Sprintf("Error: result (%s characters) exceeds maximum allowed tokens. Failed to save output to file: %v. If this MCP server provides pagination or filtering tools, use them to retrieve specific portions of the data.", formatInt(len(content)), err), nil
	}
	return fmt.Sprintf("Result (%s characters) exceeds the inline limit. Output has been saved to %s.\nFormat: %s\nIf this MCP server provides pagination or filtering tools, use them to retrieve specific portions of the data.", formatInt(len(content)), artifact.Path, resultFormatDescription(result)), &artifact
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
