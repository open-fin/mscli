package fs

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/tools"
)

// WriteTool writes or creates file contents.
type WriteTool struct {
	resolver *pathpolicy.Resolver
}

// NewWriteTool creates a new write tool.
func NewWriteTool(workDir string) *WriteTool {
	return NewWriteToolWithResolver(newWorkspaceResolver(workDir))
}

func NewWriteToolWithResolver(resolver *pathpolicy.Resolver) *WriteTool {
	return &WriteTool{resolver: resolver}
}

// Name returns the tool name.
func (t *WriteTool) Name() string {
	return "write"
}

func (t *WriteTool) Capabilities() tools.Capabilities {
	return tools.Capabilities{
		Kind:             tools.KindFilesystem,
		MutatesWorkspace: true,
		ResultTypes:      []string{tools.ResultTypeText},
		Risk:             "medium",
	}
}

// Description returns the tool description.
func (t *WriteTool) Description() string {
	return "Create a new file or overwrite an existing file with new content. Arguments must be a JSON object containing required fields path and content."
}

// Schema returns the tool parameter schema.
func (t *WriteTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"path": {
				Type:        "string",
				Description: "Required. Path to the file to write. Sprint 1 allows workspace paths only for writes. Use this exact field name; do not use file_path or filename.",
			},
			"content": {
				Type:        "string",
				Description: "Required. Full content to write to the file.",
			},
		},
		Required: []string{"path", "content"},
	}
}

type writeParams struct {
	Path     string `json:"path"`
	FilePath string `json:"file_path"`
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

// Execute executes the write tool.
func (t *WriteTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p writeParams
	if err := tools.ParseParams(params, &p); err != nil {
		return withSourceMeta(tools.ErrorResult(err)), nil
	}

	path := strings.TrimSpace(p.Path)
	if path == "" {
		path = strings.TrimSpace(p.FilePath)
	}
	if path == "" {
		path = strings.TrimSpace(p.Filename)
	}
	if path == "" {
		return withSourceMeta(tools.ErrorResultf(`invalid_write_args: missing path (required keys: "path","content"; aliases "file_path"/"filename" are fallback only)`)), nil
	}

	fullPath, denial, err := t.resolver.ResolveWritablePathForOperation("write", path, pathpolicy.ResolveOptionsFromContext(ctx))
	if err != nil {
		return withSourceMeta(tools.ErrorResult(err)), nil
	}
	if denial != nil {
		return pathpolicy.NewPathDenialResult(denial), nil
	}

	// Ensure parent directory exists
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return withSourceMeta(tools.ErrorResultf("create directory: %w", err)), nil
	}

	// Check if file already exists
	exists := false
	if _, err := os.Stat(fullPath); err == nil {
		exists = true
	}

	// Write file
	if err := os.WriteFile(fullPath, []byte(p.Content), 0644); err != nil {
		return withSourceMeta(tools.ErrorResultf("write file: %w", err)), nil
	}

	// Build result
	lines := strings.Count(p.Content, "\n")
	if !strings.HasSuffix(p.Content, "\n") && p.Content != "" {
		lines++
	}

	action := "Created"
	if exists {
		action = "Updated"
	}

	content := fmt.Sprintf("%s: %s\n+ %s", action, path, p.Content)
	summary := fmt.Sprintf("%s %d lines", action, lines)

	return withSourceMeta(tools.StringResultWithSummary(content, summary)), nil
}
