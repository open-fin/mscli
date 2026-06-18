package fs

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gitcode.com/mindspore/mscli/internal/pathpolicy"
)

func TestMemoryRootPathOptionsAllowConfiguredRootOnly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	workDir := t.TempDir()
	memoryRoot := filepath.Join(home, ".mscli", "sessions", "work-key", "memory")
	siblingSessionRoot := filepath.Join(home, ".mscli", "sessions", "work-key", "sess_000001")
	otherHomeRoot := filepath.Join(home, ".mscli", "other")
	for _, dir := range []string{memoryRoot, siblingSessionRoot, otherHomeRoot} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("MkdirAll(%s) error = %v", dir, err)
		}
	}
	if err := os.WriteFile(filepath.Join(siblingSessionRoot, "trajectory.jsonl"), []byte("needle"), 0o644); err != nil {
		t.Fatalf("WriteFile(sibling) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(otherHomeRoot, "secret.txt"), []byte("needle"), 0o644); err != nil {
		t.Fatalf("WriteFile(other home) error = %v", err)
	}

	resolver := pathpolicy.NewResolver(pathpolicy.NewPathPolicyWithWriteRoots(workDir, nil, []string{memoryRoot}, nil))
	memoryFile := filepath.Join(memoryRoot, "MEMORY.md")

	writeResult, err := NewWriteToolWithResolver(resolver).Execute(context.Background(), mustJSON(t, map[string]string{
		"path":    memoryFile,
		"content": "needle memory\n",
	}))
	if err != nil {
		t.Fatalf("write Execute() error = %v", err)
	}
	if writeResult.Error != nil {
		t.Fatalf("write result error = %v", writeResult.Error)
	}

	readResult, err := NewReadToolWithResolver(resolver).Execute(context.Background(), mustJSON(t, map[string]string{
		"path": memoryFile,
	}))
	if err != nil {
		t.Fatalf("read Execute() error = %v", err)
	}
	if readResult.Error != nil {
		t.Fatalf("read result error = %v", readResult.Error)
	}
	if !strings.Contains(readResult.Content, "needle memory") {
		t.Fatalf("read content = %q, want memory content", readResult.Content)
	}

	grepResult, err := NewGrepToolWithResolver(resolver).Execute(context.Background(), mustJSON(t, map[string]any{
		"pattern":        "needle",
		"path":           memoryRoot,
		"case_sensitive": true,
	}))
	if err != nil {
		t.Fatalf("grep Execute() error = %v", err)
	}
	if grepResult.Error != nil {
		t.Fatalf("grep result error = %v", grepResult.Error)
	}
	if !strings.Contains(grepResult.Content, "MEMORY.md:1:needle memory") {
		t.Fatalf("grep content = %q, want memory match", grepResult.Content)
	}

	globResult, err := NewGlobToolWithResolver(resolver).Execute(context.Background(), mustJSON(t, map[string]any{
		"pattern": "*.md",
		"path":    memoryRoot,
	}))
	if err != nil {
		t.Fatalf("glob Execute() error = %v", err)
	}
	if globResult.Error != nil {
		t.Fatalf("glob result error = %v", globResult.Error)
	}
	if !strings.Contains(globResult.Content, "MEMORY.md") {
		t.Fatalf("glob content = %q, want MEMORY.md", globResult.Content)
	}

	for _, path := range []string{
		filepath.Join(siblingSessionRoot, "trajectory.jsonl"),
		filepath.Join(otherHomeRoot, "secret.txt"),
	} {
		result, err := NewReadToolWithResolver(resolver).Execute(context.Background(), mustJSON(t, map[string]string{
			"path": path,
		}))
		if err != nil {
			t.Fatalf("read rejected path Execute() error = %v", err)
		}
		if result.Error == nil {
			t.Fatalf("read rejected path %s succeeded, want error", path)
		}
		if result.Error != pathpolicy.ErrExternalPathDenied {
			t.Fatalf("read rejected path error = %q, want external path denial", result.Error)
		}
	}
}

func mustJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()

	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return data
}
