package fs

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gitcode.com/mindspore/mscli/tools"
)

func TestReadTool_Execute_LargeFileOverflow(t *testing.T) {
	tmp := t.TempDir()
	largeFile := filepath.Join(tmp, "large.txt")
	content := strings.Repeat("x", tools.DefaultMaxResultBytes+200)
	if err := os.WriteFile(largeFile, []byte(content), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	tool := NewReadTool(tmp)
	result, err := tool.Execute(context.Background(), []byte(`{"path":"large.txt"}`))
	if err != nil {
		t.Fatalf("execute read tool: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}

	if !strings.Contains(result.Content, "Result too large") {
		t.Errorf("expected overflow notice, got: %s", result.Content)
	}
	if !strings.Contains(result.Summary, "truncated") {
		t.Errorf("expected truncated summary, got: %s", result.Summary)
	}
}

func TestReadTool_Execute_SmallFileNoOverflow(t *testing.T) {
	tmp := t.TempDir()
	smallFile := filepath.Join(tmp, "small.txt")
	content := "hello world\n"
	if err := os.WriteFile(smallFile, []byte(content), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	tool := NewReadTool(tmp)
	result, err := tool.Execute(context.Background(), []byte(`{"path":"small.txt"}`))
	if err != nil {
		t.Fatalf("execute read tool: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}

	if result.Content != "hello world" {
		t.Errorf("content = %q, want %q", result.Content, "hello world")
	}
	if strings.Contains(result.Summary, "truncated") {
		t.Errorf("expected non-truncated summary, got: %s", result.Summary)
	}
}

func TestReadToolReturnsPlaceholderForEmptyFile(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "empty.txt"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	tool := NewReadTool(root)
	args, err := json.Marshal(map[string]string{"path": "empty.txt"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Content != emptyFilePlaceholder {
		t.Fatalf("read content = %q, want empty-file placeholder", result.Content)
	}
	if result.Summary != "0 lines" {
		t.Fatalf("read summary = %q, want 0 lines", result.Summary)
	}
}

func TestReadToolPreservesWhitespaceOnlyFile(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "spaces.txt"), []byte("   "), 0o644); err != nil {
		t.Fatal(err)
	}

	tool := NewReadTool(root)
	args, err := json.Marshal(map[string]string{"path": "spaces.txt"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Content != "   " {
		t.Fatalf("read content = %q, want whitespace preserved", result.Content)
	}
}
