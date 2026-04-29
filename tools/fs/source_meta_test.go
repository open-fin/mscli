package fs

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"gitcode.com/mindspore/mscli/tools"
)

func TestFilesystemToolsSetSourceMeta(t *testing.T) {
	workDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(workDir, "a.txt"), []byte("hello\n"), 0644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tests := []struct {
		name string
		tool tools.Tool
		args []byte
	}{
		{name: "read", tool: NewReadTool(workDir), args: []byte(`{"path":"a.txt"}`)},
		{name: "grep", tool: NewGrepTool(workDir), args: []byte(`{"pattern":"hello","path":"."}`)},
		{name: "glob", tool: NewGlobTool(workDir), args: []byte(`{"pattern":"*.txt"}`)},
		{name: "edit", tool: NewEditTool(workDir), args: []byte(`{"path":"a.txt","old_string":"hello\n","new_string":"hello world\n"}`)},
		{name: "write", tool: NewWriteTool(workDir), args: []byte(`{"path":"b.txt","content":"new\n"}`)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.tool.Execute(context.Background(), tt.args)
			if err != nil {
				t.Fatalf("Execute error: %v", err)
			}
			if result.Error != nil {
				t.Fatalf("Result error: %v", result.Error)
			}
			if got := result.Meta[tools.MetaSource]; got != tools.SourceFS {
				t.Fatalf("source meta = %#v, want %q (meta %#v)", got, tools.SourceFS, result.Meta)
			}
		})
	}
}
