package fs

import (
	"testing"

	"gitcode.com/mindspore/mscli/tools"
)

func TestFilesystemToolCapabilities(t *testing.T) {
	tests := []struct {
		name             string
		tool             tools.Tool
		readOnly         bool
		mutatesWorkspace bool
	}{
		{name: "read", tool: NewReadTool(t.TempDir()), readOnly: true},
		{name: "grep", tool: NewGrepTool(t.TempDir()), readOnly: true},
		{name: "glob", tool: NewGlobTool(t.TempDir()), readOnly: true},
		{name: "edit", tool: NewEditTool(t.TempDir()), mutatesWorkspace: true},
		{name: "write", tool: NewWriteTool(t.TempDir()), mutatesWorkspace: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tools.CapabilitiesForTool(tt.tool)
			if got.Kind != tools.KindFilesystem {
				t.Fatalf("Kind = %q, want %q", got.Kind, tools.KindFilesystem)
			}
			if got.ReadOnly != tt.readOnly {
				t.Fatalf("ReadOnly = %v, want %v", got.ReadOnly, tt.readOnly)
			}
			if got.MutatesWorkspace != tt.mutatesWorkspace {
				t.Fatalf("MutatesWorkspace = %v, want %v", got.MutatesWorkspace, tt.mutatesWorkspace)
			}
		})
	}
}
