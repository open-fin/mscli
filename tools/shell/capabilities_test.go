package shell

import (
	"testing"

	rshell "github.com/mindspore-lab/mindspore-cli/runtime/shell"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

func TestShellToolCapabilities(t *testing.T) {
	got := tools.CapabilitiesForTool(NewShellTool(rshell.NewRunner(rshell.Config{WorkDir: t.TempDir()})))
	if got.Kind != tools.KindShell {
		t.Fatalf("Kind = %q, want %q", got.Kind, tools.KindShell)
	}
	if got.ReadOnly {
		t.Fatal("ReadOnly = true, want false")
	}
	if !got.MutatesWorkspace {
		t.Fatal("MutatesWorkspace = false, want true")
	}
	if !got.LongRunning {
		t.Fatal("LongRunning = false, want true")
	}
	if !got.SupportsStreaming {
		t.Fatal("SupportsStreaming = false, want true")
	}
}
