package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"gitcode.com/mindspore/mscli/agent/session"
	"gitcode.com/mindspore/mscli/integrations/llm"
)

func TestTrajectoryRecorderSkipsMemoryCheckpointBackups(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	memoryDir := filepath.Join(t.TempDir(), "memory")
	if err := os.MkdirAll(memoryDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(memory) error = %v", err)
	}
	memoryFile := filepath.Join(memoryDir, memoryIndexFilename)
	if err := os.WriteFile(memoryFile, []byte("old memory\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(memory) error = %v", err)
	}

	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := runtimeSession.AppendUserInput("remember this"); err != nil {
		t.Fatalf("append user input: %v", err)
	}

	args, err := json.Marshal(map[string]string{
		"path":    memoryFile,
		"content": "new memory\n",
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	recorder := newTrajectoryRecorder(
		runtimeSession,
		nil,
		workDir,
		autoMemoryConfig{Resolved: true, Enabled: true, Dir: memoryDir},
		nil,
	)

	err = recorder.PrepareFileMutation(llm.ToolCall{
		Function: llm.ToolCallFunc{
			Name:      "write",
			Arguments: args,
		},
	})
	if err != nil {
		t.Fatalf("PrepareFileMutation() error = %v", err)
	}

	checkpoints := runtimeSession.ListCheckpoints()
	if got, want := len(checkpoints), 1; got != want {
		t.Fatalf("checkpoint count = %d, want %d", got, want)
	}
	if checkpoints[0].HasCodeRestore {
		t.Fatal("memory write created checkpoint backup, want skipped")
	}
}
