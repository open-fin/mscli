package shell

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	rshell "gitcode.com/mindspore/mscli/runtime/shell"
	"gitcode.com/mindspore/mscli/tools"
)

func TestShellToolExecute_DoesNotDuplicateCommandOrExit0InContent(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 2 * time.Second,
	})
	tool := NewShellTool(runner, ".")

	result, err := tool.Execute(context.Background(), []byte(`{"command":"printf 'hello\\n'"}`))
	if err != nil {
		t.Fatalf("execute shell tool: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}

	if strings.Contains(result.Content, "$ printf") {
		t.Fatalf("expected content without command echo, got:\n%s", result.Content)
	}
	if strings.Contains(result.Content, "exit status 0") {
		t.Fatalf("expected content without exit status, got:\n%s", result.Content)
	}
	if strings.TrimSpace(result.Summary) == "exit 0" {
		t.Fatalf("expected summary not to be 'exit 0'")
	}
	if got := result.Meta[tools.MetaSource]; got != tools.SourceShell {
		t.Fatalf("source meta = %#v, want %q", got, tools.SourceShell)
	}
	if got := result.Meta[tools.MetaExitCode]; got != 0 {
		t.Fatalf("exit code meta = %#v, want 0", got)
	}
	if got := result.Meta[tools.MetaTruncated]; got != false {
		t.Fatalf("truncated meta = %#v, want false", got)
	}
}

func TestShellToolExecute_NonzeroExitSetsFailedStatus(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 2 * time.Second,
	})
	tool := NewShellTool(runner)

	result, err := tool.Execute(context.Background(), []byte(`{"command":"printf 'nope\\n' >&2; exit 7"}`))
	if err != nil {
		t.Fatalf("execute shell tool: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}
	if got, want := result.Summary, "exit 7"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
	if got := result.Meta[tools.MetaStatus]; got != tools.StatusFailed {
		t.Fatalf("status meta = %#v, want failed (meta %#v)", got, result.Meta)
	}
	if got := result.Meta[tools.MetaExitCode]; got != 7 {
		t.Fatalf("exit code meta = %#v, want 7", got)
	}
}

func TestShellToolExecute_ToolTimeoutSetsInterruptedStatus(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 5 * time.Second,
	})
	tool := NewShellTool(runner)

	result, err := tool.Execute(context.Background(), []byte(`{"command":"sleep 2","timeout":1}`))
	if err != nil {
		t.Fatalf("execute shell tool: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}
	if got := result.Meta[tools.MetaStatus]; got != tools.StatusInterrupted {
		t.Fatalf("status meta = %#v, want interrupted (summary %q, meta %#v)", got, result.Summary, result.Meta)
	}
}

func TestShellToolExecuteStream_EmitsStartedAndOutput(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 2 * time.Second,
	})
	tool := NewShellTool(runner, ".")

	var (
		mu      sync.Mutex
		updates []tools.StreamEvent
	)
	result, err := tool.ExecuteStream(context.Background(), []byte(`{"command":"printf 'hello\\n'; printf 'warn\\n' >&2"}`), func(ev tools.StreamEvent) {
		mu.Lock()
		updates = append(updates, ev)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("execute shell tool stream: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}
	if len(updates) == 0 || updates[0].Type != tools.StreamEventStarted {
		t.Fatalf("expected first update to be started, got %#v", updates)
	}

	var sawStdout, sawStderr bool
	for _, update := range updates {
		if update.Type != tools.StreamEventOutput {
			continue
		}
		if strings.Contains(update.Message, "hello") {
			sawStdout = true
		}
		if strings.Contains(update.Message, "[stderr] warn") {
			sawStderr = true
		}
	}
	if !sawStdout {
		t.Fatalf("expected stdout update, got %#v", updates)
	}
	if !sawStderr {
		t.Fatalf("expected stderr update, got %#v", updates)
	}
}

func TestShellToolExecute_LargeOutputRunnerTruncation(t *testing.T) {
	// The shell runner truncates output at 64KB (maxOutputBytes).
	// Our tool-level SpillResult threshold is 100KB, so the runner's
	// truncation fires first. This test verifies the tool handles
	// runner-truncated output gracefully without double-truncation.
	tmp := t.TempDir()
	largeFile := filepath.Join(tmp, "large.txt")
	content := strings.Repeat("x", 128*1024)
	if err := os.WriteFile(largeFile, []byte(content+"\n"), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	runner := rshell.NewRunner(rshell.Config{
		WorkDir: tmp,
		Timeout: 2 * time.Second,
	})
	tool := NewShellTool(runner, tmp)

	result, err := tool.Execute(context.Background(), []byte(`{"command":"cat large.txt"}`))
	if err != nil {
		t.Fatalf("execute shell tool: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}

	// Runner truncates at 64KB, so content should contain runner's mark
	if !strings.Contains(result.Content, "[output truncated]") {
		t.Errorf("expected runner truncation mark, got: %s", result.Content)
	}
	// Our 100KB spill should NOT fire (runner already capped it)
	if strings.Contains(result.Content, "Result too large") {
		t.Errorf("unexpected tool-level spill notice")
	}
	if strings.Contains(result.Summary, "truncated") {
		t.Errorf("unexpected tool-level truncated summary, got: %s", result.Summary)
	}
}

func TestShellToolExecuteStream_ReturnsInterruptedSummaryWithPartialOutput(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 5 * time.Second,
	})
	tool := NewShellTool(runner, ".")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var once sync.Once
	result, err := tool.ExecuteStream(ctx, []byte(`{"command":"printf 'hello\\n'; sleep 5; printf 'done\\n'"}`), func(ev tools.StreamEvent) {
		if ev.Type == tools.StreamEventOutput && strings.Contains(ev.Message, "hello") {
			once.Do(cancel)
		}
	})
	if err != nil {
		t.Fatalf("execute shell tool stream: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}
	if got, want := strings.TrimSpace(result.Summary), "interrupted"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
	if !strings.Contains(result.Content, "hello") {
		t.Fatalf("expected partial stdout preserved, got:\n%s", result.Content)
	}
	if strings.Contains(result.Content, "done") {
		t.Fatalf("expected canceled command to omit trailing output, got:\n%s", result.Content)
	}
}
