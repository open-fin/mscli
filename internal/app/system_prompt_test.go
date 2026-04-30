package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestBuildInitialUserContextPromptIncludesAutoMemoryPathAndIndex(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	workDir := t.TempDir()
	bucketDir, err := session.BucketDirForWorkDir(workDir)
	if err != nil {
		t.Fatalf("BucketDirForWorkDir() error = %v", err)
	}
	memoryDir := filepath.Join(bucketDir, "memory")
	if err := os.MkdirAll(memoryDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(memory) error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(memoryDir, memoryIndexFilename), []byte("Remember batch size defaults to 8.\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(MEMORY.md) error = %v", err)
	}

	systemPrompt, memoryCfg, err := buildEffectiveSystemPrompt(workDir, nil)
	if err != nil {
		t.Fatalf("buildEffectiveSystemPrompt() error = %v", err)
	}
	if strings.Contains(systemPrompt, "# auto memory") {
		t.Fatalf("system prompt contains memory instructions:\n%s", systemPrompt)
	}

	if !memoryCfg.Enabled {
		t.Fatal("memory config disabled, want enabled")
	}
	if got, want := memoryCfg.Dir, memoryDir; got != want {
		t.Fatalf("memory dir = %q, want %q", got, want)
	}

	prompt, err := buildInitialUserContextPrompt(workDir, memoryCfg)
	if err != nil {
		t.Fatalf("buildInitialUserContextPrompt() error = %v", err)
	}
	for _, want := range []string{
		"# auto memory",
		memoryDir,
		"Use read, write, edit, grep, and glob",
		"## Types of memory",
		"Remember batch size defaults to 8.",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestBuildInitialUserContextPromptCapsMemoryIndex(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	workDir := t.TempDir()
	bucketDir, err := session.BucketDirForWorkDir(workDir)
	if err != nil {
		t.Fatalf("BucketDirForWorkDir() error = %v", err)
	}
	memoryDir := filepath.Join(bucketDir, "memory")
	if err := os.MkdirAll(memoryDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(memory) error = %v", err)
	}
	content := strings.Repeat("a", autoMemoryIndexMaxLen) + "TAIL"
	if err := os.WriteFile(filepath.Join(memoryDir, memoryIndexFilename), []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(MEMORY.md) error = %v", err)
	}

	_, memoryCfg, err := buildEffectiveSystemPrompt(workDir, nil)
	if err != nil {
		t.Fatalf("buildEffectiveSystemPrompt() error = %v", err)
	}
	prompt, err := buildInitialUserContextPrompt(workDir, memoryCfg)
	if err != nil {
		t.Fatalf("buildInitialUserContextPrompt() error = %v", err)
	}

	if strings.Contains(prompt, "TAIL") {
		t.Fatalf("prompt contains uncapped memory tail")
	}
	if !strings.Contains(prompt, "truncated to") {
		t.Fatalf("prompt missing truncation notice")
	}
}

func TestBuildEffectiveSystemPromptRespectsMemoryDisabled(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_MEMORY_ENABLED", "false")

	workDir := t.TempDir()
	prompt, memoryCfg, err := buildEffectiveSystemPrompt(workDir, nil)
	if err != nil {
		t.Fatalf("buildEffectiveSystemPrompt() error = %v", err)
	}

	if memoryCfg.Enabled {
		t.Fatal("memory config enabled, want disabled")
	}
	if strings.Contains(prompt, "# auto memory") {
		t.Fatalf("prompt contains memory instructions while disabled:\n%s", prompt)
	}
	contextPrompt, err := buildInitialUserContextPrompt(workDir, memoryCfg)
	if err != nil {
		t.Fatalf("buildInitialUserContextPrompt() error = %v", err)
	}
	if strings.Contains(contextPrompt, "# auto memory") {
		t.Fatalf("initial user context contains memory instructions while disabled:\n%s", contextPrompt)
	}
	bucketDir, err := session.BucketDirForWorkDir(workDir)
	if err != nil {
		t.Fatalf("BucketDirForWorkDir() error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(bucketDir, "memory")); !os.IsNotExist(err) {
		t.Fatalf("memory dir stat error = %v, want not exist", err)
	}
}

func TestBuildInitialUserContextPromptUsesMemoryPathOverride(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	workDir := t.TempDir()
	overrideDir := filepath.Join(t.TempDir(), "custom-memory")
	t.Setenv("MSCLI_MEMORY_PATH", overrideDir)

	systemPrompt, memoryCfg, err := buildEffectiveSystemPrompt(workDir, nil)
	if err != nil {
		t.Fatalf("buildEffectiveSystemPrompt() error = %v", err)
	}

	if got, want := memoryCfg.Dir, overrideDir; got != want {
		t.Fatalf("memory dir = %q, want override %q", got, want)
	}
	if strings.Contains(systemPrompt, overrideDir) {
		t.Fatalf("system prompt contains override memory dir %q:\n%s", overrideDir, systemPrompt)
	}
	contextPrompt, err := buildInitialUserContextPrompt(workDir, memoryCfg)
	if err != nil {
		t.Fatalf("buildInitialUserContextPrompt() error = %v", err)
	}
	if !strings.Contains(contextPrompt, overrideDir) {
		t.Fatalf("initial user context missing override dir %q", overrideDir)
	}
	if _, err := os.Stat(overrideDir); err != nil {
		t.Fatalf("expected override memory dir created: %v", err)
	}
}

func TestBuildInitialUserContextPromptLoadsMSCLIInstructionsInPriorityOrder(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_MEMORY_ENABLED", "false")

	base := t.TempDir()
	workDir := filepath.Join(base, "repo", "pkg")
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(workDir) error = %v", err)
	}

	files := map[string]string{
		filepath.Join(home, ".mscli", "MSCLI.md"):         "global instructions",
		filepath.Join(base, "MSCLI.md"):                   "base instructions",
		filepath.Join(base, "repo", "MSCLI.md"):           "repo instructions",
		filepath.Join(base, "repo", ".mscli", "MSCLI.md"): "repo dot instructions",
		filepath.Join(workDir, "MSCLI.md"):                "pkg instructions",
	}
	for path, content := range files {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("MkdirAll(%s) error = %v", path, err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile(%s) error = %v", path, err)
		}
	}

	systemPrompt, memoryCfg, err := buildEffectiveSystemPrompt(workDir, nil)
	if err != nil {
		t.Fatalf("buildEffectiveSystemPrompt() error = %v", err)
	}
	if strings.Contains(systemPrompt, "global instructions") {
		t.Fatalf("system prompt contains MSCLI.md instructions:\n%s", systemPrompt)
	}
	prompt, err := buildInitialUserContextPrompt(workDir, memoryCfg)
	if err != nil {
		t.Fatalf("buildInitialUserContextPrompt() error = %v", err)
	}

	assertPromptOrder(t, prompt,
		"global instructions",
		"base instructions",
		"repo instructions",
		"repo dot instructions",
		"pkg instructions",
	)
}

func TestCmdClearKeepsMSCLIInstructionsOutOfSystemPrompt(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_MEMORY_ENABLED", "false")

	workDir := t.TempDir()
	instructionsPath := filepath.Join(workDir, "MSCLI.md")
	if err := os.WriteFile(instructionsPath, []byte("before clear"), 0o644); err != nil {
		t.Fatalf("WriteFile(before) error = %v", err)
	}

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("old prompt")

	app := newModelCommandTestApp()
	app.WorkDir = workDir
	app.ctxManager = ctxManager

	if err := os.WriteFile(instructionsPath, []byte("after clear"), 0o644); err != nil {
		t.Fatalf("WriteFile(after) error = %v", err)
	}

	app.cmdClear()

	_ = drainUntilEventType(t, app, model.ClearScreen)
	system := app.ctxManager.GetSystemPrompt()
	if system == nil {
		t.Fatal("system prompt nil after clear")
	}
	if strings.Contains(system.Content, "before clear") || strings.Contains(system.Content, "after clear") {
		t.Fatalf("system prompt contains MSCLI.md instructions after clear:\n%s", system.Content)
	}
}

func assertPromptOrder(t *testing.T, prompt string, parts ...string) {
	t.Helper()

	last := -1
	for _, part := range parts {
		idx := strings.Index(prompt, part)
		if idx < 0 {
			t.Fatalf("prompt missing %q:\n%s", part, prompt)
		}
		if idx <= last {
			t.Fatalf("prompt part %q appears out of order", part)
		}
		last = idx
	}
}
