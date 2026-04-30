package app

import (
	"strings"
	"testing"
	"time"

	agentctx "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/agent/session"
	"gitcode.com/mindspore/mscli/configs"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/permission"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestCmdClearRotatesSessionAndLeavesPreviousSnapshotUntouched(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})
	if err := runtimeSession.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage(user) failed: %v", err)
	}
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("hi")); err != nil {
		t.Fatalf("AddMessage(assistant) failed: %v", err)
	}
	if err := runtimeSession.SaveSnapshot("system prompt", ctxManager.GetNonSystemMessages()); err != nil {
		t.Fatalf("SaveSnapshot() error = %v", err)
	}
	beforeSystemPrompt, beforeMessages := runtimeSession.RestoreContext()

	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{
		DefaultLevel: "ask",
	})
	store := permission.NewMemoryPermissionStore()
	if err := store.SaveDecision(permission.PermissionDecision{
		Tool:      "shell",
		Action:    "npm test ./...",
		Level:     permission.PermissionAllowSession,
		Timestamp: time.Now(),
	}); err != nil {
		t.Fatalf("SaveDecision() error = %v", err)
	}
	permSvc.SetStore(store)

	app := newModelCommandTestApp()
	app.WorkDir = workDir
	app.session = runtimeSession
	app.ctxManager = ctxManager
	app.permService = permSvc
	app.sessionLLMActivity.Store(true)
	app.sessionStoreReady.Store(true)
	oldSessionID := runtimeSession.ID()

	app.cmdClear()

	ev := drainUntilClearScreen(t, app)
	if got, want := ev.Message, "Chat history cleared."; got != want {
		t.Fatalf("ClearScreen message = %q, want %q", got, want)
	}
	if got, want := ev.Summary, "Resume the previous conversation with: `/resume "+oldSessionID+"`"; got != want {
		t.Fatalf("ClearScreen summary = %q, want %q", got, want)
	}

	if got := len(ctxManager.GetNonSystemMessages()); got != 0 {
		t.Fatalf("non-system messages after clear = %d, want 0", got)
	}
	if got := app.session.ID(); got == oldSessionID {
		t.Fatalf("session id after clear = %q, want a new session id", got)
	}
	if got := app.sessionLLMActivity.Load(); got {
		t.Fatal("sessionLLMActivity should reset after clear")
	}
	if got := app.sessionStoreReady.Load(); got {
		t.Fatal("sessionStoreReady should reset after clear")
	}
	if got := permSvc.CheckCommand("npm test ./..."); got != permission.PermissionAsk {
		t.Fatalf("CheckCommand(npm test ./...) after clear = %s, want %s", got, permission.PermissionAsk)
	}

	reloadedPrevious, err := session.LoadByID(workDir, oldSessionID)
	if err != nil {
		t.Fatalf("LoadByID() error = %v", err)
	}
	t.Cleanup(func() {
		_ = reloadedPrevious.Close()
	})

	systemPrompt, restoredMessages := reloadedPrevious.RestoreContext()
	if got, want := systemPrompt, beforeSystemPrompt; got != want {
		t.Fatalf("previous system prompt = %q, want %q", got, want)
	}
	if got, want := len(restoredMessages), len(beforeMessages); got != want {
		t.Fatalf("previous restored messages after clear = %d, want %d", got, want)
	}

	reloadedCurrent, err := session.LoadByID(workDir, app.session.ID())
	if err != nil {
		t.Fatalf("LoadByID(new session) error = %v", err)
	}
	t.Cleanup(func() {
		_ = reloadedCurrent.Close()
	})

	systemPrompt, restoredMessages = reloadedCurrent.RestoreContext()
	if systemPrompt == "system prompt" {
		t.Fatalf("new session system prompt was not rebuilt")
	}
	if !strings.Contains(systemPrompt, "You are MindSpore CLI") {
		t.Fatalf("new session system prompt missing rebuilt sections: %q", systemPrompt)
	}
	if strings.Contains(systemPrompt, "# auto memory") {
		t.Fatalf("new session system prompt contains memory instructions: %q", systemPrompt)
	}
	if got := len(restoredMessages); got != 0 {
		t.Fatalf("new session restored messages after clear = %d, want 0", got)
	}
}

func drainUntilClearScreen(t *testing.T, app *Application) model.Event {
	t.Helper()

	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()

	for {
		select {
		case ev := <-app.EventCh:
			if ev.Type == model.ToolError {
				t.Fatalf("unexpected tool error while clearing: %#v", ev)
			}
			if ev.Type == model.ClearScreen {
				return ev
			}
		case <-timer.C:
			t.Fatal("timed out waiting for clear screen event")
		}
	}
}
