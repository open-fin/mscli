package app

import (
	"bytes"
	"strings"
	"testing"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/permission"
	"github.com/mindspore-lab/mindspore-cli/tools"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestRunHeadlessIssueCommandBuildsFixTaskAndEnablesYolo(t *testing.T) {
	provider := &singleReplyProvider{content: "fixed"}
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())
	engine.SetContextManager(ctxManager)

	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{})
	engine.SetPermissionService(permSvc)

	app := &Application{
		Engine:      engine,
		EventCh:     make(chan model.Event, 8),
		llmReady:    true,
		ctxManager:  ctxManager,
		permService: permSvc,
		WorkDir:     t.TempDir(),
	}

	var out bytes.Buffer
	if err := app.runHeadlessIssueCommand("fix", "broken training", &out); err != nil {
		t.Fatalf("run headless fix: %v", err)
	}
	if got := out.String(); got != "fixed\n" {
		t.Fatalf("stdout = %q, want %q", got, "fixed\n")
	}
	if got := permSvc.Check("shell", ""); got != permission.PermissionAllowAlways {
		t.Fatalf("shell permission = %s, want %s", got, permission.PermissionAllowAlways)
	}

	msgs := ctxManager.GetNonSystemMessages()
	if len(msgs) == 0 {
		t.Fatal("expected user task in context")
	}
	task := msgs[0].Content
	for _, want := range []string{
		"in fix mode",
		"User problem: broken training",
	} {
		if !strings.Contains(task, want) {
			t.Fatalf("task should contain %q, got:\n%s", want, task)
		}
	}
}

func TestRunHeadlessExecCommandRunsRawTaskAndPrintsResumeHint(t *testing.T) {
	provider := &singleReplyProvider{content: "done"}
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())
	engine.SetContextManager(ctxManager)

	permSvc := permission.NewDefaultPermissionService(configs.PermissionsConfig{})
	engine.SetPermissionService(permSvc)

	workDir := t.TempDir()
	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	sessionID := runtimeSession.ID()

	app := &Application{
		Engine:      engine,
		EventCh:     make(chan model.Event, 8),
		llmReady:    true,
		ctxManager:  ctxManager,
		permService: permSvc,
		session:     runtimeSession,
		WorkDir:     workDir,
	}

	var out bytes.Buffer
	if err := app.runHeadlessCommand("exec", "inspect workspace", &out); err != nil {
		t.Fatalf("run headless exec: %v", err)
	}
	wantHint := "Resume the previous conversation with: mscli resume " + sessionID
	if got := out.String(); got != "done\n"+wantHint+"\n" {
		t.Fatalf("stdout = %q, want %q", got, "done\n"+wantHint+"\n")
	}
	if got := permSvc.Check("shell", ""); got != permission.PermissionAllowAlways {
		t.Fatalf("shell permission = %s, want %s", got, permission.PermissionAllowAlways)
	}

	msgs := ctxManager.GetNonSystemMessages()
	if len(msgs) == 0 {
		t.Fatal("expected user task in context")
	}
	if got := msgs[0].Content; got != "inspect workspace" {
		t.Fatalf("task = %q, want %q", got, "inspect workspace")
	}
}

func TestRunTaskHeadlessWithoutLLMPrintsUnavailableMessage(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	app := &Application{
		EventCh:    make(chan model.Event, 8),
		llmReady:   false,
		ctxManager: ctxManager,
	}

	var out bytes.Buffer
	if err := app.runTaskHeadless("hello", &out); err != nil {
		t.Fatalf("run headless without llm: %v", err)
	}
	if got := strings.TrimSpace(out.String()); got != provideAPIKeyFirstMsg {
		t.Fatalf("stdout = %q, want %q", got, provideAPIKeyFirstMsg)
	}
}
