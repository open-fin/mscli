package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	ctxmanager "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/integrations/skills"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestExpandInputTextExpandsStandaloneTokensAndEscapes(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "a.txt", "alpha")
	writeTestFile(t, root, "b.txt", "beta")

	app := &Application{WorkDir: root}
	got, err := app.expandInputText("read @a.txt and @@literal then @b.txt")
	if err != nil {
		t.Fatalf("expandInputText returned error: %v", err)
	}

	if !strings.Contains(got, `[file path="`+filepath.ToSlash(filepath.Join(root, "a.txt"))+`"]`) {
		t.Fatalf("expected a.txt contents to be expanded, got %q", got)
	}
	if !strings.Contains(got, `[file path="`+filepath.ToSlash(filepath.Join(root, "b.txt"))+`"]`) {
		t.Fatalf("expected b.txt contents to be expanded, got %q", got)
	}
	if strings.Contains(got, "alpha") || strings.Contains(got, "beta") {
		t.Fatalf("expected file contents not to be inlined, got %q", got)
	}
	if !strings.Contains(got, "@literal") {
		t.Fatalf("expected @@ escape to keep literal @, got %q", got)
	}
}

func TestExpandInputTextLeavesUnsupportedAtFormsUnchanged(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "context")

	app := &Application{WorkDir: root}
	input := "see @ctx.txt, (@ctx.txt) user@ctx.txt"
	got, err := app.expandInputText(input)
	if err != nil {
		t.Fatalf("expandInputText returned error: %v", err)
	}
	if got != input {
		t.Fatalf("unsupported @ forms should stay unchanged, got %q", got)
	}
}

func TestExpandInputTextRejectsUnsafeFiles(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "dir"), 0o755); err != nil {
		t.Fatal(err)
	}

	app := &Application{WorkDir: root}
	tests := []struct {
		input string
		want  string
	}{
		{"@missing.txt", "file not found"},
		{"@dir", "path is a directory"},
		{"@../escape.txt", "path escapes working directory"},
	}

	for _, tc := range tests {
		_, err := app.expandInputText(tc.input)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("expandInputText(%q) error = %v, want substring %q", tc.input, err, tc.want)
		}
	}
}

func TestProcessInputExpandsPlainChatBeforeRunTask(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "context payload")

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
	}

	app.processInput("please read @ctx.txt")

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("expected unavailable reply, got %q", ev.Message)
	}

	msgs := app.ctxManager.GetNonSystemMessages()
	if len(msgs) < 1 || msgs[0].Role != "user" {
		t.Fatalf("expected recorded user message, got %#v", msgs)
	}
	if !strings.Contains(msgs[0].Content, `[file path="`+filepath.ToSlash(filepath.Join(root, "ctx.txt"))+`"]`) {
		t.Fatalf("expected expanded plain chat to be recorded, got %q", msgs[0].Content)
	}
	if strings.Contains(msgs[0].Content, "context payload") {
		t.Fatalf("expected file content not to be recorded inline, got %q", msgs[0].Content)
	}
}

func TestProcessInputEmitsExpandedUserInputEvent(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "context payload")

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
	}

	app.processInput("please read @ctx.txt")

	ev := drainUntilEventType(t, app, model.UserInput)
	if !strings.Contains(ev.Message, `[file path="`+filepath.ToSlash(filepath.Join(root, "ctx.txt"))+`"]`) {
		t.Fatalf("expected expanded user input event, got %q", ev.Message)
	}
	if strings.Contains(ev.Message, "context payload") {
		t.Fatalf("expected user input event not to inline file content, got %q", ev.Message)
	}
}

func TestHandleCommandFixExpandsPrompt(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "fix context")

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
	}

	app.handleCommand(`/fix ISSUE-42 @ctx.txt`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("expected unavailable reply, got %q", ev.Message)
	}
	msgs := app.ctxManager.GetNonSystemMessages()
	if !containsUserMessage(msgs, "ISSUE-42") {
		t.Fatalf("expected issue-like text to be preserved as prompt text, got %#v", msgs)
	}
	if !containsUserMessage(msgs, `[file path="`+filepath.ToSlash(filepath.Join(root, "ctx.txt"))+`"]`) {
		t.Fatalf("expected expanded prompt remainder, got %#v", msgs)
	}
	if containsUserMessage(msgs, "fix context") {
		t.Fatalf("expected fix prompt not to inline file content, got %#v", msgs)
	}
}

func TestHandleCommandFixFileFirstStaysFreeTextMode(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "fix context")

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
	}

	app.handleCommand(`/fix @ctx.txt ISSUE-42`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("expected unavailable reply, got %q", ev.Message)
	}
	msgs := app.ctxManager.GetNonSystemMessages()
	if containsUserMessage(msgs, "Issue: ISSUE-42") {
		t.Fatalf("file-first input should not switch to issue mode, got %#v", msgs)
	}
	if !containsUserMessage(msgs, filepath.ToSlash(filepath.Join(root, "ctx.txt"))) || !containsUserMessage(msgs, "ISSUE-42") {
		t.Fatalf("expected free-text mode with expanded content, got %#v", msgs)
	}
	if containsUserMessage(msgs, "fix context") {
		t.Fatalf("expected free-text mode not to inline file content, got %#v", msgs)
	}
}

func TestHandleCommandSkillAndAliasExpandOnlyRequestRemainder(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "req.txt", "skill request")
	skillDir := filepath.Join(root, "skills")
	createTestSkill(t, skillDir, "demo")

	app := &Application{
		WorkDir:     root,
		EventCh:     make(chan model.Event, 16),
		llmReady:    false,
		ctxManager:  ctxmanager.NewManager(ctxmanager.ManagerConfig{ContextWindow: 24000, ReserveTokens: 4000}),
		skillLoader: skills.NewLoader(skillDir),
	}

	app.handleCommand(`/skill demo @req.txt`)
	drainUntilEventType(t, app, model.ToolSkill)
	drainUntilEventType(t, app, model.AgentReply)

	msgs := app.ctxManager.GetNonSystemMessages()
	if !containsUserMessage(msgs, `[file path="`+filepath.ToSlash(filepath.Join(root, "req.txt"))+`"]`) {
		t.Fatalf("expected /skill request to be expanded, got %#v", msgs)
	}

	app.ctxManager.Clear()
	app.handleCommand(`/demo @req.txt`)
	drainUntilEventType(t, app, model.ToolSkill)
	drainUntilEventType(t, app, model.AgentReply)

	msgs = app.ctxManager.GetNonSystemMessages()
	if !containsUserMessage(msgs, `[file path="`+filepath.ToSlash(filepath.Join(root, "req.txt"))+`"]`) {
		t.Fatalf("expected skill alias request to be expanded, got %#v", msgs)
	}
}

func TestExpandInputTextExternalAtFileUsesReadableResolver(t *testing.T) {
	root := t.TempDir()
	external := t.TempDir()
	writeTestFile(t, external, "ctx.txt", "external context")
	resolver := pathpolicy.NewResolver(pathpolicy.NewPathPolicy(root, []string{external}, nil))
	app := &Application{WorkDir: root, pathResolver: resolver}

	got, err := app.expandInputText("read @" + filepath.Join(external, "ctx.txt"))
	if err != nil {
		t.Fatalf("expandInputText returned error: %v", err)
	}
	if !strings.Contains(got, `[file path="`+filepath.ToSlash(filepath.Join(external, "ctx.txt"))+`"]`) {
		t.Fatalf("expected external @file to be expanded, got %q", got)
	}
	if strings.Contains(got, "external context") {
		t.Fatalf("expected external file content not to be inlined, got %q", got)
	}
}

func TestProcessInputExternalAtFilePromptsAndRetriesAfterApproval(t *testing.T) {
	root := t.TempDir()
	external := t.TempDir()
	writeTestFile(t, external, "ctx.txt", "external context")
	policy := pathpolicy.NewPathPolicy(root, nil, nil)
	app := &Application{
		WorkDir:      root,
		EventCh:      make(chan model.Event, 16),
		llmReady:     false,
		pathResolver: pathpolicy.NewResolver(policy),
		ctxManager:   ctxmanager.NewManager(ctxmanager.ManagerConfig{ContextWindow: 24000, ReserveTokens: 4000}),
	}
	app.pathAuthorizer = NewPathAuthorizer(app.EventCh, policy, nil)

	app.processInput("please read @" + filepath.Join(external, "ctx.txt"))
	prompt := drainUntilEventType(t, app, model.PermissionPrompt)
	if prompt.Permission == nil || prompt.Permission.Title != "External read access" {
		t.Fatalf("expected external read prompt, got %#v", prompt.Permission)
	}

	app.processInput("1")
	drainUntilEventType(t, app, model.AgentReply)
	msgs := app.ctxManager.GetNonSystemMessages()
	if !containsUserMessage(msgs, `[file path="`+filepath.ToSlash(filepath.Join(external, "ctx.txt"))+`"]`) {
		t.Fatalf("expected approved external @file to be recorded, got %#v", msgs)
	}
	if containsUserMessage(msgs, "external context") {
		t.Fatalf("expected external file content not to be inlined, got %#v", msgs)
	}
}
func TestExpandInputTextExpandsQuotedAtPathWithSpaces(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "dir with spaces/ctx file.md", "context")

	app := &Application{WorkDir: root}
	got, err := app.expandInputText(`read @"dir with spaces/ctx file.md"`)
	if err != nil {
		t.Fatalf("expandInputText returned error: %v", err)
	}
	if !strings.Contains(got, `[file path="`+filepath.ToSlash(filepath.Join(root, "dir with spaces", "ctx file.md"))+`"]`) {
		t.Fatalf("expected quoted @file path to expand, got %q", got)
	}
}

func TestExpandInputTextRelativeEscapeDoesNotPromptForAuthorization(t *testing.T) {
	root := t.TempDir()
	policy := pathpolicy.NewPathPolicy(root, nil, nil)
	app := &Application{
		WorkDir:      root,
		EventCh:      make(chan model.Event, 4),
		pathResolver: pathpolicy.NewResolver(policy),
	}

	_, err := app.expandInputText("read @../../outside.txt")
	if err == nil || !strings.Contains(err.Error(), "path escapes working directory") {
		t.Fatalf("expected direct relative escape error, got %v", err)
	}
	if app.tryAuthorizeInputExpansion(err, nil) {
		t.Fatal("relative escape should not be treated as authorizable")
	}
}

func containsUserMessage(msgs []llm.Message, needle string) bool {
	for _, msg := range msgs {
		if msg.Role == "user" && strings.Contains(msg.Content, needle) {
			return true
		}
	}
	return false
}

func writeTestFile(t *testing.T, root, relativePath, content string) {
	t.Helper()
	path := filepath.Join(root, relativePath)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func createTestSkill(t *testing.T, skillsRoot, name string) {
	t.Helper()
	skillPath := filepath.Join(skillsRoot, name)
	if err := os.MkdirAll(skillPath, 0o755); err != nil {
		t.Fatal(err)
	}
	content := "---\nname: " + name + "\ndescription: demo skill\n---\n\nbody"
	if err := os.WriteFile(filepath.Join(skillPath, "SKILL.md"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
