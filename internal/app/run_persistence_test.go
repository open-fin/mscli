package app

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	agentctx "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/agent/loop"
	"gitcode.com/mindspore/mscli/agent/session"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/tools"
	"gitcode.com/mindspore/mscli/ui/model"
)

type singleReplyProvider struct {
	content string
	usage   llm.Usage
	lastReq *llm.CompletionRequest
}

func (p *singleReplyProvider) Name() string {
	return "single-reply"
}

func (p *singleReplyProvider) Complete(_ context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	p.lastReq = req
	return &llm.CompletionResponse{Content: p.content, FinishReason: llm.FinishStop, Usage: p.usage}, nil
}

func (p *singleReplyProvider) CompleteStream(_ context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	copied.Tools = append([]llm.Tool(nil), req.Tools...)
	p.lastReq = &copied

	return &singleReplyIterator{
		chunks: []llm.StreamChunk{
			{Content: p.content, FinishReason: llm.FinishStop, Usage: &p.usage},
		},
	}, nil
}

func (p *singleReplyProvider) SupportsTools() bool {
	return true
}

func (p *singleReplyProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type singleReplyIterator struct {
	chunks []llm.StreamChunk
	index  int
}

func (it *singleReplyIterator) Next() (*llm.StreamChunk, error) {
	if it.index >= len(it.chunks) {
		return nil, io.EOF
	}
	chunk := it.chunks[it.index]
	it.index++
	return &chunk, nil
}

func (it *singleReplyIterator) Close() error {
	return nil
}

func TestRunTaskWithoutLLMDoesNotPersistSession(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	app := &Application{
		EventCh:    make(chan model.Event),
		llmReady:   false,
		session:    runtimeSession,
		ctxManager: ctxManager,
	}

	done := make(chan struct{})
	go func() {
		app.runTask("hello")
		close(done)
	}()

	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %q, want %q", ev.Type, model.AgentReply)
	}
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("event message = %q, want %q", ev.Message, provideAPIKeyFirstMsg)
	}

	if _, err := os.Stat(runtimeSession.Path()); !os.IsNotExist(err) {
		t.Fatalf("expected no trajectory without live llm activity, got %v", err)
	}
	snapshotPath := filepath.Join(filepath.Dir(runtimeSession.Path()), "snapshot.json")
	if _, err := os.Stat(snapshotPath); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot without live llm activity, got %v", err)
	}
	if got := app.exitResumeHint(); got != "" {
		t.Fatalf("expected no resume hint without live llm activity, got %q", got)
	}

	<-done
}

func TestRunTaskPersistsSessionAfterLiveLLMReply(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	provider := &singleReplyProvider{
		content: "hi there",
		usage: llm.Usage{
			PromptTokens:     1660,
			CompletionTokens: 149,
			TotalTokens:      1809,
			Raw:              json.RawMessage(`{"prompt_tokens":1660,"completion_tokens":149,"total_tokens":1809,"cached_tokens":32}`),
		},
	}
	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())
	engine.SetContextManager(ctxManager)

	app := &Application{
		Engine:     engine,
		EventCh:    make(chan model.Event, 32),
		llmReady:   true,
		session:    runtimeSession,
		ctxManager: ctxManager,
	}
	engine.SetTrajectoryRecorder(newTrajectoryRecorder(runtimeSession, ctxManager, workDir, nil, autoMemoryConfig{Resolved: true, Enabled: false}, app.noteLiveLLMActivity))

	app.runTask("hello")

	if _, err := os.Stat(runtimeSession.Path()); err != nil {
		t.Fatalf("expected trajectory after live llm reply, got %v", err)
	}
	snapshotPath := filepath.Join(filepath.Dir(runtimeSession.Path()), "snapshot.json")
	if _, err := os.Stat(snapshotPath); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot sidecar after live llm reply, got %v", err)
	}

	trajectory, err := os.ReadFile(runtimeSession.Path())
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	if !strings.Contains(string(trajectory), `"type":"user"`) {
		t.Fatalf("expected trajectory to contain user record, got %s", string(trajectory))
	}
	if !strings.Contains(string(trajectory), `"type":"assistant"`) {
		t.Fatalf("expected trajectory to contain assistant record, got %s", string(trajectory))
	}
	if got := app.exitResumeHint(); !strings.Contains(got, "mscli resume "+runtimeSession.ID()) {
		t.Fatalf("expected resume hint with session id after live llm reply, got %q", got)
	}

	loaded, err := session.LoadByID(workDir, runtimeSession.ID())
	if err != nil {
		t.Fatalf("load session for resume: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	usage := loaded.UsageSnapshot()
	if usage == nil {
		t.Fatal("UsageSnapshot() = nil, want provider usage")
	}
	if got, want := usage.Tokens, 1809; got != want {
		t.Fatalf("usage.Tokens = %d, want %d", got, want)
	}
	if got, want := usage.TokenScope, "total"; got != want {
		t.Fatalf("usage.TokenScope = %q, want %q", got, want)
	}
	if usage.Usage == nil {
		t.Fatal("usage.Usage = nil, want persisted canonical/raw usage")
	}
	if got, want := usage.Usage.CompletionTokens, 149; got != want {
		t.Fatalf("usage.Usage.CompletionTokens = %d, want %d", got, want)
	}
	if !jsonEqualRaw(t, usage.Usage.Raw, json.RawMessage(`{"prompt_tokens":1660,"completion_tokens":149,"total_tokens":1809,"cached_tokens":32}`)) {
		t.Fatalf("usage.Usage.Raw = %s, want semantic match", string(usage.Usage.Raw))
	}
}

func TestRunTaskInjectsMemoryAndMSCLIAsSeparateInitialUserMessage(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(workDir, "MSCLI.md"), []byte("project instructions"), 0o644); err != nil {
		t.Fatalf("write MSCLI.md: %v", err)
	}
	memoryDir := filepath.Join(t.TempDir(), "memory")
	t.Setenv("MSCLI_MEMORY_PATH", memoryDir)
	memoryCfg, err := resolveAutoMemoryConfig(workDir)
	if err != nil {
		t.Fatalf("resolveAutoMemoryConfig() error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(memoryDir, memoryIndexFilename), []byte("Remember batch size defaults to 8.\n"), 0o644); err != nil {
		t.Fatalf("write MEMORY.md: %v", err)
	}

	systemPrompt := buildEffectiveSystemPromptFromSummariesWithMemory(nil, memoryCfg)
	runtimeSession, err := session.Create(workDir, systemPrompt)
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 200000,
		ReserveTokens: 20000,
	})
	ctxManager.SetSystemPrompt(systemPrompt)

	provider := &singleReplyProvider{content: "ok"}
	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 200000,
		SystemPrompt:  systemPrompt,
	}, provider, tools.NewRegistry())
	engine.SetContextManager(ctxManager)

	app := &Application{
		Engine:       engine,
		EventCh:      make(chan model.Event, 32),
		WorkDir:      workDir,
		llmReady:     true,
		session:      runtimeSession,
		ctxManager:   ctxManager,
		memoryConfig: memoryCfg,
	}
	engine.SetTrajectoryRecorder(newTrajectoryRecorder(runtimeSession, ctxManager, workDir, nil, memoryCfg, app.noteLiveLLMActivity))

	app.runTask("hello")

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if len(provider.lastReq.Messages) < 3 {
		t.Fatalf("request messages = %d, want at least system + context + user", len(provider.lastReq.Messages))
	}
	system := provider.lastReq.Messages[0]
	if system.Role != "system" {
		t.Fatalf("first request message role = %q, want system", system.Role)
	}
	for _, want := range []string{
		"# auto memory",
		"Use read, write, edit, grep, and glob",
	} {
		if !strings.Contains(system.Content, want) {
			t.Fatalf("system prompt missing %q:\n%s", want, system.Content)
		}
	}
	for _, forbidden := range []string{"Remember batch size defaults to 8.", "project instructions"} {
		if strings.Contains(system.Content, forbidden) {
			t.Fatalf("system prompt contains %q:\n%s", forbidden, system.Content)
		}
	}

	contextUser := provider.lastReq.Messages[1]
	if contextUser.Role != "user" {
		t.Fatalf("second request message role = %q, want user", contextUser.Role)
	}
	for _, want := range []string{
		initialUserContextTag,
		"## Auto Memory",
		"Remember batch size defaults to 8.",
		"project instructions",
	} {
		if !strings.Contains(contextUser.Content, want) {
			t.Fatalf("context user message missing %q:\n%s", want, contextUser.Content)
		}
	}
	for _, forbidden := range []string{
		"# auto memory",
		"Use read, write, edit, grep, and glob",
		"## Types of memory",
		"## User Request",
		"hello",
	} {
		if strings.Contains(contextUser.Content, forbidden) {
			t.Fatalf("context user message contains %q:\n%s", forbidden, contextUser.Content)
		}
	}
	if got, want := len(contextUser.ContentParts), 2; got != want {
		t.Fatalf("context user content parts = %d, want %d: %#v", got, want, contextUser.ContentParts)
	}
	if !strings.Contains(contextUser.ContentParts[0].Text, "Remember batch size defaults to 8.") {
		t.Fatalf("memory content part missing memory index:\n%s", contextUser.ContentParts[0].Text)
	}
	if strings.Contains(contextUser.ContentParts[0].Text, "project instructions") {
		t.Fatalf("memory content part contains MSCLI.md content:\n%s", contextUser.ContentParts[0].Text)
	}
	if !strings.Contains(contextUser.ContentParts[1].Text, "project instructions") {
		t.Fatalf("MSCLI.md content part missing project instructions:\n%s", contextUser.ContentParts[1].Text)
	}
	if strings.Contains(contextUser.ContentParts[1].Text, "Remember batch size defaults to 8.") {
		t.Fatalf("MSCLI.md content part contains memory index:\n%s", contextUser.ContentParts[1].Text)
	}
	realUser := provider.lastReq.Messages[2]
	if realUser.Role != "user" {
		t.Fatalf("third request message role = %q, want user", realUser.Role)
	}
	if got, want := realUser.Content, "hello"; got != want {
		t.Fatalf("real user message = %q, want %q", got, want)
	}

	app.runTask("again")

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive second completion request")
	}
	messages := provider.lastReq.Messages
	last := messages[len(messages)-1]
	if got, want := last.Role, "user"; got != want {
		t.Fatalf("last message role = %q, want %q", got, want)
	}
	if got, want := last.Content, "again"; got != want {
		t.Fatalf("second task user message = %q, want %q", got, want)
	}
	reloaded, err := session.LoadByID(workDir, runtimeSession.ID())
	if err != nil {
		t.Fatalf("LoadByID() error = %v", err)
	}
	t.Cleanup(func() {
		_ = reloaded.Close()
	})
	replay := reloaded.ReplayEvents()
	var userInputs []string
	for _, ev := range replay {
		if ev.Type == model.UserInput {
			userInputs = append(userInputs, ev.Message)
		}
	}
	if got, want := userInputs, []string{"hello", "again"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("replay user inputs = %#v, want %#v", got, want)
	}
}

func TestBuildTaskInitialMessagesDoesNotInjectAfterPlainFirstUserMessage(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	memoryDir := filepath.Join(t.TempDir(), "memory")
	t.Setenv("MSCLI_MEMORY_PATH", memoryDir)
	memoryCfg, err := resolveAutoMemoryConfig(workDir)
	if err != nil {
		t.Fatalf("resolveAutoMemoryConfig() error = %v", err)
	}

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 200000,
		ReserveTokens: 20000,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("plain first request")); err != nil {
		t.Fatalf("AddMessage(user) error = %v", err)
	}

	if err := os.WriteFile(filepath.Join(workDir, "MSCLI.md"), []byte("late project instructions"), 0o644); err != nil {
		t.Fatalf("write MSCLI.md: %v", err)
	}
	if err := os.WriteFile(filepath.Join(memoryDir, memoryIndexFilename), []byte("late memory\n"), 0o644); err != nil {
		t.Fatalf("write MEMORY.md: %v", err)
	}

	app := &Application{
		WorkDir:      workDir,
		ctxManager:   ctxManager,
		memoryConfig: memoryCfg,
	}
	msgs, err := app.buildTaskInitialMessages()
	if err != nil {
		t.Fatalf("buildTaskInitialMessages() error = %v", err)
	}
	if len(msgs) != 0 {
		t.Fatalf("initial messages = %#v, want none", msgs)
	}
}

func jsonEqualRaw(t *testing.T, got, want json.RawMessage) bool {
	t.Helper()

	var gotValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("unmarshal got json: %v", err)
	}

	var wantValue any
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("unmarshal want json: %v", err)
	}

	return reflect.DeepEqual(gotValue, wantValue)
}
