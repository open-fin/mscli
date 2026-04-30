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
	engine.SetTrajectoryRecorder(newTrajectoryRecorder(runtimeSession, ctxManager, workDir, autoMemoryConfig{Resolved: true, Enabled: false}, app.noteLiveLLMActivity))

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

func TestRunTaskInjectsMemoryAndMSCLIIntoFirstUserMessage(t *testing.T) {
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

	systemPrompt := buildBaseSystemPrompt(nil)
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
	engine.SetTrajectoryRecorder(newTrajectoryRecorder(runtimeSession, ctxManager, workDir, memoryCfg, app.noteLiveLLMActivity))

	app.runTask("hello")

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if len(provider.lastReq.Messages) < 2 {
		t.Fatalf("request messages = %d, want at least system + user", len(provider.lastReq.Messages))
	}
	system := provider.lastReq.Messages[0]
	if system.Role != "system" {
		t.Fatalf("first request message role = %q, want system", system.Role)
	}
	for _, forbidden := range []string{"# auto memory", "project instructions"} {
		if strings.Contains(system.Content, forbidden) {
			t.Fatalf("system prompt contains %q:\n%s", forbidden, system.Content)
		}
	}

	firstUser := provider.lastReq.Messages[1]
	if firstUser.Role != "user" {
		t.Fatalf("second request message role = %q, want user", firstUser.Role)
	}
	for _, want := range []string{
		initialUserContextTag,
		"# auto memory",
		"Remember batch size defaults to 8.",
		"project instructions",
		"## User Request",
		"hello",
	} {
		if !strings.Contains(firstUser.Content, want) {
			t.Fatalf("first user message missing %q:\n%s", want, firstUser.Content)
		}
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
