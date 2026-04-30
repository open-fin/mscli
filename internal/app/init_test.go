package app

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

type initCaptureProvider struct {
	reqCh chan *llm.CompletionRequest
}

func (p *initCaptureProvider) Name() string {
	return "init-capture"
}

func (p *initCaptureProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *initCaptureProvider) CompleteStream(_ context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	copied.Tools = append([]llm.Tool(nil), req.Tools...)
	p.reqCh <- &copied

	return &initCaptureIterator{sent: false}, nil
}

func (p *initCaptureProvider) SupportsTools() bool {
	return true
}

func (p *initCaptureProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type initCaptureIterator struct {
	sent bool
}

func (it *initCaptureIterator) Next() (*llm.StreamChunk, error) {
	if it.sent {
		return nil, io.EOF
	}
	it.sent = true
	return &llm.StreamChunk{Content: "ok", FinishReason: llm.FinishStop}, nil
}

func (it *initCaptureIterator) Close() error {
	return nil
}

func TestCmdInitRunsMSCLIInitPrompt(t *testing.T) {
	provider := &initCaptureProvider{reqCh: make(chan *llm.CompletionRequest, 1)}
	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt(buildBaseSystemPrompt(nil))
	engine.SetContextManager(ctxManager)

	app := &Application{
		Engine:     engine,
		EventCh:    make(chan model.Event, 8),
		llmReady:   true,
		ctxManager: ctxManager,
		memoryConfig: autoMemoryConfig{
			Resolved: true,
			Enabled:  false,
		},
	}

	app.handleCommand("/init")

	if ev := <-app.EventCh; ev.Type != model.AgentThinking {
		t.Fatalf("first event type = %s, want %s", ev.Type, model.AgentThinking)
	}

	var req *llm.CompletionRequest
	select {
	case req = <-provider.reqCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for /init completion request")
	}
	if len(req.Messages) < 2 {
		t.Fatalf("request messages = %d, want at least system + user", len(req.Messages))
	}
	user := req.Messages[1]
	if user.Role != "user" {
		t.Fatalf("second request message role = %q, want user", user.Role)
	}
	for _, want := range []string{
		"Please analyze this codebase and create a MSCLI.md file",
		"If there's already a MSCLI.md, suggest improvements to it.",
		".cursor/rules/",
		".github/copilot-instructions.md",
		"# MSCLI.md\n\nThis file provides guidance to mscli when working with code in this repository.",
	} {
		if !strings.Contains(user.Content, want) {
			t.Fatalf("init prompt missing %q:\n%s", want, user.Content)
		}
	}
}
