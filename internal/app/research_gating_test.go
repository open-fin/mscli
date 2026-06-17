package app

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"

	"gitcode.com/mindspore/mscli/agent/loop"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/tools"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestRunTaskPromptsForExplicitDecisionAfterMaxIterations(t *testing.T) {
	provider := &captureTaskProvider{}
	app := newResearchGatingTestApp(provider)

	app.runTask("inspect")

	if !app.pendingMaxIterationDecision {
		t.Fatal("pendingMaxIterationDecision = false, want true after max-iteration failure")
	}
	if !researchEventsContain(app.EventCh, "Continue", "Write now", "Stop") {
		t.Fatal("event stream does not contain explicit Continue / Write now / Stop prompt")
	}
}

func TestMaxIterationDecisionContinueDoesNotDisableResearchTools(t *testing.T) {
	provider := &captureTaskProvider{}
	app := newResearchGatingTestApp(provider)

	app.runTask("inspect")
	provider.responses = []llm.CompletionResponse{{Content: "continued", FinishReason: llm.FinishStop}}
	app.handleMaxIterationDecision("Continue")
	waitForRequests(t, provider, 2)

	if len(provider.requests) != 2 {
		t.Fatalf("provider requests = %d, want 2", len(provider.requests))
	}
	if requestHasResearchDisabledGuidance(provider.requests[1]) {
		t.Fatalf("continuation request messages = %#v, want no research-disabled guidance", provider.requests[1].Messages)
	}
	if app.pendingMaxIterationDecision {
		t.Fatal("pendingMaxIterationDecision = true, want false after Continue")
	}
}

func TestMaxIterationDecisionWriteNowDisablesResearchTools(t *testing.T) {
	provider := &captureTaskProvider{}
	registry := tools.NewRegistry()
	registry.MustRegister(stubAppTool{name: "read"})
	registry.MustRegister(stubAppTool{name: "write"})
	app := newResearchGatingTestAppWithRegistry(provider, registry)

	app.runTask("inspect")
	provider.responses = []llm.CompletionResponse{{Content: "summary", FinishReason: llm.FinishStop}}
	app.handleMaxIterationDecision("Write now")
	waitForRequests(t, provider, 2)

	if len(provider.requests) != 2 {
		t.Fatalf("provider requests = %d, want 2", len(provider.requests))
	}
	if !requestHasResearchDisabledGuidance(provider.requests[1]) {
		t.Fatalf("write-now request messages = %#v, want research-disabled guidance", provider.requests[1].Messages)
	}
	toolNames := completionToolNames(provider.requests[1].Tools)
	if researchContainsString(toolNames, "read") {
		t.Fatalf("write-now tools = %v, want read filtered", toolNames)
	}
	if !researchContainsString(toolNames, "write") {
		t.Fatalf("write-now tools = %v, want write retained", toolNames)
	}
	if app.pendingMaxIterationDecision {
		t.Fatal("pendingMaxIterationDecision = true, want false after Write now")
	}
}

func TestMaxIterationDecisionStopClearsPendingWithoutFollowup(t *testing.T) {
	provider := &captureTaskProvider{}
	app := newResearchGatingTestApp(provider)

	app.runTask("inspect")
	app.handleMaxIterationDecision("Stop")

	if app.pendingMaxIterationDecision {
		t.Fatal("pendingMaxIterationDecision = true, want false after Stop")
	}
	if len(provider.requests) != 1 {
		t.Fatalf("provider requests = %d, want no follow-up request", len(provider.requests))
	}
	if !researchEventsContain(app.EventCh, "Stopped", "context is preserved") {
		t.Fatal("event stream does not contain Stop confirmation")
	}
}

func TestMaxIterationDecisionRejectsImplicitIntent(t *testing.T) {
	provider := &captureTaskProvider{}
	app := newResearchGatingTestApp(provider)

	app.runTask("inspect")
	app.handleMaxIterationDecision("please summarize")

	if !app.pendingMaxIterationDecision {
		t.Fatal("pendingMaxIterationDecision = false, want still pending for implicit text")
	}
	if len(provider.requests) != 1 {
		t.Fatalf("provider requests = %d, want no implicit follow-up request", len(provider.requests))
	}
	if !researchEventsContain(app.EventCh, "Please choose Continue, Write now, or Stop") {
		t.Fatal("event stream does not contain explicit-choice retry prompt")
	}
}

func newResearchGatingTestApp(provider *captureTaskProvider) *Application {
	return newResearchGatingTestAppWithRegistry(provider, tools.NewRegistry())
}

func newResearchGatingTestAppWithRegistry(provider *captureTaskProvider, registry *tools.Registry) *Application {
	return &Application{
		Engine: loop.NewEngine(loop.EngineConfig{
			MaxIterations: 1,
			ContextWindow: 4096,
		}, provider, registry),
		EventCh:  make(chan model.Event, 32),
		llmReady: true,
	}
}

func requestHasResearchDisabledGuidance(req *llm.CompletionRequest) bool {
	if req == nil {
		return false
	}
	for _, msg := range req.Messages {
		if msg.Role == "system" && strings.Contains(msg.Content, "Research tools are disabled") {
			return true
		}
	}
	return false
}

func waitForRequests(t *testing.T, provider *captureTaskProvider, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if len(provider.requests) >= want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("provider requests = %d, want at least %d", len(provider.requests), want)
}

func researchEventsContain(ch <-chan model.Event, parts ...string) bool {
	for {
		select {
		case ev := <-ch:
			matched := true
			for _, part := range parts {
				if !strings.Contains(ev.Message, part) {
					matched = false
					break
				}
			}
			if matched {
				return true
			}
		default:
			return false
		}
	}
}

func completionToolNames(tools []llm.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Function.Name)
	}
	return names
}

func researchContainsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

type captureTaskProvider struct {
	requests  []*llm.CompletionRequest
	responses []llm.CompletionResponse
}

func (p *captureTaskProvider) Name() string {
	return "capture-task"
}

func (p *captureTaskProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *captureTaskProvider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	copied.Tools = append([]llm.Tool(nil), req.Tools...)
	p.requests = append(p.requests, &copied)

	if len(p.responses) == 0 {
		return &captureTaskIterator{chunks: []llm.StreamChunk{{
			ToolCalls: []llm.ToolCall{{
				ID:   "call-missing-read",
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "read",
					Arguments: json.RawMessage(`{"path":"README.md"}`),
				},
			}},
			FinishReason: llm.FinishToolCalls,
		}}}, nil
	}

	resp := p.responses[0]
	p.responses = p.responses[1:]
	return &captureTaskIterator{chunks: []llm.StreamChunk{{
		Content:      resp.Content,
		ToolCalls:    append([]llm.ToolCall(nil), resp.ToolCalls...),
		FinishReason: resp.FinishReason,
	}}}, nil
}

func (p *captureTaskProvider) SupportsTools() bool {
	return true
}

func (p *captureTaskProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type captureTaskIterator struct {
	chunks []llm.StreamChunk
	index  int
}

func (it *captureTaskIterator) Next() (*llm.StreamChunk, error) {
	if it.index >= len(it.chunks) {
		return nil, io.EOF
	}
	chunk := it.chunks[it.index]
	it.index++
	return &chunk, nil
}

func (it *captureTaskIterator) Close() error {
	return nil
}

type stubAppTool struct {
	name string
}

func (t stubAppTool) Name() string { return t.name }

func (t stubAppTool) Description() string { return t.name }

func (t stubAppTool) Schema() llm.ToolSchema { return llm.ToolSchema{Type: "object"} }

func (t stubAppTool) Execute(context.Context, json.RawMessage) (*tools.Result, error) {
	return &tools.Result{Content: t.name}, nil
}
