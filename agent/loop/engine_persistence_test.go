package loop

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"

	ctxmanager "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/integrations/llm"
	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
	loopmcp "gitcode.com/mindspore/mscli/tools/mcp"
)

type scriptedStreamProvider struct {
	mu        sync.Mutex
	responses []*llm.CompletionResponse
	requests  []*llm.CompletionRequest
}

func (p *scriptedStreamProvider) Name() string {
	return "scripted"
}

func (p *scriptedStreamProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *scriptedStreamProvider) CompleteStream(_ context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	copied.Tools = append([]llm.Tool(nil), req.Tools...)
	p.requests = append(p.requests, &copied)

	if len(p.responses) == 0 {
		return &scriptedStreamIterator{}, nil
	}

	resp := p.responses[0]
	p.responses = p.responses[1:]

	return &scriptedStreamIterator{
		chunks: []llm.StreamChunk{{
			Content:      resp.Content,
			ToolCalls:    append([]llm.ToolCall(nil), resp.ToolCalls...),
			FinishReason: resp.FinishReason,
			Usage:        &resp.Usage,
		}},
	}, nil
}

func (p *scriptedStreamProvider) SupportsTools() bool {
	return true
}

func (p *scriptedStreamProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type scriptedStreamIterator struct {
	chunks []llm.StreamChunk
	index  int
}

func (it *scriptedStreamIterator) Next() (*llm.StreamChunk, error) {
	if it.index >= len(it.chunks) {
		return nil, io.EOF
	}
	chunk := it.chunks[it.index]
	it.index++
	return &chunk, nil
}

func (it *scriptedStreamIterator) Close() error {
	return nil
}

type stubTool struct {
	name    string
	content string
	summary string
	meta    map[string]any
}

func (t stubTool) Name() string {
	return t.name
}

func (t stubTool) Description() string {
	return "stub tool"
}

func (t stubTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{Type: "object"}
}

func (t stubTool) Execute(context.Context, json.RawMessage) (*tools.Result, error) {
	return &tools.Result{Content: t.content, Summary: t.summary, Meta: t.meta}, nil
}

type errorStubTool struct {
	stubTool
	caps tools.Capabilities
	err  error
}

func (t errorStubTool) Execute(context.Context, json.RawMessage) (*tools.Result, error) {
	return nil, t.err
}

func (t errorStubTool) Capabilities() tools.Capabilities {
	return t.caps
}

type nilResultStubTool struct {
	stubTool
	caps tools.Capabilities
}

func (t nilResultStubTool) Execute(context.Context, json.RawMessage) (*tools.Result, error) {
	return nil, nil
}

func (t nilResultStubTool) Capabilities() tools.Capabilities {
	return t.caps
}

type fakeMCPCaller struct {
	result *runtimemcp.CallResult
	err    error
}

func (f *fakeMCPCaller) CallTool(context.Context, string, string, json.RawMessage) (*runtimemcp.CallResult, error) {
	return f.result, f.err
}

type fakeMCPArtifactStore struct {
	err error
}

func (f *fakeMCPArtifactStore) Write(loopmcp.ArtifactWriteRequest) (loopmcp.Artifact, error) {
	return loopmcp.Artifact{}, f.err
}

type streamingStubTool struct {
	stubTool
	updates []tools.StreamEvent
}

func (t streamingStubTool) ExecuteStream(ctx context.Context, raw json.RawMessage, emit func(tools.StreamEvent)) (*tools.Result, error) {
	if emit != nil {
		emit(tools.StreamEvent{Type: tools.StreamEventStarted})
		for _, update := range t.updates {
			emit(update)
		}
	}
	return t.Execute(ctx, raw)
}

type cancelAwareStreamingStubTool struct {
	stubTool
	started chan struct{}
}

func (t cancelAwareStreamingStubTool) ExecuteStream(ctx context.Context, raw json.RawMessage, emit func(tools.StreamEvent)) (*tools.Result, error) {
	if emit != nil {
		emit(tools.StreamEvent{Type: tools.StreamEventStarted})
		emit(tools.StreamEvent{Type: tools.StreamEventOutput, Message: t.content})
	}
	select {
	case <-t.started:
	default:
		close(t.started)
	}
	<-ctx.Done()
	return tools.StringResultWithSummary(t.content, "interrupted"), nil
}

func (t cancelAwareStreamingStubTool) Capabilities() tools.Capabilities {
	return tools.Capabilities{
		Kind: tools.KindShell,
	}
}

func newPersistenceRecorder(log *[]string) *TrajectoryRecorder {
	last := ""
	appendLog := func(entry string) {
		*log = append(*log, entry)
	}

	return &TrajectoryRecorder{
		RecordUserInput: func(string) error {
			last = "user"
			appendLog(last)
			return nil
		},
		RecordAssistant: func(string) error {
			last = "assistant"
			appendLog(last)
			return nil
		},
		RecordToolCall: func(tc llm.ToolCall) error {
			last = "tool_call:" + tc.Function.Name
			appendLog(last)
			return nil
		},
		RecordToolResult: func(tc llm.ToolCall, _ string, _ map[string]any) error {
			last = "tool_result:" + tc.Function.Name
			appendLog(last)
			return nil
		},
		RecordSkillActivate: func(skillName string) error {
			last = "skill:" + skillName
			appendLog(last)
			return nil
		},
		RecordContextCompaction: func(trigger string, _, _ int, _ string) error {
			last = "compact:" + trigger
			appendLog(last)
			return nil
		},
		PersistSnapshot: func() error {
			appendLog("snapshot:" + last)
			return nil
		},
	}
}

func requireOrder(t *testing.T, log []string, entries ...string) {
	t.Helper()

	next := 0
	for _, entry := range entries {
		found := -1
		for i := next; i < len(log); i++ {
			if log[i] == entry {
				found = i
				next = i + 1
				break
			}
		}
		if found == -1 {
			t.Fatalf("expected log to contain %q after index %d, got %v", entry, next, log)
		}
	}
}

func TestRunPersistsSnapshotBeforeStreamingTaskEvents(t *testing.T) {
	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			Content:      "ok",
			FinishReason: llm.FinishStop,
		}},
	}
	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())

	var log []string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err := engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-before-ui",
		Description: "say ok",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	requireOrder(t, log, "user", "snapshot:user", "ui:TaskStarted")
	requireOrder(t, log, "assistant", "snapshot:assistant", "ui:AgentReply")
}

func TestRunPersistsToolResultBeforeToolRender(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-read-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "read",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "done",
				FinishReason: llm.FinishStop,
			},
		},
	}

	registry := tools.NewRegistry()
	registry.MustRegister(stubTool{name: "read", content: "file contents", summary: "1 line"})

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, registry)

	var log []string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-tool-result",
		Description: "read the file",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	requireOrder(t, log, "tool_call:read", "snapshot:tool_call:read", "ui:ToolCallStart")
	requireOrder(t, log, "tool_result:read", "snapshot:tool_result:read", "ui:ToolRead")
}

func TestRunAddsStandardMetadataToReturnedToolResult(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "sample.txt"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-edit-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "edit",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{Content: "done", FinishReason: llm.FinishStop},
		},
	}

	registry := tools.NewRegistry()
	registry.MustRegister(stubTool{
		name:    "edit",
		content: "edited",
		summary: "1 line",
		meta: map[string]any{
			"edit_diff": map[string]any{"path": "sample.txt"},
		},
	})

	engine := NewEngine(EngineConfig{MaxIterations: 2, ContextWindow: 4096}, provider, registry)

	var editEvent *Event
	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "tool-result-meta",
		Description: "edit file",
	}, func(ev Event) {
		if ev.Type == EventToolEdit {
			copy := ev
			editEvent = &copy
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}
	if editEvent == nil {
		t.Fatal("missing ToolEdit event")
	}
	if got := editEvent.Meta[tools.MetaStatus]; got != tools.StatusCompleted {
		t.Fatalf("status meta = %#v, want completed (meta %#v)", got, editEvent.Meta)
	}
	if _, ok := editEvent.Meta[tools.MetaDurationMS].(int64); !ok {
		t.Fatalf("duration meta = %#v, want int64", editEvent.Meta[tools.MetaDurationMS])
	}
	if editEvent.Meta["edit_diff"] == nil {
		t.Fatalf("edit_diff metadata was not preserved: %#v", editEvent.Meta)
	}
}

func TestRunRecorderReceivesToolResultMetadata(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "sample.txt"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-read-meta",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "read",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{Content: "done", FinishReason: llm.FinishStop},
		},
	}
	registry := tools.NewRegistry()
	registry.MustRegister(stubTool{
		name:    "read",
		content: "file",
		meta:    map[string]any{tools.MetaSource: tools.SourceFS},
	})

	engine := NewEngine(EngineConfig{MaxIterations: 2, ContextWindow: 4096}, provider, registry)
	var recordedMeta map[string]any
	engine.SetTrajectoryRecorder(&TrajectoryRecorder{
		RecordToolResult: func(_ llm.ToolCall, _ string, meta map[string]any) error {
			recordedMeta = meta
			return nil
		},
		PersistSnapshot: func() error { return nil },
	})

	_, err = engine.RunWithContext(context.Background(), Task{
		ID:          "record-tool-meta",
		Description: "read file",
	})
	if err != nil {
		t.Fatalf("RunWithContext failed: %v", err)
	}
	if got := recordedMeta[tools.MetaSource]; got != tools.SourceFS {
		t.Fatalf("recorded source = %#v, want fs (meta %#v)", got, recordedMeta)
	}
	if got := recordedMeta[tools.MetaStatus]; got != tools.StatusCompleted {
		t.Fatalf("recorded status = %#v, want completed", got)
	}
	if _, ok := recordedMeta[tools.MetaDurationMS].(int64); !ok {
		t.Fatalf("recorded duration = %#v, want int64", recordedMeta[tools.MetaDurationMS])
	}
}

func TestRunFallbackToolResultUsesPersistedContentAndMetadataForEvent(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "large.txt"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-read-large",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "read",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{Content: "done", FinishReason: llm.FinishStop},
		},
	}

	large := strings.Repeat("x", 1000)
	registry := tools.NewRegistry()
	registry.MustRegister(stubTool{
		name:    "read",
		content: large,
		meta: map[string]any{
			tools.MetaSource: tools.SourceFS,
		},
	})

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 120,
	}, provider, registry)

	var recordedContent string
	var recordedMeta map[string]any
	var readEvent *Event
	var toolErrorEvents []Event
	engine.SetTrajectoryRecorder(&TrajectoryRecorder{
		RecordToolResult: func(_ llm.ToolCall, content string, meta map[string]any) error {
			recordedContent = content
			recordedMeta = meta
			return nil
		},
		PersistSnapshot: func() error { return nil },
	})

	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "fallback-result-event",
		Description: "read large file",
	}, func(ev Event) {
		if ev.Type == EventToolRead {
			copy := ev
			readEvent = &copy
		}
		if ev.Type == EventToolError {
			toolErrorEvents = append(toolErrorEvents, ev)
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}
	if readEvent == nil {
		t.Fatal("missing ToolRead event")
	}
	if strings.Contains(readEvent.Message, large[:128]) {
		t.Fatalf("event message used original oversized content")
	}
	if !strings.Contains(readEvent.Message, "tool result replaced due to context limit") {
		t.Fatalf("event message missing fallback content:\n%s", readEvent.Message)
	}
	if readEvent.Message != recordedContent {
		t.Fatalf("event message != recorded content\nmessage: %q\nrecorded: %q", readEvent.Message, recordedContent)
	}
	if got := readEvent.Meta[tools.MetaFallback]; got != true {
		t.Fatalf("event fallback meta = %#v, want true (meta %#v)", got, readEvent.Meta)
	}
	if got := recordedMeta[tools.MetaFallback]; got != true {
		t.Fatalf("recorded fallback meta = %#v, want true (meta %#v)", got, recordedMeta)
	}
	if len(toolErrorEvents) != 0 {
		t.Fatalf("success fallback emitted ToolError events: %#v", toolErrorEvents)
	}
}

func TestRunMCPLargeResultArtifactFailureEmitsToolError(t *testing.T) {
	args := json.RawMessage(`{"query":"all"}`)
	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-mcp-large",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "mcp__server__tool",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{Content: "done", FinishReason: llm.FinishStop},
		},
	}

	registry := tools.NewRegistry()
	registry.MustRegister(loopmcp.NewToolWithArtifactStore(runtimemcp.ToolDefinition{
		ServerName:       "server",
		OriginalToolName: "tool",
		Name:             "mcp__server__tool",
	}, &fakeMCPCaller{result: &runtimemcp.CallResult{
		Content: []runtimemcp.ContentBlock{{Type: "text", Text: strings.Repeat("x", 64*1024+100)}},
	}}, &fakeMCPArtifactStore{err: errors.New("disk full")}))

	engine := NewEngine(EngineConfig{MaxIterations: 2, ContextWindow: 4096}, provider, registry)

	var errorEvent *Event
	var completedEvent *Event
	err := engine.RunWithContextStream(context.Background(), Task{
		ID:          "mcp-large-artifact-failure",
		Description: "call mcp",
	}, func(ev Event) {
		if ev.Type == EventToolError {
			copy := ev
			errorEvent = &copy
		}
		if ev.ToolCallID == "call-mcp-large" && ev.Type != EventToolCallStart && ev.Type != EventToolError {
			copy := ev
			completedEvent = &copy
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}
	if errorEvent == nil {
		t.Fatal("missing ToolError event")
	}
	if got, want := errorEvent.ToolName, "mcp__server__tool"; got != want {
		t.Fatalf("ToolError ToolName = %q, want %q", got, want)
	}
	if got, want := errorEvent.ToolCallID, "call-mcp-large"; got != want {
		t.Fatalf("ToolError ToolCallID = %q, want %q", got, want)
	}
	if got := errorEvent.Meta[tools.MetaStatus]; got != tools.StatusFailed {
		t.Fatalf("ToolError status = %#v, want failed (meta %#v)", got, errorEvent.Meta)
	}
	if got := errorEvent.Meta[tools.MetaSource]; got != tools.SourceMCP {
		t.Fatalf("ToolError source = %#v, want mcp (meta %#v)", got, errorEvent.Meta)
	}
	if completedEvent != nil {
		t.Fatalf("MCP artifact failure emitted success event: %#v", completedEvent)
	}
}

func TestRunToolExecutionErrorRecordsDurationMetadata(t *testing.T) {
	args, err := json.Marshal(map[string]string{"command": "fail"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-shell-error",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "shell",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{Content: "done", FinishReason: llm.FinishStop},
		},
	}
	registry := tools.NewRegistry()
	registry.MustRegister(errorStubTool{
		stubTool: stubTool{name: "shell"},
		caps:     tools.Capabilities{Kind: tools.KindShell},
		err:      errors.New("boom"),
	})

	engine := NewEngine(EngineConfig{MaxIterations: 2, ContextWindow: 4096}, provider, registry)

	var recordedMeta map[string]any
	var errorEvent *Event
	engine.SetTrajectoryRecorder(&TrajectoryRecorder{
		RecordToolResult: func(_ llm.ToolCall, _ string, meta map[string]any) error {
			recordedMeta = meta
			return nil
		},
		PersistSnapshot: func() error { return nil },
	})

	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "tool-error-duration",
		Description: "run shell",
	}, func(ev Event) {
		if ev.Type == EventToolError {
			copy := ev
			errorEvent = &copy
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}
	if got := recordedMeta[tools.MetaStatus]; got != tools.StatusFailed {
		t.Fatalf("recorded status = %#v, want failed (meta %#v)", got, recordedMeta)
	}
	if got := recordedMeta[tools.MetaSource]; got != tools.SourceShell {
		t.Fatalf("recorded source = %#v, want shell (meta %#v)", got, recordedMeta)
	}
	if _, ok := recordedMeta[tools.MetaDurationMS].(int64); !ok {
		t.Fatalf("recorded duration = %#v, want int64 (meta %#v)", recordedMeta[tools.MetaDurationMS], recordedMeta)
	}
	if errorEvent == nil {
		t.Fatal("missing ToolError event")
	}
	if _, ok := errorEvent.Meta[tools.MetaDurationMS].(int64); !ok {
		t.Fatalf("event duration = %#v, want int64 (meta %#v)", errorEvent.Meta[tools.MetaDurationMS], errorEvent.Meta)
	}
}

func TestRunNilToolResultRecordsDurationMetadata(t *testing.T) {
	args, err := json.Marshal(map[string]string{"command": "empty"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-shell-nil",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "shell",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{Content: "done", FinishReason: llm.FinishStop},
		},
	}
	registry := tools.NewRegistry()
	registry.MustRegister(nilResultStubTool{
		stubTool: stubTool{name: "shell"},
		caps:     tools.Capabilities{Kind: tools.KindShell},
	})

	engine := NewEngine(EngineConfig{MaxIterations: 2, ContextWindow: 4096}, provider, registry)

	var recordedMeta map[string]any
	engine.SetTrajectoryRecorder(&TrajectoryRecorder{
		RecordToolResult: func(_ llm.ToolCall, _ string, meta map[string]any) error {
			recordedMeta = meta
			return nil
		},
		PersistSnapshot: func() error { return nil },
	})

	_, err = engine.RunWithContext(context.Background(), Task{
		ID:          "nil-tool-duration",
		Description: "run shell",
	})
	if err != nil {
		t.Fatalf("RunWithContext failed: %v", err)
	}
	if got := recordedMeta[tools.MetaStatus]; got != tools.StatusFailed {
		t.Fatalf("recorded status = %#v, want failed (meta %#v)", got, recordedMeta)
	}
	if got := recordedMeta[tools.MetaSource]; got != tools.SourceShell {
		t.Fatalf("recorded source = %#v, want shell (meta %#v)", got, recordedMeta)
	}
	if _, ok := recordedMeta[tools.MetaDurationMS].(int64); !ok {
		t.Fatalf("recorded duration = %#v, want int64 (meta %#v)", recordedMeta[tools.MetaDurationMS], recordedMeta)
	}
}

func TestRunShellStreamingEmitsLiveCommandEvents(t *testing.T) {
	args, err := json.Marshal(map[string]string{"command": "printf 'line-1\\nline-2\\n'"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-shell-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "shell",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "done",
				FinishReason: llm.FinishStop,
			},
		},
	}

	registry := tools.NewRegistry()
	registry.MustRegister(streamingStubTool{
		stubTool: stubTool{name: "shell", content: "line-1\nline-2", summary: "completed"},
		updates: []tools.StreamEvent{
			{Type: tools.StreamEventOutput, Message: "line-1"},
			{Type: tools.StreamEventOutput, Message: "line-2"},
		},
	})

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, registry)

	var events []Event
	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "stream-shell-events",
		Description: "run a shell command",
	}, func(ev Event) {
		switch ev.Type {
		case EventToolCallStart, EventCmdStarted, EventCmdOutput, EventCmdFinished:
			events = append(events, ev)
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	if len(events) != 5 {
		t.Fatalf("event count = %d, want 5 (%#v)", len(events), events)
	}

	wantTypes := []string{
		EventToolCallStart,
		EventCmdStarted,
		EventCmdOutput,
		EventCmdOutput,
		EventCmdFinished,
	}
	for i, want := range wantTypes {
		if got := events[i].Type; got != want {
			t.Fatalf("events[%d].Type = %q, want %q", i, got, want)
		}
		if got := events[i].ToolCallID; got != "call-shell-1" {
			t.Fatalf("events[%d].ToolCallID = %q, want call-shell-1", i, got)
		}
	}

	if got := events[4].Summary; got != "completed" {
		t.Fatalf("final summary = %q, want completed", got)
	}
	if got := events[4].Message; got != "line-1\nline-2" {
		t.Fatalf("final message = %q, want full shell output", got)
	}
}

func TestRunPersistsInterruptedToolResultBeforeInterruptedRender(t *testing.T) {
	args, err := json.Marshal(map[string]string{"command": "sleep 10"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			ToolCalls: []llm.ToolCall{{
				ID:   "call-shell-interrupt-1",
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "shell",
					Arguments: args,
				},
			}},
			FinishReason: llm.FinishToolCalls,
		}},
	}

	registry := tools.NewRegistry()
	started := make(chan struct{})
	registry.MustRegister(cancelAwareStreamingStubTool{
		stubTool: stubTool{name: "shell", content: "partial line"},
		started:  started,
	})

	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, registry)

	var log []string
	var events []Event
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-started
		cancel()
	}()

	err = engine.RunWithContextStream(ctx, Task{
		ID:          "interrupt-shell",
		Description: "run shell then interrupt",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
		events = append(events, ev)
	})
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "canceled") {
		t.Fatalf("RunWithContextStream error = %v, want context canceled", err)
	}

	requireOrder(t, log, "tool_call:shell", "snapshot:tool_call:shell", "ui:ToolCallStart")
	requireOrder(t, log, "tool_result:shell", "snapshot:tool_result:shell", "ui:ToolInterrupted")

	var interrupted *Event
	for i := range events {
		if events[i].Type == EventToolInterrupted {
			interrupted = &events[i]
			break
		}
	}
	if interrupted == nil {
		t.Fatalf("expected ToolInterrupted event, got %#v", events)
	}
	if got, want := interrupted.ToolCallID, "call-shell-interrupt-1"; got != want {
		t.Fatalf("ToolInterrupted ToolCallID = %q, want %q", got, want)
	}
	if got, want := interrupted.Summary, "interrupted"; got != want {
		t.Fatalf("ToolInterrupted summary = %q, want %q", got, want)
	}
	if got := interrupted.Meta[tools.MetaStatus]; got != tools.StatusInterrupted {
		t.Fatalf("ToolInterrupted status meta = %#v, want interrupted", got)
	}
	if got := interrupted.Meta[tools.MetaSource]; got != tools.SourceShell {
		t.Fatalf("ToolInterrupted source meta = %#v, want shell", got)
	}
	if _, ok := interrupted.Meta[tools.MetaDurationMS].(int64); !ok {
		t.Fatalf("ToolInterrupted duration meta = %#v, want int64", interrupted.Meta[tools.MetaDurationMS])
	}
	if got, want := interrupted.Message, "partial line"; got != want {
		t.Fatalf("ToolInterrupted message = %q, want %q", got, want)
	}

	messages := engine.ctxManager.GetNonSystemMessages()
	var toolResult string
	for _, msg := range messages {
		if msg.Role == "tool" && msg.ToolCallID == "call-shell-interrupt-1" {
			toolResult = msg.Content
		}
	}
	if !strings.Contains(toolResult, "status: interrupted") {
		t.Fatalf("tool result missing interrupted status, got:\n%s", toolResult)
	}
	if !strings.Contains(toolResult, "reason: user requested cancellation") {
		t.Fatalf("tool result missing cancellation reason, got:\n%s", toolResult)
	}
	if !strings.Contains(toolResult, "partial_output:") {
		t.Fatalf("tool result missing partial output section, got:\n%s", toolResult)
	}
	if !strings.Contains(toolResult, "partial line") {
		t.Fatalf("tool result missing streamed partial output, got:\n%s", toolResult)
	}
}

func TestRunInterruptedToolFallbackUsesPersistedContentForEvent(t *testing.T) {
	args, err := json.Marshal(map[string]string{"command": "sleep 10"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			ToolCalls: []llm.ToolCall{{
				ID:   "call-shell-interrupt-large",
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "shell",
					Arguments: args,
				},
			}},
			FinishReason: llm.FinishToolCalls,
		}},
	}

	registry := tools.NewRegistry()
	started := make(chan struct{})
	largePartial := strings.Repeat("x", 1000)
	registry.MustRegister(cancelAwareStreamingStubTool{
		stubTool: stubTool{name: "shell", content: largePartial},
		started:  started,
	})

	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 120,
	}, provider, registry)

	var recordedContent string
	var recordedMeta map[string]any
	var interrupted *Event
	engine.SetTrajectoryRecorder(&TrajectoryRecorder{
		RecordToolResult: func(_ llm.ToolCall, content string, meta map[string]any) error {
			recordedContent = content
			recordedMeta = meta
			return nil
		},
		PersistSnapshot: func() error { return nil },
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-started
		cancel()
	}()

	err = engine.RunWithContextStream(ctx, Task{
		ID:          "interrupt-shell-large",
		Description: "run shell then interrupt",
	}, func(ev Event) {
		if ev.Type == EventToolInterrupted {
			copy := ev
			interrupted = &copy
		}
	})
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "canceled") {
		t.Fatalf("RunWithContextStream error = %v, want context canceled", err)
	}
	if interrupted == nil {
		t.Fatal("missing ToolInterrupted event")
	}
	if strings.Contains(interrupted.Message, largePartial[:128]) {
		t.Fatalf("interrupted event used original oversized partial output")
	}
	if !strings.Contains(interrupted.Message, "tool result replaced due to context limit") {
		t.Fatalf("interrupted event missing fallback content:\n%s", interrupted.Message)
	}
	if interrupted.Message != recordedContent {
		t.Fatalf("event message != recorded content\nmessage: %q\nrecorded: %q", interrupted.Message, recordedContent)
	}
	if got := interrupted.Meta[tools.MetaFallback]; got != true {
		t.Fatalf("event fallback meta = %#v, want true (meta %#v)", got, interrupted.Meta)
	}
	if got := recordedMeta[tools.MetaFallback]; got != true {
		t.Fatalf("recorded fallback meta = %#v, want true (meta %#v)", got, recordedMeta)
	}
}

func TestRunPersistsSnapshotBeforeContextCompactionNotice(t *testing.T) {
	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			Content:      "ok",
			FinishReason: llm.FinishStop,
		}},
	}
	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 100,
	}, provider, tools.NewRegistry())

	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow:       100,
		ReserveTokens:       10,
		CompactionThreshold: 0.9,
		EnableSmartCompact:  false,
	})
	cm.SetSystemPrompt("system")
	for i := 0; i < 3; i++ {
		if err := cm.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
			t.Fatalf("preload AddMessage #%d failed: %v", i+1, err)
		}
	}
	engine.SetContextManager(cm)

	var log []string
	var compactMessage string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err := engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-context-compaction",
		Description: strings.Repeat("y", 40),
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
		if ev.Type == EventContextCompacted {
			compactMessage = ev.Message
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	requireOrder(t, log, "user", "snapshot:user", "compact:auto", "ui:ContextCompacted", "ui:TaskStarted")
	if !strings.Contains(compactMessage, "Context compacted automatically:") {
		t.Fatalf("context compaction message = %q, want automatic compaction summary", compactMessage)
	}
}

func TestRunPersistsToolErrorBeforeErrorRender(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "missing.txt"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-missing-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "missing_tool",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "done",
				FinishReason: llm.FinishStop,
			},
		},
	}

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())

	var log []string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-tool-error",
		Description: "use the missing tool",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	// ToolCallStart is emitted after permission check, so missing tools
	// skip it entirely — verify error handling order instead.
	requireOrder(t, log, "tool_call:missing_tool", "snapshot:tool_call:missing_tool")
	requireOrder(t, log, "tool_result:missing_tool", "snapshot:tool_result:missing_tool", "ui:ToolError")
}

func TestRunToolErrorEventIncludesFailureMetadata(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "missing.txt"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-missing-meta",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "missing_tool",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{Content: "done", FinishReason: llm.FinishStop},
		},
	}
	engine := NewEngine(EngineConfig{MaxIterations: 2, ContextWindow: 4096}, provider, tools.NewRegistry())

	var errorEvent *Event
	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "tool-error-meta",
		Description: "use missing tool",
	}, func(ev Event) {
		if ev.Type == EventToolError {
			copy := ev
			errorEvent = &copy
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}
	if errorEvent == nil {
		t.Fatal("missing ToolError event")
	}
	if got := errorEvent.Meta[tools.MetaStatus]; got != tools.StatusFailed {
		t.Fatalf("status meta = %#v, want failed", got)
	}
	if got := errorEvent.Meta[tools.MetaSource]; got != tools.SourceLoop {
		t.Fatalf("source meta = %#v, want loop", got)
	}
}
