package loop

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ctxmanager "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/permission"
	"gitcode.com/mindspore/mscli/tools"
)

type pathDenialAuthorizer struct {
	decision PathAuthorizationDecision
	calls    int
}

func (a *pathDenialAuthorizer) RequestPathAuthorization(context.Context, *pathpolicy.PathDenial) (PathAuthorizationDecision, error) {
	a.calls++
	return a.decision, nil
}

type pathDenialTool struct {
	calls int
	mode  string
}

func (t *pathDenialTool) Name() string {
	if t.mode == PathAuthorizationModeWrite {
		return "edit"
	}
	return "read"
}

func (t *pathDenialTool) Description() string { return "path denial tool" }

func (t *pathDenialTool) Schema() llm.ToolSchema { return llm.ToolSchema{Type: "object"} }

func (t *pathDenialTool) Execute(ctx context.Context, _ json.RawMessage) (*tools.Result, error) {
	t.calls++
	opts := pathpolicy.ResolveOptionsFromContext(ctx)
	if t.mode == PathAuthorizationModeWrite {
		if len(opts.TemporaryWriteRoots) > 0 {
			return tools.StringResultWithSummary("write allowed", "ok"), nil
		}
		return pathpolicy.NewPathDenialResult(&pathpolicy.PathDenial{
			Kind:          string(pathpolicy.DenialKindExternalWrite),
			Operation:     "edit",
			InputPath:     "/external/file.txt",
			SuggestedRoot: "/external",
		}), nil
	}
	if len(opts.TemporaryReadRoots) > 0 {
		return tools.StringResultWithSummary("allowed", "ok"), nil
	}
	return pathpolicy.NewPathDenialResult(&pathpolicy.PathDenial{
		Kind:          string(pathpolicy.DenialKindExternalRead),
		Operation:     "read",
		InputPath:     "/external/file.txt",
		SuggestedRoot: "/external",
	}), nil
}

func TestExecuteToolCallRetriesPathDenialWithTemporaryReadRoot(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "/external/file.txt"})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	tool := &pathDenialTool{}
	registry := tools.NewRegistry()
	registry.MustRegister(tool)

	engine := NewEngine(EngineConfig{ContextWindow: 4096}, nil, registry)
	engine.ctxManager = ctxmanager.NewManager(ctxmanager.ManagerConfig{ContextWindow: 4096, ReserveTokens: 100})
	authorizer := &pathDenialAuthorizer{decision: PathAuthorizationDecision{Scope: PathAuthorizationOnce, Root: "/external", Mode: "read"}}
	engine.SetPathAuthorizer(authorizer)
	engine.SetPermissionService(permission.NewNoOpPermissionService())

	ex := &executor{engine: engine}
	tc := llm.ToolCall{ID: "call-read", Type: "function", Function: llm.ToolCallFunc{Name: "read", Arguments: args}}

	if err := ex.executeToolCall(context.Background(), tc); err != nil {
		t.Fatalf("executeToolCall() error = %v", err)
	}
	if got, want := authorizer.calls, 1; got != want {
		t.Fatalf("authorizer calls = %d, want %d", got, want)
	}
	if got, want := tool.calls, 2; got != want {
		t.Fatalf("tool calls = %d, want %d", got, want)
	}
	msgs := engine.ctxManager.GetNonSystemMessages()
	if len(msgs) != 1 {
		t.Fatalf("tool messages = %d, want 1", len(msgs))
	}
	if got, want := msgs[0].Content, "allowed"; got != want {
		t.Fatalf("tool result = %q, want %q", got, want)
	}
}

func TestExecuteToolCallWritesToolResultWhenPathAuthorizationDenied(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "/external/file.txt"})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	tool := &pathDenialTool{}
	registry := tools.NewRegistry()
	registry.MustRegister(tool)

	engine := NewEngine(EngineConfig{ContextWindow: 4096}, nil, registry)
	engine.ctxManager = ctxmanager.NewManager(ctxmanager.ManagerConfig{ContextWindow: 4096, ReserveTokens: 100})
	authorizer := &pathDenialAuthorizer{decision: PathAuthorizationDecision{Scope: PathAuthorizationDeny, Root: "/external", Mode: PathAuthorizationModeRead}}
	engine.SetPathAuthorizer(authorizer)
	engine.SetPermissionService(permission.NewNoOpPermissionService())

	ex := &executor{engine: engine}
	tc := llm.ToolCall{ID: "call-read-denied", Type: "function", Function: llm.ToolCallFunc{Name: "read", Arguments: args}}

	if err := ex.executeToolCall(context.Background(), tc); err != nil {
		t.Fatalf("executeToolCall() error = %v", err)
	}
	if got, want := authorizer.calls, 1; got != want {
		t.Fatalf("authorizer calls = %d, want %d", got, want)
	}
	if got, want := tool.calls, 1; got != want {
		t.Fatalf("tool calls = %d, want %d", got, want)
	}
	msgs := engine.ctxManager.GetNonSystemMessages()
	if len(msgs) != 1 {
		t.Fatalf("tool messages = %d, want 1", len(msgs))
	}
	if got := msgs[0].Content; !strings.Contains(got, "External path access denied") || !strings.Contains(got, "/external/file.txt") {
		t.Fatalf("tool result = %q, want denial details", got)
	}
	if len(ex.events) == 0 {
		t.Fatal("expected tool error event")
	}
	last := ex.events[len(ex.events)-1]
	if last.Type != EventToolError || last.ToolCallID != tc.ID || last.ToolName != "read" {
		t.Fatalf("last event = %#v, want tool error for denied read", last)
	}
}
func TestExecuteToolCallRetriesPathDenialWithTemporaryWriteRoot(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "/external/file.txt"})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	tool := &pathDenialTool{mode: PathAuthorizationModeWrite}
	registry := tools.NewRegistry()
	registry.MustRegister(tool)

	engine := NewEngine(EngineConfig{ContextWindow: 4096}, nil, registry)
	engine.ctxManager = ctxmanager.NewManager(ctxmanager.ManagerConfig{ContextWindow: 4096, ReserveTokens: 100})
	authorizer := &pathDenialAuthorizer{decision: PathAuthorizationDecision{Scope: PathAuthorizationOnce, Root: "/external", Mode: PathAuthorizationModeWrite}}
	engine.SetPathAuthorizer(authorizer)
	engine.SetPermissionService(permission.NewNoOpPermissionService())

	ex := &executor{engine: engine}
	tc := llm.ToolCall{ID: "call-edit", Type: "function", Function: llm.ToolCallFunc{Name: "edit", Arguments: args}}

	prepareCalls := 0
	engine.SetTrajectoryRecorder(&TrajectoryRecorder{
		PrepareFileMutation: func(ctx context.Context, got llm.ToolCall) error {
			prepareCalls++
			if got.ID != tc.ID {
				t.Fatalf("prepared tool call ID = %q, want %q", got.ID, tc.ID)
			}
			opts := pathpolicy.ResolveOptionsFromContext(ctx)
			switch prepareCalls {
			case 1:
				if len(opts.TemporaryWriteRoots) != 0 {
					t.Fatalf("initial prepare write roots = %v, want none", opts.TemporaryWriteRoots)
				}
			case 2:
				if got, want := opts.TemporaryWriteRoots, []string{"/external"}; len(got) != 1 || got[0] != want[0] {
					t.Fatalf("retry prepare write roots = %v, want %v", got, want)
				}
			default:
				t.Fatalf("PrepareFileMutation called %d times, want 2", prepareCalls)
			}
			return nil
		},
	})

	if err := ex.executeToolCall(context.Background(), tc); err != nil {
		t.Fatalf("executeToolCall() error = %v", err)
	}
	if got, want := prepareCalls, 2; got != want {
		t.Fatalf("PrepareFileMutation calls = %d, want %d", got, want)
	}
	if got, want := authorizer.calls, 1; got != want {
		t.Fatalf("authorizer calls = %d, want %d", got, want)
	}
	if got, want := tool.calls, 2; got != want {
		t.Fatalf("tool calls = %d, want %d", got, want)
	}
	msgs := engine.ctxManager.GetNonSystemMessages()
	if len(msgs) != 1 {
		t.Fatalf("tool messages = %d, want 1", len(msgs))
	}
	if got, want := msgs[0].Content, "write allowed"; got != want {
		t.Fatalf("tool result = %q, want %q", got, want)
	}
}
