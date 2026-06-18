package app

import (
	"context"
	"reflect"
	"testing"
	"time"

	"gitcode.com/mindspore/mscli/agent/loop"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestPathAuthorizerHandleInputOnce(t *testing.T) {
	authorizer, events, denial := newPathAuthorizerForTest(nil, nil)
	decisionCh := requestPathAuthorizationForTest(t, authorizer, denial)
	requirePromptEvent(t, events)

	if !authorizer.HandleInput("1") {
		t.Fatal("HandleInput returned false, want true")
	}
	decision := requirePathDecision(t, decisionCh)
	if got, want := decision.Scope, loop.PathAuthorizationOnce; got != want {
		t.Fatalf("decision scope = %q, want %q", got, want)
	}
	if got, want := decision.Root, denial.SuggestedRoot; got != want {
		t.Fatalf("decision root = %q, want %q", got, want)
	}
}

func TestPathAuthorizerHandleInputSessionAddsReadRoot(t *testing.T) {
	policy := pathpolicy.NewPathPolicy(t.TempDir(), nil, nil)
	authorizer, events, denial := newPathAuthorizerForTest(policy, nil)
	decisionCh := requestPathAuthorizationForTest(t, authorizer, denial)
	requirePromptEvent(t, events)

	if !authorizer.HandleInput("2") {
		t.Fatal("HandleInput returned false, want true")
	}
	decision := requirePathDecision(t, decisionCh)
	if got, want := decision.Scope, loop.PathAuthorizationSession; got != want {
		t.Fatalf("decision scope = %q, want %q", got, want)
	}
	resolved, pathDenial, err := pathpolicy.NewResolver(policy).ResolveReadablePath(denial.ResolvedPath, pathpolicy.ResolveOptions{})
	if err != nil || pathDenial != nil {
		t.Fatalf("session root did not allow read: resolved=%q denial=%v err=%v", resolved, pathDenial, err)
	}
}

func TestPathAuthorizerHandleInputPersistentSavesConfigRoots(t *testing.T) {
	policy := pathpolicy.NewPathPolicy(t.TempDir(), []string{"/already-allowed"}, nil)
	var saved []string
	authorizer, events, denial := newPathAuthorizerForTest(policy, func(roots []string) error {
		saved = append([]string(nil), roots...)
		return nil
	})
	decisionCh := requestPathAuthorizationForTest(t, authorizer, denial)
	requirePromptEvent(t, events)

	if !authorizer.HandleInput("3") {
		t.Fatal("HandleInput returned false, want true")
	}
	decision := requirePathDecision(t, decisionCh)
	if got, want := decision.Scope, loop.PathAuthorizationPersistent; got != want {
		t.Fatalf("decision scope = %q, want %q", got, want)
	}
	want := []string{pathpolicy.NormalizeInputPath("/already-allowed"), pathpolicy.NormalizeInputPath(denial.SuggestedRoot)}
	if !reflect.DeepEqual(saved, want) {
		t.Fatalf("saved roots = %v, want %v", saved, want)
	}
}

func TestPathAuthorizerHandleInputWriteSessionAddsWriteRoot(t *testing.T) {
	policy := pathpolicy.NewPathPolicy(t.TempDir(), nil, nil)
	authorizer, events, denial := newPathAuthorizerForTest(policy, nil)
	denial.Kind = string(pathpolicy.DenialKindExternalWrite)
	denial.Operation = "edit"
	decisionCh := requestPathAuthorizationForTest(t, authorizer, denial)
	ev := requirePromptEvent(t, events)
	if got, want := ev.Permission.Title, "External write access"; got != want {
		t.Fatalf("prompt title = %q, want %q", got, want)
	}
	if got, want := len(ev.Permission.Options), 3; got != want {
		t.Fatalf("write prompt options = %d, want %d", got, want)
	}

	if !authorizer.HandleInput("2") {
		t.Fatal("HandleInput returned false, want true")
	}
	decision := requirePathDecision(t, decisionCh)
	if got, want := decision.Mode, loop.PathAuthorizationModeWrite; got != want {
		t.Fatalf("decision mode = %q, want %q", got, want)
	}
	resolved, pathDenial, err := pathpolicy.NewResolver(policy).ResolveWritablePath(denial.ResolvedPath, pathpolicy.ResolveOptions{})
	if err != nil || pathDenial != nil {
		t.Fatalf("session write root did not allow write: resolved=%q denial=%v err=%v", resolved, pathDenial, err)
	}
}
func TestPathAuthorizerHandleInputDeny(t *testing.T) {
	authorizer, events, denial := newPathAuthorizerForTest(nil, nil)
	decisionCh := requestPathAuthorizationForTest(t, authorizer, denial)
	requirePromptEvent(t, events)

	if !authorizer.HandleInput("4") {
		t.Fatal("HandleInput returned false, want true")
	}
	decision := requirePathDecision(t, decisionCh)
	if got, want := decision.Scope, loop.PathAuthorizationDeny; got != want {
		t.Fatalf("decision scope = %q, want %q", got, want)
	}
}

func TestPathAuthorizerInvalidInputReprompts(t *testing.T) {
	authorizer, events, denial := newPathAuthorizerForTest(nil, nil)
	decisionCh := requestPathAuthorizationForTest(t, authorizer, denial)
	requirePromptEvent(t, events)

	if !authorizer.HandleInput("bad") {
		t.Fatal("HandleInput returned false, want true")
	}
	reprompt := requirePromptEvent(t, events)
	if got, want := reprompt.Message, "Please choose 1, 2, 3, or 4."; got != want {
		t.Fatalf("reprompt message = %q, want %q", got, want)
	}
	select {
	case decision := <-decisionCh:
		t.Fatalf("unexpected decision after invalid input: %#v", decision)
	default:
	}
}

func newPathAuthorizerForTest(policy *pathpolicy.PathPolicy, saveRead func([]string) error) (*PathAuthorizer, chan model.Event, *pathpolicy.PathDenial) {
	events := make(chan model.Event, 4)
	denial := &pathpolicy.PathDenial{
		Kind:          string(pathpolicy.DenialKindExternalRead),
		Operation:     "read",
		InputPath:     "/external/file.txt",
		ResolvedPath:  "/external/file.txt",
		WorkDir:       "/workspace",
		SuggestedRoot: "/external",
	}
	return NewPathAuthorizer(events, policy, saveRead), events, denial
}

func requestPathAuthorizationForTest(t *testing.T, authorizer *PathAuthorizer, denial *pathpolicy.PathDenial) <-chan loop.PathAuthorizationDecision {
	t.Helper()
	decisionCh := make(chan loop.PathAuthorizationDecision, 1)
	go func() {
		decision, err := authorizer.RequestPathAuthorization(context.Background(), denial)
		if err != nil {
			t.Errorf("RequestPathAuthorization() error = %v", err)
			return
		}
		decisionCh <- decision
	}()
	return decisionCh
}

func requirePromptEvent(t *testing.T, events <-chan model.Event) model.Event {
	t.Helper()
	select {
	case ev := <-events:
		if ev.Type != model.PermissionPrompt {
			t.Fatalf("event type = %q, want %q", ev.Type, model.PermissionPrompt)
		}
		if ev.Permission == nil {
			t.Fatal("event Permission = nil, want prompt data")
		}
		return ev
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for prompt event")
		return model.Event{}
	}
}

func requirePathDecision(t *testing.T, decisionCh <-chan loop.PathAuthorizationDecision) loop.PathAuthorizationDecision {
	t.Helper()
	select {
	case decision := <-decisionCh:
		return decision
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for path decision")
		return loop.PathAuthorizationDecision{}
	}
}
