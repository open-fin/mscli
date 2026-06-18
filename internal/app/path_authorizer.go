package app

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"gitcode.com/mindspore/mscli/agent/loop"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/ui/model"
)

type pathAuthorizationDecision struct {
	decision loop.PathAuthorizationDecision
}

type pendingPathAuthorizationRequest struct {
	wait   chan pathAuthorizationDecision
	denial *pathpolicy.PathDenial
}

type PathAuthorizer struct {
	mu        sync.Mutex
	pending   *pendingPathAuthorizationRequest
	eventCh   chan<- model.Event
	policy    *pathpolicy.PathPolicy
	saveRead  func([]string) error
	saveWrite func([]string) error
}

func NewPathAuthorizer(eventCh chan<- model.Event, policy *pathpolicy.PathPolicy, saveRead func([]string) error) *PathAuthorizer {
	return NewPathAuthorizerWithWrite(eventCh, policy, saveRead, nil)
}

func NewPathAuthorizerWithWrite(eventCh chan<- model.Event, policy *pathpolicy.PathPolicy, saveRead func([]string) error, saveWrite func([]string) error) *PathAuthorizer {
	return &PathAuthorizer{eventCh: eventCh, policy: policy, saveRead: saveRead, saveWrite: saveWrite}
}

func (p *PathAuthorizer) RequestPathAuthorization(ctx context.Context, denial *pathpolicy.PathDenial) (loop.PathAuthorizationDecision, error) {
	if p == nil || denial == nil {
		return loop.PathAuthorizationDecision{Scope: loop.PathAuthorizationDeny}, nil
	}
	mode := loop.PathAuthorizationModeRead
	switch denial.Kind {
	case string(pathpolicy.DenialKindExternalRead):
		mode = loop.PathAuthorizationModeRead
	case string(pathpolicy.DenialKindExternalWrite):
		mode = loop.PathAuthorizationModeWrite
	default:
		return loop.PathAuthorizationDecision{Scope: loop.PathAuthorizationDeny}, nil
	}
	root := strings.TrimSpace(denial.SuggestedRoot)
	if root == "" {
		return loop.PathAuthorizationDecision{Scope: loop.PathAuthorizationDeny}, nil
	}

	p.mu.Lock()
	if p.pending != nil {
		p.mu.Unlock()
		return loop.PathAuthorizationDecision{}, fmt.Errorf("path authorization request already pending")
	}
	req := &pendingPathAuthorizationRequest{wait: make(chan pathAuthorizationDecision, 1), denial: denial}
	p.pending = req
	p.mu.Unlock()

	p.eventCh <- model.Event{Type: model.PermissionPrompt, Message: pathAuthorizationPromptMessage(denial), Permission: pathAuthorizationPromptData(denial)}

	select {
	case <-ctx.Done():
		p.clearPending(req)
		return loop.PathAuthorizationDecision{}, ctx.Err()
	case decision := <-req.wait:
		if decision.decision.Mode == "" {
			decision.decision.Mode = mode
		}
		return decision.decision, nil
	}
}

func (p *PathAuthorizer) HandleInput(input string) bool {
	input = strings.ToLower(strings.TrimSpace(input))
	p.mu.Lock()
	req := p.pending
	p.mu.Unlock()
	if req == nil {
		return false
	}
	root := strings.TrimSpace(req.denial.SuggestedRoot)
	mode := pathAuthorizationMode(req.denial)
	resolve := func(decision loop.PathAuthorizationDecision) bool {
		p.clearPending(req)
		if decision.Scope == loop.PathAuthorizationSession {
			p.addSessionRoot(decision.Root, mode)
		}
		if decision.Scope == loop.PathAuthorizationPersistent {
			p.addSessionRoot(decision.Root, mode)
			if err := p.savePersistentRoot(mode, decision.Root); err != nil {
				p.eventCh <- model.Event{Type: model.ToolError, Message: fmt.Sprintf("Failed to persist external path authorization; access is allowed for this session only: %v", err)}
			}
		}
		req.wait <- pathAuthorizationDecision{decision: decision}
		return true
	}
	switch input {
	case "1", "once", "y", "yes":
		return resolve(loop.PathAuthorizationDecision{Scope: loop.PathAuthorizationOnce, Root: root, Mode: mode})
	case "2", "session", "allow_session":
		return resolve(loop.PathAuthorizationDecision{Scope: loop.PathAuthorizationSession, Root: root, Mode: mode})
	case "3", "always", "persistent":
		if mode == loop.PathAuthorizationModeWrite {
			p.eventCh <- model.Event{Type: model.PermissionPrompt, Message: "Please choose 1, 2, or 4. Persistent write access is not available yet.", Permission: pathAuthorizationPromptData(req.denial)}
			return true
		}
		return resolve(loop.PathAuthorizationDecision{Scope: loop.PathAuthorizationPersistent, Root: root, Mode: mode})
	case "4", "n", "no", "deny", "esc", "escape":
		return resolve(loop.PathAuthorizationDecision{Scope: loop.PathAuthorizationDeny, Root: root, Mode: mode})
	default:
		p.eventCh <- model.Event{Type: model.PermissionPrompt, Message: "Please choose 1, 2, 3, or 4.", Permission: pathAuthorizationPromptData(req.denial)}
		return true
	}
}

func (p *PathAuthorizer) clearPending(req *pendingPathAuthorizationRequest) {
	p.mu.Lock()
	if p.pending == req {
		p.pending = nil
	}
	p.mu.Unlock()
}

func (p *PathAuthorizer) addSessionRoot(root, mode string) {
	if p == nil || p.policy == nil {
		return
	}
	if mode == loop.PathAuthorizationModeWrite {
		p.policy.AddSessionWriteRoot(root)
		return
	}
	p.policy.AddSessionReadRoot(root)
}

func (p *PathAuthorizer) savePersistentRoot(mode, root string) error {
	if p == nil || p.policy == nil {
		return nil
	}
	if mode == loop.PathAuthorizationModeWrite {
		if p.saveWrite == nil {
			return nil
		}
		return p.saveWrite(snapshotConfigRootsWithRoot(p.policy.WriteRoots, root))
	}
	if p.saveRead == nil {
		return nil
	}
	return p.saveRead(snapshotConfigRootsWithRoot(p.policy.ReadRoots, root))
}

func snapshotConfigRootsWithRoot(roots *pathpolicy.RootSet, root string) []string {
	seen := map[string]bool{}
	out := []string{}
	if roots != nil {
		for _, entry := range roots.Snapshot() {
			if entry.Source != pathpolicy.RootSourceConfig {
				continue
			}
			key := strings.ToLower(entry.Path)
			if seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, entry.Path)
		}
	}
	root = pathpolicy.NormalizeInputPath(root)
	if strings.TrimSpace(root) != "" {
		key := strings.ToLower(root)
		if !seen[key] {
			out = append(out, root)
		}
	}
	return out
}

func pathAuthorizationMode(denial *pathpolicy.PathDenial) string {
	if denial != nil && denial.Kind == string(pathpolicy.DenialKindExternalWrite) {
		return loop.PathAuthorizationModeWrite
	}
	return loop.PathAuthorizationModeRead
}

func pathAuthorizationOperation(denial *pathpolicy.PathDenial) string {
	if strings.TrimSpace(denial.Operation) != "" {
		return titlePathOperation(denial.Operation)
	}
	if pathAuthorizationMode(denial) == loop.PathAuthorizationModeWrite {
		return "Write"
	}
	return "Read"
}

func pathAuthorizationTargetLabel(denial *pathpolicy.PathDenial) string {
	if pathAuthorizationMode(denial) == loop.PathAuthorizationModeWrite {
		return "Suggested external write root"
	}
	return "Suggested external read root"
}
func titlePathOperation(operation string) string {
	operation = strings.TrimSpace(operation)
	if operation == "" {
		return ""
	}
	return strings.ToUpper(operation[:1]) + operation[1:]
}

func pathAuthorizationPromptMessage(denial *pathpolicy.PathDenial) string {
	operation := pathAuthorizationOperation(denial)
	if pathAuthorizationMode(denial) == loop.PathAuthorizationModeWrite {
		return fmt.Sprintf("%s wants to modify a file outside the current workspace.\n\nCurrent workspace:\n  %s\n\nRequested file:\n  %s\n\n%s:\n  %s\n\nThis grants write access under the selected root.\nOnly approve if you trust this task.\n\nAllow write access?\n  1. Allow once\n  2. Allow for this session\n  4. Deny\n\nEsc to cancel", operation, denial.WorkDir, denial.InputPath, pathAuthorizationTargetLabel(denial), denial.SuggestedRoot)
	}
	return fmt.Sprintf("%s wants to access a file outside the current workspace.\n\nCurrent workspace:\n  %s\n\nRequested file:\n  %s\n\n%s:\n  %s\n\nAllow read-only access?\n  1. Allow once\n  2. Allow for this session\n  3. Always allow\n  4. Deny\n\nEsc to cancel", operation, denial.WorkDir, denial.InputPath, pathAuthorizationTargetLabel(denial), denial.SuggestedRoot)
}

func pathAuthorizationPromptData(denial *pathpolicy.PathDenial) *model.PermissionPromptData {
	operation := pathAuthorizationOperation(denial)
	if pathAuthorizationMode(denial) == loop.PathAuthorizationModeWrite {
		return &model.PermissionPromptData{
			Title:   "External write access",
			Message: fmt.Sprintf("%s wants to modify a file outside the current workspace.\n\nCurrent workspace:\n  %s\n\nRequested file:\n  %s\n\n%s:\n  %s\n\nThis grants write access under the selected root.\nOnly approve if you trust this task.", operation, denial.WorkDir, denial.InputPath, pathAuthorizationTargetLabel(denial), denial.SuggestedRoot),
			Options: []model.PermissionOption{
				{Input: "1", Label: "1. Allow once"},
				{Input: "2", Label: "2. Allow for this session"},
				{Input: "4", Label: "4. Deny"},
			},
			DefaultIndex: 0,
		}
	}
	return &model.PermissionPromptData{
		Title:   "External read access",
		Message: fmt.Sprintf("%s wants to access a file outside the current workspace.\n\nCurrent workspace:\n  %s\n\nRequested file:\n  %s\n\n%s:\n  %s", operation, denial.WorkDir, denial.InputPath, pathAuthorizationTargetLabel(denial), denial.SuggestedRoot),
		Options: []model.PermissionOption{
			{Input: "1", Label: "1. Allow once"},
			{Input: "2", Label: "2. Allow for this session"},
			{Input: "3", Label: "3. Always allow"},
			{Input: "4", Label: "4. Deny"},
		},
		DefaultIndex: 0,
	}
}
