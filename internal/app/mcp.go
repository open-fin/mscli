package app

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
	mcptool "gitcode.com/mindspore/mscli/tools/mcp"
	"gitcode.com/mindspore/mscli/ui/model"
)

var resolveMCPConfig = runtimemcp.ResolveConfig

var newMCPManager = func(cfg runtimemcp.Config) runtimemcp.Manager {
	return runtimemcp.NewManager(cfg)
}

const defaultMCPStartupDiscoveryTimeout = 15 * time.Second

// MCPApprovalRequest describes a pending project MCP server approval.
type MCPApprovalRequest struct {
	WorkspaceRoot string
	ServerName    string
	ConfigHash    string
	Command       string
	Args          []string
	EnvKeys       []string
}

// MCPApprovalPrompter asks the user for a project MCP server decision.
type MCPApprovalPrompter interface {
	PromptMCPApproval(ctx context.Context, req MCPApprovalRequest) (runtimemcp.ApprovalDecision, error)
}

func initMCPTools(ctx context.Context, registry *tools.Registry, workDir string, discoveryTimeout time.Duration, prompter MCPApprovalPrompter) (runtimemcp.Manager, []model.Event, error) {
	if registry == nil {
		return nil, nil, fmt.Errorf("tool registry is nil")
	}
	var events []model.Event
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, events, fmt.Errorf("resolve home directory: %w", err)
	}
	workspaceRoot, err := filepath.Abs(workDir)
	if err != nil {
		return nil, events, fmt.Errorf("resolve workspace path: %w", err)
	}
	store := runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home))
	opts := runtimemcp.ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspaceRoot,
		ApprovalStore: store,
	}

	resolved, err := resolveMCPConfig(ctx, opts)
	if err != nil {
		return nil, events, err
	}
	emitMCPResolveWarnings(&events, resolved)

	recordedDecision := false
	for _, server := range sortedServersCopy(resolved.Pending) {
		emitMCPWarning(&events, fmt.Sprintf("mcp project server %s requires approval before it can start", server.Name))
		decision, promptErr := prompterOrDefault(prompter).PromptMCPApproval(ctx, approvalRequest(workspaceRoot, server))
		if promptErr != nil || (decision != runtimemcp.DecisionApproved && decision != runtimemcp.DecisionRejected) {
			emitMCPWarning(&events, fmt.Sprintf("mcp project server %s skipped for this startup", server.Name))
			continue
		}
		if err := store.Record(workspaceRoot, server.Name, server.Hash, decision); err != nil {
			emitMCPWarning(&events, fmt.Sprintf("record mcp approval for %s: %v", server.Name, err))
			continue
		}
		recordedDecision = true
		switch decision {
		case runtimemcp.DecisionApproved:
			emitMCPWarning(&events, fmt.Sprintf("mcp project server %s approved", server.Name))
		case runtimemcp.DecisionRejected:
			emitMCPWarning(&events, fmt.Sprintf("mcp project server %s rejected and skipped", server.Name))
		}
	}
	if recordedDecision {
		resolved, err = resolveMCPConfig(ctx, opts)
		if err != nil {
			return nil, events, err
		}
		emitMCPResolveWarnings(&events, resolved)
	}
	if discoveryTimeout <= 0 {
		discoveryTimeout = defaultMCPStartupDiscoveryTimeout
	}

	manager := newMCPManager(runtimemcp.Config{
		WorkDir:        workspaceRoot,
		ConnectTimeout: 10 * time.Second,
		CallTimeout:    discoveryTimeout,
	})

	candidates := make([]mcpToolCandidate, 0)
	for _, server := range resolved.Servers {
		if err := manager.Connect(ctx, server); err != nil {
			emitMCPWarning(&events, fmt.Sprintf("connect mcp server %s: %v", server.Name, err))
			continue
		}
		defs, err := manager.ListTools(ctx, server.Name)
		if err != nil {
			emitMCPWarning(&events, fmt.Sprintf("list mcp tools for %s: %v", server.Name, err))
			continue
		}
		for _, def := range defs {
			if strings.TrimSpace(def.ServerName) == "" {
				def.ServerName = server.Name
			}
			if strings.TrimSpace(def.Name) == "" {
				def.Name = runtimemcp.BuildToolName(def.ServerName, def.OriginalToolName)
			}
			candidates = append(candidates, mcpToolCandidate{server: server, def: def})
		}
	}

	for _, def := range dedupeMCPTools(candidates, &events) {
		for _, tool := range mcptool.WrapTools([]runtimemcp.ToolDefinition{def}, manager) {
			if err := registry.Register(tool); err != nil {
				emitMCPWarning(&events, fmt.Sprintf("register mcp tool %s: %v", tool.Name(), err))
			}
		}
	}

	return manager, events, nil
}

func emitMCPResolveWarnings(events *[]model.Event, resolved runtimemcp.ResolvedConfig) {
	for _, warning := range resolved.Warnings {
		emitMCPWarning(events, warning)
	}
	for _, server := range resolved.Rejected {
		emitMCPWarning(events, fmt.Sprintf("mcp project server %s rejected and skipped until config changes or approvals are reset", server.Name))
	}
}

func emitMCPWarning(events *[]model.Event, msg string) {
	if events == nil || strings.TrimSpace(msg) == "" {
		return
	}
	*events = append(*events, model.Event{
		Type:     model.ToolWarning,
		ToolName: "mcp",
		Message:  msg,
	})
}

func prompterOrDefault(prompter MCPApprovalPrompter) MCPApprovalPrompter {
	if prompter != nil {
		return prompter
	}
	return newTerminalMCPApprovalPrompter(os.Stdin, os.Stdout)
}

func approvalRequest(workspaceRoot string, server runtimemcp.ScopedServer) MCPApprovalRequest {
	envKeys := make([]string, 0, len(server.Config.Env))
	for key := range server.Config.Env {
		envKeys = append(envKeys, key)
	}
	sort.Strings(envKeys)
	return MCPApprovalRequest{
		WorkspaceRoot: workspaceRoot,
		ServerName:    server.Name,
		ConfigHash:    server.Hash,
		Command:       server.Config.Command,
		Args:          append([]string(nil), server.Config.Args...),
		EnvKeys:       envKeys,
	}
}

type mcpToolCandidate struct {
	server runtimemcp.ScopedServer
	def    runtimemcp.ToolDefinition
}

func dedupeMCPTools(candidates []mcpToolCandidate, events *[]model.Event) []runtimemcp.ToolDefinition {
	sort.Slice(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		if a.def.Name != b.def.Name {
			return a.def.Name < b.def.Name
		}
		if scopeRank(a.server.Scope) != scopeRank(b.server.Scope) {
			return scopeRank(a.server.Scope) > scopeRank(b.server.Scope)
		}
		if a.def.ServerName != b.def.ServerName {
			return a.def.ServerName < b.def.ServerName
		}
		return a.def.OriginalToolName < b.def.OriginalToolName
	})
	kept := make(map[string]mcpToolCandidate)
	order := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if _, ok := kept[candidate.def.Name]; !ok {
			kept[candidate.def.Name] = candidate
			order = append(order, candidate.def.Name)
			continue
		}
		winner := kept[candidate.def.Name]
		emitMCPWarning(events, fmt.Sprintf("duplicate mcp tool %s from %s/%s dropped; keeping %s/%s", candidate.def.Name, candidate.def.ServerName, candidate.def.OriginalToolName, winner.def.ServerName, winner.def.OriginalToolName))
	}
	out := make([]runtimemcp.ToolDefinition, 0, len(order))
	for _, name := range order {
		out = append(out, kept[name].def)
	}
	return out
}

func scopeRank(scope runtimemcp.Scope) int {
	switch scope {
	case runtimemcp.ScopeLocal:
		return 3
	case runtimemcp.ScopeProject:
		return 2
	case runtimemcp.ScopeUser:
		return 1
	default:
		return 0
	}
}

func sortedServersCopy(servers []runtimemcp.ScopedServer) []runtimemcp.ScopedServer {
	out := append([]runtimemcp.ScopedServer(nil), servers...)
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}
