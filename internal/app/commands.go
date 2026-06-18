package app

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	agentctx "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/configs"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/permission"
	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/ui/model"
)

func (a *Application) handleCommand(input string) {
	cmd, ok := splitRawCommand(input)
	if !ok {
		return
	}
	args := strings.Fields(cmd.Remainder)

	switch cmd.Name {
	case "/model":
		a.cmdModel(args)
	case "/effort":
		a.cmdEffort(args)
	case "/exit":
		a.cmdExit()
	case "/compact":
		a.cmdCompact(cmd.Remainder)
	case "/ctx":
		a.cmdCtx()
	case "/init":
		a.cmdInit()
	case "/clear":
		a.cmdClear()
	case "/branch", "/fork":
		a.cmdBranch(cmd.Name, args)
	case "/rewind":
		a.cmdRewind(args)
	case "/resume":
		a.cmdResume(args)
	case "/replay":
		a.cmdReplay(args)
	case "/__rewind":
		a.cmdRewindApply(args)
	case "/permissions":
		a.cmdPermissions(nil)
	case "/yolo":
		a.cmdYolo()
	case "/train":
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Coming soon."}
		return
	case "/diagnose":
		a.handleExpandableCommand(cmd.Remainder, a.expandIssueCommandInputWithOptions, a.cmdDiagnose)
	case "/fix":
		a.handleExpandableCommand(cmd.Remainder, a.expandIssueCommandInputWithOptions, a.cmdFix)
	case "/migrate":
		a.handleExpandableCommand(cmd.Remainder, a.expandIssueCommandInputWithOptions, a.cmdMigrate)
	case "/integrate":
		a.handleExpandableCommand(cmd.Remainder, a.expandIssueCommandInputWithOptions, a.cmdIntegrate)
	case "/preflight":
		a.handleExpandableCommand(cmd.Remainder, a.expandInputTextWithOptions, a.cmdPreflight)
	case "/factory":
		a.cmdFactory(cmd.Remainder)
	case "/mcp":
		a.cmdMCP(args)
	case "/skill":
		if err := a.handleRawSkillCommand(cmd.Remainder); err != nil {
			a.emitInputExpansionError(err)
		}
	case "/skill-add":
		a.cmdSkillAddInput(cmd.Remainder)
	default:
		if cmd.Name == "/permission" {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: "Command `/permission` has been removed. Use `/permissions`.",
			}
			return
		}
		if handled, err := a.handleSkillAliasCommand(cmd.Name, cmd.Remainder); handled {
			if err != nil {
				a.emitInputExpansionError(err)
			}
			return
		}
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Unknown command: %s. Type / to see available commands.", cmd.Name),
		}
	}
}

func (a *Application) handleExpandableCommand(raw string, expand func(string, pathpolicy.ResolveOptions) (string, error), run func(string)) {
	expanded, err := expand(raw, pathpolicy.ResolveOptions{})
	if err != nil {
		if a.tryAuthorizeInputExpansion(err, func(opts pathpolicy.ResolveOptions) {
			expanded, retryErr := expand(raw, opts)
			if retryErr != nil {
				a.emitInputExpansionError(retryErr)
				return
			}
			run(expanded)
		}) {
			return
		}
		a.emitInputExpansionError(err)
		return
	}
	run(expanded)
}

func (a *Application) cmdMCP(args []string) {
	if len(args) == 0 || args[0] == "no-redirect" {
		a.cmdMCPSummary()
		return
	}

	switch args[0] {
	case "reconnect":
		a.cmdMCPReconnect(strings.Join(args[1:], " "))
	case "enable":
		a.cmdMCPToggle(true, strings.Join(args[1:], " "))
	case "disable":
		a.cmdMCPToggle(false, strings.Join(args[1:], " "))
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /mcp [reconnect <server>|enable [server]|disable [server]]",
		}
	}
}

func (a *Application) cmdMCPSummary() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resolved, _, err := a.resolveMCPForCommand(ctx)
	if err != nil {
		a.emitMCPCommandError("mcp", err)
		return
	}
	if len(resolved.Servers) == 0 && len(resolved.Pending) == 0 && len(resolved.Rejected) == 0 && len(resolved.Disabled) == 0 && len(resolved.Warnings) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "No MCP servers configured."}
		return
	}

	manager := a.mcpManager

	var b strings.Builder
	b.WriteString("MCP servers:")
	for _, server := range resolved.Servers {
		status := "not connected"
		if manager == nil {
			b.WriteString(fmt.Sprintf("\n  %s [%s] %s - %s", server.Name, server.Scope, server.Config.TransportType(), status))
			continue
		}
		defs, err := manager.ListTools(ctx, server.Name)
		if err != nil {
			status = "failed: " + err.Error()
		} else {
			status = fmt.Sprintf("connected, %d tools", len(defs))
		}
		b.WriteString(fmt.Sprintf("\n  %s [%s] %s - %s", server.Name, server.Scope, server.Config.TransportType(), status))
	}
	for _, server := range resolved.Pending {
		b.WriteString(fmt.Sprintf("\n  %s [%s] %s - pending approval", server.Name, server.Scope, server.Config.TransportType()))
	}
	for _, server := range resolved.Rejected {
		b.WriteString(fmt.Sprintf("\n  %s [%s] %s - rejected", server.Name, server.Scope, server.Config.TransportType()))
	}
	for _, server := range resolved.Disabled {
		b.WriteString(fmt.Sprintf("\n  %s [%s] %s - disabled", server.Name, server.Scope, server.Config.TransportType()))
	}
	if len(resolved.Warnings) > 0 {
		b.WriteString("\n\nWarnings:")
		for _, warning := range resolved.Warnings {
			b.WriteString("\n  " + warning)
		}
	}
	a.EventCh <- model.Event{Type: model.AgentReply, Message: b.String()}
}

func (a *Application) cmdMCPReconnect(serverName string) {
	name := strings.TrimSpace(serverName)
	if name == "" {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /mcp reconnect <server>"}
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resolved, workspaceRoot, err := a.resolveMCPForCommand(ctx)
	if err != nil {
		a.emitMCPCommandError("mcp", err)
		return
	}
	server, ok := findMCPServer(resolved.Servers, name)
	if !ok {
		if containsMCPServer(resolved.Pending, name) || containsMCPServer(resolved.Rejected, name) || containsMCPServer(resolved.Disabled, name) {
			a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("Failed to reconnect to %s", name)}
			return
		}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q not found", name)}
		return
	}

	if err := a.reconnectMCPServer(ctx, workspaceRoot, server); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("Failed to reconnect to %s", name)}
		return
	}
	a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("Successfully reconnected to %s", name)}
}

func (a *Application) reconnectMCPServer(ctx context.Context, workspaceRoot string, server runtimemcp.ScopedServer) error {
	manager, _ := a.mcpCommandManager(workspaceRoot)
	if err := manager.Connect(ctx, server); err != nil {
		return err
	}
	defs, err := manager.ListTools(ctx, server.Name)
	if err != nil {
		return err
	}
	unregisterMCPServerTools(a.toolRegistry, server.Name)
	artifactStore, err := newMCPArtifactStore(workspaceRoot)
	if err != nil {
		return err
	}
	registerMCPToolDefinitions(a.toolRegistry, manager, normalizeMCPToolDefinitions(server, defs), nil, artifactStore)
	return nil
}

func (a *Application) cmdMCPToggle(enable bool, target string) {
	name := strings.TrimSpace(target)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resolved, workspaceRoot, err := a.resolveMCPForCommand(ctx)
	if err != nil {
		a.emitMCPCommandError("mcp", err)
		return
	}

	if name != "" {
		a.cmdMCPToggleOne(ctx, workspaceRoot, resolved, enable, name)
		return
	}

	var targets []runtimemcp.ScopedServer
	if enable {
		targets = resolved.Disabled
	} else {
		targets = resolved.Servers
	}
	if len(targets) == 0 {
		state := "disabled"
		if enable {
			state = "enabled"
		}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("All MCP servers are already %s", state)}
		return
	}
	enabledCount := 0
	skippedCount := 0
	for _, server := range targets {
		if enable {
			if _, enabled, err := a.enableDisabledMCPServer(ctx, workspaceRoot, server); err != nil {
				a.emitMCPCommandError("mcp", err)
				return
			} else if !enabled {
				skippedCount++
				continue
			}
			enabledCount++
			continue
		}
		if _, err := runtimemcp.SetLocalServerDisabled(workspaceRoot, server.Name, true); err != nil {
			a.emitMCPCommandError("mcp", err)
			return
		}
		if !enable && a.mcpManager != nil {
			_ = a.mcpManager.CloseServer(ctx, server.Name)
		}
		if !enable {
			unregisterMCPServerTools(a.toolRegistry, server.Name)
		}
	}
	action := "Disabled"
	count := len(targets)
	if enable {
		action = "Enabled"
		count = enabledCount
	}
	message := fmt.Sprintf("%s %d MCP server(s)", action, count)
	if enable && skippedCount > 0 {
		message = fmt.Sprintf("%s, skipped %d", message, skippedCount)
	}
	a.EventCh <- model.Event{Type: model.AgentReply, Message: message}
}

func (a *Application) cmdMCPToggleOne(ctx context.Context, workspaceRoot string, resolved runtimemcp.ResolvedConfig, enable bool, name string) {
	if server, ok := findMCPServer(resolved.Disabled, name); ok {
		if enable {
			message, _, err := a.enableDisabledMCPServer(ctx, workspaceRoot, server)
			if err != nil {
				a.emitMCPCommandError("mcp", err)
				return
			}
			a.EventCh <- model.Event{Type: model.AgentReply, Message: message}
			return
		}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q disabled", name)}
		return
	}
	if server, ok := findMCPServer(resolved.Servers, name); ok {
		if !enable {
			if _, err := runtimemcp.SetLocalServerDisabled(workspaceRoot, server.Name, true); err != nil {
				a.emitMCPCommandError("mcp", err)
				return
			}
			if a.mcpManager != nil {
				_ = a.mcpManager.CloseServer(ctx, server.Name)
			}
			unregisterMCPServerTools(a.toolRegistry, server.Name)
		}
		state := "enabled"
		if !enable {
			state = "disabled"
		}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q %s", name, state)}
		return
	}
	if server, ok := findMCPServer(resolved.Pending, name); ok {
		if enable {
			a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q is pending approval; approve it before enabling", name)}
			return
		}
		if _, err := runtimemcp.SetLocalServerDisabled(workspaceRoot, server.Name, true); err != nil {
			a.emitMCPCommandError("mcp", err)
			return
		}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q disabled", name)}
		return
	}
	if server, ok := findMCPServer(resolved.Rejected, name); ok {
		if enable {
			a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q is rejected; reset project choices or change approval before enabling", name)}
			return
		}
		if _, err := runtimemcp.SetLocalServerDisabled(workspaceRoot, server.Name, true); err != nil {
			a.emitMCPCommandError("mcp", err)
			return
		}
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q disabled", name)}
		return
	}
	a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("MCP server %q not found", name)}
}

func (a *Application) enableDisabledMCPServer(ctx context.Context, workspaceRoot string, server runtimemcp.ScopedServer) (string, bool, error) {
	if _, err := runtimemcp.SetLocalServerDisabled(workspaceRoot, server.Name, false); err != nil {
		return "", false, err
	}
	restoreDisabled := func() error {
		_, err := runtimemcp.SetLocalServerDisabled(workspaceRoot, server.Name, true)
		return err
	}
	refreshed, _, err := a.resolveMCPForCommand(ctx)
	if err != nil {
		if restoreErr := restoreDisabled(); restoreErr != nil {
			return "", false, fmt.Errorf("restore mcp server %q disabled state: %w", server.Name, restoreErr)
		}
		return "", false, err
	}
	if server, ok := findMCPServer(refreshed.Pending, server.Name); ok {
		if err := restoreDisabled(); err != nil {
			return "", false, fmt.Errorf("restore mcp server %q disabled state: %w", server.Name, err)
		}
		return fmt.Sprintf("MCP server %q is pending approval; approve it before enabling", server.Name), false, nil
	}
	if server, ok := findMCPServer(refreshed.Rejected, server.Name); ok {
		if err := restoreDisabled(); err != nil {
			return "", false, fmt.Errorf("restore mcp server %q disabled state: %w", server.Name, err)
		}
		return fmt.Sprintf("MCP server %q is rejected; reset project choices or change approval before enabling", server.Name), false, nil
	}
	active, ok := findMCPServer(refreshed.Servers, server.Name)
	if !ok {
		if err := restoreDisabled(); err != nil {
			return "", false, fmt.Errorf("restore mcp server %q disabled state: %w", server.Name, err)
		}
		return fmt.Sprintf("MCP server %q not found", server.Name), false, nil
	}
	if err := a.reconnectMCPServer(ctx, workspaceRoot, active); err != nil {
		if restoreErr := restoreDisabled(); restoreErr != nil {
			return "", false, fmt.Errorf("restore mcp server %q disabled state: %w", server.Name, restoreErr)
		}
		return fmt.Sprintf("MCP server %q failed to reconnect after enable: %v", server.Name, err), false, nil
	}
	return fmt.Sprintf("MCP server %q enabled", server.Name), true, nil
}

func (a *Application) resolveMCPForCommand(ctx context.Context) (runtimemcp.ResolvedConfig, string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return runtimemcp.ResolvedConfig{}, "", fmt.Errorf("resolve home directory: %w", err)
	}
	workspace := strings.TrimSpace(a.WorkDir)
	if workspace == "" {
		workspace, err = os.Getwd()
		if err != nil {
			return runtimemcp.ResolvedConfig{}, "", fmt.Errorf("resolve workspace: %w", err)
		}
	}
	workspaceAbs, err := filepath.Abs(workspace)
	if err != nil {
		return runtimemcp.ResolvedConfig{}, "", fmt.Errorf("resolve workspace path: %w", err)
	}
	resolved, err := resolveMCPConfig(ctx, runtimemcp.ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspaceAbs,
		ApprovalStore: runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home)),
	})
	if err != nil {
		return runtimemcp.ResolvedConfig{}, "", err
	}
	return resolved, workspaceAbs, nil
}

func (a *Application) mcpCommandManager(workspaceRoot string) (runtimemcp.Manager, bool) {
	if a.mcpManager != nil {
		return a.mcpManager, false
	}
	timeout := time.Second * 30
	if a.Config != nil && a.Config.Execution.TimeoutSec > 0 {
		timeout = time.Duration(a.Config.Execution.TimeoutSec) * time.Second
	}
	a.mcpManager = newMCPManager(runtimemcp.Config{
		WorkDir:        workspaceRoot,
		ConnectTimeout: 10 * time.Second,
		CallTimeout:    timeout,
	})
	return a.mcpManager, false
}

func findMCPServer(servers []runtimemcp.ScopedServer, name string) (runtimemcp.ScopedServer, bool) {
	for _, server := range servers {
		if server.Name == name {
			return server, true
		}
	}
	return runtimemcp.ScopedServer{}, false
}

func containsMCPServer(servers []runtimemcp.ScopedServer, name string) bool {
	_, ok := findMCPServer(servers, name)
	return ok
}

func (a *Application) emitMCPCommandError(tool string, err error) {
	a.EventCh <- model.Event{Type: model.ToolError, ToolName: tool, Message: err.Error()}
}

func (a *Application) handleRawSkillCommand(rawInput string) error {
	if strings.TrimSpace(rawInput) == "" {
		a.cmdSkill(nil)
		return nil
	}

	skillName, request := splitFirstToken(rawInput)
	if skillName == "" {
		a.cmdSkill(nil)
		return nil
	}

	run := func(expanded string) {
		a.runLoadedSkillCommand(skillName, expanded)
	}
	if request == "" {
		run("")
		return nil
	}
	expanded, err := a.expandInputTextWithOptions(request, pathpolicy.ResolveOptions{})
	if err != nil {
		if a.tryAuthorizeInputExpansion(err, func(opts pathpolicy.ResolveOptions) {
			expanded, retryErr := a.expandInputTextWithOptions(request, opts)
			if retryErr != nil {
				a.emitInputExpansionError(retryErr)
				return
			}
			run(expanded)
		}) {
			return nil
		}
		return err
	}
	run(expanded)
	return nil
}

func (a *Application) handleSkillAliasCommand(commandName, rawRemainder string) (bool, error) {
	if a.skillLoader == nil {
		return false, nil
	}

	skillName := strings.TrimPrefix(strings.TrimSpace(commandName), "/")
	if skillName == "" {
		return false, nil
	}
	if _, err := a.skillLoader.Load(skillName); err != nil {
		return false, nil
	}

	request := strings.TrimSpace(rawRemainder)
	run := func(expanded string) {
		a.runLoadedSkillCommand(skillName, expanded)
	}
	if request == "" {
		run("")
		return true, nil
	}
	expanded, err := a.expandInputTextWithOptions(request, pathpolicy.ResolveOptions{})
	if err != nil {
		if a.tryAuthorizeInputExpansion(err, func(opts pathpolicy.ResolveOptions) {
			expanded, retryErr := a.expandInputTextWithOptions(request, opts)
			if retryErr != nil {
				a.emitInputExpansionError(retryErr)
				return
			}
			run(expanded)
		}) {
			return true, nil
		}
		return true, err
	}
	run(expanded)
	return true, nil
}

func (a *Application) cmdModel(args []string) {
	a.emitModelSetupPopup(true)
}

func (a *Application) cmdEffort(args []string) {
	if a.Config == nil {
		a.Config = configs.DefaultConfig()
	}
	providerName := a.currentEffortProvider()
	options := configs.EffortOptionsForProvider(providerName)
	current := configs.NormalizeEffort(a.Config.Request.Effort)
	if current == "" {
		current = configs.DefaultRequestEffort
	}

	if len(args) == 0 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Current effort: %s (provider: %s, options: %s)", current, providerName, strings.Join(options, ", ")),
		}
		return
	}
	if len(args) != 1 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Usage: /effort <%s>", strings.Join(options, "|")),
		}
		return
	}

	effort := configs.NormalizeEffort(args[0])
	if !configs.EffortAllowedForProvider(providerName, effort) {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Unsupported effort %q for provider %s. Options: %s", args[0], providerName, strings.Join(options, ", ")),
		}
		return
	}

	a.Config.Request.Effort = effort
	if a.Engine != nil {
		a.Engine.SetEffort(effort)
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Effort set to: %s", effort),
	}
}

func (a *Application) currentEffortProvider() string {
	if a != nil && a.provider != nil {
		if name := llm.NormalizeProvider(a.provider.Name()); name != "" {
			return name
		}
	}
	if a != nil && a.Config != nil {
		if name := llm.NormalizeProvider(a.Config.Model.Provider); name != "" {
			return name
		}
	}
	return "openai-completion"
}

func (a *Application) cmdExit() {
	a.EventCh <- model.Event{Type: model.AgentReply, Message: "Goodbye!"}
	go func() {
		time.Sleep(100 * time.Millisecond)
		a.EventCh <- model.Event{Type: model.Done}
	}()
}

func (a *Application) cmdCompact(customInstructions string) {
	a.EventCh <- model.Event{Type: model.AgentThinking}

	if a.ctxManager == nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Context compaction is not available.",
		}
		return
	}
	a.EventCh <- model.Event{
		Type: model.ContextCompactStarted,
	}

	compactCtx := context.Background()
	cancel := func() {}
	if a.Config != nil && a.Config.Model.TimeoutSec > 0 {
		compactCtx, cancel = context.WithTimeout(compactCtx, time.Duration(a.Config.Model.TimeoutSec)*time.Second)
	}
	defer cancel()

	before := a.ctxManager.TokenUsage()
	if err := a.ctxManager.CompactWithContextInstructions(compactCtx, customInstructions); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "context",
			Message:  fmt.Sprintf("Failed to compact context: %v", err),
		}
		return
	}
	after := a.ctxManager.TokenUsage()
	if err := a.persistSessionSnapshot(); err != nil {
		a.emitToolError("session", "Failed to persist session snapshot: %v", err)
	}
	a.emitTokenUsageSnapshot()

	message := fmt.Sprintf("Context compacted: %d -> %d tokens.", before.Current, after.Current)
	if after.Current >= before.Current {
		message = "Context compaction had nothing to remove."
	}
	if err := a.recordContextCompaction("manual", before.Current, after.Current, message); err != nil {
		a.emitToolError("session", "Failed to record context compaction: %v", err)
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: message,
	}
}

func (a *Application) recordContextCompaction(trigger string, beforeTokens, afterTokens int, message string) error {
	if a == nil || a.session == nil {
		return nil
	}
	if err := a.noteLiveLLMActivity(); err != nil {
		return err
	}
	return a.session.AppendContextCompaction(trigger, beforeTokens, afterTokens, message)
}

func (a *Application) cmdCtx() {
	if a.ctxManager == nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Context usage is not available.",
		}
		return
	}

	a.emitTokenUsageSnapshot()
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: formatContextUsageMessage(a.displayTokenUsageDetails()),
	}
}

func formatContextUsageMessage(details agentctx.TokenUsageDetails) string {
	lines := []string{
		"Context usage:",
		"",
		fmt.Sprintf("  Current:   %d", details.Current),
		fmt.Sprintf("  Window:    %d", details.ContextWindow),
		fmt.Sprintf("  Reserved:  %d", details.Reserved),
		fmt.Sprintf("  Available: %d", details.Available),
	}

	if details.Source == agentctx.TokenUsageSourceProvider {
		if stats := formatProviderUsageStats(details.ProviderUsage, details.ProviderTokenScope); len(stats) > 0 {
			lines = append(lines, "", "Provider usage stats:")
			lines = append(lines, stats...)
		}
	}

	return strings.Join(lines, "\n")
}

func formatProviderUsageStats(usage llm.Usage, scope agentctx.ProviderTokenScope) []string {
	filtered := filterProviderUsageStats(flattenUsageRaw(usage.Raw), scope)
	if len(filtered) > 0 {
		return filtered
	}

	lines := make([]string, 0, 3)
	switch scope {
	case agentctx.ProviderTokenScopeTotal:
		if usage.PromptTokens > 0 {
			lines = append(lines, fmt.Sprintf("  prompt_tokens: %d", usage.PromptTokens))
		}
	case agentctx.ProviderTokenScopePrompt:
		if usage.CompletionTokens > 0 {
			lines = append(lines, fmt.Sprintf("  completion_tokens: %d", usage.CompletionTokens))
		}
		if usage.TotalTokens > 0 {
			lines = append(lines, fmt.Sprintf("  total_tokens: %d", usage.TotalTokens))
		}
	default:
		if usage.PromptTokens > 0 {
			lines = append(lines, fmt.Sprintf("  prompt_tokens: %d", usage.PromptTokens))
		}
		if usage.CompletionTokens > 0 {
			lines = append(lines, fmt.Sprintf("  completion_tokens: %d", usage.CompletionTokens))
		}
		if usage.TotalTokens > 0 {
			lines = append(lines, fmt.Sprintf("  total_tokens: %d", usage.TotalTokens))
		}
	}
	return lines
}

func flattenUsageRaw(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return []string{fmt.Sprintf("  raw: %s", string(raw))}
	}

	var lines []string
	appendFlattenedUsageStats(&lines, "", value)
	return lines
}

func appendFlattenedUsageStats(lines *[]string, prefix string, value any) {
	switch v := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(v))
		for key := range v {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			nextPrefix := key
			if prefix != "" {
				nextPrefix = prefix + "." + key
			}
			appendFlattenedUsageStats(lines, nextPrefix, v[key])
		}
	case []any:
		if prefix == "" {
			data, _ := json.Marshal(v)
			*lines = append(*lines, fmt.Sprintf("  value: %s", string(data)))
			return
		}
		data, _ := json.Marshal(v)
		*lines = append(*lines, fmt.Sprintf("  %s: %s", prefix, string(data)))
	case nil:
		if prefix != "" {
			*lines = append(*lines, fmt.Sprintf("  %s: null", prefix))
		}
	case string:
		if prefix != "" {
			*lines = append(*lines, fmt.Sprintf("  %s: %s", prefix, v))
		}
	case bool:
		if prefix != "" {
			*lines = append(*lines, fmt.Sprintf("  %s: %t", prefix, v))
		}
	case float64:
		if prefix != "" {
			*lines = append(*lines, fmt.Sprintf("  %s: %v", prefix, v))
		}
	default:
		if prefix != "" {
			data, _ := json.Marshal(v)
			*lines = append(*lines, fmt.Sprintf("  %s: %s", prefix, string(data)))
		}
	}
}

func filterProviderUsageStats(lines []string, scope agentctx.ProviderTokenScope) []string {
	if len(lines) == 0 {
		return nil
	}

	skipPrefixes := map[string]struct{}{}
	switch scope {
	case agentctx.ProviderTokenScopePrompt:
		skipPrefixes["prompt_tokens:"] = struct{}{}
		skipPrefixes["input_tokens:"] = struct{}{}
	case agentctx.ProviderTokenScopeTotal:
		skipPrefixes["total_tokens:"] = struct{}{}
	}

	filtered := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		skip := false
		for prefix := range skipPrefixes {
			if strings.HasPrefix(trimmed, prefix) {
				skip = true
				break
			}
		}
		if !skip {
			filtered = append(filtered, line)
		}
	}
	return filtered
}

func (a *Application) cmdClear() {
	previousSessionID := ""
	if a.session != nil {
		previousSessionID = strings.TrimSpace(a.session.ID())
		if err := a.session.Activate(); err != nil {
			a.EventCh <- model.Event{
				Type:     model.ToolError,
				ToolName: "session",
				Message:  fmt.Sprintf("Failed to preserve the current conversation: %v", err),
			}
			return
		}
		if err := a.persistSessionSnapshot(); err != nil {
			a.EventCh <- model.Event{
				Type:     model.ToolError,
				ToolName: "session",
				Message:  fmt.Sprintf("Failed to preserve the current conversation: %v", err),
			}
			return
		}
	}
	a.interruptReplay()
	a.interruptActiveTasks()
	if err := a.rotateSession(); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "session",
			Message:  fmt.Sprintf("Failed to start a fresh conversation: %v", err),
		}
		return
	}
	if a.ctxManager != nil {
		a.ctxManager.Clear()
	}
	if err := a.persistSessionSnapshot(); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "session",
			Message:  fmt.Sprintf("Failed to persist session snapshot: %v", err),
		}
	}
	a.emitTokenUsageSnapshot()
	a.EventCh <- model.Event{
		Type:    model.ClearScreen,
		Message: "Chat history cleared.",
		Summary: inlineResumeHintForSession(previousSessionID),
	}
}

func (a *Application) cmdPermissions(args []string) {
	_ = args // /permissions is single-entry: ignore all trailing arguments.
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Permission management not available in current mode.",
		}
		return
	}

	a.EventCh <- model.Event{
		Type:        model.PermissionsView,
		Permissions: a.buildPermissionsViewData(permSvc),
	}
}

func (a *Application) cmdPermissionsInternal(args []string) {
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Permission management not available in current mode.",
		}
		return
	}
	if len(args) == 0 {
		a.cmdPermissions(nil)
		return
	}
	if len(args) >= 1 && strings.EqualFold(args[0], "add") {
		a.cmdPermissionsAdd(permSvc, args[1:])
		return
	}
	if len(args) >= 1 && strings.EqualFold(args[0], "remove") {
		a.cmdPermissionsRemove(permSvc, args[1:])
		return
	}
	if len(args) >= 2 {
		tool := args[0]
		level := permission.ParsePermissionLevel(args[1])
		if err := permSvc.AddRule(tool, level); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to set permission for '%s': %v", tool, err),
			}
			return
		}
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Permission for '%s' set to: %s", tool, level),
		}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: "internal permissions command requires action",
	}
}

func (a *Application) buildPermissionsViewData(permSvc *permission.DefaultPermissionService) *model.PermissionsViewData {
	data := &model.PermissionsViewData{
		RuleSources: map[string]string{},
	}

	for _, rv := range permSvc.GetRuleViews() {
		entry := strings.TrimSpace(rv.Rule)
		if entry == "" {
			continue
		}
		switch rv.Level {
		case permission.PermissionAllowAlways, permission.PermissionAllowSession, permission.PermissionAllowOnce:
			data.Allow = append(data.Allow, entry)
		case permission.PermissionDeny:
			data.Deny = append(data.Deny, entry)
		default:
			data.Ask = append(data.Ask, entry)
		}
		if strings.TrimSpace(rv.Source) != "" {
			data.RuleSources[entry] = rv.Source
		}
	}
	return data
}

func (a *Application) cmdPermissionsAdd(permSvc *permission.DefaultPermissionService, args []string) {
	if len(args) >= 2 {
		level := permission.ParsePermissionLevel(args[0])
		scope, hasScope, rest := parsePermissionScopeArgs(args[1:])
		rule := strings.TrimSpace(strings.Join(rest, " "))
		if strings.Contains(rule, "(") || strings.HasPrefix(strings.ToLower(rule), "mcp__") {
			if err := permSvc.AddRule(rule, level); err != nil {
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Failed to add rule: %v", err),
				}
				return
			}
			if hasScope {
				path, err := a.savePermissionRuleToScope(rule, level, scope)
				if err != nil {
					a.EventCh <- model.Event{
						Type:    model.AgentReply,
						Message: fmt.Sprintf("Added rule for this session, but failed to save settings file: %v", err),
					}
					return
				}
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Added rule: %s => %s (saved to %s)", rule, level, path),
				}
				return
			}
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Added rule: %s => %s", rule, level),
			}
			return
		}
	}

	if len(args) < 3 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "invalid internal permissions add command",
		}
		return
	}

	targetType := strings.ToLower(strings.TrimSpace(args[0]))
	target := strings.TrimSpace(strings.Join(args[1:len(args)-1], " "))
	level := permission.ParsePermissionLevel(args[len(args)-1])

	switch targetType {
	case "tool":
		if err := permSvc.AddRule(permissionRuleForLegacyTarget("tool", target), level); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to add tool rule: %v", err),
			}
			return
		}
	case "command":
		if err := permSvc.AddRule(permissionRuleForLegacyTarget("command", target), level); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to add command rule: %v", err),
			}
			return
		}
	case "path":
		if err := permSvc.AddRule(permissionRuleForLegacyTarget("path", target), level); err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to add path rule: %v", err),
			}
			return
		}
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Invalid rule type. Use: tool, command, path",
		}
		return
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Added %s rule: %s => %s", targetType, target, level),
	}
}

func permissionRuleForLegacyTarget(targetType, target string) string {
	switch strings.ToLower(strings.TrimSpace(targetType)) {
	case "tool":
		return target
	case "command":
		cmd := strings.TrimSpace(target)
		if cmd == "" {
			return "Bash(*)"
		}
		if strings.HasSuffix(cmd, "*") {
			return fmt.Sprintf("Bash(%s)", cmd)
		}
		return fmt.Sprintf("Bash(%s *)", cmd)
	case "path":
		p := strings.TrimSpace(target)
		if filepath.IsAbs(p) {
			p = "//" + strings.TrimPrefix(filepath.ToSlash(p), "/")
		}
		return fmt.Sprintf("Edit(%s)", p)
	default:
		return strings.TrimSpace(target)
	}
}

func (a *Application) cmdPermissionsRemove(permSvc *permission.DefaultPermissionService, args []string) {
	if len(args) >= 1 {
		rule := strings.TrimSpace(strings.Join(args, " "))
		if strings.Contains(rule, "(") || strings.HasPrefix(strings.ToLower(rule), "mcp__") {
			ok, err := permSvc.RemoveRule(rule)
			if err != nil {
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Failed to remove rule: %v", err),
				}
				return
			}
			if !ok {
				a.EventCh <- model.Event{
					Type:    model.AgentReply,
					Message: fmt.Sprintf("Rule not found: %s", rule),
				}
				return
			}
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Removed rule: %s", rule),
			}
			return
		}
	}

	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "invalid internal permissions remove command",
		}
		return
	}

	targetType := strings.ToLower(strings.TrimSpace(args[0]))
	target := strings.TrimSpace(strings.Join(args[1:], " "))

	switch targetType {
	case "tool":
		ok, err := permSvc.RemoveRule(permissionRuleForLegacyTarget("tool", target))
		if err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to remove tool rule: %v", err),
			}
			return
		}
		if !ok {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Rule not found: %s", target),
			}
			return
		}
	case "command":
		ok, err := permSvc.RemoveRule(permissionRuleForLegacyTarget("command", target))
		if err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to remove command rule: %v", err),
			}
			return
		}
		if !ok {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Rule not found: %s", target),
			}
			return
		}
	case "path":
		ok, err := permSvc.RemoveRule(permissionRuleForLegacyTarget("path", target))
		if err != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Failed to remove path rule: %v", err),
			}
			return
		}
		if !ok {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: fmt.Sprintf("Rule not found: %s", target),
			}
			return
		}
	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Invalid rule type. Use: tool, command, path",
		}
		return
	}

	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Removed %s rule: %s", targetType, target),
	}
}

func (a *Application) cmdYolo() {
	permSvc, ok := a.permService.(*permission.DefaultPermissionService)
	if !ok {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "YOLO mode not available in current configuration.",
		}
		return
	}

	current := permSvc.Check("shell", "")
	if current == permission.PermissionAllowAlways {
		permSvc.Grant("shell", permission.PermissionAsk)
		permSvc.Grant("write", permission.PermissionAsk)
		permSvc.Grant("edit", permission.PermissionAsk)
		permSvc.Grant("load_skill", permission.PermissionAsk)
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "YOLO mode disabled. Will ask for confirmation on destructive operations.",
		}
	} else {
		permSvc.Grant("shell", permission.PermissionAllowAlways)
		permSvc.Grant("write", permission.PermissionAllowAlways)
		permSvc.Grant("edit", permission.PermissionAllowAlways)
		permSvc.Grant("read", permission.PermissionAllowAlways)
		permSvc.Grant("grep", permission.PermissionAllowAlways)
		permSvc.Grant("glob", permission.PermissionAllowAlways)
		permSvc.Grant("load_skill", permission.PermissionAllowAlways)
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "YOLO mode enabled! All operations will be auto-approved. Use with caution!",
		}
	}
}

func (a *Application) cmdSkill(args []string) {
	if a.skillLoader == nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Skills not available."}
		return
	}
	if len(args) == 0 {
		a.emitAvailableSkills(true)
		return
	}

	skillName := args[0]
	userRequest := strings.TrimSpace(strings.Join(args[1:], " "))
	a.runLoadedSkillCommand(skillName, userRequest)
}

func (a *Application) runLoadedSkillCommand(skillName, userRequest string) {
	content, err := a.skillLoader.Load(skillName)
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.ToolError,
			Message: fmt.Sprintf("Failed to load skill %q: %v", skillName, err),
		}
		return
	}

	// Inject a synthetic assistant tool_call + tool result into context so the
	// model sees the skill as already loaded and won't call load_skill again.
	toolCallID := "slash_skill_" + skillName
	argBytes, _ := json.Marshal(map[string]string{"name": skillName})
	assistantMsg := llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{
			{
				ID:   toolCallID,
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "load_skill",
					Arguments: json.RawMessage(argBytes),
				},
			},
		},
	}
	if err := a.addContextMessages(assistantMsg, llm.NewToolMessage(toolCallID, content)); err != nil {
		a.emitToolError("load_skill", "Failed to activate skill %q: %v", skillName, err)
		return
	}
	if a.session != nil {
		if err := a.session.AppendSkillActivation(skillName); err != nil {
			a.emitToolError("session", "Failed to persist skill activation: %v", err)
		}
		if err := a.persistSessionSnapshot(); err != nil {
			a.emitToolError("session", "Failed to persist session snapshot: %v", err)
		}
	}
	a.EventCh <- model.Event{
		Type:     model.ToolSkill,
		ToolName: "load_skill",
		Message:  skillName,
		Summary:  fmt.Sprintf("loaded skill: %s", skillName),
	}

	if userRequest == "" {
		userRequest = defaultSkillRequest(skillName)
	}
	go a.runTask(userRequest)
}

func defaultSkillRequest(skillName string) string {
	return fmt.Sprintf(
		`The %q skill is already loaded. Start following that skill now using the current workspace and conversation context. Begin with the first concrete step immediately, keep gathering evidence with tools, and only stop to ask the user if the skill cannot proceed without missing information.`,
		skillName,
	)
}
