package app

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
)

const mcpCLIName = "mscli mcp"

func runMCPCLI(args []string, stdout, stderr io.Writer) error {
	if stdout == nil {
		stdout = io.Discard
	}
	if stderr == nil {
		stderr = io.Discard
	}
	if len(args) == 0 {
		printMCPCLIUsage(stderr)
		return fmt.Errorf("missing mcp command")
	}

	switch args[0] {
	case "add":
		return runMCPCLIAdd(args[1:], stdout, stderr)
	case "add-json":
		return runMCPCLIAddJSON(args[1:], stdout, stderr)
	case "list":
		return runMCPCLIList(args[1:], stdout, stderr)
	case "get":
		return runMCPCLIGet(args[1:], stdout, stderr)
	case "remove":
		return runMCPCLIRemove(args[1:], stdout, stderr)
	case "reset-project-choices":
		return runMCPCLIResetProjectChoices(args[1:], stdout, stderr)
	default:
		printMCPCLIUsage(stderr)
		return fmt.Errorf("unknown mcp command %q", args[0])
	}
}

func runMCPCLIAddJSON(args []string, stdout, stderr io.Writer) error {
	fs := newMCPFlagSet("add-json", stderr)
	scopeShort, scopeLong := addScopeFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	scope, err := parseCLIFlagScope(*scopeShort, *scopeLong)
	if err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("usage: %s add-json [-s local|user|project] <name> <json>", mcpCLIName)
	}

	name := fs.Arg(0)
	cfg, err := serverConfigFromJSON(fs.Arg(1))
	if err != nil {
		return err
	}
	if err := runtimemcp.ValidateStdioServerConfig(cfg); err != nil {
		return err
	}
	home, workspace, err := mcpCLIHomeWorkspace()
	if err != nil {
		return err
	}
	path, err := runtimemcp.AddServerConfig(home, workspace, name, cfg, scope)
	if err != nil {
		return err
	}
	printMCPAdded(stdout, name, cfg.TransportType(), scope, path)
	return nil
}

func runMCPCLIAdd(args []string, stdout, _ io.Writer) error {
	parsed, err := parseMCPAddArgs(args)
	if err != nil {
		return err
	}
	cfg := runtimemcp.ServerConfig{
		Type: parsed.transport,
		Env:  parsed.env,
		URL:  parsed.url,
	}
	if parsed.transport == "http" {
		cfg.Command = ""
		cfg.Args = nil
	} else {
		cfg.Command = parsed.command[0]
		cfg.Args = append([]string(nil), parsed.command[1:]...)
	}
	home, workspace, err := mcpCLIHomeWorkspace()
	if err != nil {
		return err
	}
	path, err := runtimemcp.AddServerConfig(home, workspace, parsed.name, cfg, parsed.scope)
	if err != nil {
		return err
	}
	if parsed.transport == "http" {
		fmt.Fprintf(stdout, "Added http MCP server %s with URL: %s to %s config\n", parsed.name, parsed.url, parsed.scope)
	} else {
		fmt.Fprintf(stdout, "Added stdio MCP server %s with command: %s to %s config\n", parsed.name, strings.Join(parsed.command, " "), parsed.scope)
	}
	fmt.Fprintf(stdout, "File modified: %s\n", path)
	fmt.Fprintln(stdout, "Restart mscli to load tool registry changes.")
	return nil
}

func runMCPCLIRemove(args []string, stdout, stderr io.Writer) error {
	parsed, err := parseMCPRemoveArgs(args)
	if err != nil {
		return err
	}
	home, workspace, err := mcpCLIHomeWorkspace()
	if err != nil {
		return err
	}

	scope := parsed.scope
	if !parsed.scopeSet {
		matches, err := runtimemcp.FindServerScopes(home, workspace, parsed.name)
		if err != nil {
			return err
		}
		switch len(matches) {
		case 0:
			return fmt.Errorf("No MCP server found with name: %s", parsed.name)
		case 1:
			scope = matches[0].Scope
		default:
			fmt.Fprintf(stderr, "MCP server %q exists in multiple scopes:\n", parsed.name)
			for _, match := range sortedServersByScope(matches) {
				path, pathErr := runtimemcp.ConfigPathForScope(home, workspace, match.Scope)
				if pathErr != nil {
					return pathErr
				}
				fmt.Fprintf(stderr, "  - %s (%s)\n", match.Scope, path)
			}
			fmt.Fprintln(stderr)
			fmt.Fprintln(stderr, "To remove from a specific scope, use:")
			for _, match := range sortedServersByScope(matches) {
				fmt.Fprintf(stderr, "  %s remove %q -s %s\n", mcpCLIName, parsed.name, match.Scope)
			}
			return fmt.Errorf("MCP server %q exists in multiple scopes", parsed.name)
		}
	}

	path, err := runtimemcp.RemoveServerConfig(home, workspace, parsed.name, scope)
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Removed MCP server %q from %s config\n", parsed.name, scope)
	fmt.Fprintf(stdout, "File modified: %s\n", path)
	fmt.Fprintln(stdout, "Restart mscli to update tool registry changes.")
	return nil
}

func runMCPCLIResetProjectChoices(args []string, stdout, stderr io.Writer) error {
	fs := newMCPFlagSet("reset-project-choices", stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: %s reset-project-choices", mcpCLIName)
	}
	home, workspace, err := mcpCLIHomeWorkspace()
	if err != nil {
		return err
	}
	store := runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home))
	removed, err := store.ResetWorkspace(workspace)
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "%d MCP project approvals and rejections have been reset for %s\n", removed, workspace)
	return nil
}

func runMCPCLIList(args []string, stdout, stderr io.Writer) error {
	fs := newMCPFlagSet("list", stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: %s list", mcpCLIName)
	}

	home, workspace, err := mcpCLIHomeWorkspace()
	if err != nil {
		return err
	}
	resolved, err := resolveMCPConfig(context.Background(), runtimemcp.ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home)),
	})
	if err != nil {
		return err
	}
	if len(resolved.Servers) == 0 && len(resolved.Pending) == 0 && len(resolved.Rejected) == 0 && len(resolved.Disabled) == 0 && len(resolved.Warnings) == 0 {
		fmt.Fprintln(stdout, "No MCP servers configured.")
		return nil
	}

	manager := newMCPManager(runtimemcp.Config{
		WorkDir:        workspace,
		ConnectTimeout: 10 * time.Second,
		CallTimeout:    30 * time.Second,
	})
	defer closeMCPCLIManager(manager)

	for _, server := range sortedServersCopy(resolved.Servers) {
		fmt.Fprintf(stdout, "  %s - %s\n", mcpServerSummary(server), mcpHealthStatus(context.Background(), manager, server))
	}
	for _, server := range sortedServersCopy(resolved.Pending) {
		fmt.Fprintf(stdout, "  %s - pending approval\n", mcpServerSummary(server))
	}
	for _, server := range sortedServersCopy(resolved.Rejected) {
		fmt.Fprintf(stdout, "  %s - rejected\n", mcpServerSummary(server))
	}
	for _, server := range sortedServersCopy(resolved.Disabled) {
		fmt.Fprintf(stdout, "  %s - disabled\n", mcpServerSummary(server))
	}
	printMCPWarnings(stdout, resolved.Warnings)
	return nil
}

func runMCPCLIGet(args []string, stdout, stderr io.Writer) error {
	fs := newMCPFlagSet("get", stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: %s get <name>", mcpCLIName)
	}
	name := fs.Arg(0)
	if err := runtimemcp.ValidateServerName(name); err != nil {
		return err
	}

	home, workspace, err := mcpCLIHomeWorkspace()
	if err != nil {
		return err
	}
	resolved, err := resolveMCPConfig(context.Background(), runtimemcp.ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home)),
	})
	if err != nil {
		return err
	}

	server, status, ok := findResolvedMCPServer(resolved, name)
	if !ok {
		return fmt.Errorf("No MCP server found with name: %s", name)
	}
	path, err := runtimemcp.ConfigPathForScope(home, workspace, server.Scope)
	if err != nil {
		return err
	}
	if status == "active" {
		manager := newMCPManager(runtimemcp.Config{
			WorkDir:        workspace,
			ConnectTimeout: 10 * time.Second,
			CallTimeout:    30 * time.Second,
		})
		status = mcpHealthStatus(context.Background(), manager, server)
		closeMCPCLIManager(manager)
	}

	fmt.Fprintf(stdout, "Name: %s\n", server.Name)
	fmt.Fprintf(stdout, "Scope: %s\n", server.Scope)
	fmt.Fprintf(stdout, "Status: %s\n", status)
	fmt.Fprintf(stdout, "Type: %s\n", server.Config.TransportType())
	if server.Config.TransportType() == "http" {
		fmt.Fprintf(stdout, "URL: %s\n", server.Config.URL)
	} else {
		fmt.Fprintf(stdout, "Command: %s\n", server.Config.Command)
		if len(server.Config.Args) > 0 {
			fmt.Fprintf(stdout, "Args: %s\n", strings.Join(server.Config.Args, " "))
		} else {
			fmt.Fprintln(stdout, "Args: <none>")
		}
	}
	envKeys := sortedEnvKeys(server.Config.Env)
	if len(envKeys) > 0 {
		fmt.Fprintf(stdout, "Env keys: %s\n", strings.Join(envKeys, ", "))
	} else {
		fmt.Fprintln(stdout, "Env keys: <none>")
	}
	if server.Hash != "" {
		fmt.Fprintf(stdout, "Config hash: %s\n", server.Hash)
	}
	fmt.Fprintf(stdout, "Config file: %s\n", path)
	fmt.Fprintf(stdout, "To remove this server, run: %s remove %q -s %s\n", mcpCLIName, server.Name, server.Scope)
	printMCPWarnings(stdout, resolved.Warnings)
	return nil
}

type mcpAddArgs struct {
	scope     runtimemcp.Scope
	transport string
	name      string
	url       string
	env       map[string]string
	command   []string
}

func parseMCPAddArgs(args []string) (mcpAddArgs, error) {
	out := mcpAddArgs{scope: runtimemcp.ScopeLocal, transport: "stdio"}
	var nameParts []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			out.command = append([]string(nil), args[i+1:]...)
			break
		}
		switch arg {
		case "-s", "--scope":
			if i+1 >= len(args) {
				return out, fmt.Errorf("%s add: %s requires a value", mcpCLIName, arg)
			}
			scope, err := runtimemcp.ParseScope(args[i+1])
			if err != nil {
				return out, err
			}
			out.scope = scope
			i++
		case "-t", "--transport":
			if i+1 >= len(args) {
				return out, fmt.Errorf("%s add: %s requires a value", mcpCLIName, arg)
			}
			out.transport = strings.ToLower(strings.TrimSpace(args[i+1]))
			i++
		case "-e", "--env":
			if i+1 >= len(args) {
				return out, fmt.Errorf("%s add: %s requires a value", mcpCLIName, arg)
			}
			key, value, ok := strings.Cut(args[i+1], "=")
			if !ok || strings.TrimSpace(key) == "" {
				return out, fmt.Errorf("invalid MCP env %q: expected KEY=VALUE", args[i+1])
			}
			if out.env == nil {
				out.env = make(map[string]string)
			}
			out.env[key] = value
			i++
		default:
			if strings.HasPrefix(arg, "-") {
				return out, fmt.Errorf("unknown %s add flag %q", mcpCLIName, arg)
			}
			nameParts = append(nameParts, arg)
		}
	}
	if out.transport == "http" {
		if len(nameParts) != 2 {
			return out, fmt.Errorf("usage: %s add [-s local|user|project] -t http <name> <url>", mcpCLIName)
		}
		if len(out.command) > 0 {
			return out, fmt.Errorf("%s add -t http does not accept -- command arguments", mcpCLIName)
		}
		out.name = nameParts[0]
		out.url = nameParts[1]
		return out, runtimemcp.ValidateServerConfig(runtimemcp.ServerConfig{Type: out.transport, URL: out.url})
	}
	if out.transport != "stdio" {
		return out, runtimemcp.ValidateServerConfig(runtimemcp.ServerConfig{Type: out.transport})
	}
	if len(nameParts) != 1 {
		return out, fmt.Errorf("usage: %s add [-s local|user|project] [-t stdio] [-e KEY=VALUE] <name> -- <command> [args...]", mcpCLIName)
	}
	if len(out.command) == 0 {
		return out, fmt.Errorf("%s add requires -- before the server command", mcpCLIName)
	}
	out.name = nameParts[0]
	return out, nil
}

type mcpRemoveArgs struct {
	name     string
	scope    runtimemcp.Scope
	scopeSet bool
}

func parseMCPRemoveArgs(args []string) (mcpRemoveArgs, error) {
	out := mcpRemoveArgs{scope: runtimemcp.ScopeLocal}
	var names []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch arg {
		case "-s", "--scope":
			if i+1 >= len(args) {
				return out, fmt.Errorf("%s remove: %s requires a value", mcpCLIName, arg)
			}
			scope, err := runtimemcp.ParseScope(args[i+1])
			if err != nil {
				return out, err
			}
			out.scope = scope
			out.scopeSet = true
			i++
		default:
			if strings.HasPrefix(arg, "-") {
				return out, fmt.Errorf("unknown %s remove flag %q", mcpCLIName, arg)
			}
			names = append(names, arg)
		}
	}
	if len(names) != 1 {
		return out, fmt.Errorf("usage: %s remove [-s local|user|project] <name>", mcpCLIName)
	}
	out.name = names[0]
	return out, nil
}

func serverConfigFromJSON(input string) (runtimemcp.ServerConfig, error) {
	var raw map[string]any
	dec := json.NewDecoder(bytes.NewBufferString(input))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil {
		return runtimemcp.ServerConfig{}, fmt.Errorf("parse mcp server json: %w", err)
	}
	if len(raw) == 0 {
		return runtimemcp.ServerConfig{}, fmt.Errorf("mcp server json is empty")
	}
	cfg := runtimemcp.ServerConfig{Raw: raw}
	if value, ok, err := stringFromRaw(raw, "type"); err != nil {
		return cfg, err
	} else if ok {
		cfg.Type = value
	}
	if value, ok, err := stringFromRaw(raw, "command"); err != nil {
		return cfg, err
	} else if ok {
		cfg.Command = value
	}
	if value, ok, err := stringFromRaw(raw, "url"); err != nil {
		return cfg, err
	} else if ok {
		cfg.URL = value
	}
	if value, ok, err := stringSliceFromRaw(raw, "args"); err != nil {
		return cfg, err
	} else if ok {
		cfg.Args = value
	}
	if value, ok, err := stringMapFromRaw(raw, "env"); err != nil {
		return cfg, err
	} else if ok {
		cfg.Env = value
	}
	return cfg, nil
}

func stringFromRaw(raw map[string]any, key string) (string, bool, error) {
	value, ok := raw[key]
	if !ok || value == nil {
		return "", false, nil
	}
	str, ok := value.(string)
	if !ok {
		return "", true, fmt.Errorf("mcp server field %q must be a string", key)
	}
	return str, true, nil
}

func stringSliceFromRaw(raw map[string]any, key string) ([]string, bool, error) {
	value, ok := raw[key]
	if !ok || value == nil {
		return nil, false, nil
	}
	items, ok := value.([]any)
	if !ok {
		return nil, true, fmt.Errorf("mcp server field %q must be an array", key)
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		str, ok := item.(string)
		if !ok {
			return nil, true, fmt.Errorf("mcp server field %q must contain strings", key)
		}
		out = append(out, str)
	}
	return out, true, nil
}

func stringMapFromRaw(raw map[string]any, key string) (map[string]string, bool, error) {
	value, ok := raw[key]
	if !ok || value == nil {
		return nil, false, nil
	}
	items, ok := value.(map[string]any)
	if !ok {
		return nil, true, fmt.Errorf("mcp server field %q must be an object", key)
	}
	out := make(map[string]string, len(items))
	for itemKey, itemValue := range items {
		str, ok := itemValue.(string)
		if !ok {
			return nil, true, fmt.Errorf("mcp server env %q must be a string", itemKey)
		}
		out[itemKey] = str
	}
	return out, true, nil
}

func addScopeFlags(fs *flag.FlagSet) (*string, *string) {
	short := fs.String("s", "", "MCP config scope: local, user, or project")
	long := fs.String("scope", "", "MCP config scope: local, user, or project")
	return short, long
}

func parseCLIFlagScope(short, long string) (runtimemcp.Scope, error) {
	if short != "" && long != "" && short != long {
		return "", fmt.Errorf("scope specified twice with different values")
	}
	if long != "" {
		return runtimemcp.ParseScope(long)
	}
	return runtimemcp.ParseScope(short)
}

func newMCPFlagSet(name string, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet(mcpCLIName+" "+name, flag.ContinueOnError)
	if stderr == nil {
		stderr = io.Discard
	}
	fs.SetOutput(stderr)
	return fs
}

func mcpCLIHomeWorkspace() (string, string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", "", fmt.Errorf("resolve home directory: %w", err)
	}
	workspace, err := os.Getwd()
	if err != nil {
		return "", "", fmt.Errorf("resolve workspace: %w", err)
	}
	workspaceAbs, err := filepath.Abs(workspace)
	if err != nil {
		return "", "", fmt.Errorf("resolve workspace path: %w", err)
	}
	return home, workspaceAbs, nil
}

func printMCPAdded(stdout io.Writer, name, transport string, scope runtimemcp.Scope, path string) {
	fmt.Fprintf(stdout, "Added %s MCP server %s to %s config\n", transport, name, scope)
	fmt.Fprintf(stdout, "File modified: %s\n", path)
	fmt.Fprintln(stdout, "Restart mscli to load tool registry changes.")
}

func printMCPCLIUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage: mscli mcp <command> [options]")
	fmt.Fprintln(w, "Commands: add, add-json, list, get, remove, reset-project-choices")
}

func mcpServerSummary(server runtimemcp.ScopedServer) string {
	return fmt.Sprintf("%s [%s] %s: %s", server.Name, server.Scope, server.Config.TransportType(), mcpCommandLine(server.Config))
}

func mcpCommandLine(cfg runtimemcp.ServerConfig) string {
	if cfg.TransportType() == "http" {
		return cfg.URL
	}
	parts := []string{cfg.Command}
	parts = append(parts, cfg.Args...)
	return strings.Join(parts, " ")
}

func mcpHealthStatus(ctx context.Context, manager runtimemcp.Manager, server runtimemcp.ScopedServer) string {
	if err := manager.Connect(ctx, server); err != nil {
		return "failed: " + err.Error()
	}
	defs, err := manager.ListTools(ctx, server.Name)
	if err != nil {
		return "failed: " + err.Error()
	}
	return fmt.Sprintf("connected, %d tools", len(defs))
}

func closeMCPCLIManager(manager runtimemcp.Manager) {
	if manager == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	_ = manager.Close(ctx)
	cancel()
}

func findResolvedMCPServer(resolved runtimemcp.ResolvedConfig, name string) (runtimemcp.ScopedServer, string, bool) {
	if server, ok := findResolvedMCPServerIn(resolved.Servers, name); ok {
		return server, "active", true
	}
	if server, ok := findResolvedMCPServerIn(resolved.Pending, name); ok {
		return server, "pending approval", true
	}
	if server, ok := findResolvedMCPServerIn(resolved.Rejected, name); ok {
		return server, "rejected", true
	}
	if server, ok := findResolvedMCPServerIn(resolved.Disabled, name); ok {
		return server, "disabled", true
	}
	return runtimemcp.ScopedServer{}, "", false
}

func findResolvedMCPServerIn(servers []runtimemcp.ScopedServer, name string) (runtimemcp.ScopedServer, bool) {
	for _, server := range servers {
		if server.Name == name {
			return server, true
		}
	}
	return runtimemcp.ScopedServer{}, false
}

func sortedServersByScope(servers []runtimemcp.ScopedServer) []runtimemcp.ScopedServer {
	out := append([]runtimemcp.ScopedServer(nil), servers...)
	sort.Slice(out, func(i, j int) bool {
		if scopeRank(out[i].Scope) != scopeRank(out[j].Scope) {
			return scopeRank(out[i].Scope) > scopeRank(out[j].Scope)
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func sortedEnvKeys(env map[string]string) []string {
	keys := make([]string, 0, len(env))
	for key := range env {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func printMCPWarnings(stdout io.Writer, warnings []string) {
	for _, warning := range warnings {
		if strings.TrimSpace(warning) == "" {
			continue
		}
		fmt.Fprintf(stdout, "Warning: %s\n", warning)
	}
}
