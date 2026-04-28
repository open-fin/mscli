package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ResolveOptions controls MCP config discovery and approval filtering.
type ResolveOptions struct {
	HomeDir       string
	WorkspaceRoot string
	ApprovalStore *ApprovalStore
}

// ResolvedConfig is the result of loading, filtering, and merging MCP configs.
type ResolvedConfig struct {
	Servers  []ScopedServer
	Pending  []ScopedServer
	Rejected []ScopedServer
	Disabled []ScopedServer
	Warnings []string
}

type mcpConfigFile struct {
	MCPServers         map[string]json.RawMessage `json:"mcpServers"`
	DisabledMCPServers []string                   `json:"disabledMcpServers"`
}

// UserConfigPath returns the user-scope MCP config path.
func UserConfigPath(home string) string {
	return filepath.Join(home, ".mscli", "mcp.json")
}

// ProjectConfigPath returns the project-scope MCP config path.
func ProjectConfigPath(workspace string) string {
	return filepath.Join(workspace, ".mscli", "mcp.json")
}

// LocalConfigPath returns the machine-local project MCP config path.
func LocalConfigPath(workspace string) string {
	return filepath.Join(workspace, ".mscli", "mcp.local.json")
}

// ResolveConfig loads MCP configs, applies project approvals, and merges servers.
func ResolveConfig(ctx context.Context, opts ResolveOptions) (ResolvedConfig, error) {
	if err := ctx.Err(); err != nil {
		return ResolvedConfig{}, err
	}

	home := strings.TrimSpace(opts.HomeDir)
	if home == "" {
		var err error
		home, err = os.UserHomeDir()
		if err != nil {
			return ResolvedConfig{}, fmt.Errorf("resolve home directory: %w", err)
		}
	}
	workspace := strings.TrimSpace(opts.WorkspaceRoot)
	if workspace == "" {
		var err error
		workspace, err = os.Getwd()
		if err != nil {
			return ResolvedConfig{}, fmt.Errorf("resolve workspace: %w", err)
		}
	}
	workspaceAbs, err := filepath.Abs(workspace)
	if err != nil {
		return ResolvedConfig{}, fmt.Errorf("resolve workspace path: %w", err)
	}

	var resolved ResolvedConfig
	userServers := readMCPConfigSource(&resolved, UserConfigPath(home), ScopeUser)
	projectServers := readMCPConfigSource(&resolved, ProjectConfigPath(workspaceAbs), ScopeProject)
	localServers := readMCPConfigSource(&resolved, LocalConfigPath(workspaceAbs), ScopeLocal)
	disabledNames, disabledErr := readLocalDisabledServersFile(LocalConfigPath(workspaceAbs))
	disabledSet := stringSet(disabledNames)

	approvedProject := make([]ScopedServer, 0, len(projectServers))
	for _, server := range projectServers {
		server.Hash = CanonicalConfigHash(server.Config)
		if opts.ApprovalStore == nil {
			resolved.Pending = append(resolved.Pending, server)
			continue
		}
		decision, ok, err := opts.ApprovalStore.Lookup(workspaceAbs, server.Name, server.Hash)
		if err != nil {
			resolved.Warnings = append(resolved.Warnings, fmt.Sprintf("lookup mcp approval for %s: %v", server.Name, err))
			resolved.Pending = append(resolved.Pending, server)
			continue
		}
		if !ok {
			resolved.Pending = append(resolved.Pending, server)
			continue
		}
		switch decision {
		case DecisionApproved:
			approvedProject = append(approvedProject, server)
		case DecisionRejected:
			resolved.Rejected = append(resolved.Rejected, server)
		default:
			resolved.Warnings = append(resolved.Warnings, fmt.Sprintf("unsupported mcp approval decision for %s: %s", server.Name, decision))
			resolved.Pending = append(resolved.Pending, server)
		}
	}

	merged := make(map[string]ScopedServer)
	for _, server := range userServers {
		merged[server.Name] = server
	}
	for _, server := range approvedProject {
		merged[server.Name] = server
	}
	for _, server := range localServers {
		merged[server.Name] = server
	}
	resolved.Servers = sortedServerValues(merged)
	if disabledErr == nil && len(disabledNames) > 0 {
		resolved.Servers, resolved.Disabled = splitDisabledServers(resolved.Servers, stringSet(disabledNames))
		resolved.Pending, resolved.Disabled = splitDisabledServersInto(resolved.Pending, disabledSet, resolved.Disabled)
		resolved.Rejected, resolved.Disabled = splitDisabledServersInto(resolved.Rejected, disabledSet, resolved.Disabled)
	} else if disabledErr != nil && !os.IsNotExist(disabledErr) {
		// The local config source has already emitted parse warnings. Avoid
		// duplicating that warning while keeping startup non-fatal.
	}
	sortServers(resolved.Pending)
	sortServers(resolved.Rejected)
	sortServers(resolved.Disabled)
	return resolved, nil
}

// ParseConfigFile parses a single MCP config file for one scope.
func ParseConfigFile(path string, scope Scope) ([]ScopedServer, []string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var root mcpConfigFile
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&root); err != nil {
		return nil, nil, fmt.Errorf("parse mcp config %s: %w", path, err)
	}
	if len(root.MCPServers) == 0 {
		return nil, nil, nil
	}

	names := make([]string, 0, len(root.MCPServers))
	for name := range root.MCPServers {
		names = append(names, name)
	}
	sort.Strings(names)

	var servers []ScopedServer
	var warnings []string
	for _, name := range names {
		cfg, cfgWarnings, ok := decodeServerConfig(root.MCPServers[name])
		for _, warning := range cfgWarnings {
			warnings = append(warnings, fmt.Sprintf("%s %s: %s", scope, name, warning))
		}
		if !ok {
			continue
		}
		if err := ValidateServerConfig(cfg); err != nil {
			warnings = append(warnings, fmt.Sprintf("%s %s: %v", scope, name, err))
			continue
		}
		servers = append(servers, ScopedServer{
			Name:   name,
			Scope:  scope,
			Config: cfg,
			Hash:   CanonicalConfigHash(cfg),
		})
	}
	return servers, warnings, nil
}

// ReadLocalDisabledServers returns the project-local disabled MCP server names.
func ReadLocalDisabledServers(workspace string) ([]string, error) {
	workspaceAbs, err := filepath.Abs(strings.TrimSpace(workspace))
	if err != nil {
		return nil, fmt.Errorf("resolve workspace path: %w", err)
	}
	return readLocalDisabledServersFile(LocalConfigPath(workspaceAbs))
}

// SetLocalServerDisabled updates the project-local disabled state for one MCP server.
func SetLocalServerDisabled(workspace, serverName string, disabled bool) (string, error) {
	name := strings.TrimSpace(serverName)
	if name == "" {
		return "", fmt.Errorf("mcp server name is empty")
	}
	workspaceAbs, err := filepath.Abs(strings.TrimSpace(workspace))
	if err != nil {
		return "", fmt.Errorf("resolve workspace path: %w", err)
	}
	path := LocalConfigPath(workspaceAbs)

	root := make(map[string]json.RawMessage)
	data, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) {
			return "", fmt.Errorf("read local mcp config: %w", err)
		}
	} else if len(data) > 0 {
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.UseNumber()
		if err := dec.Decode(&root); err != nil {
			return "", fmt.Errorf("parse mcp config %s: %w", path, err)
		}
	}
	if root == nil {
		root = make(map[string]json.RawMessage)
	}
	if _, ok := root["mcpServers"]; !ok {
		root["mcpServers"] = json.RawMessage(`{}`)
	}

	disabledNames := normalizeDisabledNames(nil)
	if raw := root["disabledMcpServers"]; len(raw) > 0 {
		var parsed []string
		if err := json.Unmarshal(raw, &parsed); err != nil {
			return "", fmt.Errorf("parse disabled mcp servers %s: %w", path, err)
		}
		disabledNames = normalizeDisabledNames(parsed)
	}
	disabledSet := stringSet(disabledNames)
	if disabled {
		disabledSet[name] = struct{}{}
	} else {
		delete(disabledSet, name)
	}
	disabledNames = setValues(disabledSet)
	rawDisabled, err := json.Marshal(disabledNames)
	if err != nil {
		return "", fmt.Errorf("encode disabled mcp servers: %w", err)
	}
	root["disabledMcpServers"] = rawDisabled

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", fmt.Errorf("create local mcp config directory: %w", err)
	}
	out, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode local mcp config: %w", err)
	}
	if err := os.WriteFile(path, append(out, '\n'), 0644); err != nil {
		return "", fmt.Errorf("write local mcp config: %w", err)
	}
	return path, nil
}

func readLocalDisabledServersFile(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var root mcpConfigFile
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&root); err != nil {
		return nil, fmt.Errorf("parse mcp config %s: %w", path, err)
	}
	return normalizeDisabledNames(root.DisabledMCPServers), nil
}

func splitDisabledServers(servers []ScopedServer, disabled map[string]struct{}) ([]ScopedServer, []ScopedServer) {
	return splitDisabledServersInto(servers, disabled, nil)
}

func splitDisabledServersInto(servers []ScopedServer, disabled map[string]struct{}, disabledServers []ScopedServer) ([]ScopedServer, []ScopedServer) {
	active := make([]ScopedServer, 0, len(servers))
	disabledNames := make(map[string]struct{}, len(disabledServers))
	for _, server := range disabledServers {
		disabledNames[server.Name] = struct{}{}
	}
	for _, server := range servers {
		if _, ok := disabled[server.Name]; ok {
			if _, exists := disabledNames[server.Name]; !exists {
				disabledServers = append(disabledServers, server)
				disabledNames[server.Name] = struct{}{}
			}
			continue
		}
		active = append(active, server)
	}
	return active, disabledServers
}

func normalizeDisabledNames(names []string) []string {
	set := make(map[string]struct{}, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		set[name] = struct{}{}
	}
	return setValues(set)
}

func stringSet(values []string) map[string]struct{} {
	set := make(map[string]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return set
}

func setValues(set map[string]struct{}) []string {
	values := make([]string, 0, len(set))
	for value := range set {
		values = append(values, value)
	}
	sort.Strings(values)
	return values
}

func readMCPConfigSource(resolved *ResolvedConfig, path string, scope Scope) []ScopedServer {
	servers, warnings, err := ParseConfigFile(path, scope)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		resolved.Warnings = append(resolved.Warnings, err.Error())
		return nil
	}
	resolved.Warnings = append(resolved.Warnings, warnings...)
	return servers
}

func decodeServerConfig(raw json.RawMessage) (ServerConfig, []string, bool) {
	var decoded map[string]any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&decoded); err != nil {
		return ServerConfig{}, []string{fmt.Sprintf("parse server config: %v", err)}, false
	}
	if decoded == nil {
		return ServerConfig{}, []string{"server config must be an object"}, false
	}

	var warnings []string
	cfg := ServerConfig{Raw: decoded}
	if value, ok := decoded["type"]; ok {
		cfg.Type, ok = stringValue(value)
		if !ok {
			warnings = append(warnings, "type must be a string")
			return ServerConfig{}, warnings, false
		}
	}
	if value, ok := decoded["command"]; ok {
		cfg.Command, ok = stringValue(value)
		if !ok {
			warnings = append(warnings, "command must be a string")
			return ServerConfig{}, warnings, false
		}
	}
	if value, ok := decoded["args"]; ok {
		args, ok := stringSliceValue(value)
		if !ok {
			warnings = append(warnings, "args must be an array of strings")
			return ServerConfig{}, warnings, false
		}
		cfg.Args = args
	}
	if value, ok := decoded["env"]; ok {
		env, ok := stringMapValue(value)
		if !ok {
			warnings = append(warnings, "env must be an object of strings")
			return ServerConfig{}, warnings, false
		}
		cfg.Env = env
	}
	if value, ok := decoded["url"]; ok {
		cfg.URL, ok = stringValue(value)
		if !ok {
			warnings = append(warnings, "url must be a string")
			return ServerConfig{}, warnings, false
		}
	}
	return cfg, warnings, true
}

func stringValue(v any) (string, bool) {
	s, ok := v.(string)
	return s, ok
}

func stringSliceValue(v any) ([]string, bool) {
	items, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		s, ok := item.(string)
		if !ok {
			return nil, false
		}
		out = append(out, s)
	}
	return out, true
}

func stringMapValue(v any) (map[string]string, bool) {
	items, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	out := make(map[string]string, len(items))
	for key, item := range items {
		s, ok := item.(string)
		if !ok {
			return nil, false
		}
		out[key] = s
	}
	return out, true
}

func sortedServerValues(servers map[string]ScopedServer) []ScopedServer {
	names := make([]string, 0, len(servers))
	for name := range servers {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]ScopedServer, 0, len(names))
	for _, name := range names {
		out = append(out, servers[name])
	}
	return out
}

func sortServers(servers []ScopedServer) {
	sort.Slice(servers, func(i, j int) bool {
		return servers[i].Name < servers[j].Name
	})
}
