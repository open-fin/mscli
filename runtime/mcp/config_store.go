package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var validServerNamePattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// ParseScope parses an MCP config scope, defaulting to local.
func ParseScope(value string) (Scope, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "local":
		return ScopeLocal, nil
	case "user":
		return ScopeUser, nil
	case "project":
		return ScopeProject, nil
	default:
		return "", fmt.Errorf("invalid scope %q: must be one of: local, user, project", value)
	}
}

// ConfigPathForScope returns the MCP config path for a scope.
func ConfigPathForScope(home, workspace string, scope Scope) (string, error) {
	switch scope {
	case ScopeUser:
		home = strings.TrimSpace(home)
		if home == "" {
			var err error
			home, err = os.UserHomeDir()
			if err != nil {
				return "", fmt.Errorf("resolve home directory: %w", err)
			}
		}
		return UserConfigPath(home), nil
	case ScopeProject:
		workspaceAbs, err := absWorkspace(workspace)
		if err != nil {
			return "", err
		}
		return ProjectConfigPath(workspaceAbs), nil
	case ScopeLocal:
		workspaceAbs, err := absWorkspace(workspace)
		if err != nil {
			return "", err
		}
		return LocalConfigPath(workspaceAbs), nil
	default:
		return "", fmt.Errorf("invalid mcp scope %q", scope)
	}
}

// ValidateServerName validates a user-visible MCP server name.
func ValidateServerName(name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("invalid MCP server name: name is required")
	}
	if !validServerNamePattern.MatchString(name) {
		return fmt.Errorf("invalid MCP server name %q: names can only contain letters, numbers, hyphens, and underscores", name)
	}
	return nil
}

// ValidateServerConfig validates a supported MCP server config.
func ValidateServerConfig(cfg ServerConfig) error {
	transport := cfg.TransportType()
	switch transport {
	case "stdio":
		if strings.TrimSpace(cfg.Command) == "" {
			return fmt.Errorf("mcp stdio server missing command")
		}
		return nil
	case "http":
		if strings.TrimSpace(cfg.URL) == "" {
			return fmt.Errorf("mcp http server missing url")
		}
		if !strings.HasPrefix(cfg.URL, "http://") && !strings.HasPrefix(cfg.URL, "https://") {
			return fmt.Errorf("mcp http server url must start with http:// or https://")
		}
		return nil
	default:
		return fmt.Errorf("unsupported MCP transport %q: mscli currently supports stdio and http", transport)
	}
}

// ValidateStdioServerConfig validates a supported MCP server shape.
func ValidateStdioServerConfig(cfg ServerConfig) error {
	return ValidateServerConfig(cfg)
}

// AddServerConfig writes one server config into the selected scope.
func AddServerConfig(home, workspace, name string, cfg ServerConfig, scope Scope) (string, error) {
	if err := ValidateServerName(name); err != nil {
		return "", err
	}
	if err := ValidateServerConfig(cfg); err != nil {
		return "", err
	}
	path, err := ConfigPathForScope(home, workspace, scope)
	if err != nil {
		return "", err
	}
	root, err := readConfigStoreFile(path)
	if err != nil {
		return "", err
	}
	if _, ok := root.MCPServers[name]; ok {
		return "", fmt.Errorf("MCP server %s already exists in %s config", name, scope)
	}
	raw, err := json.Marshal(rawFromConfig(cfg))
	if err != nil {
		return "", fmt.Errorf("encode mcp server config: %w", err)
	}
	root.MCPServers[name] = raw
	if err := writeConfigStoreFile(path, root); err != nil {
		return "", err
	}
	return path, nil
}

// RemoveServerConfig removes one server config from the selected scope.
func RemoveServerConfig(home, workspace, name string, scope Scope) (string, error) {
	if err := ValidateServerName(name); err != nil {
		return "", err
	}
	path, err := ConfigPathForScope(home, workspace, scope)
	if err != nil {
		return "", err
	}
	root, err := readConfigStoreFile(path)
	if err != nil {
		return "", err
	}
	if _, ok := root.MCPServers[name]; !ok {
		return "", fmt.Errorf("No MCP server found with name: %s in %s config", name, scope)
	}
	delete(root.MCPServers, name)
	if err := writeConfigStoreFile(path, root); err != nil {
		return "", err
	}
	return path, nil
}

// FindServerScopes returns all scopes that contain the server name.
func FindServerScopes(home, workspace, name string) ([]ScopedServer, error) {
	if err := ValidateServerName(name); err != nil {
		return nil, err
	}
	var out []ScopedServer
	for _, scope := range []Scope{ScopeLocal, ScopeProject, ScopeUser} {
		path, err := ConfigPathForScope(home, workspace, scope)
		if err != nil {
			return nil, err
		}
		servers, _, err := ParseConfigFile(path, scope)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		if server, ok := findScopedServer(servers, name); ok {
			out = append(out, server)
		}
	}
	return out, nil
}

// GetServerConfig returns the effective server config by local > project > user precedence.
func GetServerConfig(home, workspace, name string) (ScopedServer, string, bool, error) {
	if err := ValidateServerName(name); err != nil {
		return ScopedServer{}, "", false, err
	}
	for _, scope := range []Scope{ScopeLocal, ScopeProject, ScopeUser} {
		path, err := ConfigPathForScope(home, workspace, scope)
		if err != nil {
			return ScopedServer{}, "", false, err
		}
		servers, _, err := ParseConfigFile(path, scope)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return ScopedServer{}, "", false, err
		}
		if server, ok := findScopedServer(servers, name); ok {
			return server, path, true, nil
		}
	}
	return ScopedServer{}, "", false, nil
}

type configStoreFile struct {
	Fields     map[string]json.RawMessage
	MCPServers map[string]json.RawMessage
}

func readConfigStoreFile(path string) (configStoreFile, error) {
	file := configStoreFile{
		Fields:     make(map[string]json.RawMessage),
		MCPServers: make(map[string]json.RawMessage),
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return file, nil
		}
		return file, fmt.Errorf("read mcp config: %w", err)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return file, nil
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&file.Fields); err != nil {
		return file, fmt.Errorf("parse mcp config %s: %w", path, err)
	}
	if raw := file.Fields["mcpServers"]; len(raw) > 0 {
		if err := json.Unmarshal(raw, &file.MCPServers); err != nil {
			return file, fmt.Errorf("parse mcpServers %s: %w", path, err)
		}
	}
	if file.MCPServers == nil {
		file.MCPServers = make(map[string]json.RawMessage)
	}
	return file, nil
}

func writeConfigStoreFile(path string, file configStoreFile) error {
	if file.Fields == nil {
		file.Fields = make(map[string]json.RawMessage)
	}
	if file.MCPServers == nil {
		file.MCPServers = make(map[string]json.RawMessage)
	}
	rawServers, err := json.Marshal(file.MCPServers)
	if err != nil {
		return fmt.Errorf("encode mcpServers: %w", err)
	}
	file.Fields["mcpServers"] = rawServers
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create mcp config directory: %w", err)
	}
	out, err := json.MarshalIndent(file.Fields, "", "  ")
	if err != nil {
		return fmt.Errorf("encode mcp config: %w", err)
	}
	if err := os.WriteFile(path, append(out, '\n'), 0644); err != nil {
		return fmt.Errorf("write mcp config: %w", err)
	}
	return nil
}

func findScopedServer(servers []ScopedServer, name string) (ScopedServer, bool) {
	for _, server := range servers {
		if server.Name == name {
			return server, true
		}
	}
	return ScopedServer{}, false
}

func absWorkspace(workspace string) (string, error) {
	workspace = strings.TrimSpace(workspace)
	if workspace == "" {
		var err error
		workspace, err = os.Getwd()
		if err != nil {
			return "", fmt.Errorf("resolve workspace: %w", err)
		}
	}
	workspaceAbs, err := filepath.Abs(workspace)
	if err != nil {
		return "", fmt.Errorf("resolve workspace path: %w", err)
	}
	return workspaceAbs, nil
}
