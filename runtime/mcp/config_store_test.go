package mcp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseScopeDefaultsAndValidates(t *testing.T) {
	for input, want := range map[string]Scope{
		"":        ScopeLocal,
		"local":   ScopeLocal,
		"user":    ScopeUser,
		"project": ScopeProject,
	} {
		got, err := ParseScope(input)
		if err != nil {
			t.Fatalf("ParseScope(%q) err = %v", input, err)
		}
		if got != want {
			t.Fatalf("ParseScope(%q) = %q, want %q", input, got, want)
		}
	}
	if _, err := ParseScope("remote"); err == nil || !strings.Contains(err.Error(), "local, user, project") {
		t.Fatalf("ParseScope(remote) err = %v, want allowed scopes", err)
	}
}

func TestValidateServerName(t *testing.T) {
	for _, name := range []string{"echo", "my-server", "my_server", "S1"} {
		if err := ValidateServerName(name); err != nil {
			t.Fatalf("ValidateServerName(%q) err = %v", name, err)
		}
	}
	for _, name := range []string{"", "bad name", "bad.name", "../x"} {
		if err := ValidateServerName(name); err == nil {
			t.Fatalf("ValidateServerName(%q) err = nil, want error", name)
		}
	}
}

func TestAddGetRemoveServerConfigByScope(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	cfg := ServerConfig{
		Type:    "stdio",
		Command: "python",
		Args:    []string{"server.py"},
		Env:     map[string]string{"API_KEY": "secret"},
	}
	path, err := AddServerConfig(home, workspace, "echo", cfg, ScopeLocal)
	if err != nil {
		t.Fatalf("AddServerConfig() err = %v", err)
	}
	if path != LocalConfigPath(workspace) {
		t.Fatalf("path = %q, want %q", path, LocalConfigPath(workspace))
	}

	got, gotPath, ok, err := GetServerConfig(home, workspace, "echo")
	if err != nil {
		t.Fatalf("GetServerConfig() err = %v", err)
	}
	if !ok {
		t.Fatal("GetServerConfig ok = false, want true")
	}
	if gotPath != path || got.Name != "echo" || got.Scope != ScopeLocal || got.Config.Command != "python" {
		t.Fatalf("server/path = %#v %q", got, gotPath)
	}

	scopes, err := FindServerScopes(home, workspace, "echo")
	if err != nil {
		t.Fatalf("FindServerScopes() err = %v", err)
	}
	if len(scopes) != 1 || scopes[0].Scope != ScopeLocal {
		t.Fatalf("scopes = %#v, want local", scopes)
	}

	if _, err := AddServerConfig(home, workspace, "echo", cfg, ScopeLocal); err == nil {
		t.Fatal("duplicate AddServerConfig err = nil, want error")
	}
	if _, err := RemoveServerConfig(home, workspace, "echo", ScopeLocal); err != nil {
		t.Fatalf("RemoveServerConfig() err = %v", err)
	}
	if _, _, ok, err := GetServerConfig(home, workspace, "echo"); err != nil || ok {
		t.Fatalf("Get after remove ok=%v err=%v, want false nil", ok, err)
	}
}

func TestAddServerConfigAcceptsHTTP(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	path, err := AddServerConfig(home, workspace, "remote", ServerConfig{Type: "http", URL: "https://example.com/mcp"}, ScopeUser)
	if err != nil {
		t.Fatalf("AddServerConfig http err = %v", err)
	}
	if path != UserConfigPath(home) {
		t.Fatalf("path = %q, want %q", path, UserConfigPath(home))
	}
	server, _, ok, err := GetServerConfig(home, workspace, "remote")
	if err != nil || !ok {
		t.Fatalf("GetServerConfig ok=%v err=%v", ok, err)
	}
	if server.Config.TransportType() != "http" || server.Config.URL != "https://example.com/mcp" {
		t.Fatalf("server config = %#v", server.Config)
	}
}

func TestAddServerConfigRejectsUnsupportedTransport(t *testing.T) {
	_, err := AddServerConfig(t.TempDir(), t.TempDir(), "remote", ServerConfig{Type: "sse", URL: "https://example.com/mcp"}, ScopeLocal)
	if err == nil || !strings.Contains(err.Error(), `unsupported MCP transport "sse"`) {
		t.Fatalf("AddServerConfig unsupported err = %v", err)
	}
}

func TestConfigStorePreservesDisabledMCPServers(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	writeMCPConfigWithDisabled(t, LocalConfigPath(workspace), map[string]any{
		"old": stdioRaw("old"),
	}, []string{"old"})

	if _, err := AddServerConfig(home, workspace, "new", ServerConfig{Type: "stdio", Command: "new"}, ScopeLocal); err != nil {
		t.Fatalf("AddServerConfig() err = %v", err)
	}

	data, err := os.ReadFile(LocalConfigPath(workspace))
	if err != nil {
		t.Fatal(err)
	}
	var root struct {
		Disabled []string `json:"disabledMcpServers"`
	}
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(root.Disabled, ","); got != "old" {
		t.Fatalf("disabled = %q, want old", got)
	}
	if _, err := RemoveServerConfig(home, workspace, "new", ScopeLocal); err != nil {
		t.Fatalf("RemoveServerConfig() err = %v", err)
	}
	if _, err := os.Stat(filepath.Dir(LocalConfigPath(workspace))); err != nil {
		t.Fatalf("config dir missing: %v", err)
	}
}
