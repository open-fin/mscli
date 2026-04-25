package mcp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveConfigMergesUserApprovedProjectAndLocal(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	store := NewApprovalStore(filepath.Join(home, ".mscli", "mcp_approvals.json"))

	userServers := map[string]any{
		"shared":   stdioRaw("user-server"),
		"userOnly": stdioRaw("user-only"),
	}
	projectServers := map[string]any{
		"shared":      stdioRaw("project-server"),
		"projectOnly": stdioRaw("project-only"),
	}
	localServers := map[string]any{
		"shared":    stdioRaw("local-server"),
		"localOnly": stdioRaw("local-only"),
	}
	writeMCPConfig(t, UserConfigPath(home), userServers)
	writeMCPConfig(t, ProjectConfigPath(workspace), projectServers)
	writeMCPConfig(t, LocalConfigPath(workspace), localServers)

	for name, raw := range projectServers {
		hash := CanonicalConfigHash(ServerConfig{Raw: raw.(map[string]any)})
		if err := store.Record(workspace, name, hash, DecisionApproved); err != nil {
			t.Fatalf("Record approval: %v", err)
		}
	}

	resolved, err := ResolveConfig(context.Background(), ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: store,
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	got := serverCommands(resolved.Servers)
	want := map[string]string{
		"localOnly":   "local-only",
		"projectOnly": "project-only",
		"shared":      "local-server",
		"userOnly":    "user-only",
	}
	if !equalStringMap(got, want) {
		t.Fatalf("commands = %#v, want %#v", got, want)
	}
	if len(resolved.Pending) != 0 || len(resolved.Rejected) != 0 {
		t.Fatalf("pending/rejected = %d/%d, want 0/0", len(resolved.Pending), len(resolved.Rejected))
	}
}

func TestResolveConfigLeavesUnapprovedProjectPending(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	store := NewApprovalStore(filepath.Join(home, ".mscli", "mcp_approvals.json"))
	writeMCPConfig(t, ProjectConfigPath(workspace), map[string]any{
		"projectOnly": stdioRaw("project-only"),
	})

	resolved, err := ResolveConfig(context.Background(), ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: store,
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	if len(resolved.Servers) != 0 {
		t.Fatalf("Servers len = %d, want 0", len(resolved.Servers))
	}
	if got, want := len(resolved.Pending), 1; got != want {
		t.Fatalf("Pending len = %d, want %d", got, want)
	}
	if resolved.Pending[0].Name != "projectOnly" {
		t.Fatalf("Pending[0].Name = %q", resolved.Pending[0].Name)
	}
}

func TestResolveConfigSuppressesRejectedProject(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	store := NewApprovalStore(filepath.Join(home, ".mscli", "mcp_approvals.json"))
	raw := stdioRaw("project-only")
	writeMCPConfig(t, ProjectConfigPath(workspace), map[string]any{"projectOnly": raw})
	hash := CanonicalConfigHash(ServerConfig{Raw: raw})
	if err := store.Record(workspace, "projectOnly", hash, DecisionRejected); err != nil {
		t.Fatalf("Record rejection: %v", err)
	}

	resolved, err := ResolveConfig(context.Background(), ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: store,
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	if len(resolved.Servers) != 0 || len(resolved.Pending) != 0 {
		t.Fatalf("servers/pending = %d/%d, want 0/0", len(resolved.Servers), len(resolved.Pending))
	}
	if got, want := len(resolved.Rejected), 1; got != want {
		t.Fatalf("Rejected len = %d, want %d", got, want)
	}
}

func TestResolveConfigAcceptsHTTPAndSkipsUnsupportedTransports(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	writeMCPConfig(t, UserConfigPath(home), map[string]any{
		"remote": map[string]any{"type": "sse", "url": "https://example.com/mcp"},
		"http":   map[string]any{"type": "http", "url": "https://example.com/mcp"},
		"stdio":  stdioRaw("ok"),
	})

	resolved, err := ResolveConfig(context.Background(), ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: NewApprovalStore(filepath.Join(home, ".mscli", "mcp_approvals.json")),
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	got := serverTransports(resolved.Servers)
	want := map[string]string{"http": "http", "stdio": "stdio"}
	if !equalStringMap(got, want) {
		t.Fatalf("servers = %#v, want %#v", got, want)
	}
	if !warningsContain(resolved.Warnings, "unsupported MCP transport") {
		t.Fatalf("warnings = %#v, want unsupported type warning", resolved.Warnings)
	}
}

func TestResolveConfigInvalidJSONReturnsWarningNotFatalForSingleFile(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	writeMCPConfig(t, UserConfigPath(home), map[string]any{"userOnly": stdioRaw("user-only")})
	if err := os.MkdirAll(filepath.Dir(ProjectConfigPath(workspace)), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ProjectConfigPath(workspace), []byte(`{"mcpServers":`), 0644); err != nil {
		t.Fatal(err)
	}
	writeMCPConfig(t, LocalConfigPath(workspace), map[string]any{"localOnly": stdioRaw("local-only")})

	resolved, err := ResolveConfig(context.Background(), ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: NewApprovalStore(filepath.Join(home, ".mscli", "mcp_approvals.json")),
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	got := serverCommands(resolved.Servers)
	want := map[string]string{"localOnly": "local-only", "userOnly": "user-only"}
	if !equalStringMap(got, want) {
		t.Fatalf("commands = %#v, want %#v", got, want)
	}
	if !warningsContain(resolved.Warnings, "parse mcp config") {
		t.Fatalf("warnings = %#v, want parse warning", resolved.Warnings)
	}
}

func TestResolveConfigMovesLocalDisabledServersOutOfActiveServers(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	writeMCPConfig(t, UserConfigPath(home), map[string]any{
		"echo": stdioRaw("echo-user"),
	})
	writeMCPConfigWithDisabled(t, LocalConfigPath(workspace), map[string]any{
		"echo": stdioRaw("echo-local"),
		"ok":   stdioRaw("ok-local"),
	}, []string{"echo"})

	resolved, err := ResolveConfig(context.Background(), ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: NewApprovalStore(filepath.Join(home, ".mscli", "mcp_approvals.json")),
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	if got, want := serverCommands(resolved.Servers), map[string]string{"ok": "ok-local"}; !equalStringMap(got, want) {
		t.Fatalf("servers = %#v, want %#v", got, want)
	}
	if got, want := serverCommands(resolved.Disabled), map[string]string{"echo": "echo-local"}; !equalStringMap(got, want) {
		t.Fatalf("disabled = %#v, want %#v", got, want)
	}
}

func TestResolveConfigIgnoresUserAndProjectDisabledLists(t *testing.T) {
	home := t.TempDir()
	workspace := t.TempDir()
	raw := stdioRaw("project-only")
	writeMCPConfigWithDisabled(t, UserConfigPath(home), map[string]any{
		"userOnly": stdioRaw("user-only"),
	}, []string{"userOnly"})
	writeMCPConfigWithDisabled(t, ProjectConfigPath(workspace), map[string]any{
		"projectOnly": raw,
	}, []string{"projectOnly"})

	store := NewApprovalStore(filepath.Join(home, ".mscli", "mcp_approvals.json"))
	if err := store.Record(workspace, "projectOnly", CanonicalConfigHash(ServerConfig{Raw: raw}), DecisionApproved); err != nil {
		t.Fatalf("Record approval: %v", err)
	}

	resolved, err := ResolveConfig(context.Background(), ResolveOptions{
		HomeDir:       home,
		WorkspaceRoot: workspace,
		ApprovalStore: store,
	})
	if err != nil {
		t.Fatalf("ResolveConfig() err = %v", err)
	}
	if len(resolved.Disabled) != 0 {
		t.Fatalf("Disabled len = %d, want 0", len(resolved.Disabled))
	}
	if got, want := serverCommands(resolved.Servers), map[string]string{
		"projectOnly": "project-only",
		"userOnly":    "user-only",
	}; !equalStringMap(got, want) {
		t.Fatalf("servers = %#v, want %#v", got, want)
	}
}

func TestSetLocalServerDisabledWritesSortedUniqueNamesAndPreservesServers(t *testing.T) {
	workspace := t.TempDir()
	writeMCPConfigWithDisabled(t, LocalConfigPath(workspace), map[string]any{
		"echo": stdioRaw("echo"),
	}, []string{"zeta", "echo"})

	path, err := SetLocalServerDisabled(workspace, "alpha", true)
	if err != nil {
		t.Fatalf("SetLocalServerDisabled() err = %v", err)
	}
	if path != LocalConfigPath(workspace) {
		t.Fatalf("path = %q, want %q", path, LocalConfigPath(workspace))
	}

	disabled, err := ReadLocalDisabledServers(workspace)
	if err != nil {
		t.Fatalf("ReadLocalDisabledServers() err = %v", err)
	}
	if got, want := strings.Join(disabled, ","), "alpha,echo,zeta"; got != want {
		t.Fatalf("disabled = %q, want %q", got, want)
	}
	servers, _, err := ParseConfigFile(LocalConfigPath(workspace), ScopeLocal)
	if err != nil {
		t.Fatalf("ParseConfigFile() err = %v", err)
	}
	if len(servers) != 1 || servers[0].Name != "echo" {
		t.Fatalf("servers = %#v, want echo preserved", servers)
	}
}

func TestSetLocalServerDisabledCanEnableByRemovingName(t *testing.T) {
	workspace := t.TempDir()
	writeMCPConfigWithDisabled(t, LocalConfigPath(workspace), nil, []string{"alpha", "echo"})

	if _, err := SetLocalServerDisabled(workspace, "echo", false); err != nil {
		t.Fatalf("SetLocalServerDisabled() err = %v", err)
	}
	disabled, err := ReadLocalDisabledServers(workspace)
	if err != nil {
		t.Fatalf("ReadLocalDisabledServers() err = %v", err)
	}
	if got, want := strings.Join(disabled, ","), "alpha"; got != want {
		t.Fatalf("disabled = %q, want %q", got, want)
	}
}

func stdioRaw(command string) map[string]any {
	return map[string]any{
		"type":    "stdio",
		"command": command,
		"args":    []any{"--flag"},
		"env":     map[string]any{"A": "1"},
	}
}

func writeMCPConfigWithDisabled(t *testing.T, path string, servers map[string]any, disabled []string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(map[string]any{
		"mcpServers":         servers,
		"disabledMcpServers": disabled,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func writeMCPConfig(t *testing.T, path string, servers map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(map[string]any{"mcpServers": servers})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func serverCommands(servers []ScopedServer) map[string]string {
	out := make(map[string]string, len(servers))
	for _, server := range servers {
		out[server.Name] = server.Config.Command
	}
	return out
}

func serverTransports(servers []ScopedServer) map[string]string {
	out := make(map[string]string, len(servers))
	for _, server := range servers {
		out[server.Name] = server.Config.TransportType()
	}
	return out
}

func equalStringMap(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		if b[key] != value {
			return false
		}
	}
	return true
}

func warningsContain(warnings []string, needle string) bool {
	for _, warning := range warnings {
		if strings.Contains(warning, needle) {
			return true
		}
	}
	return false
}
