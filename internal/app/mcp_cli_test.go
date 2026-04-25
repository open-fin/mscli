package app

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
)

func TestMCPCLIRunDispatchesBeforeTUIBootstrap(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)

	oldStdout := os.Stdout
	readStdout, writeStdout, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = writeStdout
	defer func() {
		os.Stdout = oldStdout
	}()

	if err := Run([]string{"mcp", "add-json", "echo", `{"type":"stdio","command":"python","args":["echo.py"]}`}); err != nil {
		t.Fatalf("Run mcp add-json err = %v", err)
	}
	if err := writeStdout.Close(); err != nil {
		t.Fatal(err)
	}
	os.Stdout = oldStdout
	_, _ = io.ReadAll(readStdout)

	server, _, ok, err := runtimemcp.GetServerConfig(home, workDir, "echo")
	if err != nil || !ok {
		t.Fatalf("GetServerConfig ok=%v err=%v", ok, err)
	}
	if server.Scope != runtimemcp.ScopeLocal || server.Config.Command != "python" {
		t.Fatalf("server = %#v", server)
	}
}

func TestMCPCLIAddJSONListGetRemove(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)

	var stdout, stderr bytes.Buffer
	if err := runMCPCLI([]string{"add-json", "echo", `{"type":"stdio","command":"python","args":["echo.py"],"env":{"API_KEY":"secret"}}`}, &stdout, &stderr); err != nil {
		t.Fatalf("add-json err = %v stderr=%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Added stdio MCP server echo") {
		t.Fatalf("add-json stdout = %q", stdout.String())
	}

	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["echo"] = []runtimemcp.ToolDefinition{mcpDef("echo", "tool")}
	restore := stubMCPRuntime(t, runtimemcp.ResolveConfig, func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr })
	defer restore()

	stdout.Reset()
	stderr.Reset()
	if err := runMCPCLI([]string{"list"}, &stdout, &stderr); err != nil {
		t.Fatalf("list err = %v stderr=%s", err, stderr.String())
	}
	for _, want := range []string{"echo [local] stdio: python echo.py - connected, 1 tools"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("list stdout missing %q:\n%s", want, stdout.String())
		}
	}

	stdout.Reset()
	stderr.Reset()
	if err := runMCPCLI([]string{"get", "echo"}, &stdout, &stderr); err != nil {
		t.Fatalf("get err = %v stderr=%s", err, stderr.String())
	}
	for _, want := range []string{"Scope: local", "Command: python", "Args: echo.py", "Env keys: API_KEY", "To remove this server"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("get stdout missing %q:\n%s", want, stdout.String())
		}
	}

	stdout.Reset()
	stderr.Reset()
	if err := runMCPCLI([]string{"remove", "echo"}, &stdout, &stderr); err != nil {
		t.Fatalf("remove err = %v stderr=%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), `Removed MCP server "echo" from local config`) {
		t.Fatalf("remove stdout = %q", stdout.String())
	}
}

func TestMCPCLIAddCommandAndEnv(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)

	var stdout, stderr bytes.Buffer
	err := runMCPCLI([]string{"add", "-s", "user", "-e", "API_KEY=secret", "echo", "--", "python", "echo.py", "--flag"}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("add err = %v stderr=%s", err, stderr.String())
	}
	server, _, ok, err := runtimemcp.GetServerConfig(home, workDir, "echo")
	if err != nil || !ok {
		t.Fatalf("GetServerConfig ok=%v err=%v", ok, err)
	}
	if server.Scope != runtimemcp.ScopeUser || server.Config.Command != "python" || strings.Join(server.Config.Args, " ") != "echo.py --flag" || server.Config.Env["API_KEY"] != "secret" {
		t.Fatalf("server = %#v", server)
	}
}

func TestMCPCLIAddHTTPJSON(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)
	var stdout, stderr bytes.Buffer
	err := runMCPCLI([]string{"add-json", "remote", `{"type":"http","url":"https://example.com/mcp"}`}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("add-json http err = %v stderr=%s", err, stderr.String())
	}
	server, _, ok, err := runtimemcp.GetServerConfig(home, workDir, "remote")
	if err != nil || !ok {
		t.Fatalf("GetServerConfig ok=%v err=%v", ok, err)
	}
	if server.Config.TransportType() != "http" || server.Config.URL != "https://example.com/mcp" {
		t.Fatalf("server = %#v", server)
	}
}

func TestMCPCLIRejectsUnsupportedJSON(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Chdir(t.TempDir())
	var stdout, stderr bytes.Buffer
	err := runMCPCLI([]string{"add-json", "remote", `{"type":"sse","url":"https://example.com/mcp"}`}, &stdout, &stderr)
	if err == nil {
		t.Fatal("add-json sse err = nil, want error")
	}
	if !strings.Contains(err.Error(), `unsupported MCP transport "sse"`) {
		t.Fatalf("err = %v", err)
	}
}

func TestMCPCLIAddHTTPTransport(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)
	var stdout, stderr bytes.Buffer
	err := runMCPCLI([]string{"add", "-s", "user", "-t", "http", "deepwiki", "https://mcp.deepwiki.com/mcp"}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("add http err = %v stderr=%s", err, stderr.String())
	}
	server, _, ok, err := runtimemcp.GetServerConfig(home, workDir, "deepwiki")
	if err != nil || !ok {
		t.Fatalf("GetServerConfig ok=%v err=%v", ok, err)
	}
	if server.Scope != runtimemcp.ScopeUser || server.Config.TransportType() != "http" || server.Config.URL != "https://mcp.deepwiki.com/mcp" {
		t.Fatalf("server = %#v", server)
	}

	stdout.Reset()
	stderr.Reset()
	fakeMgr := newFakeMCPManager()
	restore := stubMCPRuntime(t, runtimemcp.ResolveConfig, func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr })
	defer restore()
	if err := runMCPCLI([]string{"get", "deepwiki"}, &stdout, &stderr); err != nil {
		t.Fatalf("get http err = %v stderr=%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "URL: https://mcp.deepwiki.com/mcp") {
		t.Fatalf("get stdout = %q, want URL", stdout.String())
	}
}

func TestMCPCLIRemoveMultipleScopesRequiresScope(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)
	cfg := runtimemcp.ServerConfig{Type: "stdio", Command: "server"}
	if _, err := runtimemcp.AddServerConfig(home, workDir, "echo", cfg, runtimemcp.ScopeUser); err != nil {
		t.Fatal(err)
	}
	if _, err := runtimemcp.AddServerConfig(home, workDir, "echo", cfg, runtimemcp.ScopeLocal); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	err := runMCPCLI([]string{"remove", "echo"}, &stdout, &stderr)
	if err == nil {
		t.Fatal("remove multiple scopes err = nil, want error")
	}
	if !strings.Contains(stderr.String(), `exists in multiple scopes`) || !strings.Contains(stderr.String(), `-s local`) || !strings.Contains(stderr.String(), `-s user`) {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestMCPCLIResetProjectChoices(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)
	store := runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home))
	if err := store.Record(workDir, "echo", "sha256:1", runtimemcp.DecisionApproved); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if err := runMCPCLI([]string{"reset-project-choices"}, &stdout, &stderr); err != nil {
		t.Fatalf("reset err = %v stderr=%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "approvals and rejections have been reset") {
		t.Fatalf("stdout = %q", stdout.String())
	}
	if _, ok, err := store.Lookup(workDir, "echo", "sha256:1"); err != nil || ok {
		t.Fatalf("lookup after reset ok=%v err=%v, want false nil", ok, err)
	}
}

func TestMCPCLIListDoesNotConnectInactiveServers(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)
	pending := mcpServer("pending", runtimemcp.ScopeProject)
	rejected := mcpServer("rejected", runtimemcp.ScopeProject)
	disabled := mcpServer("disabled", runtimemcp.ScopeLocal)
	fakeMgr := newFakeMCPManager()
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{
				Servers:  []runtimemcp.ScopedServer{mcpServer("active", runtimemcp.ScopeUser)},
				Pending:  []runtimemcp.ScopedServer{pending},
				Rejected: []runtimemcp.ScopedServer{rejected},
				Disabled: []runtimemcp.ScopedServer{disabled},
			}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	var stdout, stderr bytes.Buffer
	if err := runMCPCLI([]string{"list"}, &stdout, &stderr); err != nil {
		t.Fatalf("list err = %v stderr=%s", err, stderr.String())
	}
	if got := strings.Join(fakeMgr.connected, ","); got != "active" {
		t.Fatalf("connected = %q, want active only", got)
	}
	for _, want := range []string{"pending approval", "rejected", "disabled"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("stdout missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestMCPCLIRealStdioEchoSmoke(t *testing.T) {
	home := t.TempDir()
	workDir := t.TempDir()
	t.Setenv("HOME", home)
	t.Chdir(workDir)
	serverPath := filepath.Join(t.TempDir(), "echo_mcp_server.py")
	if err := os.WriteFile(serverPath, []byte(echoMCPServerPython), 0755); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if err := runMCPCLI([]string{"add-json", "echo", mustJSON(t, map[string]any{
		"type":    "stdio",
		"command": serverPath,
	})}, &stdout, &stderr); err != nil {
		t.Fatalf("add-json err = %v stderr=%s", err, stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if err := runMCPCLI([]string{"list"}, &stdout, &stderr); err != nil {
		t.Fatalf("list err = %v stderr=%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "echo [local] stdio: "+serverPath+" - connected, 1 tools") {
		t.Fatalf("list stdout = %q", stdout.String())
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}
