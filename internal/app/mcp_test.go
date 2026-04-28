package app

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"gitcode.com/mindspore/mscli/agent/session"
	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	"gitcode.com/mindspore/mscli/tools"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestInitMCPRegistersListedTools(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	workDir := t.TempDir()
	registry := tools.NewRegistry()
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["server"] = []runtimemcp.ToolDefinition{mcpDef("server", "echo")}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{mcpServer("server", runtimemcp.ScopeUser)}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	manager, _, err := initMCPTools(context.Background(), registry, workDir, time.Second, nil)
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	if manager == nil {
		t.Fatal("manager = nil")
	}
	if _, ok := registry.Get("mcp__server__echo"); !ok {
		t.Fatal("registry missing mcp__server__echo")
	}
}

func TestWireUsesBoundedMCPDiscoveryTimeout(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Chdir(t.TempDir())
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["server"] = []runtimemcp.ToolDefinition{mcpDef("server", "echo")}
	var gotTimeout time.Duration
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{mcpServer("server", runtimemcp.ScopeUser)}}, nil
		},
		func(cfg runtimemcp.Config) runtimemcp.Manager {
			gotTimeout = cfg.CallTimeout
			return fakeMgr
		},
	)
	defer restore()

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() err = %v", err)
	}
	app.closeRuntimeResources()
	if gotTimeout <= 0 || gotTimeout > 30*time.Second {
		t.Fatalf("MCP discovery CallTimeout = %v, want bounded startup timeout <= 30s", gotTimeout)
	}
}

func TestWireReplaySkipsLiveMCPStartup(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_BASE_URL", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	recordedWorkDir := t.TempDir()
	runtimeSession, err := session.Create(recordedWorkDir, "system prompt")
	if err != nil {
		t.Fatalf("session.Create() err = %v", err)
	}
	if err := runtimeSession.Activate(); err != nil {
		t.Fatalf("runtimeSession.Activate() err = %v", err)
	}
	if err := runtimeSession.Close(); err != nil {
		t.Fatalf("runtimeSession.Close() err = %v", err)
	}

	t.Chdir(t.TempDir())
	resolved := false
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			resolved = true
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{mcpServer("live", runtimemcp.ScopeLocal)}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager {
			t.Fatal("newMCPManager called during replay")
			return newFakeMCPManager()
		},
	)
	defer restore()

	app, err := Wire(BootstrapConfig{Replay: true, ReplaySessionID: runtimeSession.Path()})
	if err != nil {
		t.Fatalf("Wire() err = %v", err)
	}
	app.closeRuntimeResources()
	if resolved {
		t.Fatal("MCP config resolved during replay")
	}
	if app.mcpManager != nil {
		t.Fatal("mcpManager initialized during replay")
	}
}

func TestWireDoesNotBlockWhenMCPStartupEmitsManyWarnings(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Chdir(t.TempDir())
	pending := make([]runtimemcp.ScopedServer, 80)
	for i := range pending {
		pending[i] = mcpServer("pending"+strconv.Itoa(i), runtimemcp.ScopeProject)
	}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Pending: pending}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return newFakeMCPManager() },
	)
	defer restore()

	done := make(chan error, 1)
	go func() {
		app, err := Wire(BootstrapConfig{MCPApprovalPrompter: fixedMCPPrompter{}})
		if app != nil {
			app.closeRuntimeResources()
		}
		done <- err
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Wire() err = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Wire() blocked while emitting MCP startup warnings")
	}
}

func TestInitMCPPromptsPendingProjectServersAndResolvesAgain(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	workDir := t.TempDir()
	registry := tools.NewRegistry()
	pending := mcpServer("project", runtimemcp.ScopeProject)
	pending.Hash = "sha256:abc"
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["project"] = []runtimemcp.ToolDefinition{mcpDef("project", "echo")}
	calls := 0
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			calls++
			if calls == 1 {
				return runtimemcp.ResolvedConfig{Pending: []runtimemcp.ScopedServer{pending}}, nil
			}
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{pending}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	_, events, err := initMCPTools(context.Background(), registry, workDir, time.Second, fixedMCPPrompter{decision: runtimemcp.DecisionApproved})
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	if calls != 2 {
		t.Fatalf("resolve calls = %d, want 2", calls)
	}
	store := runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home))
	decision, ok, err := store.Lookup(workDir, "project", "sha256:abc")
	if err != nil || !ok || decision != runtimemcp.DecisionApproved {
		t.Fatalf("approval = %q %v %v, want approved", decision, ok, err)
	}
	if !eventsContain(events, "approved") {
		t.Fatal("events missing approval warning")
	}
}

func TestApprovalRequestIncludesHTTPTransportAndURL(t *testing.T) {
	server := runtimemcp.ScopedServer{
		Name:  "remote",
		Scope: runtimemcp.ScopeProject,
		Config: runtimemcp.ServerConfig{
			Type: "http",
			URL:  "https://example.test/mcp",
			Raw:  map[string]any{"type": "http", "url": "https://example.test/mcp"},
		},
		Hash: "sha256:remote",
	}

	req := approvalRequest("/workspace", server)
	if req.Transport != "http" {
		t.Fatalf("Transport = %q, want http", req.Transport)
	}
	if req.URL != "https://example.test/mcp" {
		t.Fatalf("URL = %q, want https://example.test/mcp", req.URL)
	}
}

func TestInitMCPRejectPersistsAndSkips(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	workDir := t.TempDir()
	registry := tools.NewRegistry()
	pending := mcpServer("project", runtimemcp.ScopeProject)
	pending.Hash = "sha256:abc"
	fakeMgr := newFakeMCPManager()
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Pending: []runtimemcp.ScopedServer{pending}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	_, events, err := initMCPTools(context.Background(), registry, workDir, time.Second, fixedMCPPrompter{decision: runtimemcp.DecisionRejected})
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	if len(fakeMgr.connected) != 0 {
		t.Fatalf("connected = %#v, want none", fakeMgr.connected)
	}
	store := runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home))
	decision, ok, err := store.Lookup(workDir, "project", "sha256:abc")
	if err != nil || !ok || decision != runtimemcp.DecisionRejected {
		t.Fatalf("approval = %q %v %v, want rejected", decision, ok, err)
	}
	if !eventsContain(events, "rejected") {
		t.Fatal("events missing rejection warning")
	}
}

func TestInitMCPSkipPendingServerEmitsWarning(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	workDir := t.TempDir()
	pending := mcpServer("project", runtimemcp.ScopeProject)
	pending.Hash = "sha256:abc"
	fakeMgr := newFakeMCPManager()
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Pending: []runtimemcp.ScopedServer{pending}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	_, events, err := initMCPTools(context.Background(), tools.NewRegistry(), workDir, time.Second, fixedMCPPrompter{})
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	store := runtimemcp.NewApprovalStore(runtimemcp.DefaultApprovalStorePath(home))
	if _, ok, err := store.Lookup(workDir, "project", "sha256:abc"); err != nil || ok {
		t.Fatalf("lookup after skip ok=%v err=%v, want no record", ok, err)
	}
	if !eventsContain(events, "skipped") {
		t.Fatal("events missing skipped warning")
	}
}

func TestInitMCPExistingRejectedServerEmitsWarning(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	rejected := mcpServer("project", runtimemcp.ScopeProject)
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Rejected: []runtimemcp.ScopedServer{rejected}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return newFakeMCPManager() },
	)
	defer restore()

	_, events, err := initMCPTools(context.Background(), tools.NewRegistry(), t.TempDir(), time.Second, nil)
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	if !eventsContain(events, "rejected") {
		t.Fatal("events missing rejected warning")
	}
}

func TestInitMCPServerFailureEmitsWarningAndContinues(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	registry := tools.NewRegistry()
	fakeMgr := newFakeMCPManager()
	fakeMgr.connectErr["bad"] = errors.New("boom")
	fakeMgr.tools["good"] = []runtimemcp.ToolDefinition{mcpDef("good", "echo")}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{
				mcpServer("bad", runtimemcp.ScopeUser),
				mcpServer("good", runtimemcp.ScopeUser),
			}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	_, events, err := initMCPTools(context.Background(), registry, t.TempDir(), time.Second, nil)
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	if _, ok := registry.Get("mcp__good__echo"); !ok {
		t.Fatal("registry missing good tool")
	}
	if !eventsContain(events, "connect mcp server bad") {
		t.Fatal("events missing bad server warning")
	}
}

func TestInitMCPDedupesNormalizedToolNamesByScopePrecedence(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	registry := tools.NewRegistry()
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["same"] = []runtimemcp.ToolDefinition{mcpDef("same", "echo")}
	fakeMgr.tools["same!"] = []runtimemcp.ToolDefinition{mcpDef("same!", "echo")}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{
				mcpServer("same", runtimemcp.ScopeUser),
				mcpServer("same!", runtimemcp.ScopeLocal),
			}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	_, events, err := initMCPTools(context.Background(), registry, t.TempDir(), time.Second, nil)
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	tool, ok := registry.Get("mcp__same__echo")
	if !ok {
		t.Fatal("registry missing deduped tool")
	}
	_, _ = tool.Execute(context.Background(), json.RawMessage(`{}`))
	if fakeMgr.calledServer != "same!" {
		t.Fatalf("calledServer = %q, want local same!", fakeMgr.calledServer)
	}
	if !eventsContain(events, "duplicate mcp tool") {
		t.Fatal("events missing duplicate warning")
	}
}

func TestInitMCPDedupesNormalizedToolNamesDeterministicallyOnTie(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	registry := tools.NewRegistry()
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["a!"] = []runtimemcp.ToolDefinition{mcpDef("a!", "echo")}
	fakeMgr.tools["a@"] = []runtimemcp.ToolDefinition{mcpDef("a@", "echo")}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{
				mcpServer("a@", runtimemcp.ScopeUser),
				mcpServer("a!", runtimemcp.ScopeUser),
			}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	_, _, err := initMCPTools(context.Background(), registry, t.TempDir(), time.Second, nil)
	if err != nil {
		t.Fatalf("initMCPTools() err = %v", err)
	}
	tool, ok := registry.Get("mcp__a__echo")
	if !ok {
		t.Fatal("registry missing deduped tool")
	}
	_, _ = tool.Execute(context.Background(), json.RawMessage(`{}`))
	if fakeMgr.calledServer != "a!" {
		t.Fatalf("calledServer = %q, want first deterministic a!", fakeMgr.calledServer)
	}
}

func TestApplicationCloseRuntimeResourcesClosesMCPManager(t *testing.T) {
	fakeMgr := newFakeMCPManager()
	app := &Application{mcpManager: fakeMgr}
	app.closeRuntimeResources()
	if !fakeMgr.closed {
		t.Fatal("mcp manager not closed")
	}
}

func TestCmdMCPSummaryReportsNoServers(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: t.TempDir(), mcpManager: newFakeMCPManager()}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return newFakeMCPManager() },
	)
	defer restore()

	app.handleCommand("/mcp")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, "No MCP servers configured.") {
		t.Fatalf("message = %q", ev.Message)
	}
}

func TestCmdMCPSummaryShowsResolvedStates(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["active"] = []runtimemcp.ToolDefinition{mcpDef("active", "echo")}
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: t.TempDir(), mcpManager: fakeMgr}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{
				Servers:  []runtimemcp.ScopedServer{mcpServer("active", runtimemcp.ScopeUser)},
				Pending:  []runtimemcp.ScopedServer{mcpServer("pending", runtimemcp.ScopeProject)},
				Rejected: []runtimemcp.ScopedServer{mcpServer("rejected", runtimemcp.ScopeProject)},
				Disabled: []runtimemcp.ScopedServer{mcpServer("disabled", runtimemcp.ScopeLocal)},
			}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	app.handleCommand("/mcp")
	ev := <-app.EventCh
	for _, want := range []string{
		"active [user] stdio - connected, 1 tools",
		"pending [project] stdio - pending approval",
		"rejected [project] stdio - rejected",
		"disabled [local] stdio - disabled",
	} {
		if !strings.Contains(ev.Message, want) {
			t.Fatalf("message missing %q:\n%s", want, ev.Message)
		}
	}
}

func TestCmdMCPReconnectReportsNotFound(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: t.TempDir(), mcpManager: newFakeMCPManager()}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return newFakeMCPManager() },
	)
	defer restore()

	app.handleCommand("/mcp reconnect missing")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, `MCP server "missing" not found`) {
		t.Fatalf("message = %q", ev.Message)
	}
}

func TestCmdMCPReconnectSuccess(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["echo"] = []runtimemcp.ToolDefinition{mcpDef("echo", "tool")}
	registry := tools.NewRegistry()
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: t.TempDir(), mcpManager: fakeMgr, toolRegistry: registry}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{mcpServer("echo", runtimemcp.ScopeLocal)}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	app.handleCommand("/mcp reconnect echo")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, "Successfully reconnected to echo") {
		t.Fatalf("message = %q", ev.Message)
	}
	if _, ok := registry.Get("mcp__echo__tool"); !ok {
		t.Fatal("registry missing mcp__echo__tool after reconnect")
	}
}

func TestCmdMCPReconnectReplacesStaleServerTools(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	fakeMgr := newFakeMCPManager()
	registry := tools.NewRegistry()
	registerMCPToolDefinitions(registry, fakeMgr, []runtimemcp.ToolDefinition{mcpDef("echo", "old")}, nil)
	fakeMgr.tools["echo"] = []runtimemcp.ToolDefinition{mcpDef("echo", "new")}
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: t.TempDir(), mcpManager: fakeMgr, toolRegistry: registry}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{Servers: []runtimemcp.ScopedServer{mcpServer("echo", runtimemcp.ScopeLocal)}}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	app.handleCommand("/mcp reconnect echo")
	<-app.EventCh
	if _, ok := registry.Get("mcp__echo__old"); ok {
		t.Fatal("registry still has stale mcp__echo__old after reconnect")
	}
	if _, ok := registry.Get("mcp__echo__new"); !ok {
		t.Fatal("registry missing mcp__echo__new after reconnect")
	}
}

func TestCmdMCPDisableWritesLocalStateAndClosesServer(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	workDir := t.TempDir()
	writeAppMCPConfig(t, runtimemcp.LocalConfigPath(workDir), map[string]any{"echo": appStdioRaw("echo")}, nil)
	fakeMgr := newFakeMCPManager()
	registry := tools.NewRegistry()
	registerMCPToolDefinitions(registry, fakeMgr, []runtimemcp.ToolDefinition{
		mcpDef("echo", "tool"),
		mcpDef("other", "tool"),
	}, nil)
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: workDir, mcpManager: fakeMgr, toolRegistry: registry}

	app.handleCommand("/mcp disable echo")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, `MCP server "echo" disabled`) {
		t.Fatalf("message = %q", ev.Message)
	}
	disabled, err := runtimemcp.ReadLocalDisabledServers(workDir)
	if err != nil {
		t.Fatalf("ReadLocalDisabledServers: %v", err)
	}
	if strings.Join(disabled, ",") != "echo" {
		t.Fatalf("disabled = %#v, want echo", disabled)
	}
	if got := strings.Join(fakeMgr.closedServers, ","); got != "echo" {
		t.Fatalf("closedServers = %q, want echo", got)
	}
	if _, ok := registry.Get("mcp__echo__tool"); ok {
		t.Fatal("registry still has disabled server tool mcp__echo__tool")
	}
	if _, ok := registry.Get("mcp__other__tool"); !ok {
		t.Fatal("registry removed unrelated MCP server tool")
	}
}

func TestCmdMCPEnableRemovesLocalDisabledState(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	workDir := t.TempDir()
	writeAppMCPConfig(t, runtimemcp.LocalConfigPath(workDir), map[string]any{"echo": appStdioRaw("echo")}, []string{"echo"})
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["echo"] = []runtimemcp.ToolDefinition{mcpDef("echo", "tool")}
	registry := tools.NewRegistry()
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: workDir, mcpManager: fakeMgr, toolRegistry: registry}

	app.handleCommand("/mcp enable echo")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, `MCP server "echo" enabled`) {
		t.Fatalf("message = %q", ev.Message)
	}
	disabled, err := runtimemcp.ReadLocalDisabledServers(workDir)
	if err != nil {
		t.Fatalf("ReadLocalDisabledServers: %v", err)
	}
	if len(disabled) != 0 {
		t.Fatalf("disabled = %#v, want empty", disabled)
	}
	if got := strings.Join(fakeMgr.connected, ","); got != "echo" {
		t.Fatalf("connected = %q, want echo", got)
	}
	if _, ok := registry.Get("mcp__echo__tool"); !ok {
		t.Fatal("registry missing mcp__echo__tool after enable")
	}
}

func TestCmdMCPEnableDisabledPendingProjectDoesNotReconnectWithoutApproval(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	workDir := t.TempDir()
	writeAppMCPConfig(t, runtimemcp.LocalConfigPath(workDir), nil, []string{"pending"})
	fakeMgr := newFakeMCPManager()
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: workDir, mcpManager: fakeMgr, toolRegistry: tools.NewRegistry()}
	resolveCalls := 0
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			resolveCalls++
			if resolveCalls == 1 {
				return runtimemcp.ResolvedConfig{
					Disabled: []runtimemcp.ScopedServer{mcpServer("pending", runtimemcp.ScopeProject)},
				}, nil
			}
			return runtimemcp.ResolvedConfig{
				Pending: []runtimemcp.ScopedServer{mcpServer("pending", runtimemcp.ScopeProject)},
			}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	app.handleCommand("/mcp enable pending")
	ev := <-app.EventCh
	if strings.Contains(ev.Message, `"pending" enabled`) || !strings.Contains(ev.Message, "pending approval") {
		t.Fatalf("enable disabled pending message = %q", ev.Message)
	}
	if len(fakeMgr.connected) != 0 {
		t.Fatalf("connected = %#v, want none", fakeMgr.connected)
	}
	disabled, err := runtimemcp.ReadLocalDisabledServers(workDir)
	if err != nil {
		t.Fatalf("ReadLocalDisabledServers: %v", err)
	}
	if strings.Join(disabled, ",") != "pending" {
		t.Fatalf("disabled = %#v, want pending preserved", disabled)
	}
}

func TestCmdMCPEnableAllContinuesAfterDisabledPendingProject(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	workDir := t.TempDir()
	writeAppMCPConfig(t, runtimemcp.LocalConfigPath(workDir), nil, []string{"active", "pending"})
	fakeMgr := newFakeMCPManager()
	fakeMgr.tools["active"] = []runtimemcp.ToolDefinition{mcpDef("active", "tool")}
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: workDir, mcpManager: fakeMgr, toolRegistry: tools.NewRegistry()}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			disabled, err := runtimemcp.ReadLocalDisabledServers(workDir)
			if err != nil {
				return runtimemcp.ResolvedConfig{}, err
			}
			disabledSet := make(map[string]struct{}, len(disabled))
			for _, name := range disabled {
				disabledSet[name] = struct{}{}
			}
			var resolved runtimemcp.ResolvedConfig
			if _, ok := disabledSet["pending"]; ok {
				resolved.Disabled = append(resolved.Disabled, mcpServer("pending", runtimemcp.ScopeProject))
			} else {
				resolved.Pending = append(resolved.Pending, mcpServer("pending", runtimemcp.ScopeProject))
			}
			if _, ok := disabledSet["active"]; ok {
				resolved.Disabled = append(resolved.Disabled, mcpServer("active", runtimemcp.ScopeLocal))
			} else {
				resolved.Servers = append(resolved.Servers, mcpServer("active", runtimemcp.ScopeLocal))
			}
			return resolved, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return fakeMgr },
	)
	defer restore()

	app.handleCommand("/mcp enable")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, "Enabled 1 MCP server(s)") || !strings.Contains(ev.Message, "skipped 1") {
		t.Fatalf("enable all message = %q", ev.Message)
	}
	if got := strings.Join(fakeMgr.connected, ","); got != "active" {
		t.Fatalf("connected = %q, want active", got)
	}
	disabled, err := runtimemcp.ReadLocalDisabledServers(workDir)
	if err != nil {
		t.Fatalf("ReadLocalDisabledServers: %v", err)
	}
	if strings.Join(disabled, ",") != "pending" {
		t.Fatalf("disabled = %#v, want pending preserved", disabled)
	}
}

func TestCmdMCPEnablePendingOrRejectedDoesNotReportSuccess(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: t.TempDir(), mcpManager: newFakeMCPManager()}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{
				Pending:  []runtimemcp.ScopedServer{mcpServer("pending", runtimemcp.ScopeProject)},
				Rejected: []runtimemcp.ScopedServer{mcpServer("rejected", runtimemcp.ScopeProject)},
			}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return newFakeMCPManager() },
	)
	defer restore()

	app.handleCommand("/mcp enable pending")
	pending := <-app.EventCh
	if strings.Contains(pending.Message, `"pending" enabled`) || !strings.Contains(pending.Message, "pending approval") {
		t.Fatalf("pending enable message = %q", pending.Message)
	}

	app.handleCommand("/mcp enable rejected")
	rejected := <-app.EventCh
	if strings.Contains(rejected.Message, `"rejected" enabled`) || !strings.Contains(rejected.Message, "rejected") {
		t.Fatalf("rejected enable message = %q", rejected.Message)
	}
}

func TestCmdMCPDisablePendingWritesLocalDisabledState(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	workDir := t.TempDir()
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: workDir, mcpManager: newFakeMCPManager()}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{
				Pending: []runtimemcp.ScopedServer{mcpServer("pending", runtimemcp.ScopeProject)},
			}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return newFakeMCPManager() },
	)
	defer restore()

	app.handleCommand("/mcp disable pending")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, `"pending" disabled`) {
		t.Fatalf("disable pending message = %q", ev.Message)
	}
	disabled, err := runtimemcp.ReadLocalDisabledServers(workDir)
	if err != nil {
		t.Fatalf("ReadLocalDisabledServers: %v", err)
	}
	if strings.Join(disabled, ",") != "pending" {
		t.Fatalf("disabled = %#v, want pending", disabled)
	}
}

func TestCmdMCPNoRedirectShowsSummary(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	app := &Application{EventCh: make(chan model.Event, 8), WorkDir: t.TempDir(), mcpManager: newFakeMCPManager()}
	restore := stubMCPRuntime(t,
		func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error) {
			return runtimemcp.ResolvedConfig{}, nil
		},
		func(runtimemcp.Config) runtimemcp.Manager { return newFakeMCPManager() },
	)
	defer restore()

	app.handleCommand("/mcp no-redirect")
	ev := <-app.EventCh
	if !strings.Contains(ev.Message, "No MCP servers configured.") {
		t.Fatalf("message = %q", ev.Message)
	}
}

func stubMCPRuntime(t *testing.T, resolver func(context.Context, runtimemcp.ResolveOptions) (runtimemcp.ResolvedConfig, error), managerFactory func(runtimemcp.Config) runtimemcp.Manager) func() {
	t.Helper()
	oldResolver := resolveMCPConfig
	oldManager := newMCPManager
	resolveMCPConfig = resolver
	newMCPManager = managerFactory
	return func() {
		resolveMCPConfig = oldResolver
		newMCPManager = oldManager
	}
}

func mcpServer(name string, scope runtimemcp.Scope) runtimemcp.ScopedServer {
	return runtimemcp.ScopedServer{
		Name:  name,
		Scope: scope,
		Config: runtimemcp.ServerConfig{
			Type:    "stdio",
			Command: "server",
			Raw:     map[string]any{"type": "stdio", "command": "server"},
		},
		Hash: "sha256:" + name,
	}
}

func mcpDef(serverName, toolName string) runtimemcp.ToolDefinition {
	return runtimemcp.ToolDefinition{
		ServerName:       serverName,
		OriginalToolName: toolName,
		Name:             runtimemcp.BuildToolName(serverName, toolName),
		Description:      "test tool",
		InputSchema:      map[string]any{"type": "object"},
	}
}

func appStdioRaw(command string) map[string]any {
	return map[string]any{
		"type":    "stdio",
		"command": command,
		"args":    []any{"--flag"},
		"env":     map[string]any{"A": "1"},
	}
}

func writeAppMCPConfig(t *testing.T, path string, servers map[string]any, disabled []string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	root := map[string]any{"mcpServers": servers}
	if disabled != nil {
		root["disabledMcpServers"] = disabled
	}
	data, err := json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

type fixedMCPPrompter struct {
	decision runtimemcp.ApprovalDecision
	err      error
}

func (p fixedMCPPrompter) PromptMCPApproval(context.Context, MCPApprovalRequest) (runtimemcp.ApprovalDecision, error) {
	return p.decision, p.err
}

type fakeMCPManager struct {
	connected      []string
	connectErr     map[string]error
	tools          map[string][]runtimemcp.ToolDefinition
	calledServer   string
	closed         bool
	closedServers  []string
	closeServerErr map[string]error
}

func newFakeMCPManager() *fakeMCPManager {
	return &fakeMCPManager{
		connectErr:     make(map[string]error),
		tools:          make(map[string][]runtimemcp.ToolDefinition),
		closeServerErr: make(map[string]error),
	}
}

func (m *fakeMCPManager) Connect(ctx context.Context, server runtimemcp.ScopedServer) error {
	if err := m.connectErr[server.Name]; err != nil {
		return err
	}
	m.connected = append(m.connected, server.Name)
	return nil
}

func (m *fakeMCPManager) ListTools(ctx context.Context, serverName string) ([]runtimemcp.ToolDefinition, error) {
	return m.tools[serverName], nil
}

func (m *fakeMCPManager) CallTool(ctx context.Context, serverName, toolName string, args json.RawMessage) (*runtimemcp.CallResult, error) {
	m.calledServer = serverName
	return &runtimemcp.CallResult{Content: []runtimemcp.ContentBlock{{Type: "text", Text: "ok"}}}, nil
}

func (m *fakeMCPManager) CloseServer(ctx context.Context, serverName string) error {
	if err := m.closeServerErr[serverName]; err != nil {
		return err
	}
	m.closedServers = append(m.closedServers, serverName)
	return nil
}

func (m *fakeMCPManager) Close(ctx context.Context) error {
	m.closed = true
	return nil
}

func eventsContain(events []model.Event, needle string) bool {
	for _, ev := range events {
		if strings.Contains(ev.Message, needle) {
			return true
		}
	}
	return false
}
