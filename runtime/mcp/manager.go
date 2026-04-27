package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// Manager owns MCP server connections and calls.
type Manager interface {
	Connect(ctx context.Context, server ScopedServer) error
	ListTools(ctx context.Context, serverName string) ([]ToolDefinition, error)
	CallTool(ctx context.Context, serverName, toolName string, args json.RawMessage) (*CallResult, error)
	CloseServer(ctx context.Context, serverName string) error
	Close(ctx context.Context) error
}

type manager struct {
	cfg Config

	mu      sync.Mutex
	servers map[string]ScopedServer
	clients map[string]mcpClient
	closed  bool
}

type mcpClient interface {
	Connect(ctx context.Context) error
	ListTools(ctx context.Context) ([]ToolDefinition, error)
	CallTool(ctx context.Context, toolName string, args json.RawMessage) (*CallResult, error)
	Close(ctx context.Context) error
	healthy() bool
}

// NewManager creates an MCP runtime manager.
func NewManager(cfg Config) Manager {
	return &manager{
		cfg:     cfg,
		servers: make(map[string]ScopedServer),
		clients: make(map[string]mcpClient),
	}
}

func (m *manager) Connect(ctx context.Context, server ScopedServer) error {
	client, err := newClientForServer(server, m.cfg)
	if err != nil {
		return err
	}
	if err := client.Connect(ctx); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		_ = client.Close(context.Background())
		return fmt.Errorf("mcp manager is closed")
	}
	if old := m.clients[server.Name]; old != nil {
		_ = old.Close(context.Background())
	}
	m.servers[server.Name] = server
	m.clients[server.Name] = client
	return nil
}

func (m *manager) ListTools(ctx context.Context, serverName string) ([]ToolDefinition, error) {
	client, err := m.ensureClient(ctx, serverName)
	if err != nil {
		return nil, err
	}
	tools, err := client.ListTools(ctx)
	if isContextErr(err) {
		m.dropClient(serverName, client)
	}
	return tools, err
}

func (m *manager) CallTool(ctx context.Context, serverName, toolName string, args json.RawMessage) (*CallResult, error) {
	client, err := m.ensureClient(ctx, serverName)
	if err != nil {
		return nil, err
	}
	result, err := client.CallTool(ctx, toolName, args)
	if isContextErr(err) {
		m.dropClient(serverName, client)
	}
	return result, err
}

func (m *manager) CloseServer(ctx context.Context, serverName string) error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return fmt.Errorf("mcp manager is closed")
	}
	if _, ok := m.servers[serverName]; !ok {
		m.mu.Unlock()
		return fmt.Errorf("unknown mcp server %q", serverName)
	}
	client := m.clients[serverName]
	delete(m.clients, serverName)
	delete(m.servers, serverName)
	m.mu.Unlock()

	if client == nil {
		return nil
	}
	return client.Close(ctx)
}

func (m *manager) Close(ctx context.Context) error {
	m.mu.Lock()
	clients := make([]mcpClient, 0, len(m.clients))
	for _, client := range m.clients {
		clients = append(clients, client)
	}
	m.clients = make(map[string]mcpClient)
	m.closed = true
	m.mu.Unlock()

	var errs []string
	for _, client := range clients {
		if err := client.Close(ctx); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func (m *manager) ensureClient(ctx context.Context, serverName string) (mcpClient, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, fmt.Errorf("mcp manager is closed")
	}
	server, ok := m.servers[serverName]
	if !ok {
		m.mu.Unlock()
		return nil, fmt.Errorf("unknown mcp server %q", serverName)
	}
	client := m.clients[serverName]
	if client != nil && client.healthy() {
		m.mu.Unlock()
		return client, nil
	}
	m.mu.Unlock()

	next, err := newClientForServer(server, m.cfg)
	if err != nil {
		return nil, err
	}
	if err := next.Connect(ctx); err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		_ = next.Close(context.Background())
		return nil, fmt.Errorf("mcp manager is closed")
	}
	if old := m.clients[serverName]; old != nil && old != client {
		_ = old.Close(context.Background())
	}
	m.clients[serverName] = next
	return next, nil
}

func (m *manager) dropClient(serverName string, client mcpClient) {
	m.mu.Lock()
	if current := m.clients[serverName]; current == client {
		delete(m.clients, serverName)
	}
	m.mu.Unlock()
}

func newClientForServer(server ScopedServer, cfg Config) (mcpClient, error) {
	switch server.Config.TransportType() {
	case "stdio":
		return newStdioClient(server, cfg), nil
	case "http":
		return newStreamableHTTPClient(server, cfg), nil
	default:
		return nil, fmt.Errorf("unsupported mcp server type %q for %s", server.Config.TransportType(), server.Name)
	}
}
