package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	mcpProtocolVersion = "2025-11-25"
	mcpSessionHeader   = "MCP-Session-Id"
)

type streamableHTTPClient struct {
	server ScopedServer
	cfg    Config
	http   *http.Client

	opMu   sync.Mutex
	nextID atomic.Int64

	mu        sync.Mutex
	sessionID string
	closed    bool
	unhealthy bool
	connected bool
}

func newStreamableHTTPClient(server ScopedServer, cfg Config) *streamableHTTPClient {
	return &streamableHTTPClient{
		server: server,
		cfg:    cfg,
		http:   http.DefaultClient,
	}
}

func (c *streamableHTTPClient) Connect(ctx context.Context) error {
	c.opMu.Lock()
	defer c.opMu.Unlock()

	if c.healthy() {
		return nil
	}
	connectCtx, cancel := withDefaultTimeout(ctx, c.connectTimeout())
	defer cancel()

	var result map[string]any
	if err := c.call(connectCtx, "initialize", map[string]any{
		"protocolVersion": mcpProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    "mscli",
			"version": "0",
		},
	}, &result); err != nil {
		c.markUnhealthy()
		return err
	}
	notifyCtx, cancelNotify := context.WithTimeout(connectCtx, 500*time.Millisecond)
	_ = c.notify(notifyCtx, "notifications/initialized", map[string]any{})
	cancelNotify()
	c.mu.Lock()
	c.connected = true
	c.closed = false
	c.unhealthy = false
	c.mu.Unlock()
	return nil
}

func (c *streamableHTTPClient) ListTools(ctx context.Context) ([]ToolDefinition, error) {
	c.opMu.Lock()
	defer c.opMu.Unlock()

	if err := c.ensureConnected(); err != nil {
		return nil, err
	}
	callCtx, cancel := withDefaultTimeout(ctx, c.callTimeout())
	defer cancel()

	var result struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := c.call(callCtx, "tools/list", map[string]any{}, &result); err != nil {
		if isContextErr(err) {
			c.markUnhealthy()
		}
		return nil, err
	}
	return toolDefinitionsFromRaw(c.server.Name, result.Tools)
}

func (c *streamableHTTPClient) CallTool(ctx context.Context, toolName string, args json.RawMessage) (*CallResult, error) {
	c.opMu.Lock()
	defer c.opMu.Unlock()

	if err := c.ensureConnected(); err != nil {
		return nil, err
	}
	arguments, err := decodeArguments(args)
	if err != nil {
		return nil, err
	}
	callCtx, cancel := withDefaultTimeout(ctx, c.callTimeout())
	defer cancel()

	var rawResult json.RawMessage
	if err := c.call(callCtx, "tools/call", map[string]any{
		"name":      toolName,
		"arguments": arguments,
	}, &rawResult); err != nil {
		if isContextErr(err) {
			c.markUnhealthy()
		}
		return nil, err
	}
	return decodeCallResult(rawResult)
}

func (c *streamableHTTPClient) Close(ctx context.Context) error {
	c.mu.Lock()
	sessionID := c.sessionID
	closed := c.closed
	c.closed = true
	c.connected = false
	c.unhealthy = true
	c.mu.Unlock()

	if closed || sessionID == "" {
		return nil
	}
	closeCtx, cancel := withDefaultTimeout(ctx, c.connectTimeout())
	defer cancel()
	req, err := http.NewRequestWithContext(closeCtx, http.MethodDelete, c.server.Config.URL, nil)
	if err != nil {
		return fmt.Errorf("create mcp http close request: %w", err)
	}
	req.Header.Set(mcpSessionHeader, sessionID)
	req.Header.Set("MCP-Protocol-Version", mcpProtocolVersion)
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("close mcp http session: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 && resp.StatusCode != http.StatusNotFound && resp.StatusCode != http.StatusMethodNotAllowed {
		return fmt.Errorf("close mcp http session: status %d", resp.StatusCode)
	}
	return nil
}

func (c *streamableHTTPClient) healthy() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.closed && !c.unhealthy && c.connected
}

func (c *streamableHTTPClient) ensureConnected() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("mcp client %s is closed", c.server.Name)
	}
	if c.unhealthy || !c.connected {
		return fmt.Errorf("mcp client %s is not connected", c.server.Name)
	}
	return nil
}

func (c *streamableHTTPClient) call(ctx context.Context, method string, params any, result any) error {
	id := c.nextID.Add(1)
	reqBody, err := json.Marshal(jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	})
	if err != nil {
		return fmt.Errorf("encode mcp http request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.server.Config.URL, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("create mcp http request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("MCP-Protocol-Version", mcpProtocolVersion)
	if sessionID := c.currentSessionID(); sessionID != "" {
		req.Header.Set(mcpSessionHeader, sessionID)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("send mcp http request: %w", err)
	}
	defer resp.Body.Close()
	if sessionID := resp.Header.Get(mcpSessionHeader); sessionID != "" {
		c.setSessionID(sessionID)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		msg := strings.TrimSpace(string(body))
		if msg == "" {
			msg = http.StatusText(resp.StatusCode)
		}
		return fmt.Errorf("mcp http status %d: %s", resp.StatusCode, msg)
	}

	rpcResp, err := decodeHTTPRPCResponse(resp, id)
	if err != nil {
		return err
	}
	if rpcResp.Error != nil {
		return fmt.Errorf("mcp json-rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}
	if result != nil {
		if err := json.Unmarshal(rpcResp.Result, result); err != nil {
			return fmt.Errorf("decode mcp json-rpc result: %w", err)
		}
	}
	return nil
}

func (c *streamableHTTPClient) notify(ctx context.Context, method string, params any) error {
	reqBody, err := json.Marshal(struct {
		JSONRPC string `json:"jsonrpc"`
		Method  string `json:"method"`
		Params  any    `json:"params,omitempty"`
	}{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	})
	if err != nil {
		return fmt.Errorf("encode mcp http notification: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.server.Config.URL, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("create mcp http notification: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("MCP-Protocol-Version", mcpProtocolVersion)
	if sessionID := c.currentSessionID(); sessionID != "" {
		req.Header.Set(mcpSessionHeader, sessionID)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("send mcp http notification: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		msg := strings.TrimSpace(string(body))
		if msg == "" {
			msg = http.StatusText(resp.StatusCode)
		}
		return fmt.Errorf("mcp http notification status %d: %s", resp.StatusCode, msg)
	}
	return nil
}

func (c *streamableHTTPClient) currentSessionID() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sessionID
}

func (c *streamableHTTPClient) setSessionID(sessionID string) {
	c.mu.Lock()
	c.sessionID = sessionID
	c.mu.Unlock()
}

func (c *streamableHTTPClient) markUnhealthy() {
	c.mu.Lock()
	c.unhealthy = true
	c.connected = false
	c.mu.Unlock()
}

func (c *streamableHTTPClient) connectTimeout() time.Duration {
	if c.cfg.ConnectTimeout > 0 {
		return c.cfg.ConnectTimeout
	}
	return defaultConnectTimeout
}

func (c *streamableHTTPClient) callTimeout() time.Duration {
	if c.cfg.CallTimeout > 0 {
		return c.cfg.CallTimeout
	}
	return defaultCallTimeout
}

func decodeHTTPRPCResponse(resp *http.Response, id int64) (jsonrpcResponse, error) {
	mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mediaType == "text/event-stream" {
		return decodeSSERPCResponse(resp.Body, id)
	}
	var rpcResp jsonrpcResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return jsonrpcResponse{}, fmt.Errorf("decode mcp http json response: %w", err)
	}
	if rpcResp.ID != id {
		return jsonrpcResponse{}, fmt.Errorf("mcp json-rpc response id %d does not match request id %d", rpcResp.ID, id)
	}
	return rpcResp, nil
}

func decodeSSERPCResponse(r io.Reader, id int64) (jsonrpcResponse, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var dataLines []string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if len(dataLines) == 0 {
				continue
			}
			resp, ok, err := decodeSSERPCData(strings.Join(dataLines, "\n"), id)
			if err != nil || ok {
				return resp, err
			}
			dataLines = nil
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return jsonrpcResponse{}, fmt.Errorf("read mcp http event stream: %w", err)
	}
	if len(dataLines) > 0 {
		resp, ok, err := decodeSSERPCData(strings.Join(dataLines, "\n"), id)
		if err != nil || ok {
			return resp, err
		}
	}
	return jsonrpcResponse{}, fmt.Errorf("mcp event stream ended before response id %d", id)
}

func decodeSSERPCData(data string, id int64) (jsonrpcResponse, bool, error) {
	if strings.TrimSpace(data) == "" {
		return jsonrpcResponse{}, false, nil
	}
	var rpcResp jsonrpcResponse
	if err := json.Unmarshal([]byte(data), &rpcResp); err != nil {
		return jsonrpcResponse{}, false, fmt.Errorf("decode mcp http event data: %w", err)
	}
	if rpcResp.ID != id {
		return jsonrpcResponse{}, false, nil
	}
	return rpcResp, true, nil
}
