package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
	mcpSessionHeader = "MCP-Session-Id"
)

var errMCPSessionExpired = errors.New("mcp http session expired")

type sseResumeError struct {
	lastEventID string
	retry       time.Duration
}

func (e sseResumeError) Error() string {
	return "mcp event stream ended before response"
}

type streamableHTTPClient struct {
	server ScopedServer
	cfg    Config
	http   *http.Client

	opMu   sync.Mutex
	nextID atomic.Int64

	mu        sync.Mutex
	sessionID string
	protocol  string
	closed    bool
	unhealthy bool
	connected bool
}

func newStreamableHTTPClient(server ScopedServer, cfg Config) *streamableHTTPClient {
	return &streamableHTTPClient{
		server:   server,
		cfg:      cfg,
		http:     http.DefaultClient,
		protocol: mcpProtocolVersion,
	}
}

func (c *streamableHTTPClient) Connect(ctx context.Context) error {
	c.opMu.Lock()
	defer c.opMu.Unlock()

	return c.connectLocked(ctx)
}

func (c *streamableHTTPClient) connectLocked(ctx context.Context) error {
	if c.healthy() {
		return nil
	}
	connectCtx, cancel := withDefaultTimeout(ctx, c.connectTimeout())
	defer cancel()

	var result struct {
		ProtocolVersion string         `json:"protocolVersion"`
		Capabilities    map[string]any `json:"capabilities"`
	}
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
	negotiated := strings.TrimSpace(result.ProtocolVersion)
	if negotiated == "" {
		negotiated = mcpProtocolVersion
	}
	if !supportedMCPProtocolVersion(negotiated) {
		c.markUnhealthy()
		return fmt.Errorf("unsupported mcp protocol version %q", negotiated)
	}
	c.setProtocolVersion(negotiated)
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

	var result struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := c.callWithSessionRecoveryLocked(ctx, "tools/list", map[string]any{}, &result); err != nil {
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
	var rawResult json.RawMessage
	if err := c.callWithSessionRecoveryLocked(ctx, "tools/call", map[string]any{
		"name":      toolName,
		"arguments": arguments,
	}, &rawResult); err != nil {
		return nil, err
	}
	return decodeCallResult(rawResult)
}

func (c *streamableHTTPClient) callWithSessionRecoveryLocked(ctx context.Context, method string, params any, result any) error {
	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		if err := c.ensureConnected(); err != nil {
			return err
		}
		callCtx, cancel := withDefaultTimeout(ctx, c.callTimeout())
		err := c.call(callCtx, method, params, result)
		cancel()
		if err == nil {
			return nil
		}
		if errors.Is(err, errMCPSessionExpired) && attempt == 0 {
			lastErr = err
			c.markSessionExpired()
			if err := c.connectLocked(ctx); err != nil {
				return err
			}
			continue
		}
		if isContextErr(err) {
			c.markUnhealthy()
		}
		return err
	}
	return lastErr
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
	req.Header.Set("MCP-Protocol-Version", c.currentProtocolVersion())
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
	req.Header.Set("MCP-Protocol-Version", c.currentProtocolVersion())
	sessionID := c.currentSessionID()
	if sessionID != "" {
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
		if resp.StatusCode == http.StatusNotFound && sessionID != "" {
			return fmt.Errorf("%w: status %d: %s", errMCPSessionExpired, resp.StatusCode, msg)
		}
		return fmt.Errorf("mcp http status %d: %s", resp.StatusCode, msg)
	}

	rpcResp, err := c.decodeHTTPRPCResponse(ctx, resp, id)
	var resumeErr sseResumeError
	if errors.As(err, &resumeErr) {
		rpcResp, err = c.resumeSSE(ctx, id, resumeErr)
	}
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
	req.Header.Set("MCP-Protocol-Version", c.currentProtocolVersion())
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

func (c *streamableHTTPClient) resumeSSE(ctx context.Context, id int64, resume sseResumeError) (jsonrpcResponse, error) {
	lastEventID := resume.lastEventID
	retry := resume.retry
	for {
		if lastEventID == "" {
			return jsonrpcResponse{}, fmt.Errorf("mcp event stream ended before response id %d", id)
		}
		if retry > 0 {
			timer := time.NewTimer(retry)
			select {
			case <-ctx.Done():
				timer.Stop()
				return jsonrpcResponse{}, ctx.Err()
			case <-timer.C:
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.server.Config.URL, nil)
		if err != nil {
			return jsonrpcResponse{}, fmt.Errorf("create mcp http resume request: %w", err)
		}
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("Last-Event-ID", lastEventID)
		req.Header.Set("MCP-Protocol-Version", c.currentProtocolVersion())
		if sessionID := c.currentSessionID(); sessionID != "" {
			req.Header.Set(mcpSessionHeader, sessionID)
		}
		resp, err := c.http.Do(req)
		if err != nil {
			return jsonrpcResponse{}, fmt.Errorf("send mcp http resume request: %w", err)
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			_ = resp.Body.Close()
			msg := strings.TrimSpace(string(body))
			if msg == "" {
				msg = http.StatusText(resp.StatusCode)
			}
			return jsonrpcResponse{}, fmt.Errorf("mcp http resume status %d: %s", resp.StatusCode, msg)
		}
		rpcResp, err := c.decodeHTTPRPCResponse(ctx, resp, id)
		_ = resp.Body.Close()
		if err == nil {
			return rpcResp, nil
		}
		var nextResume sseResumeError
		if !errors.As(err, &nextResume) {
			return jsonrpcResponse{}, err
		}
		lastEventID = nextResume.lastEventID
		retry = nextResume.retry
	}
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

func (c *streamableHTTPClient) currentProtocolVersion() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.protocol == "" {
		return mcpProtocolVersion
	}
	return c.protocol
}

func (c *streamableHTTPClient) setProtocolVersion(protocol string) {
	c.mu.Lock()
	c.protocol = protocol
	c.mu.Unlock()
}

func (c *streamableHTTPClient) markUnhealthy() {
	c.mu.Lock()
	c.unhealthy = true
	c.connected = false
	c.mu.Unlock()
}

func (c *streamableHTTPClient) markSessionExpired() {
	c.mu.Lock()
	c.sessionID = ""
	c.protocol = mcpProtocolVersion
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

func (c *streamableHTTPClient) decodeHTTPRPCResponse(ctx context.Context, resp *http.Response, id int64) (jsonrpcResponse, error) {
	mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mediaType == "text/event-stream" {
		return decodeSSERPCResponseWithHandler(resp.Body, id, c.respondToServerRequest(ctx))
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

func (c *streamableHTTPClient) respondToServerRequest(ctx context.Context) func(jsonrpcMessage) error {
	return func(msg jsonrpcMessage) error {
		if len(msg.ID) == 0 {
			return nil
		}
		var result json.RawMessage
		var rpcErr *jsonrpcError
		switch msg.Method {
		case "ping":
			result = json.RawMessage(`{}`)
		default:
			rpcErr = &jsonrpcError{Code: -32601, Message: "method not found"}
		}
		return c.sendHTTPResponse(ctx, msg.ID, result, rpcErr)
	}
}

func (c *streamableHTTPClient) sendHTTPResponse(ctx context.Context, id json.RawMessage, result json.RawMessage, rpcErr *jsonrpcError) error {
	reqBody, err := json.Marshal(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Result  json.RawMessage `json:"result,omitempty"`
		Error   *jsonrpcError   `json:"error,omitempty"`
	}{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
		Error:   rpcErr,
	})
	if err != nil {
		return fmt.Errorf("encode mcp http response: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.server.Config.URL, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("create mcp http response request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("MCP-Protocol-Version", c.currentProtocolVersion())
	if sessionID := c.currentSessionID(); sessionID != "" {
		req.Header.Set(mcpSessionHeader, sessionID)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("send mcp http response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		msg := strings.TrimSpace(string(body))
		if msg == "" {
			msg = http.StatusText(resp.StatusCode)
		}
		return fmt.Errorf("mcp http response status %d: %s", resp.StatusCode, msg)
	}
	return nil
}

func decodeHTTPRPCResponse(resp *http.Response, id int64) (jsonrpcResponse, error) {
	return decodeHTTPRPCResponseWithHandler(resp, id, nil)
}

func decodeHTTPRPCResponseWithHandler(resp *http.Response, id int64, handleServerRequest func(jsonrpcMessage) error) (jsonrpcResponse, error) {
	mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mediaType == "text/event-stream" {
		return decodeSSERPCResponseWithHandler(resp.Body, id, handleServerRequest)
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
	return decodeSSERPCResponseWithHandler(r, id, nil)
}

func decodeSSERPCResponseWithHandler(r io.Reader, id int64, handleServerRequest func(jsonrpcMessage) error) (jsonrpcResponse, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var dataLines []string
	var lastEventID string
	var retry time.Duration
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if len(dataLines) == 0 {
				continue
			}
			resp, ok, err := decodeSSERPCData(strings.Join(dataLines, "\n"), id, handleServerRequest)
			if err != nil || ok {
				return resp, err
			}
			dataLines = nil
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			continue
		}
		if strings.HasPrefix(line, "id:") {
			lastEventID = strings.TrimSpace(strings.TrimPrefix(line, "id:"))
			continue
		}
		if strings.HasPrefix(line, "retry:") {
			if retryMillis, err := time.ParseDuration(strings.TrimSpace(strings.TrimPrefix(line, "retry:")) + "ms"); err == nil && retryMillis > 0 {
				retry = retryMillis
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return jsonrpcResponse{}, fmt.Errorf("read mcp http event stream: %w", err)
	}
	if len(dataLines) > 0 {
		resp, ok, err := decodeSSERPCData(strings.Join(dataLines, "\n"), id, handleServerRequest)
		if err != nil || ok {
			return resp, err
		}
	}
	if lastEventID != "" {
		return jsonrpcResponse{}, sseResumeError{lastEventID: lastEventID, retry: retry}
	}
	return jsonrpcResponse{}, fmt.Errorf("mcp event stream ended before response id %d", id)
}

func decodeSSERPCData(data string, id int64, handleServerRequest func(jsonrpcMessage) error) (jsonrpcResponse, bool, error) {
	if strings.TrimSpace(data) == "" {
		return jsonrpcResponse{}, false, nil
	}
	var msg jsonrpcMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		return jsonrpcResponse{}, false, fmt.Errorf("decode mcp http event data: %w", err)
	}
	if msg.Method != "" {
		if handleServerRequest != nil {
			if err := handleServerRequest(msg); err != nil {
				return jsonrpcResponse{}, false, err
			}
		}
		return jsonrpcResponse{}, false, nil
	}
	msgID, ok := jsonrpcNumericID(msg.ID)
	if !ok || msgID != id {
		return jsonrpcResponse{}, false, nil
	}
	return jsonrpcResponse{
		JSONRPC: msg.JSONRPC,
		ID:      msgID,
		Result:  msg.Result,
		Error:   msg.Error,
	}, true, nil
}
