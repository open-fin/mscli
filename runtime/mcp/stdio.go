package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

const (
	defaultConnectTimeout = 10 * time.Second
	defaultCallTimeout    = 5 * time.Minute
	stderrLimit           = 64 * 1024
)

type stdioClient struct {
	server ScopedServer
	cfg    Config

	opMu sync.Mutex
	mu   sync.Mutex

	cmd       *exec.Cmd
	stdin     io.WriteCloser
	stdout    io.ReadCloser
	waitCh    chan error
	rpc       *rpcConn
	stderr    *boundedBuffer
	closed    bool
	unhealthy bool
}

func newStdioClient(server ScopedServer, cfg Config) *stdioClient {
	return &stdioClient{
		server: server,
		cfg:    cfg,
		stderr: newBoundedBuffer(stderrLimit),
	}
}

func (c *stdioClient) Connect(ctx context.Context) error {
	c.opMu.Lock()
	defer c.opMu.Unlock()

	if c.healthy() {
		return nil
	}

	connectCtx, cancel := withDefaultTimeout(ctx, c.connectTimeout())
	defer cancel()

	cmd := exec.Command(c.server.Config.Command, c.server.Config.Args...)
	if strings.TrimSpace(c.cfg.WorkDir) != "" {
		cmd.Dir = c.cfg.WorkDir
	}
	cmd.Env = mergeEnv(os.Environ(), c.server.Config.Env)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("open mcp stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return fmt.Errorf("open mcp stdout: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return fmt.Errorf("open mcp stderr: %w", err)
	}
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return fmt.Errorf("start mcp server %s: %w", c.server.Name, err)
	}

	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
	}()
	stderrBuf := newBoundedBuffer(stderrLimit)
	go func() {
		_, _ = io.Copy(stderrBuf, stderrPipe)
	}()

	rpc := newRPCConn(stdin, stdout, func() {
		_ = c.retire(context.Background(), false)
	})

	c.mu.Lock()
	c.cmd = cmd
	c.stdin = stdin
	c.stdout = stdout
	c.waitCh = waitCh
	c.rpc = rpc
	c.stderr = stderrBuf
	c.closed = false
	c.unhealthy = false
	c.mu.Unlock()

	var result map[string]any
	err = rpc.call(connectCtx, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    "mscli",
			"version": "0",
		},
	}, &result)
	if err != nil {
		_ = c.retire(context.Background(), false)
		return c.withStderr(err)
	}
	return nil
}

func (c *stdioClient) ListTools(ctx context.Context) ([]ToolDefinition, error) {
	c.opMu.Lock()
	defer c.opMu.Unlock()

	rpc, err := c.rpcForOperation()
	if err != nil {
		return nil, err
	}
	callCtx, cancel := withDefaultTimeout(ctx, c.callTimeout())
	defer cancel()

	var result struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := rpc.call(callCtx, "tools/list", map[string]any{}, &result); err != nil {
		if isContextErr(err) {
			_ = c.retire(context.Background(), false)
		}
		return nil, c.withStderr(err)
	}

	return toolDefinitionsFromRaw(c.server.Name, result.Tools)
}

func (c *stdioClient) CallTool(ctx context.Context, toolName string, args json.RawMessage) (*CallResult, error) {
	c.opMu.Lock()
	defer c.opMu.Unlock()

	rpc, err := c.rpcForOperation()
	if err != nil {
		return nil, err
	}
	arguments, err := decodeArguments(args)
	if err != nil {
		return nil, err
	}
	callCtx, cancel := withDefaultTimeout(ctx, c.callTimeout())
	defer cancel()

	var rawResult json.RawMessage
	if err := rpc.call(callCtx, "tools/call", map[string]any{
		"name":      toolName,
		"arguments": arguments,
	}, &rawResult); err != nil {
		if isContextErr(err) {
			_ = c.retire(context.Background(), false)
		}
		return nil, c.withStderr(err)
	}
	result, err := decodeCallResult(rawResult)
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (c *stdioClient) Close(ctx context.Context) error {
	return c.retire(ctx, true)
}

func (c *stdioClient) healthy() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.closed && !c.unhealthy && c.cmd != nil && c.rpc != nil
}

func (c *stdioClient) stderrString() string {
	c.mu.Lock()
	stderr := c.stderr
	c.mu.Unlock()
	if stderr == nil {
		return ""
	}
	return stderr.String()
}

func (c *stdioClient) rpcForOperation() (*rpcConn, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, fmt.Errorf("mcp client %s is closed", c.server.Name)
	}
	if c.unhealthy || c.rpc == nil {
		return nil, fmt.Errorf("mcp client %s is not connected", c.server.Name)
	}
	return c.rpc, nil
}

func (c *stdioClient) retire(ctx context.Context, closed bool) error {
	c.mu.Lock()
	cmd := c.cmd
	stdin := c.stdin
	stdout := c.stdout
	waitCh := c.waitCh
	c.cmd = nil
	c.stdin = nil
	c.stdout = nil
	c.waitCh = nil
	c.rpc = nil
	if closed {
		c.closed = true
	}
	c.unhealthy = true
	c.mu.Unlock()

	if stdin != nil {
		_ = stdin.Close()
	}
	if stdout != nil {
		_ = stdout.Close()
	}
	if cmd == nil || waitCh == nil {
		return nil
	}

	select {
	case <-waitCh:
		return nil
	case <-time.After(100 * time.Millisecond):
	}

	if cmd.Process != nil {
		_ = cmd.Process.Kill()
	}

	waitCtx := ctx
	if waitCtx == nil {
		waitCtx = context.Background()
	}
	select {
	case <-waitCh:
		return nil
	case <-waitCtx.Done():
		return waitCtx.Err()
	case <-time.After(time.Second):
		return fmt.Errorf("timeout waiting for mcp server %s to exit", c.server.Name)
	}
}

func (c *stdioClient) connectTimeout() time.Duration {
	if c.cfg.ConnectTimeout > 0 {
		return c.cfg.ConnectTimeout
	}
	return defaultConnectTimeout
}

func (c *stdioClient) callTimeout() time.Duration {
	if c.cfg.CallTimeout > 0 {
		return c.cfg.CallTimeout
	}
	return defaultCallTimeout
}

func (c *stdioClient) withStderr(err error) error {
	if err == nil {
		return nil
	}
	stderr := strings.TrimSpace(c.stderrString())
	if stderr == "" {
		return err
	}
	if len(stderr) > 4096 {
		stderr = stderr[:4096] + "\n... stderr truncated ..."
	}
	return fmt.Errorf("%w; stderr: %s", err, stderr)
}

func withDefaultTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	if _, ok := ctx.Deadline(); ok || timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func mergeEnv(base []string, overrides map[string]string) []string {
	if len(overrides) == 0 {
		return base
	}
	env := make([]string, 0, len(base)+len(overrides))
	seen := make(map[string]struct{}, len(overrides))
	for key := range overrides {
		seen[key] = struct{}{}
	}
	for _, item := range base {
		key, _, ok := strings.Cut(item, "=")
		if ok {
			if _, overridden := seen[key]; overridden {
				continue
			}
		}
		env = append(env, item)
	}
	for key, value := range overrides {
		env = append(env, key+"="+value)
	}
	return env
}

func toolDefinitionsFromRaw(serverName string, tools []json.RawMessage) ([]ToolDefinition, error) {
	defs := make([]ToolDefinition, 0, len(tools))
	for _, rawTool := range tools {
		var tool struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			InputSchema map[string]any `json:"inputSchema"`
		}
		if err := decodeJSONRaw(rawTool, &tool); err != nil {
			return nil, fmt.Errorf("decode mcp tool definition: %w", err)
		}
		if strings.TrimSpace(tool.Name) == "" {
			continue
		}
		rawMap := make(map[string]any)
		_ = decodeJSONRaw(rawTool, &rawMap)
		defs = append(defs, ToolDefinition{
			ServerName:       serverName,
			OriginalToolName: tool.Name,
			Name:             BuildToolName(serverName, tool.Name),
			Description:      tool.Description,
			InputSchema:      tool.InputSchema,
			Raw:              rawMap,
		})
	}
	return defs, nil
}

func decodeArguments(raw json.RawMessage) (any, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return map[string]any{}, nil
	}
	var out any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&out); err != nil {
		return nil, fmt.Errorf("decode mcp tool arguments: %w", err)
	}
	if out == nil {
		return map[string]any{}, nil
	}
	return out, nil
}

func decodeCallResult(raw json.RawMessage) (*CallResult, error) {
	var wire struct {
		Content           []json.RawMessage `json:"content"`
		StructuredContent any               `json:"structuredContent"`
		IsError           bool              `json:"isError"`
	}
	if err := decodeJSONRaw(raw, &wire); err != nil {
		return nil, fmt.Errorf("decode mcp call result: %w", err)
	}
	result := &CallResult{
		Content:           make([]ContentBlock, 0, len(wire.Content)),
		StructuredContent: wire.StructuredContent,
		IsError:           wire.IsError,
		Raw:               raw,
	}
	for _, rawBlock := range wire.Content {
		var blockMap map[string]any
		if err := decodeJSONRaw(rawBlock, &blockMap); err != nil {
			return nil, fmt.Errorf("decode mcp content block: %w", err)
		}
		block := ContentBlock{Data: blockMap}
		if value, ok := blockMap["type"].(string); ok {
			block.Type = value
		}
		if value, ok := blockMap["text"].(string); ok {
			block.Text = value
		}
		result.Content = append(result.Content, block)
	}
	return result, nil
}

func decodeJSONRaw(raw json.RawMessage, v any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	return dec.Decode(v)
}

func isContextErr(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

type boundedBuffer struct {
	mu    sync.Mutex
	limit int
	data  []byte
}

func newBoundedBuffer(limit int) *boundedBuffer {
	return &boundedBuffer{limit: limit}
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = append(b.data, p...)
	if b.limit > 0 && len(b.data) > b.limit {
		b.data = append([]byte(nil), b.data[len(b.data)-b.limit:]...)
	}
	return len(p), nil
}

func (b *boundedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(append([]byte(nil), b.data...))
}
