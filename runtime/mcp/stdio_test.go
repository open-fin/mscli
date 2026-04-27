package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func TestStdioClientInitializeAndListTools(t *testing.T) {
	client := newTestStdioClient(t, "normal")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	defer client.Close(context.Background())

	tools, err := client.ListTools(ctx)
	if err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
	if got, want := len(tools), 1; got != want {
		t.Fatalf("ListTools len = %d, want %d", got, want)
	}
	if got, want := tools[0].Name, "mcp__fake__echo"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if got, want := tools[0].OriginalToolName, "echo"; got != want {
		t.Fatalf("original tool name = %q, want %q", got, want)
	}
}

func TestStdioClientSendsInitializedNotificationBeforeListTools(t *testing.T) {
	client := newTestStdioClient(t, "require-initialized")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	defer client.Close(context.Background())

	if _, err := client.ListTools(ctx); err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
}

func TestStdioClientRespondsToServerPingBeforeResponse(t *testing.T) {
	client := newTestStdioClient(t, "ping-before-list")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	defer client.Close(context.Background())

	if _, err := client.ListTools(ctx); err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
}

func TestStdioClientSendsLatestProtocolVersion(t *testing.T) {
	client := newTestStdioClient(t, "require-latest-protocol")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	defer client.Close(context.Background())
}

func TestStdioClientAcceptsSupportedNegotiatedProtocolVersion(t *testing.T) {
	client := newTestStdioClient(t, "negotiate-2025-03-26")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	defer client.Close(context.Background())
}

func TestStdioClientRejectsUnsupportedNegotiatedProtocolVersion(t *testing.T) {
	client := newTestStdioClient(t, "negotiate-2024-11-05")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	err := client.Connect(ctx)
	if err == nil {
		t.Fatal("Connect() err = nil, want unsupported protocol error")
	}
	if !strings.Contains(err.Error(), `unsupported mcp protocol version "2024-11-05"`) {
		t.Fatalf("Connect() err = %v", err)
	}
}

func TestStdioClientCallTool(t *testing.T) {
	client := newTestStdioClient(t, "normal")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	defer client.Close(context.Background())

	result, err := client.CallTool(ctx, "echo", json.RawMessage(`{"text":"hello"}`))
	if err != nil {
		t.Fatalf("CallTool() err = %v", err)
	}
	if result.IsError {
		t.Fatal("CallTool result IsError = true")
	}
	if got, want := len(result.Content), 1; got != want {
		t.Fatalf("content len = %d, want %d", got, want)
	}
	if got, want := result.Content[0].Text, "echo: hello"; got != want {
		t.Fatalf("content text = %q, want %q", got, want)
	}
	if result.StructuredContent == nil {
		t.Fatal("StructuredContent = nil, want object")
	}
}

func TestStdioClientPropagatesJSONRPCError(t *testing.T) {
	client := newTestStdioClient(t, "rpc-error")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	defer client.Close(context.Background())

	_, err := client.CallTool(ctx, "echo", json.RawMessage(`{"text":"hello"}`))
	if err == nil {
		t.Fatal("CallTool() err = nil, want error")
	}
	if !strings.Contains(err.Error(), "mcp json-rpc") {
		t.Fatalf("CallTool() err = %v, want json-rpc error", err)
	}
}

func TestStdioClientCapturesBoundedStderr(t *testing.T) {
	client := newTestStdioClient(t, "stderr")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	err := client.Connect(ctx)
	if err == nil {
		t.Fatal("Connect() err = nil, want error")
	}
	if !strings.Contains(err.Error(), "stderr:") {
		t.Fatalf("Connect() err = %v, want stderr content", err)
	}
	if len(client.stderrString()) > 64*1024+128 {
		t.Fatalf("stderr buffer too large: %d", len(client.stderrString()))
	}
}

func TestStdioClientConnectTimeoutTerminatesProcess(t *testing.T) {
	client := newTestStdioClient(t, "hang-init")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	client.cfg.ConnectTimeout = 50 * time.Millisecond
	err := client.Connect(ctx)
	if err == nil {
		t.Fatal("Connect() err = nil, want timeout")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Connect() err = %v, want deadline exceeded", err)
	}
	if client.healthy() {
		t.Fatal("client healthy after connect timeout")
	}
}

func TestStdioClientCallToolCancellationClosesClient(t *testing.T) {
	client := newTestStdioClient(t, "block-call")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	callCtx, cancelCall := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancelCall()
	_, err := client.CallTool(callCtx, "echo", json.RawMessage(`{"text":"hello"}`))
	if err == nil {
		t.Fatal("CallTool() err = nil, want timeout")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("CallTool() err = %v, want deadline exceeded", err)
	}
	if client.healthy() {
		t.Fatal("client healthy after canceled call")
	}
}

func TestStdioClientCloseTerminatesProcess(t *testing.T) {
	client := newTestStdioClient(t, "normal")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	if err := client.Close(ctx); err != nil {
		t.Fatalf("Close() err = %v", err)
	}
	if client.healthy() {
		t.Fatal("client healthy after close")
	}
}

func newTestStdioClient(t *testing.T, mode string) *stdioClient {
	t.Helper()
	server := ScopedServer{
		Name:  "fake",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type:    "stdio",
			Command: os.Args[0],
			Args:    []string{"-test.run=TestMCPHelperProcess", "--", mode},
			Env:     map[string]string{"GO_WANT_HELPER_PROCESS": "1"},
			Raw:     map[string]any{"type": "stdio", "command": os.Args[0]},
		},
	}
	return newStdioClient(server, Config{
		ConnectTimeout: time.Second,
		CallTimeout:    time.Second,
	})
}

func TestMCPHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	mode := os.Args[len(os.Args)-1]
	dec := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(os.Stdout)
	initialized := false
	for {
		var req map[string]any
		if err := dec.Decode(&req); err != nil {
			os.Exit(0)
		}
		id := req["id"]
		method, _ := req["method"].(string)
		switch method {
		case "initialize":
			if mode == "hang-init" {
				for {
					time.Sleep(time.Hour)
				}
			}
			if mode == "require-latest-protocol" {
				params, _ := req["params"].(map[string]any)
				if got := params["protocolVersion"]; got != "2025-11-25" {
					writeRPCError(enc, id, -32602, fmt.Sprintf("protocolVersion = %v, want 2025-11-25", got))
					continue
				}
			}
			if mode == "stderr" {
				fmt.Fprint(os.Stderr, strings.Repeat("stderr line\n", 8000))
				writeRPCError(enc, id, -32000, "initialize failed")
				continue
			}
			protocolVersion := "2025-11-25"
			if mode == "negotiate-2025-03-26" {
				protocolVersion = "2025-03-26"
			}
			if mode == "negotiate-2024-11-05" {
				protocolVersion = "2024-11-05"
			}
			writeRPCResult(enc, id, map[string]any{
				"protocolVersion": protocolVersion,
				"capabilities":    map[string]any{},
				"serverInfo":      map[string]any{"name": "fake", "version": "1.0.0"},
			})
		case "notifications/initialized":
			initialized = true
		case "tools/list":
			if mode == "require-initialized" && !initialized {
				writeRPCError(enc, id, -32002, "initialized notification required before tools/list")
				continue
			}
			if mode == "ping-before-list" {
				pingID := 99
				_ = enc.Encode(map[string]any{
					"jsonrpc": "2.0",
					"id":      pingID,
					"method":  "ping",
				})
				var resp map[string]any
				if err := dec.Decode(&resp); err != nil {
					os.Exit(2)
				}
				if int(resp["id"].(float64)) != pingID {
					writeRPCError(enc, id, -32003, "ping response id mismatch")
					continue
				}
				if _, ok := resp["result"].(map[string]any); !ok {
					writeRPCError(enc, id, -32003, "ping response missing empty result")
					continue
				}
			}
			writeRPCResult(enc, id, map[string]any{
				"tools": []any{
					map[string]any{
						"name":        "echo",
						"description": "echo input",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"text": map[string]any{"type": "string"},
							},
							"required": []any{"text"},
						},
					},
				},
			})
		case "tools/call":
			if mode == "block-call" {
				for {
					time.Sleep(time.Hour)
				}
			}
			if mode == "block-once" && shouldBlockOnce() {
				for {
					time.Sleep(time.Hour)
				}
			}
			if mode == "rpc-error" {
				writeRPCError(enc, id, -32001, "call failed")
				continue
			}
			text := toolCallText(req)
			writeRPCResult(enc, id, map[string]any{
				"content": []any{
					map[string]any{"type": "text", "text": "echo: " + text},
				},
				"structuredContent": map[string]any{"echoed": text},
				"isError":           false,
			})
		default:
			writeRPCError(enc, id, -32601, "method not found")
		}
	}
}

func writeRPCResult(enc *json.Encoder, id any, result any) {
	_ = enc.Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"result":  result,
	})
}

func writeRPCError(enc *json.Encoder, id any, code int, message string) {
	_ = enc.Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"error": map[string]any{
			"code":    code,
			"message": message,
		},
	})
}

func toolCallText(req map[string]any) string {
	params, _ := req["params"].(map[string]any)
	args, _ := params["arguments"].(map[string]any)
	text, _ := args["text"].(string)
	return text
}

func shouldBlockOnce() bool {
	path := os.Getenv("MCP_BLOCK_ONCE_FILE")
	if path == "" {
		return false
	}
	if _, err := os.Stat(path); err == nil {
		return false
	}
	_ = os.WriteFile(path, []byte("blocked"), 0644)
	return true
}
