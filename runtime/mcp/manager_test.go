package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestManagerConnectListCallAndClose(t *testing.T) {
	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	server := managerTestServer("fake", "normal", nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := mgr.Connect(ctx, server); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	tools, err := mgr.ListTools(ctx, "fake")
	if err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
	if got, want := tools[0].Name, "mcp__fake__echo"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	result, err := mgr.CallTool(ctx, "fake", "echo", json.RawMessage(`{"text":"hi"}`))
	if err != nil {
		t.Fatalf("CallTool() err = %v", err)
	}
	if got, want := result.Content[0].Text, "echo: hi"; got != want {
		t.Fatalf("result text = %q, want %q", got, want)
	}
	if err := mgr.Close(ctx); err != nil {
		t.Fatalf("Close() err = %v", err)
	}
}

func TestManagerSkipsUnsupportedTransport(t *testing.T) {
	mgr := NewManager(Config{})
	err := mgr.Connect(context.Background(), ScopedServer{
		Name:   "remote",
		Scope:  ScopeUser,
		Config: ServerConfig{Type: "sse", URL: "https://example.com/mcp"},
	})
	if err == nil {
		t.Fatal("Connect() err = nil, want unsupported transport")
	}
	if !strings.Contains(err.Error(), "unsupported mcp server type") {
		t.Fatalf("Connect() err = %v", err)
	}
}

func TestManagerHTTPConnectListCallAndClose(t *testing.T) {
	sessionID := "test-session"
	var sawInitialized, sawSessionOnList, sawDelete bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			if got := r.Header.Get("MCP-Session-Id"); got != sessionID {
				t.Errorf("DELETE session header = %q, want %q", got, sessionID)
			}
			sawDelete = true
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if got := r.Header.Get("Content-Type"); !strings.Contains(got, "application/json") {
			t.Errorf("Content-Type = %q, want application/json", got)
		}
		if got := r.Header.Get("Accept"); !strings.Contains(got, "application/json") || !strings.Contains(got, "text/event-stream") {
			t.Errorf("Accept = %q, want json and event-stream", got)
		}
		var req jsonrpcRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch req.Method {
		case "initialize":
			w.Header().Set("MCP-Session-Id", sessionID)
			writeJSONRPCResult(t, w, req.ID, map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{},
			})
		case "notifications/initialized":
			if got := r.Header.Get("MCP-Session-Id"); got == sessionID {
				sawInitialized = true
			}
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			if got := r.Header.Get("MCP-Session-Id"); got == sessionID {
				sawSessionOnList = true
			}
			writeJSONRPCResult(t, w, req.ID, map[string]any{
				"tools": []map[string]any{{
					"name":        "echo",
					"description": "Echo text",
					"inputSchema": map[string]any{"type": "object"},
				}},
			})
		case "tools/call":
			writeJSONRPCResult(t, w, req.ID, map[string]any{
				"content": []map[string]any{{"type": "text", "text": "echo: hi"}},
			})
		default:
			t.Errorf("unexpected method %q", req.Method)
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	defer server.Close()

	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	remote := ScopedServer{
		Name:  "remote",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type: "http",
			URL:  server.URL + "/mcp",
		},
	}
	if err := mgr.Connect(ctx, remote); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	tools, err := mgr.ListTools(ctx, "remote")
	if err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
	if got, want := tools[0].Name, "mcp__remote__echo"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	result, err := mgr.CallTool(ctx, "remote", "echo", json.RawMessage(`{"text":"hi"}`))
	if err != nil {
		t.Fatalf("CallTool() err = %v", err)
	}
	if got, want := result.Content[0].Text, "echo: hi"; got != want {
		t.Fatalf("result text = %q, want %q", got, want)
	}
	if err := mgr.Close(ctx); err != nil {
		t.Fatalf("Close() err = %v", err)
	}
	if !sawSessionOnList {
		t.Fatal("tools/list did not include MCP-Session-Id header")
	}
	if !sawInitialized {
		t.Fatal("Connect did not send initialized notification")
	}
	if !sawDelete {
		t.Fatal("Close did not send DELETE")
	}
}

func TestManagerHTTPUsesNegotiatedProtocolVersion(t *testing.T) {
	const negotiated = "2025-03-26"
	sessionID := "test-session"
	var sawNegotiatedList, sawNegotiatedCall, sawNegotiatedDelete bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			if got := r.Header.Get("MCP-Protocol-Version"); got == negotiated {
				sawNegotiatedDelete = true
			}
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var req jsonrpcRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch req.Method {
		case "initialize":
			w.Header().Set("MCP-Session-Id", sessionID)
			writeJSONRPCResult(t, w, req.ID, map[string]any{
				"protocolVersion": negotiated,
				"capabilities":    map[string]any{},
			})
		case "notifications/initialized":
			if got := r.Header.Get("MCP-Protocol-Version"); got != negotiated {
				t.Errorf("initialized protocol header = %q, want %q", got, negotiated)
			}
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			if got := r.Header.Get("MCP-Protocol-Version"); got != negotiated {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte("wrong protocol version"))
				return
			}
			sawNegotiatedList = true
			writeJSONRPCResult(t, w, req.ID, map[string]any{
				"tools": []map[string]any{{
					"name":        "echo",
					"description": "Echo text",
					"inputSchema": map[string]any{"type": "object"},
				}},
			})
		case "tools/call":
			if got := r.Header.Get("MCP-Protocol-Version"); got != negotiated {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte("wrong protocol version"))
				return
			}
			sawNegotiatedCall = true
			writeJSONRPCResult(t, w, req.ID, map[string]any{
				"content": []map[string]any{{"type": "text", "text": "echo: hi"}},
			})
		default:
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	defer server.Close()

	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	remote := ScopedServer{
		Name:  "remote",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type: "http",
			URL:  server.URL + "/mcp",
		},
	}
	if err := mgr.Connect(ctx, remote); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	if _, err := mgr.ListTools(ctx, "remote"); err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
	if _, err := mgr.CallTool(ctx, "remote", "echo", json.RawMessage(`{"text":"hi"}`)); err != nil {
		t.Fatalf("CallTool() err = %v", err)
	}
	if err := mgr.Close(ctx); err != nil {
		t.Fatalf("Close() err = %v", err)
	}
	if !sawNegotiatedList || !sawNegotiatedCall || !sawNegotiatedDelete {
		t.Fatalf("negotiated protocol flags list=%v call=%v delete=%v", sawNegotiatedList, sawNegotiatedCall, sawNegotiatedDelete)
	}
}

func TestManagerHTTPRejectsUnsupportedNegotiatedProtocolVersion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req jsonrpcRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.Method != "initialize" {
			t.Errorf("method = %q, want initialize", req.Method)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		writeJSONRPCResult(t, w, req.ID, map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{},
		})
	}))
	defer server.Close()

	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	err := mgr.Connect(ctx, ScopedServer{
		Name:  "remote",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type: "http",
			URL:  server.URL + "/mcp",
		},
	})
	if err == nil {
		t.Fatal("Connect() err = nil, want unsupported protocol error")
	}
	if !strings.Contains(err.Error(), `unsupported mcp protocol version "2024-11-05"`) {
		t.Fatalf("Connect() err = %v", err)
	}
}

func TestManagerHTTPSSEReconnectsWithLastEventIDAfterEarlyClose(t *testing.T) {
	sessionID := "test-session"
	var sawResume bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var req jsonrpcRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode request: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			switch req.Method {
			case "initialize":
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("MCP-Session-Id", sessionID)
				writeJSONRPCResult(t, w, req.ID, map[string]any{
					"protocolVersion": "2025-11-25",
					"capabilities":    map[string]any{},
				})
			case "notifications/initialized":
				w.WriteHeader(http.StatusAccepted)
			case "tools/list":
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("id: cursor-1\nretry: 1\ndata:\n\n"))
			default:
				w.WriteHeader(http.StatusBadRequest)
			}
		case http.MethodGet:
			if got := r.Header.Get("Last-Event-ID"); got != "cursor-1" {
				t.Errorf("Last-Event-ID = %q, want cursor-1", got)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if got := r.Header.Get("MCP-Session-Id"); got != sessionID {
				t.Errorf("GET session = %q, want %q", got, sessionID)
			}
			sawResume = true
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{\"type\":\"object\"}}]}}\n\n"))
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := mgr.Connect(ctx, ScopedServer{
		Name:  "remote",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type: "http",
			URL:  server.URL + "/mcp",
		},
	}); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	tools, err := mgr.ListTools(ctx, "remote")
	if err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
	if got, want := tools[0].Name, "mcp__remote__echo"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if !sawResume {
		t.Fatal("server did not receive SSE resume GET")
	}
}

func TestManagerHTTPSSERespondsToServerPingBeforeFinalResponse(t *testing.T) {
	sessionID := "test-session"
	pingResponse := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var msg jsonrpcMessage
			if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
				t.Errorf("decode request: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			switch msg.Method {
			case "initialize":
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("MCP-Session-Id", sessionID)
				writeJSONRPCResult(t, w, 1, map[string]any{
					"protocolVersion": "2025-11-25",
					"capabilities":    map[string]any{},
				})
			case "notifications/initialized":
				w.WriteHeader(http.StatusAccepted)
			case "tools/list":
				w.Header().Set("Content-Type", "text/event-stream")
				flusher, _ := w.(http.Flusher)
				_, _ = w.Write([]byte("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":99,\"method\":\"ping\"}\n\n"))
				if flusher != nil {
					flusher.Flush()
				}
				select {
				case <-pingResponse:
				case <-time.After(time.Second):
					t.Error("timed out waiting for ping response")
				}
				_, _ = w.Write([]byte("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{\"type\":\"object\"}}]}}\n\n"))
			case "":
				if string(msg.ID) == "99" && len(msg.Result) > 0 {
					pingResponse <- struct{}{}
					w.WriteHeader(http.StatusAccepted)
					return
				}
				t.Errorf("unexpected response message id=%s result=%s error=%v", msg.ID, msg.Result, msg.Error)
				w.WriteHeader(http.StatusBadRequest)
			default:
				w.WriteHeader(http.StatusBadRequest)
			}
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: 2 * time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := mgr.Connect(ctx, ScopedServer{
		Name:  "remote",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type: "http",
			URL:  server.URL + "/mcp",
		},
	}); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	tools, err := mgr.ListTools(ctx, "remote")
	if err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
	if got, want := tools[0].Name, "mcp__remote__echo"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
}

func TestManagerHTTPSSERespondsMethodNotFoundForUnsupportedServerRequest(t *testing.T) {
	sessionID := "test-session"
	errorResponse := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var msg jsonrpcMessage
			if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
				t.Errorf("decode request: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			switch msg.Method {
			case "initialize":
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("MCP-Session-Id", sessionID)
				writeJSONRPCResult(t, w, 1, map[string]any{
					"protocolVersion": "2025-11-25",
					"capabilities":    map[string]any{},
				})
			case "notifications/initialized":
				w.WriteHeader(http.StatusAccepted)
			case "tools/list":
				w.Header().Set("Content-Type", "text/event-stream")
				flusher, _ := w.(http.Flusher)
				_, _ = w.Write([]byte("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":99,\"method\":\"unsupported/request\"}\n\n"))
				if flusher != nil {
					flusher.Flush()
				}
				select {
				case <-errorResponse:
				case <-time.After(time.Second):
					t.Error("timed out waiting for unsupported request response")
				}
				_, _ = w.Write([]byte("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{\"type\":\"object\"}}]}}\n\n"))
			case "":
				if string(msg.ID) == "99" && msg.Error != nil && msg.Error.Code == -32601 {
					errorResponse <- struct{}{}
					w.WriteHeader(http.StatusAccepted)
					return
				}
				t.Errorf("unexpected response message id=%s result=%s error=%v", msg.ID, msg.Result, msg.Error)
				w.WriteHeader(http.StatusBadRequest)
			default:
				w.WriteHeader(http.StatusBadRequest)
			}
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: 2 * time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := mgr.Connect(ctx, ScopedServer{
		Name:  "remote",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type: "http",
			URL:  server.URL + "/mcp",
		},
	}); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	if _, err := mgr.ListTools(ctx, "remote"); err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
}

func TestManagerHTTPReconnectsAfterSessionExpires(t *testing.T) {
	var initializeCount int
	var initializedNewSession bool
	var sawExpiredSessionOnList bool
	var sawNewSessionOnList bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var req jsonrpcRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch req.Method {
		case "initialize":
			initializeCount++
			if initializeCount == 2 && r.Header.Get("MCP-Session-Id") != "" {
				t.Errorf("second initialize included stale session %q", r.Header.Get("MCP-Session-Id"))
			}
			w.Header().Set("MCP-Session-Id", fmt.Sprintf("session-%d", initializeCount))
			writeJSONRPCResult(t, w, req.ID, map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{},
			})
		case "notifications/initialized":
			if r.Header.Get("MCP-Session-Id") == "session-2" {
				initializedNewSession = true
			}
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			switch r.Header.Get("MCP-Session-Id") {
			case "session-1":
				sawExpiredSessionOnList = true
				w.WriteHeader(http.StatusNotFound)
			case "session-2":
				sawNewSessionOnList = true
				writeJSONRPCResult(t, w, req.ID, map[string]any{
					"tools": []map[string]any{{
						"name":        "echo",
						"description": "Echo text",
						"inputSchema": map[string]any{"type": "object"},
					}},
				})
			default:
				t.Errorf("tools/list session = %q, want session-1 or session-2", r.Header.Get("MCP-Session-Id"))
				w.WriteHeader(http.StatusBadRequest)
			}
		default:
			t.Errorf("unexpected method %q", req.Method)
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	defer server.Close()

	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := mgr.Connect(ctx, ScopedServer{
		Name:  "remote",
		Scope: ScopeUser,
		Config: ServerConfig{
			Type: "http",
			URL:  server.URL + "/mcp",
		},
	}); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}
	tools, err := mgr.ListTools(ctx, "remote")
	if err != nil {
		t.Fatalf("ListTools() err = %v", err)
	}
	if got, want := tools[0].Name, "mcp__remote__echo"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if initializeCount != 2 {
		t.Fatalf("initialize count = %d, want 2", initializeCount)
	}
	if !sawExpiredSessionOnList || !sawNewSessionOnList || !initializedNewSession {
		t.Fatalf("session recovery flags expired=%v newList=%v initialized=%v", sawExpiredSessionOnList, sawNewSessionOnList, initializedNewSession)
	}
}

func TestManagerReturnsHelpfulErrorForUnknownServer(t *testing.T) {
	mgr := NewManager(Config{})
	_, err := mgr.CallTool(context.Background(), "missing", "echo", nil)
	if err == nil {
		t.Fatal("CallTool() err = nil, want unknown server")
	}
	if !strings.Contains(err.Error(), "unknown mcp server") {
		t.Fatalf("CallTool() err = %v", err)
	}
}

func TestManagerCloseClosesAllClients(t *testing.T) {
	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	for _, name := range []string{"one", "two"} {
		if err := mgr.Connect(ctx, managerTestServer(name, "normal", nil)); err != nil {
			t.Fatalf("Connect(%s) err = %v", name, err)
		}
	}
	if err := mgr.Close(ctx); err != nil {
		t.Fatalf("Close() err = %v", err)
	}
	_, err := mgr.ListTools(ctx, "one")
	if err == nil {
		t.Fatal("ListTools after Close err = nil, want error")
	}
}

func TestManagerCloseServerClosesOnlySelectedClient(t *testing.T) {
	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	concrete := mgr.(*manager)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := mgr.Connect(ctx, managerTestServer("first", "normal", nil)); err != nil {
		t.Fatalf("Connect first: %v", err)
	}
	if err := mgr.Connect(ctx, managerTestServer("second", "normal", nil)); err != nil {
		t.Fatalf("Connect second: %v", err)
	}
	if err := mgr.CloseServer(ctx, "first"); err != nil {
		t.Fatalf("CloseServer first: %v", err)
	}

	concrete.mu.Lock()
	_, firstClient := concrete.clients["first"]
	_, secondClient := concrete.clients["second"]
	_, firstServer := concrete.servers["first"]
	_, secondServer := concrete.servers["second"]
	concrete.mu.Unlock()
	if firstClient {
		t.Fatal("first client still present after CloseServer")
	}
	if !secondClient {
		t.Fatal("second client missing after CloseServer first")
	}
	if firstServer {
		t.Fatal("first server still present after CloseServer")
	}
	if !secondServer {
		t.Fatal("second server missing after CloseServer first")
	}
	if _, err := mgr.ListTools(ctx, "first"); err == nil || !strings.Contains(err.Error(), "unknown mcp server") {
		t.Fatalf("ListTools first after CloseServer err = %v, want unknown server", err)
	}
	if _, err := mgr.ListTools(ctx, "second"); err != nil {
		t.Fatalf("ListTools second after close first: %v", err)
	}
	if err := mgr.CloseServer(ctx, "missing"); err == nil {
		t.Fatal("CloseServer missing err = nil, want unknown server error")
	}
}

func TestManagerReconnectsAfterCanceledCall(t *testing.T) {
	mgr := NewManager(Config{ConnectTimeout: time.Second, CallTimeout: time.Second})
	statePath := filepath.Join(t.TempDir(), "block-once")
	server := managerTestServer("fake", "block-once", map[string]string{
		"MCP_BLOCK_ONCE_FILE": statePath,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := mgr.Connect(ctx, server); err != nil {
		t.Fatalf("Connect() err = %v", err)
	}

	callCtx, cancelCall := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancelCall()
	_, err := mgr.CallTool(callCtx, "fake", "echo", json.RawMessage(`{"text":"first"}`))
	if err == nil {
		t.Fatal("first CallTool() err = nil, want timeout")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("first CallTool() err = %v, want deadline exceeded", err)
	}

	result, err := mgr.CallTool(ctx, "fake", "echo", json.RawMessage(`{"text":"second"}`))
	if err != nil {
		t.Fatalf("second CallTool() err = %v", err)
	}
	if got, want := result.Content[0].Text, "echo: second"; got != want {
		t.Fatalf("second result = %q, want %q", got, want)
	}
}

func managerTestServer(name, mode string, env map[string]string) ScopedServer {
	mergedEnv := map[string]string{"GO_WANT_HELPER_PROCESS": "1"}
	for key, value := range env {
		mergedEnv[key] = value
	}
	return ScopedServer{
		Name:  name,
		Scope: ScopeUser,
		Config: ServerConfig{
			Type:    "stdio",
			Command: os.Args[0],
			Args:    []string{"-test.run=TestMCPHelperProcess", "--", mode},
			Env:     mergedEnv,
			Raw:     map[string]any{"type": "stdio", "command": os.Args[0]},
		},
	}
}

func writeJSONRPCResult(t *testing.T, w http.ResponseWriter, id int64, result any) {
	t.Helper()
	if err := json.NewEncoder(w).Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"result":  result,
	}); err != nil {
		t.Fatalf("write response: %v", err)
	}
}
