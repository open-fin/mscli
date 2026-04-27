package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"
)

func TestRPCConnCallReturnsResult(t *testing.T) {
	rpc, serverIn, serverOut := newTestRPCConn()
	defer serverIn.Close()
	defer serverOut.Close()

	go func() {
		var req jsonrpcRequest
		_ = json.NewDecoder(serverIn).Decode(&req)
		_ = json.NewEncoder(serverOut).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      req.ID,
			"result":  map[string]any{"ok": true},
		})
	}()

	var result map[string]bool
	if err := rpc.call(context.Background(), "ping", map[string]any{}, &result); err != nil {
		t.Fatalf("call() err = %v", err)
	}
	if !result["ok"] {
		t.Fatalf("result = %#v, want ok", result)
	}
}

func TestRPCConnCallReturnsError(t *testing.T) {
	rpc, serverIn, serverOut := newTestRPCConn()
	defer serverIn.Close()
	defer serverOut.Close()

	go func() {
		var req jsonrpcRequest
		_ = json.NewDecoder(serverIn).Decode(&req)
		_ = json.NewEncoder(serverOut).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      req.ID,
			"error":   map[string]any{"code": -32000, "message": "failed"},
		})
	}()

	err := rpc.call(context.Background(), "ping", nil, nil)
	if err == nil {
		t.Fatal("call() err = nil, want error")
	}
	if !strings.Contains(err.Error(), "mcp json-rpc error -32000: failed") {
		t.Fatalf("call() err = %v", err)
	}
}

func TestRPCConnCallIgnoresNotificationBeforeResponse(t *testing.T) {
	rpc, serverIn, serverOut := newTestRPCConn()
	defer serverIn.Close()
	defer serverOut.Close()

	go func() {
		var req jsonrpcRequest
		_ = json.NewDecoder(serverIn).Decode(&req)
		_ = json.NewEncoder(serverOut).Encode(map[string]any{
			"jsonrpc": "2.0",
			"method":  "notifications/tools/list_changed",
			"params":  map[string]any{},
		})
		_ = json.NewEncoder(serverOut).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      req.ID,
			"result":  map[string]any{"ok": true},
		})
	}()

	var result map[string]bool
	if err := rpc.call(context.Background(), "ping", nil, &result); err != nil {
		t.Fatalf("call() err = %v", err)
	}
	if !result["ok"] {
		t.Fatalf("result = %#v, want ok", result)
	}
}

func TestRPCConnCallHandlesRequestBeforeResponse(t *testing.T) {
	rpc, serverIn, serverOut := newTestRPCConn()
	defer serverIn.Close()
	defer serverOut.Close()

	serverDone := make(chan error, 1)
	go func() {
		dec := json.NewDecoder(serverIn)
		enc := json.NewEncoder(serverOut)
		var req jsonrpcRequest
		if err := dec.Decode(&req); err != nil {
			serverDone <- err
			return
		}
		if err := enc.Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      99,
			"method":  "ping",
			"params":  map[string]any{},
		}); err != nil {
			serverDone <- err
			return
		}
		var clientResp jsonrpcResponse
		if err := dec.Decode(&clientResp); err != nil {
			serverDone <- err
			return
		}
		if clientResp.ID != 99 || clientResp.Error != nil || string(clientResp.Result) != "{}" {
			serverDone <- fmt.Errorf("client response = %#v, want empty result response for ping request 99", clientResp)
			return
		}
		if err := enc.Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      req.ID,
			"result":  map[string]any{"ok": true},
		}); err != nil {
			serverDone <- err
			return
		}
		serverDone <- nil
	}()

	var result map[string]bool
	if err := rpc.call(context.Background(), "ping", nil, &result); err != nil {
		t.Fatalf("call() err = %v", err)
	}
	if !result["ok"] {
		t.Fatalf("result = %#v, want ok", result)
	}
	if err := <-serverDone; err != nil {
		t.Fatalf("server err = %v", err)
	}
}

func TestRPCConnCallInvokesCancelHook(t *testing.T) {
	rpc, serverIn, serverOut := newTestRPCConn()
	defer serverIn.Close()
	defer serverOut.Close()
	canceled := make(chan struct{}, 1)
	rpc.onCancel = func() {
		canceled <- struct{}{}
		_ = serverOut.Close()
	}
	go func() {
		var req jsonrpcRequest
		_ = json.NewDecoder(serverIn).Decode(&req)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err := rpc.call(ctx, "ping", nil, nil)
	if err == nil {
		t.Fatal("call() err = nil, want timeout")
	}
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("cancel hook was not invoked")
	}
}

func newTestRPCConn() (*rpcConn, io.ReadCloser, io.WriteCloser) {
	serverIn, clientOut := io.Pipe()
	clientIn, serverOut := io.Pipe()
	return newRPCConn(clientOut, clientIn, nil), serverIn, serverOut
}
