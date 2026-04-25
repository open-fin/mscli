package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

type rpcConn struct {
	enc      *json.Encoder
	dec      *json.Decoder
	mu       sync.Mutex
	nextID   atomic.Int64
	onCancel func()
}

type jsonrpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonrpcError   `json:"error,omitempty"`
}

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func newRPCConn(writer io.Writer, reader io.Reader, onCancel func()) *rpcConn {
	return &rpcConn{
		enc:      json.NewEncoder(writer),
		dec:      json.NewDecoder(reader),
		onCancel: onCancel,
	}
}

func (c *rpcConn) call(ctx context.Context, method string, params any, result any) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	id := c.nextID.Add(1)
	req := jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	if err := c.enc.Encode(req); err != nil {
		return fmt.Errorf("write mcp json-rpc request: %w", err)
	}

	type decodeResult struct {
		resp jsonrpcResponse
		err  error
	}
	done := make(chan decodeResult, 1)
	go func() {
		var resp jsonrpcResponse
		err := c.dec.Decode(&resp)
		done <- decodeResult{resp: resp, err: err}
	}()

	select {
	case <-ctx.Done():
		if c.onCancel != nil {
			c.onCancel()
		}
		return ctx.Err()
	case decoded := <-done:
		if decoded.err != nil {
			return fmt.Errorf("read mcp json-rpc response: %w", decoded.err)
		}
		if decoded.resp.ID != id {
			return fmt.Errorf("mcp json-rpc response id %d does not match request id %d", decoded.resp.ID, id)
		}
		if decoded.resp.Error != nil {
			return fmt.Errorf("mcp json-rpc error %d: %s", decoded.resp.Error.Code, decoded.resp.Error.Message)
		}
		if result != nil {
			if err := json.Unmarshal(decoded.resp.Result, result); err != nil {
				return fmt.Errorf("decode mcp json-rpc result: %w", err)
			}
		}
		return nil
	}
}
