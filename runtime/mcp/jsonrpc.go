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

type jsonrpcNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonrpcError   `json:"error,omitempty"`
}

type jsonrpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
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
		resp, err := c.readMatchingResponse(id)
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

func (c *rpcConn) readMatchingResponse(id int64) (jsonrpcResponse, error) {
	for {
		var msg jsonrpcMessage
		if err := c.dec.Decode(&msg); err != nil {
			return jsonrpcResponse{}, err
		}
		if msg.Method != "" {
			if len(msg.ID) > 0 {
				if err := c.respondToServerRequest(msg); err != nil {
					return jsonrpcResponse{}, fmt.Errorf("write mcp json-rpc response: %w", err)
				}
			}
			continue
		}
		msgID, ok := jsonrpcNumericID(msg.ID)
		if !ok || msgID != id {
			continue
		}
		return jsonrpcResponse{
			JSONRPC: msg.JSONRPC,
			ID:      msgID,
			Result:  msg.Result,
			Error:   msg.Error,
		}, nil
	}
}

func (c *rpcConn) respondToServerRequest(msg jsonrpcMessage) error {
	switch msg.Method {
	case "ping":
		return c.writeRequestResult(msg.ID, map[string]any{})
	default:
		return c.writeUnsupportedRequest(msg.ID)
	}
}

func (c *rpcConn) writeRequestResult(id json.RawMessage, result any) error {
	return c.enc.Encode(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Result  any             `json:"result"`
	}{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	})
}

func (c *rpcConn) writeUnsupportedRequest(id json.RawMessage) error {
	return c.enc.Encode(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Error   *jsonrpcError   `json:"error"`
	}{
		JSONRPC: "2.0",
		ID:      id,
		Error: &jsonrpcError{
			Code:    -32601,
			Message: "method not found",
		},
	})
}

func jsonrpcNumericID(raw json.RawMessage) (int64, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var id int64
	if err := json.Unmarshal(raw, &id); err != nil {
		return 0, false
	}
	return id, true
}

func (c *rpcConn) notify(ctx context.Context, method string, params any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.enc.Encode(jsonrpcNotification{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}); err != nil {
		return fmt.Errorf("write mcp json-rpc notification: %w", err)
	}
	return nil
}
