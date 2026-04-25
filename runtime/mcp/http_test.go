package mcp

import (
	"strings"
	"testing"
)

func TestDecodeSSERPCResponse(t *testing.T) {
	resp, err := decodeSSERPCResponse(strings.NewReader("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"ok\":true}}\n\n"), 2)
	if err != nil {
		t.Fatalf("decodeSSERPCResponse err = %v", err)
	}
	if got := string(resp.Result); got != `{"ok":true}` {
		t.Fatalf("result = %s, want ok object", got)
	}
}
