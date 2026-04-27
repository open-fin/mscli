package mcp

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestDecodeHTTPRPCResponseJSON(t *testing.T) {
	resp, err := decodeHTTPRPCResponse(&http.Response{
		Header: http.Header{"Content-Type": []string{"application/json"}},
		Body:   io.NopCloser(strings.NewReader(`{"jsonrpc":"2.0","id":7,"result":{"ok":true}}`)),
	}, 7)
	if err != nil {
		t.Fatalf("decodeHTTPRPCResponse err = %v", err)
	}
	if got := string(resp.Result); got != `{"ok":true}` {
		t.Fatalf("result = %s, want ok object", got)
	}
}

func TestDecodeHTTPRPCResponseSSE(t *testing.T) {
	resp, err := decodeHTTPRPCResponse(&http.Response{
		Header: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(strings.NewReader(
			"event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":6,\"result\":{\"ignored\":true}}\n\n" +
				"event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\n\n",
		)),
	}, 7)
	if err != nil {
		t.Fatalf("decodeHTTPRPCResponse err = %v", err)
	}
	if got := string(resp.Result); got != `{"ok":true}` {
		t.Fatalf("result = %s, want ok object", got)
	}
}

func TestDecodeSSERPCResponse(t *testing.T) {
	resp, err := decodeSSERPCResponse(strings.NewReader("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"ok\":true}}\n\n"), 2)
	if err != nil {
		t.Fatalf("decodeSSERPCResponse err = %v", err)
	}
	if got := string(resp.Result); got != `{"ok":true}` {
		t.Fatalf("result = %s, want ok object", got)
	}
}

func TestDecodeSSERPCResponseIgnoresInitialEmptyPrimingEvent(t *testing.T) {
	resp, err := decodeSSERPCResponse(strings.NewReader("id: cursor-1\ndata:\n\n"+"event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"ok\":true}}\n\n"), 2)
	if err != nil {
		t.Fatalf("decodeSSERPCResponse err = %v", err)
	}
	if got := string(resp.Result); got != `{"ok":true}` {
		t.Fatalf("result = %s, want ok object", got)
	}
}
