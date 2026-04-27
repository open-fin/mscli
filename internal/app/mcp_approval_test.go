package app

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestTerminalMCPApprovalPromptShowsHTTPURL(t *testing.T) {
	var out bytes.Buffer
	prompter := newTerminalMCPApprovalPrompter(strings.NewReader("s\n"), &out)

	if _, err := prompter.PromptMCPApproval(context.Background(), MCPApprovalRequest{
		WorkspaceRoot: "/workspace",
		ServerName:    "remote",
		Transport:     "http",
		URL:           "https://example.test/mcp",
		ConfigHash:    "sha256:remote",
	}); err != nil {
		t.Fatalf("PromptMCPApproval() err = %v", err)
	}
	text := out.String()
	if !strings.Contains(text, "Transport: http") {
		t.Fatalf("prompt missing transport:\n%s", text)
	}
	if !strings.Contains(text, "URL: https://example.test/mcp") {
		t.Fatalf("prompt missing URL:\n%s", text)
	}
	if strings.Contains(text, "Command: \n") {
		t.Fatalf("prompt contains empty command for HTTP server:\n%s", text)
	}
}
