package loop

import (
	"strings"
	"testing"
)

func TestDefaultSystemPromptIncludesWriteArgumentValidationRules(t *testing.T) {
	prompt := DefaultSystemPrompt()

	if !strings.Contains(prompt, `verify arguments contain BOTH "path" and "content"`) {
		t.Fatalf("DefaultSystemPrompt() missing write arg validation rule: %q", prompt)
	}
	if !strings.Contains(prompt, "Never call write with empty JSON arguments ({})") {
		t.Fatalf("DefaultSystemPrompt() missing empty-args guard: %q", prompt)
	}
	if !strings.Contains(prompt, "Additional MCP tools may be available with names beginning mcp__") {
		t.Fatalf("DefaultSystemPrompt() missing MCP tool guidance: %q", prompt)
	}
}
