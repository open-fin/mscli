package mcp

import "testing"

func TestNormalizeNameKeepsProviderSafeCharacters(t *testing.T) {
	tests := map[string]string{
		"puppeteer":      "puppeteer",
		"git-server":     "git-server",
		"train_tools":    "train_tools",
		"GitHub MCP":     "GitHub_MCP",
		"docs/search.v1": "docs_search_v1",
		"  ":             "",
	}
	for in, want := range tests {
		if got := NormalizeName(in); got != want {
			t.Fatalf("NormalizeName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBuildToolName(t *testing.T) {
	got := BuildToolName("GitHub MCP", "search.code")
	want := "mcp__GitHub_MCP__search_code"
	if got != want {
		t.Fatalf("BuildToolName() = %q, want %q", got, want)
	}
}
