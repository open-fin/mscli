package mcp

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestCanonicalConfigHashIgnoresMapOrder(t *testing.T) {
	a := ServerConfig{Raw: map[string]any{
		"type":    "stdio",
		"command": "npx",
		"args":    []any{"-y", "pkg"},
		"env":     map[string]any{"B": "2", "A": "1"},
	}}
	b := ServerConfig{Raw: map[string]any{
		"env":     map[string]any{"A": "1", "B": "2"},
		"args":    []any{"-y", "pkg"},
		"command": "npx",
		"type":    "stdio",
	}}
	if got, want := CanonicalConfigHash(a), CanonicalConfigHash(b); got != want {
		t.Fatalf("hashes differ: %s vs %s", got, want)
	}
}

func TestCanonicalConfigHashIncludesFutureFields(t *testing.T) {
	base := ServerConfig{Raw: map[string]any{
		"type":    "stdio",
		"command": "server",
	}}
	withFuture := ServerConfig{Raw: map[string]any{
		"type":    "stdio",
		"command": "server",
		"url":     "https://example.com/mcp",
	}}
	if got, want := CanonicalConfigHash(base), CanonicalConfigHash(withFuture); got == want {
		t.Fatalf("hashes equal despite future field: %s", got)
	}
}

func TestCanonicalConfigHashHasSHA256Prefix(t *testing.T) {
	got := CanonicalConfigHash(ServerConfig{Raw: map[string]any{"command": "server"}})
	if !strings.HasPrefix(got, "sha256:") {
		t.Fatalf("hash = %q, want sha256 prefix", got)
	}
}

func TestApprovalStoreRecordsApprovedAndRejected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp_approvals.json")
	store := NewApprovalStore(path)
	err := store.Record("workspace", "server", "sha256:abc", DecisionApproved)
	if err != nil {
		t.Fatalf("Record approved: %v", err)
	}
	decision, ok, err := store.Lookup("workspace", "server", "sha256:abc")
	if err != nil || !ok || decision != DecisionApproved {
		t.Fatalf("Lookup approved = %q, %v, %v", decision, ok, err)
	}
	err = store.Record("workspace", "server", "sha256:def", DecisionRejected)
	if err != nil {
		t.Fatalf("Record rejected: %v", err)
	}
	decision, ok, err = store.Lookup("workspace", "server", "sha256:def")
	if err != nil || !ok || decision != DecisionRejected {
		t.Fatalf("Lookup rejected = %q, %v, %v", decision, ok, err)
	}
}

func TestApprovalStoreReplacesExistingDecision(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp_approvals.json")
	store := NewApprovalStore(path)
	if err := store.Record("workspace", "server", "sha256:abc", DecisionRejected); err != nil {
		t.Fatalf("Record rejected: %v", err)
	}
	if err := store.Record("workspace", "server", "sha256:abc", DecisionApproved); err != nil {
		t.Fatalf("Record approved: %v", err)
	}
	decision, ok, err := store.Lookup("workspace", "server", "sha256:abc")
	if err != nil || !ok || decision != DecisionApproved {
		t.Fatalf("Lookup replaced = %q, %v, %v", decision, ok, err)
	}
}

func TestApprovalStoreMissingFileIsEmpty(t *testing.T) {
	store := NewApprovalStore(filepath.Join(t.TempDir(), "missing.json"))
	decision, ok, err := store.Lookup("workspace", "server", "sha256:abc")
	if err != nil {
		t.Fatalf("Lookup missing file err = %v", err)
	}
	if ok || decision != "" {
		t.Fatalf("Lookup missing file = %q, %v; want empty false", decision, ok)
	}
}

func TestApprovalStoreResetWorkspaceRemovesOnlyMatchingWorkspace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp_approvals.json")
	store := NewApprovalStore(path)
	if err := store.Record("/workspace/a", "one", "sha256:1", DecisionApproved); err != nil {
		t.Fatalf("Record a one: %v", err)
	}
	if err := store.Record("/workspace/a", "two", "sha256:2", DecisionRejected); err != nil {
		t.Fatalf("Record a two: %v", err)
	}
	if err := store.Record("/workspace/b", "one", "sha256:1", DecisionApproved); err != nil {
		t.Fatalf("Record b one: %v", err)
	}

	removed, err := store.ResetWorkspace("/workspace/a")
	if err != nil {
		t.Fatalf("ResetWorkspace() err = %v", err)
	}
	if removed != 2 {
		t.Fatalf("removed = %d, want 2", removed)
	}
	if _, ok, err := store.Lookup("/workspace/a", "one", "sha256:1"); err != nil || ok {
		t.Fatalf("lookup removed a/one ok=%v err=%v, want false nil", ok, err)
	}
	if decision, ok, err := store.Lookup("/workspace/b", "one", "sha256:1"); err != nil || !ok || decision != DecisionApproved {
		t.Fatalf("lookup b/one = %q %v %v, want approved true nil", decision, ok, err)
	}
}
