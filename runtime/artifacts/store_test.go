package artifacts

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestStoreWriteScopesArtifactsByWorkspaceAndSanitizesName(t *testing.T) {
	home := t.TempDir()
	workspace := filepath.Join(t.TempDir(), "my:workspace")
	store := Store{HomeDir: home, WorkspaceRoot: workspace}

	artifact, err := store.Write(WriteRequest{
		Prefix:      "mcp",
		Name:        "server/tool result?.txt",
		ContentType: "text/plain",
		Data:        []byte("hello"),
	})
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	wantDir := filepath.Join(home, ".mscli", "projects", WorkspaceKey(workspace), "tool-results")
	if !strings.HasPrefix(artifact.Path, wantDir+string(filepath.Separator)) {
		t.Fatalf("artifact path = %q, want under %q", artifact.Path, wantDir)
	}
	if strings.Contains(filepath.Base(artifact.Path), "/") || strings.Contains(filepath.Base(artifact.Path), "?") {
		t.Fatalf("artifact filename not sanitized: %q", filepath.Base(artifact.Path))
	}
	if got, want := artifact.RelativePath, filepath.Join("projects", WorkspaceKey(workspace), "tool-results", filepath.Base(artifact.Path)); got != want {
		t.Fatalf("RelativePath = %q, want %q", got, want)
	}
	if got := artifact.Size; got != int64(len("hello")) {
		t.Fatalf("Size = %d, want %d", got, len("hello"))
	}
	if got := artifact.ContentType; got != "text/plain" {
		t.Fatalf("ContentType = %q, want text/plain", got)
	}
}

func TestWorkspaceKeyPreservesCurrentMCPAlgorithm(t *testing.T) {
	got := WorkspaceKey(filepath.Join(string(filepath.Separator), "tmp", "a:b", "workspace"))
	if strings.Contains(got, string(filepath.Separator)) || strings.Contains(got, ":") {
		t.Fatalf("WorkspaceKey contains raw separator or colon: %q", got)
	}
	if got == "" || got == "workspace" {
		t.Fatalf("WorkspaceKey = %q, want path-derived key", got)
	}
	if WorkspaceKey("") != "workspace" {
		t.Fatalf("WorkspaceKey(empty) = %q, want workspace", WorkspaceKey(""))
	}
}
