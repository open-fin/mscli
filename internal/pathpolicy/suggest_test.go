package pathpolicy

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSuggestRootUsesGitDirectory(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "sub", "file.txt")

	got := SuggestRoot(path)
	if got.Kind != RootKindGitRepo || got.Path != root {
		t.Fatalf("SuggestRoot() = %#v, want git root %q", got, root)
	}
}

func TestSuggestRootUsesGitFile(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".git"), []byte("gitdir: ../repo.git\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "sub", "file.txt")

	got := SuggestRoot(path)
	if got.Kind != RootKindGitRepo || got.Path != root {
		t.Fatalf("SuggestRoot() = %#v, want git root %q", got, root)
	}
}
