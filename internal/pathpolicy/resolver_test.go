package pathpolicy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"gitcode.com/mindspore/mscli/tools"
)

func TestResolveReadablePathAllowsWorkspaceAndExternalReadRoot(t *testing.T) {
	workDir := t.TempDir()
	external := t.TempDir()
	resolver := NewResolver(NewPathPolicy(workDir, []string{external}, nil))

	got, denial, err := resolver.ResolveReadablePath("local.txt", ResolveOptions{})
	if err != nil || denial != nil {
		t.Fatalf("workspace readable err=%v denial=%v", err, denial)
	}
	if want := filepath.Join(workDir, "local.txt"); got != want {
		t.Fatalf("workspace path = %q, want %q", got, want)
	}

	got, denial, err = resolver.ResolveReadablePath(filepath.Join(external, "file.txt"), ResolveOptions{})
	if err != nil || denial != nil {
		t.Fatalf("external readable err=%v denial=%v", err, denial)
	}
	if want := filepath.Join(external, "file.txt"); got != want {
		t.Fatalf("external path = %q, want %q", got, want)
	}
}

func TestResolveReadablePathReturnsDenialForUnauthorizedExternalPath(t *testing.T) {
	workDir := t.TempDir()
	external := filepath.Join(t.TempDir(), "file.txt")
	resolver := NewResolver(NewPathPolicy(workDir, nil, nil))

	got, denial, err := resolver.ResolveReadablePath(external, ResolveOptions{})
	if err != nil {
		t.Fatalf("ResolveReadablePath error = %v", err)
	}
	if got != "" {
		t.Fatalf("resolved path = %q, want empty", got)
	}
	if denial == nil || denial.Kind != string(DenialKindExternalRead) {
		t.Fatalf("denial = %#v, want external read denial", denial)
	}
}

func TestResolveReadablePathAllowsTemporaryReadRootFromOptions(t *testing.T) {
	workDir := t.TempDir()
	external := t.TempDir()
	resolver := NewResolver(NewPathPolicy(workDir, nil, nil))

	got, denial, err := resolver.ResolveReadablePath(filepath.Join(external, "file.txt"), ResolveOptions{TemporaryReadRoots: []string{external}})
	if err != nil || denial != nil {
		t.Fatalf("temporary external readable err=%v denial=%v", err, denial)
	}
	if want := filepath.Join(external, "file.txt"); got != want {
		t.Fatalf("temporary external path = %q, want %q", got, want)
	}
}
func TestResolveWritablePathAllowsExternalWriteRootAndWriteRootAllowsRead(t *testing.T) {
	workDir := t.TempDir()
	external := t.TempDir()
	resolver := NewResolver(NewPathPolicyWithWriteRoots(workDir, nil, []string{external}, nil))

	writePath := filepath.Join(external, "file.txt")
	got, denial, err := resolver.ResolveWritablePath(writePath, ResolveOptions{})
	if err != nil || denial != nil {
		t.Fatalf("external writable err=%v denial=%v", err, denial)
	}
	if got != writePath {
		t.Fatalf("external writable path = %q, want %q", got, writePath)
	}

	got, denial, err = resolver.ResolveReadablePath(writePath, ResolveOptions{})
	if err != nil || denial != nil {
		t.Fatalf("write root readable err=%v denial=%v", err, denial)
	}
	if got != writePath {
		t.Fatalf("write root readable path = %q, want %q", got, writePath)
	}
}

func TestResolveWritablePathRejectsExternalReadRoot(t *testing.T) {
	workDir := t.TempDir()
	external := t.TempDir()
	resolver := NewResolver(NewPathPolicy(workDir, []string{external}, nil))

	_, denial, err := resolver.ResolveWritablePath(filepath.Join(external, "file.txt"), ResolveOptions{})
	if err != nil {
		t.Fatalf("ResolveWritablePath error = %v", err)
	}
	if denial == nil || denial.Kind != string(DenialKindExternalWrite) {
		t.Fatalf("denial = %#v, want external write denial", denial)
	}
}

func TestResolveReadablePathRejectsWorkspaceSymlinkEscape(t *testing.T) {
	workDir := t.TempDir()
	external := t.TempDir()
	if err := os.WriteFile(filepath.Join(external, "outside.txt"), []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(workDir, "link")
	if err := os.Symlink(external, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	resolver := NewResolver(NewPathPolicy(workDir, nil, nil))

	got, denial, err := resolver.ResolveReadablePath(filepath.Join("link", "outside.txt"), ResolveOptions{})
	if err != nil {
		t.Fatalf("ResolveReadablePath error = %v", err)
	}
	if got != "" || denial == nil || denial.Kind != string(DenialKindExternalRead) {
		t.Fatalf("symlink escape got=%q denial=%#v, want external read denial", got, denial)
	}
}

func TestResolveWritablePathRejectsWorkspaceSymlinkEscape(t *testing.T) {
	workDir := t.TempDir()
	external := t.TempDir()
	link := filepath.Join(workDir, "link")
	if err := os.Symlink(external, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	resolver := NewResolver(NewPathPolicy(workDir, nil, nil))

	got, denial, err := resolver.ResolveWritablePath(filepath.Join("link", "new.txt"), ResolveOptions{})
	if err != nil {
		t.Fatalf("ResolveWritablePath error = %v", err)
	}
	if got != "" || denial == nil || denial.Kind != string(DenialKindExternalWrite) {
		t.Fatalf("symlink escape got=%q denial=%#v, want external write denial", got, denial)
	}
}

func TestResolveReadablePathRejectsExternalRootSymlinkEscape(t *testing.T) {
	workDir := t.TempDir()
	allowed := t.TempDir()
	external := t.TempDir()
	if err := os.WriteFile(filepath.Join(external, "outside.txt"), []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(allowed, "link")
	if err := os.Symlink(external, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	resolver := NewResolver(NewPathPolicy(workDir, []string{allowed}, nil))

	got, denial, err := resolver.ResolveReadablePath(filepath.Join(link, "outside.txt"), ResolveOptions{})
	if err != nil {
		t.Fatalf("ResolveReadablePath error = %v", err)
	}
	if got != "" || denial == nil || denial.Kind != string(DenialKindExternalRead) {
		t.Fatalf("external root symlink escape got=%q denial=%#v, want external read denial", got, denial)
	}
}

func TestResolveReadablePathRejectsRelativeEscapeWithoutAuthorization(t *testing.T) {
	workDir := t.TempDir()
	resolver := NewResolver(NewPathPolicy(workDir, nil, nil))

	got, denial, err := resolver.ResolveReadablePath("../../outside.txt", ResolveOptions{})
	if err == nil {
		t.Fatal("ResolveReadablePath error = nil, want relative escape error")
	}
	if got != "" || denial != nil {
		t.Fatalf("relative escape got=%q denial=%#v, want direct error without denial", got, denial)
	}
}

func TestExtractPathDenialSupportsPointerStructAndMap(t *testing.T) {
	denial := &PathDenial{Kind: string(DenialKindExternalRead), Operation: "read", InputPath: "/x"}
	got, ok := ExtractPathDenial(NewPathDenialResult(denial))
	if !ok || got.Kind != denial.Kind {
		t.Fatalf("ExtractPathDenial pointer got=%#v ok=%v", got, ok)
	}

	data, err := json.Marshal(denial)
	if err != nil {
		t.Fatal(err)
	}
	var asMap map[string]any
	if err := json.Unmarshal(data, &asMap); err != nil {
		t.Fatal(err)
	}
	got, ok = ExtractPathDenial(&tools.Result{Meta: map[string]any{PathDenialMetaKey: asMap}})
	if !ok || got.InputPath != denial.InputPath {
		t.Fatalf("ExtractPathDenial map got=%#v ok=%v", got, ok)
	}

	if got, ok := ExtractPathDenial(nil); ok || got != nil {
		t.Fatalf("ExtractPathDenial nil got=%#v ok=%v", got, ok)
	}
}
