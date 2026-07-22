package update

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInstallRejectsTamperedBinaryBeforeReplacingTarget(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	target := BinaryPath()
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(target, []byte("current binary"), 0755); err != nil {
		t.Fatalf("WriteFile(target) error = %v", err)
	}

	downloaded := filepath.Join(t.TempDir(), "downloaded-mscli")
	if err := os.WriteFile(downloaded, []byte("tampered binary"), 0600); err != nil {
		t.Fatalf("WriteFile(downloaded) error = %v", err)
	}
	expected := sha256.Sum256([]byte("authentic binary"))

	err := Install(downloaded, hex.EncodeToString(expected[:]))
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("Install() error = %v, want checksum mismatch", err)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("ReadFile(target) error = %v", err)
	}
	if string(got) != "current binary" {
		t.Fatalf("target contents = %q, want original binary", got)
	}
	if _, err := os.Stat(downloaded); err != nil {
		t.Fatalf("downloaded binary changed before verification completed: %v", err)
	}
}

func TestInstallReplacesTargetAfterChecksumVerification(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	target := BinaryPath()
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(target, []byte("current binary"), 0755); err != nil {
		t.Fatalf("WriteFile(target) error = %v", err)
	}

	contents := []byte("authentic binary")
	downloaded := filepath.Join(t.TempDir(), "downloaded-mscli")
	if err := os.WriteFile(downloaded, contents, 0600); err != nil {
		t.Fatalf("WriteFile(downloaded) error = %v", err)
	}
	expected := sha256.Sum256(contents)

	if err := Install(downloaded, hex.EncodeToString(expected[:])); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("ReadFile(target) error = %v", err)
	}
	if string(got) != string(contents) {
		t.Fatalf("target contents = %q, want verified binary", got)
	}
}
