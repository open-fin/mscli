package update

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestManifestURLsEmptyByDefault(t *testing.T) {
	t.Setenv("MSCLI_MANIFEST_URL", "")

	got := ManifestURLs()
	if len(got) != 0 {
		t.Fatalf("ManifestURLs() len = %d, want 0", len(got))
	}
}

func TestManifestURLsEnvOverride(t *testing.T) {
	t.Setenv("MSCLI_MANIFEST_URL", "http://example.test/manifest.json")

	got := ManifestURLs()
	if len(got) != 1 {
		t.Fatalf("ManifestURLs() len = %d, want 1", len(got))
	}
	if got[0] != "http://example.test/manifest.json" {
		t.Fatalf("ManifestURLs()[0] = %q, want env override", got[0])
	}
}

func TestFetchManifestFallsBackToNextURL(t *testing.T) {
	success := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"latest":"0.5.0-beta.2","min_allowed":"","download_base":"http://mirror.example/mscli/releases"}`))
	}))
	defer success.Close()

	got, err := fetchManifestFromURLs(context.Background(), []string{
		"http://127.0.0.1:1/latest/manifest.json",
		success.URL,
	})
	if err != nil {
		t.Fatalf("fetchManifestFromURLs() error = %v", err)
	}
	if got.Latest != "0.5.0-beta.2" {
		t.Fatalf("manifest latest = %q, want %q", got.Latest, "0.5.0-beta.2")
	}
	if got.DownloadBase != "http://mirror.example/mscli/releases" {
		t.Fatalf("manifest download_base = %q", got.DownloadBase)
	}
}

func TestCompareSemverPrereleaseOrdering(t *testing.T) {
	tests := []struct {
		name string
		a    string
		b    string
		want int
	}{
		{name: "beta increments", a: "0.5.0-beta.2", b: "0.5.0-beta.3", want: -1},
		{name: "beta before rc", a: "0.5.0-beta.3", b: "0.5.0-rc.1", want: -1},
		{name: "rc before stable", a: "0.5.0-rc.1", b: "0.5.0", want: -1},
		{name: "stable after prerelease", a: "0.5.0", b: "0.5.0-rc.1", want: 1},
		{name: "patch beats prerelease", a: "0.5.1", b: "0.5.0-rc.9", want: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := compareSemver(tt.a, tt.b); got != tt.want {
				t.Fatalf("compareSemver(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestCheckDetectsPrereleaseUpdate(t *testing.T) {
	checksum := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	success := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"latest":"0.5.0-beta.3","min_allowed":"","download_base":"https://mirror.example/mscli/releases","checksums":{"%s":"%s"}}`, binaryAssetName(), checksum)
	}))
	defer success.Close()

	t.Setenv("MSCLI_MANIFEST_URL", success.URL)

	result, err := Check(context.Background(), "0.5.0-beta.2")
	if err != nil {
		t.Fatalf("Check() error = %v", err)
	}
	if result == nil {
		t.Fatalf("Check() returned nil result")
	}
	if !result.UpdateAvailable {
		t.Fatalf("Check() UpdateAvailable = false, want true")
	}
	if result.LatestVersion != "0.5.0-beta.3" {
		t.Fatalf("Check() LatestVersion = %q", result.LatestVersion)
	}
	if result.DownloadURL != "https://mirror.example/mscli/releases/v0.5.0-beta.3/"+binaryAssetName() {
		t.Fatalf("Check() DownloadURL = %q", result.DownloadURL)
	}
	if result.SHA256 != checksum {
		t.Fatalf("Check() SHA256 = %q, want %q", result.SHA256, checksum)
	}
}

func TestCheckRejectsUpdateWithoutPlatformChecksum(t *testing.T) {
	success := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"latest":"1.0.0","download_base":"https://mirror.example/mscli/releases","checksums":{}}`))
	}))
	defer success.Close()

	t.Setenv("MSCLI_MANIFEST_URL", success.URL)
	if _, err := Check(context.Background(), "0.9.0"); err == nil {
		t.Fatal("Check() error = nil, want missing checksum error")
	}
}
