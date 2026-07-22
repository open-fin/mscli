package update

// Manifest represents the release manifest hosted alongside binaries.
type Manifest struct {
	Latest       string            `json:"latest"`
	MinAllowed   string            `json:"min_allowed"`
	DownloadBase string            `json:"download_base"`
	Checksums    map[string]string `json:"checksums"`
}

// CheckResult holds the outcome of an update check.
type CheckResult struct {
	CurrentVersion  string
	LatestVersion   string
	UpdateAvailable bool
	ForceUpdate     bool
	DownloadURL     string
	SHA256          string
	ReleaseNotes    string
}
