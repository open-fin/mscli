package pathpolicy

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	RootKindGitRepo      = "git_repo"
	RootKindParentDir    = "parent_dir"
	RootKindRequestedDir = "requested_dir"
	RootKindNone         = "none"
)

type SuggestedRoot struct {
	Path string
	Kind string
}

func SuggestRoot(path string) SuggestedRoot {
	cleaned := filepath.Clean(path)
	if root := nearestGitRoot(cleaned); root != "" && !isOverbroadRoot(root) {
		return SuggestedRoot{Path: root, Kind: RootKindGitRepo}
	}
	if info, err := os.Stat(cleaned); err == nil {
		if info.IsDir() {
			if !isOverbroadRoot(cleaned) {
				return SuggestedRoot{Path: cleaned, Kind: RootKindRequestedDir}
			}
			return SuggestedRoot{Kind: RootKindNone}
		}
		parent := filepath.Dir(cleaned)
		if !isOverbroadRoot(parent) {
			return SuggestedRoot{Path: parent, Kind: RootKindParentDir}
		}
		return SuggestedRoot{Kind: RootKindNone}
	}
	parent := filepath.Dir(cleaned)
	if !isOverbroadRoot(parent) {
		return SuggestedRoot{Path: parent, Kind: RootKindParentDir}
	}
	return SuggestedRoot{Kind: RootKindNone}
}

func nearestGitRoot(path string) string {
	candidate := path
	if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
		candidate = filepath.Dir(candidate)
	}
	for {
		if candidate == "." || candidate == "" {
			return ""
		}
		if info, err := os.Stat(filepath.Join(candidate, ".git")); err == nil && (info.IsDir() || info.Mode().IsRegular()) {
			return candidate
		}
		parent := filepath.Dir(candidate)
		if parent == candidate {
			return ""
		}
		candidate = parent
	}
}

func isOverbroadRoot(path string) bool {
	cleaned := filepath.Clean(strings.TrimSpace(path))
	if cleaned == "" || cleaned == "." {
		return true
	}
	volume := filepath.VolumeName(cleaned)
	if volume != "" && samePath(cleaned, volume+string(filepath.Separator)) {
		return true
	}
	if cleaned == string(filepath.Separator) {
		return true
	}
	slash := filepath.ToSlash(cleaned)
	if slash == "/home" || slash == "/Users" {
		return true
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" && samePath(cleaned, home) {
		return true
	}
	return false
}
