package pathpolicy

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func NormalizeInputPath(input string) string {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return ""
	}
	if strings.HasPrefix(filepath.ToSlash(trimmed), "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, filepath.FromSlash(strings.TrimPrefix(filepath.ToSlash(trimmed), "~/")))
		}
	}
	if runtime.GOOS == "windows" {
		if converted, ok := normalizeMSYSPath(trimmed); ok {
			return filepath.Clean(converted)
		}
	}
	return filepath.Clean(trimmed)
}

func normalizeMSYSPath(input string) (string, bool) {
	if len(input) < 3 {
		return "", false
	}
	if input[0] != '/' && input[0] != '\\' {
		return "", false
	}
	drive := input[1]
	if !((drive >= 'a' && drive <= 'z') || (drive >= 'A' && drive <= 'Z')) {
		return "", false
	}
	if input[2] != '/' && input[2] != '\\' {
		return "", false
	}
	rest := strings.TrimLeft(input[3:], `/\`)
	return strings.ToUpper(string(drive)) + `:\` + filepath.FromSlash(strings.ReplaceAll(rest, `\`, `/`)), true
}
