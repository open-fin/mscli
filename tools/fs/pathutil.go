package fs

import (
	"fmt"
	"strings"

	"gitcode.com/mindspore/mscli/internal/pathpolicy"
)

func resolveSafePath(workDir, input string) (string, error) {
	resolved, denial, err := newWorkspaceResolver(workDir).ResolveReadablePath(input, pathpolicy.ResolveOptions{})
	if err != nil {
		return "", err
	}
	if denial != nil {
		return "", fmt.Errorf("absolute paths are not allowed: %s", input)
	}
	return resolved, nil
}

// ResolveSafePath exposes the same path validation used by filesystem tools.
func ResolveSafePath(workDir, input string) (string, error) {
	return resolveSafePath(workDir, input)
}

func isIgnoredGitName(name string) bool {
	return pathpolicy.IsIgnoredGitName(name)
}

func isIgnoredGitPath(path string) bool {
	cleaned := strings.ReplaceAll(path, `\\`, `/`)
	for _, part := range strings.Split(cleaned, "/") {
		if part == ".git" {
			return true
		}
	}
	return false
}
