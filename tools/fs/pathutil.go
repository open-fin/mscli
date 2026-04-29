package fs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var allowedAbsoluteHomePaths = []string{
	"~/.mscli/skills",
	"~/.mscli/mindspore-skills",
}

// PathOptions extends filesystem tool path access without changing the
// default workspace-only behavior.
type PathOptions struct {
	AllowedAbsoluteRoots []string
}

type resolvedPathOptions struct {
	allowedAbsoluteRoots []string
}

func resolveSafePath(workDir, input string) (string, error) {
	return resolveSafePathWithOptions(workDir, input, PathOptions{})
}

func resolveSafePathWithOptions(workDir, input string, opts PathOptions) (string, error) {
	if strings.TrimSpace(input) == "" {
		return "", fmt.Errorf("path is required")
	}

	baseAbs, err := filepath.Abs(workDir)
	if err != nil {
		return "", fmt.Errorf("resolve working directory: %w", err)
	}

	resolvedOpts, err := resolvePathOptions(opts)
	if err != nil {
		return "", err
	}

	cleaned := filepath.Clean(input)
	normalized, err := normalizeAllowedAbsolutePath(cleaned, resolvedOpts)
	if err != nil {
		return "", err
	}
	cleaned = normalized

	if filepath.IsAbs(cleaned) {
		absPath, err := filepath.Abs(cleaned)
		if err != nil {
			return "", fmt.Errorf("resolve path: %w", err)
		}
		if pathWithinBase(baseAbs, absPath) {
			if isIgnoredGitPath(absPath) {
				return "", fmt.Errorf("path is ignored: %s", input)
			}
			return absPath, nil
		}

		allowed, err := isAllowedAbsolutePath(absPath, resolvedOpts)
		if err != nil {
			return "", err
		}
		if !allowed {
			return "", fmt.Errorf("absolute paths are not allowed: %s", input)
		}
		if isIgnoredGitPath(absPath) {
			return "", fmt.Errorf("path is ignored: %s", input)
		}
		return absPath, nil
	}

	fullAbs, err := filepath.Abs(filepath.Join(baseAbs, cleaned))
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}

	rel, err := filepath.Rel(baseAbs, fullAbs)
	if err != nil {
		return "", fmt.Errorf("check path: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes working directory: %s", input)
	}
	if isIgnoredGitPath(rel) {
		return "", fmt.Errorf("path is ignored: %s", input)
	}

	return fullAbs, nil
}

// ResolveSafePath exposes the same path validation used by filesystem tools.
func ResolveSafePath(workDir, input string) (string, error) {
	return resolveSafePath(workDir, input)
}

// ResolveSafePathWithOptions exposes path validation with explicit additional
// absolute roots for app-owned tool wiring.
func ResolveSafePathWithOptions(workDir, input string, opts PathOptions) (string, error) {
	return resolveSafePathWithOptions(workDir, input, opts)
}

func normalizeAllowedAbsolutePath(input string, opts resolvedPathOptions) (string, error) {
	expanded, ok, err := expandHomePath(input)
	if err != nil {
		return "", err
	}
	if !ok {
		return input, nil
	}

	allowed, err := isAllowedAbsolutePath(expanded, opts)
	if err != nil {
		return "", err
	}
	if !allowed {
		return input, nil
	}

	return expanded, nil
}

func isAllowedAbsolutePath(input string, opts resolvedPathOptions) (bool, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return false, fmt.Errorf("resolve home directory: %w", err)
	}

	for _, allowedRoot := range allowedAbsoluteHomePaths {
		trimmed := strings.TrimPrefix(allowedRoot, "~/")
		base := filepath.Join(homeDir, filepath.FromSlash(trimmed))
		if pathWithinBase(base, input) {
			return true, nil
		}
	}

	for _, allowedRoot := range opts.allowedAbsoluteRoots {
		if pathWithinBase(allowedRoot, input) {
			return true, nil
		}
	}

	return false, nil
}

func resolvePathOptions(opts PathOptions) (resolvedPathOptions, error) {
	roots := make([]string, 0, len(opts.AllowedAbsoluteRoots))
	seen := make(map[string]struct{}, len(opts.AllowedAbsoluteRoots))
	for _, root := range opts.AllowedAbsoluteRoots {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		expanded, ok, err := expandHomePath(root)
		if err != nil {
			return resolvedPathOptions{}, err
		}
		if ok {
			root = expanded
		}
		if !filepath.IsAbs(root) {
			abs, err := filepath.Abs(root)
			if err != nil {
				return resolvedPathOptions{}, fmt.Errorf("resolve allowed absolute root: %w", err)
			}
			root = abs
		}
		root = filepath.Clean(root)
		if _, exists := seen[root]; exists {
			continue
		}
		seen[root] = struct{}{}
		roots = append(roots, root)
	}
	return resolvedPathOptions{allowedAbsoluteRoots: roots}, nil
}

func expandHomePath(input string) (string, bool, error) {
	cleanedSlash := filepath.ToSlash(filepath.Clean(input))
	if cleanedSlash != "~" && !strings.HasPrefix(cleanedSlash, "~/") {
		return input, false, nil
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", false, fmt.Errorf("resolve home directory: %w", err)
	}
	if cleanedSlash == "~" {
		return homeDir, true, nil
	}

	trimmed := strings.TrimPrefix(cleanedSlash, "~/")
	return filepath.Join(homeDir, filepath.FromSlash(trimmed)), true, nil
}

func pathWithinBase(base, target string) bool {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func isIgnoredGitPath(path string) bool {
	cleaned := filepath.ToSlash(filepath.Clean(path))
	for _, part := range strings.Split(cleaned, "/") {
		if part == ".git" {
			return true
		}
	}
	return false
}

func isIgnoredGitName(name string) bool {
	return name == ".git"
}
