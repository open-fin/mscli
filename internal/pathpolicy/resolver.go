package pathpolicy

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Resolver struct {
	policy *PathPolicy
}

type ResolveOptions struct {
	TemporaryReadRoots  []string
	TemporaryWriteRoots []string
}

type resolveOptionsContextKey struct{}

func ContextWithResolveOptions(ctx context.Context, opts ResolveOptions) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, resolveOptionsContextKey{}, opts)
}

func ResolveOptionsFromContext(ctx context.Context) ResolveOptions {
	if ctx == nil {
		return ResolveOptions{}
	}
	if opts, ok := ctx.Value(resolveOptionsContextKey{}).(ResolveOptions); ok {
		return opts
	}
	return ResolveOptions{}
}

func NewResolver(policy *PathPolicy) *Resolver {
	return &Resolver{policy: policy}
}

func (r *Resolver) WorkDir() string {
	if r == nil || r.policy == nil {
		return ""
	}
	return r.policy.WorkDir
}

func (r *Resolver) ResolveReadablePath(input string, opts ResolveOptions) (string, *PathDenial, error) {
	return r.resolvePath("read", string(DenialKindExternalRead), input, opts, true)
}

func (r *Resolver) ResolveReadablePathForOperation(operation, input string, opts ResolveOptions) (string, *PathDenial, error) {
	return r.resolvePath(operation, string(DenialKindExternalRead), input, opts, true)
}

func (r *Resolver) ResolveWritablePath(input string, opts ResolveOptions) (string, *PathDenial, error) {
	return r.resolvePath("write", string(DenialKindExternalWrite), input, opts, false)
}

func (r *Resolver) ResolveWritablePathForOperation(operation, input string, opts ResolveOptions) (string, *PathDenial, error) {
	return r.resolvePath(operation, string(DenialKindExternalWrite), input, opts, false)
}

func (r *Resolver) resolvePath(operation, denialKind, input string, opts ResolveOptions, readable bool) (string, *PathDenial, error) {
	if strings.TrimSpace(input) == "" {
		return "", nil, fmt.Errorf("path is required")
	}
	if r == nil || r.policy == nil {
		return "", nil, fmt.Errorf("path policy is not configured")
	}

	workDir, err := filepath.Abs(NormalizeInputPath(r.policy.WorkDir))
	if err != nil {
		return "", nil, fmt.Errorf("resolve working directory: %w", err)
	}
	cleaned := NormalizeInputPath(input)

	inputWasAbs := filepath.IsAbs(cleaned) || filepath.IsAbs(input) || filepath.IsAbs(filepath.FromSlash(input))
	var fullAbs string
	if inputWasAbs {
		fullAbs, err = filepath.Abs(cleaned)
	} else {
		fullAbs, err = filepath.Abs(filepath.Join(workDir, cleaned))
	}
	if err != nil {
		return "", nil, fmt.Errorf("resolve path: %w", err)
	}

	if isIgnoredGitPath(fullAbs) {
		return "", nil, fmt.Errorf("path is ignored: %s", input)
	}
	if !inputWasAbs && pathEscapesLexically(cleaned) {
		return "", nil, fmt.Errorf("path escapes working directory: %s", input)
	}
	if allowed, err := canonicalPathWithinBase(workDir, fullAbs, readable); err != nil {
		return "", nil, err
	} else if allowed {
		return fullAbs, nil, nil
	}
	if !inputWasAbs {
		return "", nil, fmt.Errorf("path escapes working directory: %s", input)
	}

	if readable && r.allowedByRoots(fullAbs, opts.TemporaryReadRoots, true) {
		return fullAbs, nil, nil
	}
	if !readable && r.allowedByRoots(fullAbs, opts.TemporaryWriteRoots, false) {
		return fullAbs, nil, nil
	}

	suggestion := SuggestRoot(fullAbs)
	reason := "The path is outside the current workspace and is not in external_read_roots."
	if !readable {
		reason = "The path is outside the current workspace and is not in external_write_roots."
	}
	return "", &PathDenial{
		Kind:          denialKind,
		Operation:     operation,
		InputPath:     input,
		ResolvedPath:  fullAbs,
		WorkDir:       workDir,
		SuggestedRoot: suggestion.Path,
		RootKind:      suggestion.Kind,
		Reason:        reason,
	}, nil
}

func (r *Resolver) allowedByRoots(target string, temporaryRoots []string, readable bool) bool {
	for _, root := range temporaryRoots {
		if rootAllowsMode(root, target, readable) {
			return true
		}
	}
	if r == nil || r.policy == nil {
		return false
	}
	if readable {
		for _, entry := range r.policy.ReadRoots.Snapshot() {
			if rootAllowsMode(entry.Path, target, true) {
				return true
			}
		}
		for _, entry := range r.policy.WriteRoots.Snapshot() {
			if rootAllowsMode(entry.Path, target, true) {
				return true
			}
		}
		return false
	}
	for _, entry := range r.policy.WriteRoots.Snapshot() {
		if rootAllowsMode(entry.Path, target, false) {
			return true
		}
	}
	return false
}

func pathEscapesLexically(path string) bool {
	cleaned := filepath.Clean(path)
	return cleaned == ".." || strings.HasPrefix(filepath.ToSlash(cleaned), "../")
}

func rootAllowsMode(root, target string, readable bool) bool {
	cleanedRoot := NormalizeInputPath(root)
	if strings.TrimSpace(cleanedRoot) == "" {
		return false
	}
	rootAbs, err := filepath.Abs(cleanedRoot)
	if err != nil {
		return false
	}
	allowed, err := canonicalPathWithinBase(rootAbs, target, readable)
	return err == nil && allowed
}

func canonicalBasePath(base string) (string, error) {
	if real, err := filepath.EvalSymlinks(base); err == nil {
		return real, nil
	} else if !os.IsNotExist(err) {
		return "", err
	}
	return canonicalTargetPath(base, false)
}

func canonicalPathWithinBase(base, target string, targetMustExist bool) (bool, error) {
	baseAbs, err := filepath.Abs(base)
	if err != nil {
		return false, err
	}
	targetAbs, err := filepath.Abs(target)
	if err != nil {
		return false, err
	}
	baseReal, err := canonicalBasePath(baseAbs)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	targetReal, err := canonicalTargetPath(targetAbs, targetMustExist)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if isIgnoredGitPath(targetReal) {
		return false, fmt.Errorf("path is ignored: %s", target)
	}
	return pathWithinBase(baseReal, targetReal), nil
}

func canonicalTargetPath(target string, mustExist bool) (string, error) {
	if mustExist {
		if real, err := filepath.EvalSymlinks(target); err == nil {
			return real, nil
		} else if !os.IsNotExist(err) {
			return "", err
		}
	}
	if real, err := filepath.EvalSymlinks(target); err == nil {
		return real, nil
	} else if !os.IsNotExist(err) {
		return "", err
	}

	parent := target
	suffix := []string{}
	for {
		real, err := filepath.EvalSymlinks(parent)
		if err == nil {
			parts := append([]string{real}, reverseStrings(suffix)...)
			return filepath.Join(parts...), nil
		}
		if !os.IsNotExist(err) {
			return "", err
		}
		next := filepath.Dir(parent)
		if next == parent {
			return "", err
		}
		suffix = append(suffix, filepath.Base(parent))
		parent = next
	}
}

func reverseStrings(values []string) []string {
	out := make([]string, len(values))
	for i := range values {
		out[i] = values[len(values)-1-i]
	}
	return out
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

func IsIgnoredGitName(name string) bool {
	return name == ".git"
}
