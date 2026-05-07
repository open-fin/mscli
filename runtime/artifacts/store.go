package artifacts

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var safeNamePattern = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

type Store struct {
	HomeDir       string
	WorkspaceRoot string
}

type WriteRequest struct {
	Prefix      string
	Name        string
	ContentType string
	Data        []byte
}

type Artifact struct {
	Path         string
	RelativePath string
	Size         int64
	ContentType  string
}

func (s Store) Write(req WriteRequest) (Artifact, error) {
	home := strings.TrimSpace(s.HomeDir)
	if home == "" {
		var err error
		home, err = os.UserHomeDir()
		if err != nil {
			return Artifact{}, fmt.Errorf("resolve home directory: %w", err)
		}
	}
	workspace := strings.TrimSpace(s.WorkspaceRoot)
	if workspace == "" {
		workspace = "."
	}
	workspaceAbs, err := filepath.Abs(workspace)
	if err != nil {
		return Artifact{}, fmt.Errorf("resolve workspace path: %w", err)
	}

	relativeDir := filepath.Join("projects", WorkspaceKey(workspaceAbs), "tool-results")
	dir := filepath.Join(home, ".mscli", relativeDir)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return Artifact{}, fmt.Errorf("create tool-results directory: %w", err)
	}

	contentType := strings.TrimSpace(req.ContentType)
	path, name, err := writeArtifactFile(dir, req.Prefix, req.Name, contentType, req.Data)
	if err != nil {
		return Artifact{}, fmt.Errorf("write tool result: %w", err)
	}

	return Artifact{
		Path:         path,
		RelativePath: filepath.Join(relativeDir, name),
		Size:         int64(len(req.Data)),
		ContentType:  contentType,
	}, nil
}

func WorkspaceKey(workspace string) string {
	key := strings.ReplaceAll(workspace, string(filepath.Separator), "-")
	key = strings.ReplaceAll(key, ":", "-")
	key = strings.Trim(key, "-")
	if key == "" {
		return "workspace"
	}
	return key
}

func writeArtifactFile(dir, prefix, name, contentType string, data []byte) (string, string, error) {
	pattern := artifactFilePattern(prefix, name, contentType)
	file, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return "", "", err
	}
	path := file.Name()
	if _, err := file.Write(data); err != nil {
		_ = file.Close()
		_ = os.Remove(path)
		return "", "", err
	}
	if err := file.Chmod(0644); err != nil {
		_ = file.Close()
		_ = os.Remove(path)
		return "", "", err
	}
	if err := file.Close(); err != nil {
		_ = os.Remove(path)
		return "", "", err
	}
	return path, filepath.Base(path), nil
}

func artifactFilePattern(prefix, name, contentType string) string {
	prefix = safeName(prefix)
	name = safeName(name)
	return fmt.Sprintf("%s-%s-*%s", prefix, name, extensionForContentType(contentType))
}

func extensionForContentType(contentType string) string {
	switch strings.ToLower(strings.TrimSpace(contentType)) {
	case "application/json":
		return ".json"
	default:
		return ".txt"
	}
}

func safeName(name string) string {
	name = safeNamePattern.ReplaceAllString(strings.TrimSpace(name), "_")
	name = strings.Trim(name, "_")
	if name == "" {
		return "tool"
	}
	return name
}
