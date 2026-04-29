package app

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/skills"
)

const (
	memoryIndexFilename   = "MEMORY.md"
	autoMemoryIndexMaxLen = 20_000
)

type autoMemoryConfig struct {
	Resolved bool
	Enabled  bool
	Dir      string
}

type mscliInstructionFile struct {
	Path    string
	Content string
}

func buildBaseSystemPrompt(summaries []skills.SkillSummary) string {
	systemPrompt := loop.DefaultSystemPrompt()
	if len(summaries) == 0 {
		return systemPrompt
	}
	return systemPrompt + "\n\n## Available Skills\n\n" +
		"Use the load_skill tool to load a skill when the user's task matches one:\n\n" +
		skills.FormatSummaries(summaries)
}

func buildEffectiveSystemPrompt(workDir string, summaries []skills.SkillSummary) (string, autoMemoryConfig, error) {
	memory, err := resolveAutoMemoryConfig(workDir)
	if err != nil {
		return "", autoMemoryConfig{}, err
	}
	prompt, err := buildEffectiveSystemPromptWithMemory(workDir, summaries, memory)
	if err != nil {
		return "", autoMemoryConfig{}, err
	}
	return prompt, memory, nil
}

func buildEffectiveSystemPromptWithMemory(workDir string, summaries []skills.SkillSummary, memory autoMemoryConfig) (string, error) {
	parts := []string{buildBaseSystemPrompt(summaries)}

	if memory.Enabled {
		memoryPrompt, err := buildAutoMemoryPrompt(memory.Dir)
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(memoryPrompt) != "" {
			parts = append(parts, memoryPrompt)
		}
	}

	mscliPrompt, err := buildMSCLIInstructionsPrompt(workDir)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(mscliPrompt) != "" {
		parts = append(parts, mscliPrompt)
	}

	return strings.Join(parts, "\n\n"), nil
}

func (a *Application) rebuildSystemPrompt() (string, error) {
	if a == nil {
		return "", nil
	}
	memory, err := a.activeAutoMemoryConfig()
	if err != nil {
		return "", err
	}
	prompt, err := buildEffectiveSystemPromptWithMemory(a.WorkDir, a.currentSkillSummaries(), memory)
	if err != nil {
		return "", err
	}
	return prompt, nil
}

func (a *Application) activeAutoMemoryConfig() (autoMemoryConfig, error) {
	if a == nil {
		return autoMemoryConfig{Resolved: true, Enabled: false}, nil
	}
	if a.memoryConfig.Resolved {
		return a.memoryConfig, nil
	}
	memory, err := resolveAutoMemoryConfig(a.WorkDir)
	if err != nil {
		return autoMemoryConfig{}, err
	}
	a.memoryConfig = memory
	return memory, nil
}

func (a *Application) currentSkillSummaries() []skills.SkillSummary {
	if a == nil || a.skillLoader == nil {
		return nil
	}
	return a.skillLoader.List()
}

func resolveAutoMemoryConfig(workDir string) (autoMemoryConfig, error) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv("MSCLI_MEMORY_ENABLED")), "false") {
		return autoMemoryConfig{Resolved: true, Enabled: false}, nil
	}

	memoryDir := strings.TrimSpace(os.Getenv("MSCLI_MEMORY_PATH"))
	if memoryDir == "" {
		bucketDir, err := session.BucketDirForWorkDir(workDir)
		if err != nil {
			return autoMemoryConfig{}, err
		}
		memoryDir = filepath.Join(bucketDir, "memory")
	}

	expanded, err := expandPromptPath(memoryDir)
	if err != nil {
		return autoMemoryConfig{}, err
	}
	memoryDir = expanded
	if err := os.MkdirAll(memoryDir, 0o755); err != nil {
		return autoMemoryConfig{}, fmt.Errorf("create memory directory: %w", err)
	}

	return autoMemoryConfig{Resolved: true, Enabled: true, Dir: memoryDir}, nil
}

func buildAutoMemoryPrompt(memoryDir string) (string, error) {
	memoryDir = strings.TrimSpace(memoryDir)
	if memoryDir == "" {
		return "", nil
	}

	var b strings.Builder
	b.WriteString("## Persistent Memory\n\n")
	b.WriteString("A persistent memory directory is available at:\n")
	b.WriteString(memoryDir)
	b.WriteString("\n\n")
	b.WriteString("Use read, write, edit, grep, and glob on this directory to maintain durable context across sessions. ")
	b.WriteString("Keep ")
	b.WriteString(memoryIndexFilename)
	b.WriteString(" as the concise index. Proactively update memory when you learn stable user preferences, project facts, or workflow decisions that will help future sessions. ")
	b.WriteString("Do not store secrets, credentials, private keys, or short-lived conversation details.\n")

	indexPath := filepath.Join(memoryDir, memoryIndexFilename)
	data, err := os.ReadFile(indexPath)
	if err != nil {
		if os.IsNotExist(err) {
			b.WriteString("\nCurrent MEMORY.md: not found.")
			return b.String(), nil
		}
		return "", fmt.Errorf("read memory index: %w", err)
	}

	content, truncated := truncatePromptContent(string(data), autoMemoryIndexMaxLen)
	b.WriteString("\nCurrent MEMORY.md")
	if truncated {
		fmt.Fprintf(&b, " (truncated to %d characters)", autoMemoryIndexMaxLen)
	}
	b.WriteString(":\n\n```markdown\n")
	b.WriteString(content)
	if content != "" && !strings.HasSuffix(content, "\n") {
		b.WriteByte('\n')
	}
	b.WriteString("```")
	return b.String(), nil
}

func buildMSCLIInstructionsPrompt(workDir string) (string, error) {
	files, err := loadMSCLIInstructionFiles(workDir)
	if err != nil {
		return "", err
	}
	if len(files) == 0 {
		return "", nil
	}

	var b strings.Builder
	b.WriteString("## MSCLI.md Instructions\n\n")
	b.WriteString("The following instruction files were loaded in low-to-high priority order. Later files have higher priority when instructions conflict.\n")
	for _, file := range files {
		b.WriteString("\n### ")
		b.WriteString(file.Path)
		b.WriteString("\n\n```markdown\n")
		b.WriteString(file.Content)
		if file.Content != "" && !strings.HasSuffix(file.Content, "\n") {
			b.WriteByte('\n')
		}
		b.WriteString("```\n")
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

func loadMSCLIInstructionFiles(workDir string) ([]mscliInstructionFile, error) {
	absWorkDir, err := filepath.Abs(workDir)
	if err != nil {
		return nil, fmt.Errorf("resolve workdir: %w", err)
	}
	absWorkDir = filepath.Clean(absWorkDir)

	var candidates []string
	if homeDir, err := os.UserHomeDir(); err == nil && strings.TrimSpace(homeDir) != "" {
		candidates = append(candidates, filepath.Join(homeDir, ".mscli", "MSCLI.md"))
	}
	for _, dir := range ancestorDirs(absWorkDir) {
		candidates = append(candidates,
			filepath.Join(dir, "MSCLI.md"),
			filepath.Join(dir, ".mscli", "MSCLI.md"),
		)
	}

	seen := make(map[string]struct{}, len(candidates))
	files := make([]mscliInstructionFile, 0, len(candidates))
	for _, candidate := range candidates {
		path, err := filepath.Abs(candidate)
		if err != nil {
			return nil, fmt.Errorf("resolve instruction path: %w", err)
		}
		path = filepath.Clean(path)
		if _, exists := seen[path]; exists {
			continue
		}
		seen[path] = struct{}{}

		info, err := os.Stat(path)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("stat instruction file %s: %w", path, err)
		}
		if info.IsDir() {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read instruction file %s: %w", path, err)
		}
		files = append(files, mscliInstructionFile{Path: path, Content: string(data)})
	}
	return files, nil
}

func ancestorDirs(path string) []string {
	path = filepath.Clean(path)
	var reversed []string
	for {
		reversed = append(reversed, path)
		parent := filepath.Dir(path)
		if parent == path {
			break
		}
		path = parent
	}

	dirs := make([]string, 0, len(reversed))
	for i := len(reversed) - 1; i >= 0; i-- {
		dirs = append(dirs, reversed[i])
	}
	return dirs
}

func expandPromptPath(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("path cannot be empty")
	}
	cleanedSlash := filepath.ToSlash(filepath.Clean(path))
	if cleanedSlash == "~" || strings.HasPrefix(cleanedSlash, "~/") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
		if cleanedSlash == "~" {
			path = homeDir
		} else {
			path = filepath.Join(homeDir, filepath.FromSlash(strings.TrimPrefix(cleanedSlash, "~/")))
		}
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	return filepath.Clean(abs), nil
}

func truncatePromptContent(content string, maxLen int) (string, bool) {
	if maxLen <= 0 {
		return "", content != ""
	}
	if utf8.RuneCountInString(content) <= maxLen {
		return content, false
	}

	var b strings.Builder
	b.Grow(maxLen)
	count := 0
	for _, r := range content {
		if count >= maxLen {
			break
		}
		b.WriteRune(r)
		count++
	}
	return b.String(), true
}
