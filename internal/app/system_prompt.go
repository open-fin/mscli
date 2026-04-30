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
	initialUserContextTag = "## MSCLI Session Context"
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
	return buildEffectiveSystemPromptFromSummariesWithMemory(summaries, memory), memory, nil
}

func buildEffectiveSystemPromptFromSummaries(summaries []skills.SkillSummary) string {
	return buildEffectiveSystemPromptFromSummariesWithMemory(summaries, autoMemoryConfig{})
}

func buildEffectiveSystemPromptFromSummariesWithMemory(summaries []skills.SkillSummary, memory autoMemoryConfig) string {
	prompt := buildBaseSystemPrompt(summaries)
	if !memory.Enabled {
		return prompt
	}
	memoryPrompt := buildAutoMemoryInstructionsPrompt(memory.Dir)
	if strings.TrimSpace(memoryPrompt) == "" {
		return prompt
	}
	return prompt + "\n\n" + memoryPrompt
}

func (a *Application) rebuildSystemPrompt() (string, error) {
	if a == nil {
		return "", nil
	}
	memory, err := a.activeAutoMemoryConfig()
	if err != nil {
		return "", err
	}
	return buildEffectiveSystemPromptFromSummariesWithMemory(a.currentSkillSummaries(), memory), nil
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

func buildAutoMemoryInstructionsPrompt(memoryDir string) string {
	memoryDir = strings.TrimSpace(memoryDir)
	if memoryDir == "" {
		return ""
	}

	var b strings.Builder
	writeAutoMemoryInstructions(&b, memoryDir)
	return strings.TrimRight(b.String(), "\n")
}

func buildAutoMemoryContextPrompt(memoryDir string) (string, error) {
	memoryDir = strings.TrimSpace(memoryDir)
	if memoryDir == "" {
		return "", nil
	}
	indexPath := filepath.Join(memoryDir, memoryIndexFilename)
	data, err := os.ReadFile(indexPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read memory index: %w", err)
	}

	var b strings.Builder
	content, truncated := truncatePromptContent(string(data), autoMemoryIndexMaxLen)
	b.WriteString("## Auto Memory\n\n")
	b.WriteString("The following memory index was loaded from ")
	b.WriteString(indexPath)
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

func buildInitialUserContextPrompt(workDir string, memory autoMemoryConfig) (string, error) {
	var parts []string

	if memory.Enabled {
		memoryPrompt, err := buildAutoMemoryContextPrompt(memory.Dir)
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

func buildInitialUserMessage(workDir string, memory autoMemoryConfig, userInput string) (string, error) {
	contextPrompt, err := buildInitialUserContextPrompt(workDir, memory)
	if err != nil {
		return "", err
	}
	return injectInitialUserContext(userInput, contextPrompt), nil
}

func injectInitialUserContext(userInput, contextPrompt string) string {
	contextPrompt = strings.TrimSpace(contextPrompt)
	if contextPrompt == "" || strings.Contains(userInput, initialUserContextTag) {
		return userInput
	}

	var b strings.Builder
	b.WriteString(initialUserContextTag)
	b.WriteString("\n\n")
	b.WriteString("The following context was loaded by MSCLI before the user's first request. Use it as session context and instructions, not as a separate request.\n\n")
	b.WriteString(contextPrompt)
	b.WriteString("\n\n## User Request\n\n")
	b.WriteString(userInput)
	return b.String()
}

func writeAutoMemoryInstructions(b *strings.Builder, memoryDir string) {
	b.WriteString("# auto memory\n\n")
	b.WriteString("You have a persistent, file-based memory system at ")
	b.WriteString(memoryDir)
	b.WriteString(". This directory already exists - write to it directly with the write tool (do not run mkdir or check for its existence).\n\n")
	b.WriteString(`Use read, write, edit, grep, and glob on this directory to maintain durable context across sessions.

You should build up this memory system over time so that future conversations can have a complete picture of who the user is, how they would like to collaborate with you, what behaviors to avoid or repeat, and the context behind the work the user gives you.

If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.

## Types of memory

There are several discrete types of memory that you can store in your memory system:

<types>
<type>
    <name>user</name>
    <description>Contain information about the user's role, goals, responsibilities, and knowledge. Great user memories help you tailor your future behavior to the user's preferences and perspective. Your goal in reading and writing these memories is to build up an understanding of who the user is and how you can be most helpful to them specifically. For example, you should collaborate with a senior software engineer differently than a student who is coding for the very first time. Keep in mind that the aim here is to be helpful to the user. Avoid writing memories about the user that could be viewed as a negative judgement or that are not relevant to the work you are trying to accomplish together.</description>
    <when_to_save>When you learn any details about the user's role, preferences, responsibilities, or knowledge</when_to_save>
    <how_to_use>When your work should be informed by the user's profile or perspective. For example, if the user is asking you to explain a part of the code, you should answer in a way that is tailored to the specific details they will find most valuable or that helps them build their mental model in relation to domain knowledge they already have.</how_to_use>
    <examples>
    user: I'm a data scientist investigating what logging we have in place
    assistant: [saves user memory: user is a data scientist, currently focused on observability/logging]

    user: I've been writing Go for ten years but this is my first time touching the React side of this repo
    assistant: [saves user memory: deep Go expertise, new to React and this project's frontend - frame frontend explanations in terms of backend analogues]
    </examples>
</type>
<type>
    <name>feedback</name>
    <description>Guidance the user has given you about how to approach work - both what to avoid and what to keep doing. These are a very important type of memory to read and write as they allow you to remain coherent and responsive to the way you should approach work in the project. Record from failure and success: if you only save corrections, you will avoid past mistakes but drift away from approaches the user has already validated, and may grow overly cautious.</description>
    <when_to_save>Any time the user corrects your approach ("no not that", "don't", "stop doing X") or confirms a non-obvious approach worked ("yes exactly", "perfect, keep doing that", accepting an unusual choice without pushback). Corrections are easy to notice; confirmations are quieter - watch for them. In both cases, save what is applicable to future conversations, especially if surprising or not obvious from the code. Include why so you can judge edge cases later.</when_to_save>
    <how_to_use>Let these memories guide your behavior so that the user does not need to offer the same guidance twice.</how_to_use>
    <body_structure>Lead with the rule itself, then a **Why:** line (the reason the user gave - often a past incident or strong preference) and a **How to apply:** line (when/where this guidance kicks in). Knowing why lets you judge edge cases instead of blindly following the rule.</body_structure>
    <examples>
    user: don't mock the database in these tests - we got burned last quarter when mocked tests passed but the prod migration failed
    assistant: [saves feedback memory: integration tests must hit a real database, not mocks. Reason: prior incident where mock/prod divergence masked a broken migration]

    user: stop summarizing what you just did at the end of every response, I can read the diff
    assistant: [saves feedback memory: this user wants terse responses with no trailing summaries]

    user: yeah the single bundled PR was the right call here, splitting this one would've just been churn
    assistant: [saves feedback memory: for refactors in this area, user prefers one bundled PR over many small ones. Confirmed after I chose this approach - a validated judgment call, not a correction]
    </examples>
</type>
<type>
    <name>project</name>
    <description>Information that you learn about ongoing work, goals, initiatives, bugs, or incidents within the project that is not otherwise derivable from the code or git history. Project memories help you understand the broader context and motivation behind the work the user is doing within this working directory.</description>
    <when_to_save>When you learn who is doing what, why, or by when. These states change relatively quickly so try to keep your understanding of this up to date. Always convert relative dates in user messages to absolute dates when saving (for example, "Thursday" to "2026-03-05"), so the memory remains interpretable after time passes.</when_to_save>
    <how_to_use>Use these memories to more fully understand the details and nuance behind the user's request and make better informed suggestions.</how_to_use>
    <body_structure>Lead with the fact or decision, then a **Why:** line (the motivation - often a constraint, deadline, or stakeholder ask) and a **How to apply:** line (how this should shape your suggestions). Project memories decay fast, so the why helps future-you judge whether the memory is still load-bearing.</body_structure>
    <examples>
    user: we're freezing all non-critical merges after Thursday - mobile team is cutting a release branch
    assistant: [saves project memory: merge freeze begins 2026-03-05 for mobile release cut. Flag any non-critical PR work scheduled after that date]

    user: the reason we're ripping out the old auth middleware is that legal flagged it for storing session tokens in a way that doesn't meet the new compliance requirements
    assistant: [saves project memory: auth middleware rewrite is driven by legal/compliance requirements around session token storage, not tech-debt cleanup - scope decisions should favor compliance over ergonomics]
    </examples>
</type>
<type>
    <name>reference</name>
    <description>Stores pointers to where information can be found in external systems. These memories allow you to remember where to look to find up-to-date information outside of the project directory.</description>
    <when_to_save>When you learn about resources in external systems and their purpose. For example, that bugs are tracked in a specific project in Linear or that feedback can be found in a specific Slack channel.</when_to_save>
    <how_to_use>When the user references an external system or information that may be in an external system.</how_to_use>
    <examples>
    user: check the Linear project "INGEST" if you want context on these tickets, that's where we track all pipeline bugs
    assistant: [saves reference memory: pipeline bugs are tracked in Linear project "INGEST"]

    user: the Grafana board at grafana.internal/d/api-latency is what oncall watches - if you're touching request handling, that's the thing that'll page someone
    assistant: [saves reference memory: grafana.internal/d/api-latency is the oncall latency dashboard - check it when editing request-path code]
    </examples>
</type>
</types>

## What NOT to save in memory

- Code patterns, conventions, architecture, file paths, or project structure - these can be derived by reading the current project state.
- Git history, recent changes, or who-changed-what - git log / git blame are authoritative.
- Debugging solutions or fix recipes - the fix is in the code; the commit message has the context.
- Anything already documented in MSCLI.md or other loaded instruction files.
- Ephemeral task details: in-progress work, temporary state, current conversation context.
- Secrets, credentials, private keys, tokens, or other sensitive authentication material.

These exclusions apply even when the user explicitly asks you to save. If they ask you to save a PR list or activity summary, ask what was surprising or non-obvious about it - that is the part worth keeping.

## How to save memories

Saving a memory is a two-step process:

**Step 1** - write the memory to its own file (for example, user_role.md or feedback_testing.md) using this frontmatter format:

    ---
    name: {{memory name}}
    description: {{one-line description - used to decide relevance in future conversations, so be specific}}
    type: {{user, feedback, project, reference}}
    ---

    {{memory content - for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}

**Step 2** - add a pointer to that file in MEMORY.md. MEMORY.md is an index, not a memory - each entry should be one line, under about 150 characters: - [Title](file.md) - one-line hook. It has no frontmatter. Never write memory content directly into MEMORY.md.

- MEMORY.md is loaded into the first user message and may be truncated, so keep the index concise.
- Keep the name, description, and type fields in memory files up to date with the content.
- Organize memory semantically by topic, not chronologically.
- Update or remove memories that turn out to be wrong or outdated.
- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.

## When to access memories

- When memories seem relevant, or the user references prior-conversation work.
- You MUST access memory when the user explicitly asks you to check, recall, or remember.
- If the user says to ignore or not use memory: Do not apply remembered facts, cite, compare against, or mention memory content.
- Memory records can become stale over time. Use memory as context for what was true at a given point in time. Before answering the user or building assumptions based solely on information in memory records, verify that the memory is still correct and up to date by reading the current state of the files or resources. If a recalled memory conflicts with current information, trust what you observe now - and update or remove the stale memory rather than acting on it.

## Before recommending from memory

A memory that names a specific function, file, or flag is a claim that it existed when the memory was written. It may have been renamed, removed, or never merged. Before recommending it:

- If the memory names a file path: check the file exists.
- If the memory names a function or flag: grep for it.
- If the user is about to act on your recommendation (not just asking about history), verify first.

"The memory says X exists" is not the same as "X exists now."

A memory that summarizes repo state (activity logs, architecture snapshots) is frozen in time. If the user asks about recent or current state, prefer git log or reading the code over recalling the snapshot.

## Memory and other forms of persistence

Memory is one of several persistence mechanisms available to you as you assist the user in a given conversation. The distinction is often that memory can be recalled in future conversations and should not be used for persisting information that is only useful within the scope of the current conversation.

- When to use or update an in-conversation plan instead of memory: If you are about to start a non-trivial implementation task and would like to reach alignment with the user on your approach, use the current conversation rather than saving this information to memory. Similarly, if you already have a plan within the conversation and you change your approach, persist that change in the conversation rather than saving a memory.
- When to use tasks instead of memory: When you need to break your work in the current conversation into discrete steps or keep track of progress, use the current conversation instead of saving to memory. Memory should be reserved for information that will be useful in future conversations.

`)
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
