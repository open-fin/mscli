// Package shell provides shell command execution with workspace context,
// environment management, timeouts, and safety checks.
package shell

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"mvdan.cc/sh/v3/syntax"
)

// Config holds the shell runner configuration.
type Config struct {
	WorkDir        string
	Timeout        time.Duration
	AllowedCmds    []string // Whitelist (empty = allow all)
	BlockedCmds    []string // Blacklist
	RequireConfirm []string // Commands requiring confirmation
	Env            map[string]string
}

// Result is the result of a command execution.
type Result struct {
	Stdout          string
	Stderr          string
	ExitCode        int
	Error           error
	StdoutTruncated bool
	StderrTruncated bool
}

// OutputChunk is a single line emitted while a command is running.
type OutputChunk struct {
	Stream string
	Text   string
}

// Runner executes shell commands within a configured workspace.
type Runner struct {
	config Config
}

const (
	maxScannerTokenSize = 1024 * 1024
	maxOutputBytes      = 64 * 1024
	StreamStdout        = "stdout"
	StreamStderr        = "stderr"
	outputTruncatedMark = "[output truncated]"
)

// NewRunner creates a new shell runner.
func NewRunner(cfg Config) *Runner {
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &Runner{config: cfg}
}

// Run executes a command and returns the result.
func (r *Runner) Run(ctx context.Context, command string) (*Result, error) {
	return r.RunStream(ctx, command, nil)
}

// RunStream executes a command, emitting output lines as they arrive.
func (r *Runner) RunStream(ctx context.Context, command string, emit func(OutputChunk)) (*Result, error) {
	if reason := r.checkAllowed(command); reason != "" {
		return &Result{
			ExitCode: -1,
			Error:    fmt.Errorf("command not allowed: %s", reason),
		}, nil
	}

	buildCmd := func(execCtx context.Context) *exec.Cmd {
		cmd := exec.Command("sh", "-c", command)
		configureCmdForCancel(cmd)
		cmd.Dir = r.config.WorkDir
		cmd.Env = os.Environ()
		for k, v := range r.config.Env {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
		}
		return cmd
	}

	cmd := buildCmd(ctx)

	if _, hasDeadline := ctx.Deadline(); !hasDeadline && r.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.config.Timeout)
		defer cancel()
		cmd = buildCmd(ctx)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("create stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start command: %w", err)
	}
	cmdDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			terminateCmd(cmd)
		case <-cmdDone:
		}
	}()

	var stdoutOut, stderrOut string
	var stdoutTruncated, stderrTruncated bool
	var stdoutErr, stderrErr error

	stdoutDone := make(chan struct{})
	go func() {
		stdoutOut, stdoutTruncated, stdoutErr = readCapped(stdout, maxOutputBytes, func(line string) {
			if emit != nil {
				emit(OutputChunk{Stream: StreamStdout, Text: line})
			}
		})
		close(stdoutDone)
	}()

	stderrDone := make(chan struct{})
	go func() {
		stderrOut, stderrTruncated, stderrErr = readCapped(stderr, maxOutputBytes, func(line string) {
			if emit != nil {
				emit(OutputChunk{Stream: StreamStderr, Text: line})
			}
		})
		close(stderrDone)
	}()

	<-stdoutDone
	<-stderrDone
	if stdoutErr != nil {
		return nil, fmt.Errorf("read stdout: %w", stdoutErr)
	}
	if stderrErr != nil {
		return nil, fmt.Errorf("read stderr: %w", stderrErr)
	}

	err = cmd.Wait()
	close(cmdDone)

	result := &Result{
		Stdout:          stdoutOut,
		Stderr:          stderrOut,
		ExitCode:        0,
		StdoutTruncated: stdoutTruncated,
		StderrTruncated: stderrTruncated,
	}

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = -1
			result.Error = err
		}
	}

	return result, nil
}

func readCapped(r io.Reader, maxBytes int, emit func(string)) (string, bool, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), maxScannerTokenSize)

	content := ""
	truncated := false
	for scanner.Scan() {
		line := scanner.Text()
		if emit != nil {
			emit(line)
		}
		var nextTruncated bool
		content, nextTruncated = appendOutputWindow(content, line, maxBytes)
		truncated = truncated || nextTruncated
	}
	if err := scanner.Err(); err != nil {
		return "", false, err
	}
	if truncated {
		if strings.TrimSpace(content) == "" {
			return outputTruncatedMark, true, nil
		}
		return outputTruncatedMark + "\n" + content, true, nil
	}
	return content, false, nil
}

func appendOutputWindow(content, chunk string, maxBytes int) (string, bool) {
	if maxBytes <= 0 {
		return "", strings.TrimSpace(content) != "" || strings.TrimSpace(chunk) != ""
	}

	switch {
	case content == "":
		content = chunk
	case chunk != "":
		content += "\n" + chunk
	}

	if len(content) <= maxBytes {
		return content, false
	}

	start := len(content) - maxBytes
	window := content[start:]
	if start > 0 {
		if idx := strings.IndexByte(window, '\n'); idx >= 0 && idx < len(window)-1 {
			window = window[idx+1:]
		}
	}
	if window == "" {
		window = content[len(content)-maxBytes:]
	}
	return window, true
}

// IsDangerous checks if a command might be dangerous.
func (r *Runner) IsDangerous(command string) bool {
	dangerous := []string{
		"rm -rf /", "rm -rf ~", "rm -rf /*",
		"> /dev/sda", "mkfs.", "dd if=",
		":(){ :|:& };:", // fork bomb
	}

	lower := strings.ToLower(command)
	for _, d := range dangerous {
		if strings.Contains(lower, d) {
			return true
		}
	}

	if strings.HasPrefix(lower, "rm ") && strings.Contains(lower, "-rf") {
		return true
	}

	return false
}

// checkAllowed checks if a command is allowed.
func (r *Runner) checkAllowed(command string) string {
	if len(r.config.BlockedCmds) == 0 && len(r.config.AllowedCmds) == 0 {
		return ""
	}

	blocked, err := parseCommandPatterns(r.config.BlockedCmds)
	if err != nil {
		return fmt.Sprintf("invalid blocked command pattern: %v", err)
	}
	allowed, err := parseCommandPatterns(r.config.AllowedCmds)
	if err != nil {
		return fmt.Sprintf("invalid allowed command pattern: %v", err)
	}

	file, err := syntax.NewParser(syntax.Variant(syntax.LangPOSIX)).Parse(strings.NewReader(command), "")
	if err != nil {
		return fmt.Sprintf("cannot parse shell command: %v", err)
	}

	reason := ""
	syntax.Walk(file, func(node syntax.Node) bool {
		if reason != "" {
			return false
		}
		call, ok := node.(*syntax.CallExpr)
		if !ok || len(call.Args) == 0 {
			return true
		}

		argv, ok := staticCommandArgs(call.Args)
		if !ok {
			reason = "contains dynamic shell expansion"
			return false
		}
		if len(blocked) > 0 && obscuresNestedCommand(argv) {
			reason = fmt.Sprintf("command %q can execute an uninspectable nested command", argv[0])
			return false
		}
		for i, pattern := range blocked {
			if commandPatternMatches(pattern, argv) {
				reason = fmt.Sprintf("matches blocked pattern: %s", r.config.BlockedCmds[i])
				return false
			}
		}
		if len(allowed) > 0 {
			for _, pattern := range allowed {
				if commandPatternMatches(pattern, argv) {
					return true
				}
			}
			reason = fmt.Sprintf("command %q is not in allowed commands list", argv[0])
			return false
		}
		return true
	})
	return reason
}

func parseCommandPatterns(rawPatterns []string) ([][]string, error) {
	patterns := make([][]string, 0, len(rawPatterns))
	for _, raw := range rawPatterns {
		file, err := syntax.NewParser(syntax.Variant(syntax.LangPOSIX)).Parse(strings.NewReader(raw), "")
		if err != nil {
			return nil, fmt.Errorf("%q: %w", raw, err)
		}

		var calls [][]string
		valid := true
		syntax.Walk(file, func(node syntax.Node) bool {
			call, ok := node.(*syntax.CallExpr)
			if !ok {
				return true
			}
			if len(call.Assigns) > 0 || len(call.Args) == 0 {
				valid = false
				return false
			}
			argv, static := staticCommandArgs(call.Args)
			if !static {
				valid = false
				return false
			}
			calls = append(calls, argv)
			return false
		})
		if !valid || len(calls) != 1 {
			return nil, fmt.Errorf("%q must be one static command", raw)
		}
		patterns = append(patterns, calls[0])
	}
	return patterns, nil
}

func staticCommandArgs(words []*syntax.Word) ([]string, bool) {
	args := make([]string, 0, len(words))
	for _, word := range words {
		value, ok := staticWordParts(word.Parts, false)
		if !ok {
			return nil, false
		}
		args = append(args, value)
	}
	if len(args) > 0 {
		args[0] = normalizeCommandName(args[0])
	}
	return args, true
}

func staticWordParts(parts []syntax.WordPart, quoted bool) (string, bool) {
	var value strings.Builder
	for _, part := range parts {
		switch part := part.(type) {
		case *syntax.Lit:
			if !quoted && strings.ContainsAny(part.Value, "*?[") {
				return "", false
			}
			value.WriteString(part.Value)
		case *syntax.SglQuoted:
			value.WriteString(part.Value)
		case *syntax.DblQuoted:
			nested, ok := staticWordParts(part.Parts, true)
			if !ok {
				return "", false
			}
			value.WriteString(nested)
		default:
			return "", false
		}
	}
	result := value.String()
	if !quoted && strings.HasPrefix(result, "~") {
		return "", false
	}
	return result, true
}

func normalizeCommandName(command string) string {
	command = strings.ToLower(strings.TrimSpace(command))
	if strings.Contains(command, "/") {
		command = filepath.Base(command)
	}
	return command
}

func commandPatternMatches(pattern, command []string) bool {
	if len(pattern) == 0 || len(pattern) > len(command) {
		return false
	}
	for i := range pattern {
		if !strings.EqualFold(pattern[i], command[i]) {
			return false
		}
	}
	return true
}

func obscuresNestedCommand(argv []string) bool {
	if len(argv) == 0 {
		return false
	}
	switch argv[0] {
	case ".", "source", "eval", "sh", "bash", "dash", "ash", "zsh", "ksh",
		"busybox", "command", "builtin", "exec", "env", "nohup", "nice", "timeout",
		"stdbuf", "sudo", "doas", "xargs", "parallel", "python", "python3", "node",
		"perl", "ruby", "php", "lua", "awk":
		return true
	case "find":
		for _, arg := range argv[1:] {
			switch strings.ToLower(arg) {
			case "-exec", "-execdir", "-ok", "-okdir":
				return true
			}
		}
	}
	return false
}

// RequiresConfirm checks if a command requires user confirmation.
func (r *Runner) RequiresConfirm(command string) bool {
	cmd := strings.TrimSpace(strings.ToLower(command))

	for _, prefix := range r.config.RequireConfirm {
		if strings.HasPrefix(cmd, strings.ToLower(prefix)) {
			return true
		}
	}

	destructive := []string{"rm ", "mv ", "cp -r", "> ", ">> "}
	for _, d := range destructive {
		if strings.HasPrefix(cmd, d) {
			return true
		}
	}

	return false
}

// GetWorkDir returns the working directory.
func (r *Runner) GetWorkDir() string {
	return r.config.WorkDir
}

// SanitizePath sanitizes a path for use in commands.
func SanitizePath(path string) string {
	path = strings.ReplaceAll(path, ";", "")
	path = strings.ReplaceAll(path, "&", "")
	path = strings.ReplaceAll(path, "|", "")
	path = strings.ReplaceAll(path, "`", "")
	path = strings.ReplaceAll(path, "$", "")
	return filepath.Clean(path)
}
