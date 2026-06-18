package app

import (
	"context"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"gitcode.com/mindspore/mscli/agent/loop"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/internal/workspacefile"
	"gitcode.com/mindspore/mscli/ui/model"
)

var atFilePathPattern = regexp.MustCompile(`^[A-Za-z0-9.:_/\\-]+$`)

type pendingInputExpansion struct {
	denial *pathpolicy.PathDenial
	resume func(pathpolicy.ResolveOptions)
}

func (a *Application) expandInputText(text string) (string, error) {
	return a.expandAtFiles(text)
}

func (a *Application) expandInputTextWithOptions(text string, opts pathpolicy.ResolveOptions) (string, error) {
	workDir := ""
	var resolver *pathpolicy.Resolver
	if a != nil {
		workDir = a.WorkDir
		resolver = a.pathResolver
	}
	return expandAtFilesWithOptions(workDir, text, resolver, opts)
}

func (a *Application) expandAtFiles(text string) (string, error) {
	workDir := ""
	var resolver *pathpolicy.Resolver
	if a != nil {
		workDir = a.WorkDir
		resolver = a.pathResolver
	}
	return expandAtFilesWithOptions(workDir, text, resolver, pathpolicy.ResolveOptions{})
}

func expandAtFiles(workDir, text string, resolver *pathpolicy.Resolver) (string, error) {
	return expandAtFilesWithOptions(workDir, text, resolver, pathpolicy.ResolveOptions{})
}

func expandAtFilesWithOptions(workDir, text string, resolver *pathpolicy.Resolver, opts pathpolicy.ResolveOptions) (string, error) {
	var out strings.Builder

	for i := 0; i < len(text); {
		r := rune(text[i])
		if r < utf8RuneSelf && !isASCIIWhitespace(byte(r)) {
			j := i + 1
			if text[i] == '@' && j < len(text) && text[j] == '"' {
				j++
				for j < len(text) && text[j] != '"' {
					j++
				}
				if j < len(text) {
					j++
				}
			} else {
				for j < len(text) && !isASCIIWhitespace(text[j]) {
					j++
				}
			}
			token := text[i:j]
			replaced, err := replaceAtFileToken(workDir, token, resolver, opts)
			if err != nil {
				return "", err
			}
			out.WriteString(replaced)
			i = j
			continue
		}

		runeValue, size := utf8.DecodeRuneInString(text[i:])
		if !unicode.IsSpace(runeValue) {
			j := i + size
			for j < len(text) {
				nextRune, nextSize := utf8.DecodeRuneInString(text[j:])
				if unicode.IsSpace(nextRune) {
					break
				}
				j += nextSize
			}
			token := text[i:j]
			replaced, err := replaceAtFileToken(workDir, token, resolver, opts)
			if err != nil {
				return "", err
			}
			out.WriteString(replaced)
			i = j
			continue
		}

		out.WriteRune(runeValue)
		i += size
	}

	return out.String(), nil
}

func replaceAtFileToken(workDir, token string, resolver *pathpolicy.Resolver, opts pathpolicy.ResolveOptions) (string, error) {
	switch {
	case token == "":
		return token, nil
	case strings.HasPrefix(token, "@@"):
		return token[1:], nil
	case !strings.HasPrefix(token, "@") || len(token) == 1:
		return token, nil
	}

	path := token[1:]
	if strings.HasPrefix(path, `"`) && strings.HasSuffix(path, `"`) && len(path) >= 2 {
		path = strings.TrimSuffix(strings.TrimPrefix(path, `"`), `"`)
	} else if !atFilePathPattern.MatchString(path) {
		return token, nil
	}

	fullPath, err := resolveAtFilePath(workDir, path, resolver, opts)
	if err != nil {
		return "", err
	}

	return formatExpandedFilePath(fullPath), nil
}

func resolveAtFilePath(workDir, path string, resolver *pathpolicy.Resolver, opts pathpolicy.ResolveOptions) (string, error) {
	if resolver == nil {
		return workspacefile.ResolveExistingFilePath(workDir, path)
	}
	fullPath, denial, err := resolver.ResolveReadablePathForOperation("@file", path, opts)
	if err != nil {
		return "", err
	}
	if denial != nil {
		return "", pathpolicy.NewPathDenialError(denial)
	}
	if err := workspacefile.CheckExistingTextFile(fullPath, path, workspacefile.DefaultMaxInlineBytes); err != nil {
		return "", err
	}
	return fullPath, nil
}

func formatExpandedFilePath(path string) string {
	return `[file path="` + filepath.ToSlash(filepath.Clean(path)) + `"]`
}

func (a *Application) requestInputExpansionAuthorization(denial *pathpolicy.PathDenial, resume func(pathpolicy.ResolveOptions)) {
	if a == nil || a.pathAuthorizer == nil || denial == nil {
		a.emitInputExpansionError(pathpolicy.NewPathDenialError(denial))
		return
	}
	a.inputExpansionMu.Lock()
	a.pendingInputExpansion = &pendingInputExpansion{denial: denial, resume: resume}
	a.inputExpansionMu.Unlock()
	go func() {
		decision, err := a.pathAuthorizer.RequestPathAuthorization(context.Background(), denial)
		if err != nil {
			a.EventCh <- model.Event{Type: model.ToolError, Message: err.Error()}
			return
		}
		a.applyInputExpansionDecision(decision)
	}()
}

func (a *Application) applyInputExpansionDecision(decision loop.PathAuthorizationDecision) {
	a.inputExpansionMu.Lock()
	pending := a.pendingInputExpansion
	a.pendingInputExpansion = nil
	a.inputExpansionMu.Unlock()
	if pending == nil {
		return
	}
	if decision.Scope == loop.PathAuthorizationDeny || strings.TrimSpace(decision.Root) == "" {
		a.emitInputExpansionError(pathpolicy.NewPathDenialError(pending.denial))
		return
	}
	if pending.resume == nil {
		return
	}
	// Resume directly so chat and slash-command continuations keep their original dispatch path.
	pending.resume(pathpolicy.ResolveOptions{TemporaryReadRoots: []string{decision.Root}})
}

func (a *Application) processExpandedInput(expanded string) {
	a.EventCh <- model.Event{Type: model.UserInput, Message: expanded}
	go a.runTask(expanded)
}

func (a *Application) tryAuthorizeInputExpansion(err error, resume func(pathpolicy.ResolveOptions)) bool {
	denial, ok := pathpolicy.ExtractPathDenialFromError(err)
	if !ok {
		return false
	}
	a.requestInputExpansionAuthorization(denial, resume)
	return true
}

type rawCommand struct {
	Name      string
	Remainder string
}

func splitRawCommand(input string) (rawCommand, bool) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" || !strings.HasPrefix(trimmed, "/") {
		return rawCommand{}, false
	}

	if idx := strings.IndexAny(trimmed, " \t\r\n"); idx >= 0 {
		return rawCommand{
			Name:      trimmed[:idx],
			Remainder: strings.TrimSpace(trimmed[idx+1:]),
		}, true
	}

	return rawCommand{Name: trimmed}, true
}

func splitFirstToken(input string) (string, string) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return "", ""
	}

	for i, r := range trimmed {
		if unicode.IsSpace(r) {
			return trimmed[:i], strings.TrimSpace(trimmed[i:])
		}
	}

	return trimmed, ""
}

func (a *Application) expandIssueCommandInputWithOptions(rawInput string, opts pathpolicy.ResolveOptions) (string, error) {
	return a.expandInputTextWithOptions(strings.TrimSpace(rawInput), opts)
}

func (a *Application) expandIssueCommandInput(rawInput string) (string, error) {
	return a.expandIssueCommandInputWithOptions(rawInput, pathpolicy.ResolveOptions{})
}

func (a *Application) expandReportInputWithOptions(rawInput string, opts pathpolicy.ResolveOptions) (string, error) {
	first, remainder := splitFirstToken(rawInput)
	if first == "" || remainder == "" {
		return strings.TrimSpace(rawInput), nil
	}
	expanded, err := a.expandInputTextWithOptions(remainder, opts)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(first + " " + expanded), nil
}

func (a *Application) expandReportInput(rawInput string) (string, error) {
	return a.expandReportInputWithOptions(rawInput, pathpolicy.ResolveOptions{})
}

func (a *Application) emitInputExpansionError(err error) {
	a.emitToolError("input", "Failed to expand @file input: %v", err)
}

func isASCIIWhitespace(b byte) bool {
	switch b {
	case ' ', '\t', '\n', '\r', '\f', '\v':
		return true
	default:
		return false
	}
}

const utf8RuneSelf = 0x80
