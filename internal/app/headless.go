package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
)

func (a *Application) runHeadlessIssueCommand(command, rawInput string, out io.Writer) error {
	return a.runHeadlessCommand(command, rawInput, out)
}

func (a *Application) runHeadlessCommand(command, rawInput string, out io.Writer) error {
	if a == nil {
		return nil
	}
	defer func() {
		if a.session != nil {
			_ = a.session.Close()
		}
	}()
	if out == nil {
		out = io.Discard
	}
	command = strings.TrimPrefix(strings.TrimSpace(command), "/")
	if !isHeadlessBootstrapCommand(command) {
		return fmt.Errorf("unsupported headless command: %s", command)
	}
	if !a.enableYoloMode() {
		return fmt.Errorf("yolo mode not available in current configuration")
	}

	task, err := a.headlessTask(command, rawInput)
	if err != nil {
		return err
	}
	if err := a.runTaskHeadless(task, out); err != nil {
		return err
	}
	a.printHeadlessResumeHint(command, out)
	return nil
}

func (a *Application) headlessTask(command, rawInput string) (string, error) {
	if command == "exec" {
		return a.headlessExecTask(rawInput)
	}
	return a.headlessIssueTask(command, rawInput)
}

func (a *Application) headlessExecTask(rawInput string) (string, error) {
	expanded, err := a.expandInputText(strings.TrimSpace(rawInput))
	if err != nil {
		return "", fmt.Errorf("expand input: %w", err)
	}
	expanded = strings.TrimSpace(expanded)
	if expanded == "" {
		return "", fmt.Errorf("usage: mscli exec [flags] <task>")
	}
	return expanded, nil
}

func (a *Application) printHeadlessResumeHint(command string, out io.Writer) {
	if out == nil {
		return
	}
	hint := a.headlessResumeHint(command)
	if hint == "" {
		return
	}
	_, _ = fmt.Fprintln(out, hint)
}

func (a *Application) headlessResumeHint(command string) string {
	if a == nil || a.session == nil {
		return ""
	}
	if command == "exec" {
		return fmt.Sprintf("Resume this conversation with: mscli resume %s", a.session.ID())
	}
	return cliResumeHintForSession(a.session.ID())
}

func (a *Application) headlessIssueTask(command, rawInput string) (string, error) {
	expanded, err := a.expandIssueCommandInput(rawInput)
	if err != nil {
		return "", fmt.Errorf("expand input: %w", err)
	}
	target, err := parseIssueCommandTarget(expanded, "/"+command)
	if err != nil {
		return "", err
	}
	if target.HasIssue {
		if err := a.ensureIssueServiceHeadless(); err != nil {
			return "", err
		}
	}
	return a.buildSkillTask(target, command)
}

func (a *Application) ensureIssueServiceHeadless() error {
	if a.issueService != nil {
		return nil
	}
	cred, err := loadCredentials()
	if err != nil {
		return fmt.Errorf("not logged in. run /login <token> first")
	}
	a.issueService = issuepkg.NewService(issuepkg.NewRemoteStore(cred.ServerURL, cred.Token))
	a.issueUser = cred.User
	a.issueRole = cred.Role
	return nil
}

func (a *Application) runTaskHeadless(description string, out io.Writer) error {
	printer := newHeadlessPrinter(out)
	persistSnapshot := func() error {
		if err := a.persistSessionSnapshot(); err != nil {
			return fmt.Errorf("persist session snapshot: %w", err)
		}
		return nil
	}

	if !a.llmReady {
		if err := a.recordUnavailableTurn(description, provideAPIKeyFirstMsg); err != nil {
			return fmt.Errorf("record local turn: %w", err)
		}
		if err := persistSnapshot(); err != nil {
			return err
		}
		printer.println(provideAPIKeyFirstMsg)
		return nil
	}

	task := loop.Task{
		ID:          generateTaskID(),
		Description: description,
	}
	if a.ctxManager != nil && a.ctxManager.ShouldCompactAfterAdding(llm.NewUserMessage(description)) {
		printer.println(loop.ContextCompactStartMessage)
	}

	ctx, runID := a.beginTaskRun()
	defer a.finishTaskRun(runID)

	err := a.Engine.RunWithContextStream(ctx, task, func(ev loop.Event) {
		printer.printLoopEvent(ev)
	})
	printer.finish()
	if errors.Is(err, context.Canceled) {
		_ = persistSnapshot()
		return nil
	}
	if err != nil {
		if persistErr := persistSnapshot(); persistErr != nil {
			return persistErr
		}
		return err
	}
	return persistSnapshot()
}

type headlessPrinter struct {
	out                 io.Writer
	replyStreamed       bool
	lastWriteEndedLine  bool
	streamedShellOutput map[string]bool
}

func newHeadlessPrinter(out io.Writer) *headlessPrinter {
	if out == nil {
		out = io.Discard
	}
	return &headlessPrinter{
		out:                 out,
		lastWriteEndedLine:  true,
		streamedShellOutput: map[string]bool{},
	}
}

func (p *headlessPrinter) printLoopEvent(ev loop.Event) {
	switch ev.Type {
	case loop.EventAgentReplyDelta:
		p.replyStreamed = true
		p.print(ev.Message)
	case loop.EventAgentReply:
		if p.replyStreamed {
			p.finishLine()
			p.replyStreamed = false
			return
		}
		p.println(ev.Message)
	case loop.EventCmdOutput:
		p.finishReply()
		p.streamedShellOutput[ev.ToolCallID] = true
		p.print(ev.Message)
	case loop.EventCmdFinished:
		p.finishReply()
		if !p.streamedShellOutput[ev.ToolCallID] {
			p.println(firstNonEmpty(ev.Summary, ev.Message))
		}
	case loop.EventToolEdit, loop.EventToolWrite, loop.EventToolSkill, loop.EventToolInterrupted, loop.EventToolError, loop.EventTaskFailed:
		p.finishReply()
		p.println(firstNonEmpty(ev.Summary, ev.Message))
	case loop.EventContextCompacted:
		p.finishReply()
		p.println(ev.Message)
	}
}

func (p *headlessPrinter) finish() {
	p.finishReply()
	p.finishLine()
}

func (p *headlessPrinter) finishReply() {
	if !p.replyStreamed {
		return
	}
	p.finishLine()
	p.replyStreamed = false
}

func (p *headlessPrinter) print(text string) {
	if text == "" {
		return
	}
	_, _ = io.WriteString(p.out, text)
	p.lastWriteEndedLine = strings.HasSuffix(text, "\n")
}

func (p *headlessPrinter) println(text string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}
	if !p.lastWriteEndedLine {
		_, _ = io.WriteString(p.out, "\n")
	}
	_, _ = io.WriteString(p.out, text)
	_, _ = io.WriteString(p.out, "\n")
	p.lastWriteEndedLine = true
}

func (p *headlessPrinter) finishLine() {
	if p.lastWriteEndedLine {
		return
	}
	_, _ = io.WriteString(p.out, "\n")
	p.lastWriteEndedLine = true
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
