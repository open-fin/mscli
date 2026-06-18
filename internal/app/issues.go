package app

import (
	"fmt"
	"strings"

	"gitcode.com/mindspore/mscli/internal/factory/pack"
	factoryruntime "gitcode.com/mindspore/mscli/internal/factory/runtime"
	"gitcode.com/mindspore/mscli/ui/model"
)

const (
	issueKindFailure     = "failure"
	issueKindAccuracy    = "accuracy"
	issueKindPerformance = "performance"
)

func (a *Application) cmdDiagnose(input string) {
	a.runSkillCommand(input, "/diagnose")
}

func (a *Application) cmdFix(input string) {
	a.runSkillCommand(input, "/fix")
}

func (a *Application) cmdMigrate(input string) {
	task := fmt.Sprintf(
		"Load skill migrate-agent.\n\nUser request: %s",
		strings.TrimSpace(input),
	)
	a.EventCh <- model.Event{Type: model.AgentThinking}
	go a.runTask(task)
}

func (a *Application) cmdIntegrate(input string) {
	task := fmt.Sprintf(
		"Load the appropriate skill (algorithm-agent or operator-agent) based on the user request.\n\nUser request: %s",
		strings.TrimSpace(input),
	)
	a.EventCh <- model.Event{Type: model.AgentThinking}
	go a.runTask(task)
}

func (a *Application) cmdPreflight(input string) {
	task := "Load skill readiness-agent."
	if prompt := strings.TrimSpace(input); prompt != "" {
		task += "\n\nUser request: " + prompt
	}
	a.EventCh <- model.Event{Type: model.AgentThinking}
	go a.runTask(task)
}

func (a *Application) runSkillCommand(input, command string) {
	mode := strings.TrimPrefix(command, "/")
	target, err := parseIssueCommandTarget(input, command)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}

	task, err := a.buildSkillTask(target, mode)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	if mode == "diagnose" {
		task = a.enrichDiagnoseTask(task, target)
	}
	if mode == "fix" {
		a.storeFixRunSummary(task, target)
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}
	go a.runTask(task)
}

func (a *Application) enrichDiagnoseTask(task string, target issueCommandTarget) string {
	ctx := buildDiagnosticContext(target, task)
	enrichment, err := factoryruntime.BuildFactoryEnrichment(ctx, a.factoryPackLoadConfig())
	summary := factoryruntime.BuildDiagnoseRunSummary(ctx, enrichment.Matches)
	a.latestDiagnoseSummary = &summary
	a.latestRunKind = "diagnose"
	if err != nil || strings.TrimSpace(enrichment.HintBlock) == "" {
		return task
	}
	return task + "\n\n" + enrichment.HintBlock
}

func (a *Application) storeFixRunSummary(task string, target issueCommandTarget) {
	text := boundedDiagnosticText(firstNonEmptyText(target.Prompt, task))
	summary := factoryruntime.BuildFixRunSummary(factoryruntime.FixRunSummaryInput{
		Topic:              firstNonEmptyText(inferMainError(text), text),
		UserProblemSummary: text,
		PlannedFixSummary:  "Requested fix plan generated from bounded /fix input; execution results are not verified by this summary.",
		KeyEvidence:        fixSummaryEvidence(text),
	})
	a.latestFixSummary = &summary
	a.latestRunKind = "fix"
}

func fixSummaryEvidence(text string) []string {
	values := []string{inferMainError(text), inferProblemType(text), inferStage(text), inferAccelerator(text)}
	values = append(values, inferDiagnosticKeywords(text)...)
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
		if len(out) >= 8 {
			break
		}
	}
	return out
}

func (a *Application) factoryPackLoadConfig() pack.LoadConfig {
	return pack.LoadConfig{}
}

func buildDiagnosticContext(target issueCommandTarget, taskText string) pack.DiagnosticContext {
	text := boundedDiagnosticText(firstNonEmptyText(target.Prompt, taskText))
	ctx := pack.DiagnosticContext{
		Command:   "/diagnose",
		UserInput: text,
	}
	ctx.Signals.MainError = inferMainError(text)
	ctx.Signals.Keywords = inferDiagnosticKeywords(text)
	ctx.Signals.StackKeywords = inferStackKeywords(text)
	ctx.Problem.InferredType = inferProblemType(text)
	ctx.Problem.InferredStage = inferStage(text)
	ctx.Environment.Frameworks = inferFrameworks(text)
	ctx.Environment.Hardware.Accelerator = inferAccelerator(text)
	return ctx
}

func firstNonEmptyText(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func boundedDiagnosticText(value string) string {
	fields := strings.Fields(value)
	if len(fields) <= 240 {
		return strings.TrimSpace(value)
	}
	return strings.Join(fields[:240], " ")
}

func inferMainError(text string) string {
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)
		if trimmed != "" && (strings.Contains(lower, "error") || strings.Contains(lower, "exception") || strings.Contains(lower, "traceback") || strings.Contains(lower, "failed")) {
			return trimmed
		}
	}
	return ""
}

func inferDiagnosticKeywords(text string) []string {
	return presentSignals(text, []string{"torch_npu", "ascend", "cann", "mindspore", "importerror", "runtimeerror", "acl", "ge", "oom"}, 8)
}

func inferStackKeywords(text string) []string {
	return presentSignals(text, []string{"traceback", "importerror", "runtimeerror", "modulenotfounderror", "segmentation fault", "stack"}, 6)
}

func inferProblemType(text string) string {
	lower := strings.ToLower(text)
	if strings.Contains(lower, "accuracy") || strings.Contains(lower, "loss") || strings.Contains(lower, "nan") || strings.Contains(lower, "precision") {
		return issueKindAccuracy
	}
	if strings.Contains(lower, "performance") || strings.Contains(lower, "throughput") || strings.Contains(lower, "latency") || strings.Contains(lower, "slow") {
		return issueKindPerformance
	}
	if strings.Contains(lower, "error") || strings.Contains(lower, "failed") || strings.Contains(lower, "exception") || strings.Contains(lower, "crash") || strings.Contains(lower, "oom") {
		return issueKindFailure
	}
	return ""
}

func inferStage(text string) string {
	lower := strings.ToLower(text)
	stageSignals := []struct {
		stage   string
		signals []string
	}{
		{stage: "import", signals: []string{"importerror", "import ", "module not found", "modulenotfounderror"}},
		{stage: "compile", signals: []string{"compile", "graph compile", "build graph"}},
		{stage: "train", signals: []string{"train", "training", "loss", "backward"}},
		{stage: "eval", signals: []string{"eval", "evaluation", "validation"}},
		{stage: "infer", signals: []string{"infer", "inference", "predict"}},
		{stage: "data", signals: []string{"dataset", "dataloader", "data loader"}},
		{stage: "setup", signals: []string{"install", "setup", "environment", "env var"}},
	}
	for _, candidate := range stageSignals {
		for _, signal := range candidate.signals {
			if strings.Contains(lower, signal) {
				return candidate.stage
			}
		}
	}
	return ""
}

func inferFrameworks(text string) []pack.DiagnosticFramework {
	frameworks := make([]pack.DiagnosticFramework, 0, 2)
	for _, name := range presentSignals(text, []string{"mindspore", "torch", "torch_npu", "tensorflow"}, 4) {
		frameworks = append(frameworks, pack.DiagnosticFramework{Name: name})
	}
	return frameworks
}

func inferAccelerator(text string) string {
	lower := strings.ToLower(text)
	if strings.Contains(lower, "ascend") || strings.Contains(lower, "npu") || strings.Contains(lower, "cann") || strings.Contains(lower, "acl") {
		return "ascend"
	}
	if strings.Contains(lower, "cuda") || strings.Contains(lower, "gpu") {
		return "gpu"
	}
	return ""
}

func presentSignals(text string, signals []string, limit int) []string {
	lower := strings.ToLower(text)
	out := make([]string, 0, limit)
	seen := make(map[string]struct{}, len(signals))
	for _, signal := range signals {
		if strings.Contains(lower, signal) {
			value := strings.TrimSpace(signal)
			if _, ok := seen[value]; ok {
				continue
			}
			seen[value] = struct{}{}
			out = append(out, value)
			if len(out) >= limit {
				break
			}
		}
	}
	return out
}

// buildSkillTask constructs a task description that instructs the agent to load
// the appropriate diagnosis skill in the given mode (diagnose or fix).
func (a *Application) buildSkillTask(target issueCommandTarget, mode string) (string, error) {
	return fmt.Sprintf(
		"Load the appropriate diagnosis skill (failure-agent, accuracy-agent, or performance-agent) in %s mode.\n\nUser problem: %s",
		mode, target.Prompt,
	), nil
}

type issueCommandTarget struct {
	Prompt string
}

func parseIssueCommandTarget(input string, command string) (issueCommandTarget, error) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return issueCommandTarget{}, fmt.Errorf("Usage: %s <problem text>", command)
	}

	return issueCommandTarget{Prompt: trimmed}, nil
}
