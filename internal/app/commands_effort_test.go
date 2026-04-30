package app

import (
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/tools"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdEffort_NoArgsShowsCurrentProviderOptions(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "openai-responses"
	app.Config.Request.Effort = "minimal"

	app.cmdEffort(nil)

	ev := drainUntilEventType(t, app, model.AgentReply)
	for _, want := range []string{"Current effort: minimal", "provider: openai-responses", "none", "xhigh"} {
		if !strings.Contains(ev.Message, want) {
			t.Fatalf("message = %q, want to contain %q", ev.Message, want)
		}
	}
}

func TestCmdEffort_RejectsOpenAIMax(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "openai-responses"
	app.Config.Request.Effort = "high"

	app.cmdEffort([]string{"max"})

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "Unsupported effort") {
		t.Fatalf("message = %q, want unsupported effort", ev.Message)
	}
	if got, want := app.Config.Request.Effort, "high"; got != want {
		t.Fatalf("request.effort = %q, want %q", got, want)
	}
}

func TestCmdEffort_AcceptsAnthropicMax(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "anthropic"
	app.Config.Request.Effort = "high"

	app.cmdEffort([]string{"max"})

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "Effort set to: max") {
		t.Fatalf("message = %q, want effort set confirmation", ev.Message)
	}
	if got, want := app.Config.Request.Effort, "max"; got != want {
		t.Fatalf("request.effort = %q, want %q", got, want)
	}
}

func TestCmdEffort_UpdatesEngine(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "openai-responses"
	provider := &captureStreamProvider{}
	app.Engine = loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 8000,
		Effort:        "high",
	}, provider, tools.NewRegistry())

	app.cmdEffort([]string{"minimal"})
	drainUntilEventType(t, app, model.AgentReply)

	_, err := app.Engine.Run(loop.Task{
		ID:          "effort-command",
		Description: "ping",
	})
	if err != nil {
		t.Fatalf("Engine.Run() error = %v", err)
	}
	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if got, want := provider.lastReq.Effort, "minimal"; got != want {
		t.Fatalf("provider.lastReq.Effort = %q, want %q", got, want)
	}
}
