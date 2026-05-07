package app

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestRenderBootstrapHelpRoot(t *testing.T) {
	got := renderBootstrapHelp(bootstrapHelpTopicRoot)

	for _, want := range []string{
		"Usage:\n  mscli [flags] [command]",
		"Commands:\n  resume    Resume a saved session; opens the session picker UI by default\n  replay    Replay a saved session or trajectory; opens the session picker UI by default\n  factory   Run non-interactive Factory commands for external agents, shell scripts, and CI",
		"fix       Run /fix headlessly with YOLO enabled and print stdout",
		"diagnose  Run /diagnose headlessly with YOLO enabled and print stdout",
		"exec      Run a free-form task headlessly with YOLO enabled and print stdout",
		"-v, --version",
		"type / to browse slash commands",
		"mscli \"why does this MindSpore run fail?\"",
		"mscli exec \"inspect this repository\"",
		"mscli fix \"fix the training failure in ./train.py\"",
		"MSCLI_PROVIDER=openai-completion MSCLI_API_KEY=sk-... MSCLI_MODEL=gpt-4o mscli",
		"mscli replay trajectory.json 2x",
		"MSCLI_PROVIDER",
		"MSCLI_API_KEY",
		"MSCLI_MODEL",
		"MSCLI_BASE_URL",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("root help should contain %q, got:\n%s", want, got)
		}
	}

	for _, unwanted := range []string{
		"MSCLI_CONTEXT_WINDOW",
		"mindspore-cli replay",
	} {
		if strings.Contains(got, unwanted) {
			t.Fatalf("root help should not contain %q, got:\n%s", unwanted, got)
		}
	}
}

func TestRenderBootstrapHelpResume(t *testing.T) {
	got := renderBootstrapHelp(bootstrapHelpTopicResume)

	for _, want := range []string{
		"Usage:\n  mscli resume [flags] [sess_xxx]",
		"Optional session ID. Omit to open the session picker UI.",
		"--url string",
		"--model string",
		"--api-key string",
		"--debug",
		"mscli resume sess_123",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("resume help should contain %q, got:\n%s", want, got)
		}
	}
}

func TestRenderBootstrapHelpReplay(t *testing.T) {
	got := renderBootstrapHelp(bootstrapHelpTopicReplay)

	for _, want := range []string{
		"Usage:\n  mscli replay [flags] [sess_xxx|trajectory.json|trajectory.jsonl] [speed]",
		"Optional replay target. Omit to open the session picker UI.",
		"Optional replay speed multiplier such as 0.5x, 1.5x, or 2x.",
		"--debug",
		"mscli replay trajectory.json 2x",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("replay help should contain %q, got:\n%s", want, got)
		}
	}

	if strings.Contains(got, "mindspore-cli replay") {
		t.Fatalf("replay help should use mscli command name, got:\n%s", got)
	}
}

func TestRenderBootstrapHelpHeadless(t *testing.T) {
	got := renderBootstrapHelp(bootstrapHelpTopicFix)

	for _, want := range []string{
		"Run /fix without opening the TUI.",
		"Usage:\n  mscli fix [flags] <problem text|ISSUE-id>",
		"YOLO mode is enabled",
		"mscli fix ISSUE-42",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("fix help should contain %q, got:\n%s", want, got)
		}
	}
}

func TestRenderBootstrapHelpHeadlessExec(t *testing.T) {
	got := renderBootstrapHelp(bootstrapHelpTopicExec)

	for _, want := range []string{
		"Run a free-form task without opening the TUI.",
		"Usage:\n  mscli exec [flags] <task>",
		"YOLO mode is enabled",
		"mscli exec \"inspect this repository\"",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("exec help should contain %q, got:\n%s", want, got)
		}
	}
}

func TestParseBootstrapConfigHelpTopics(t *testing.T) {
	tests := []struct {
		name  string
		args  []string
		topic bootstrapHelpTopic
	}{
		{name: "root", args: []string{"--help"}, topic: bootstrapHelpTopicRoot},
		{name: "resume", args: []string{"resume", "--help"}, topic: bootstrapHelpTopicResume},
		{name: "replay", args: []string{"replay", "--help"}, topic: bootstrapHelpTopicReplay},
		{name: "fix", args: []string{"fix", "--help"}, topic: bootstrapHelpTopicFix},
		{name: "diagnose", args: []string{"diagnose", "--help"}, topic: bootstrapHelpTopicDiagnose},
		{name: "exec", args: []string{"exec", "--help"}, topic: bootstrapHelpTopicExec},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseBootstrapConfig(tt.args)
			if err == nil {
				t.Fatal("parseBootstrapConfig() error = nil, want help error")
			}

			var helpErr bootstrapHelpError
			if !errors.As(err, &helpErr) {
				t.Fatalf("parseBootstrapConfig() error = %v, want bootstrapHelpError", err)
			}
			if helpErr.topic != tt.topic {
				t.Fatalf("help topic = %q, want %q", helpErr.topic, tt.topic)
			}
		})
	}
}

func TestRunHelpReturnsNil(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "root", args: []string{"--help"}, want: "Usage:\n  mscli [flags] [command]"},
		{name: "resume", args: []string{"resume", "--help"}, want: "Usage:\n  mscli resume [flags] [sess_xxx]"},
		{name: "replay", args: []string{"replay", "--help"}, want: "Usage:\n  mscli replay [flags] [sess_xxx|trajectory.json|trajectory.jsonl] [speed]"},
		{name: "factory", args: []string{"factory", "--help"}, want: "Factory commands:"},
		{name: "fix", args: []string{"fix", "--help"}, want: "Usage:\n  mscli fix [flags] <problem text|ISSUE-id>"},
		{name: "diagnose", args: []string{"diagnose", "--help"}, want: "Usage:\n  mscli diagnose [flags] <problem text|ISSUE-id>"},
		{name: "exec", args: []string{"exec", "--help"}, want: "Usage:\n  mscli exec [flags] <task>"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			previous := cliStdout
			cliStdout = &buf
			t.Cleanup(func() {
				cliStdout = previous
			})

			if err := Run(tt.args); err != nil {
				t.Fatalf("Run(%v) error = %v, want nil", tt.args, err)
			}
			if !strings.Contains(buf.String(), tt.want) {
				t.Fatalf("Run(%v) output should contain %q, got:\n%s", tt.args, tt.want, buf.String())
			}
		})
	}
}
