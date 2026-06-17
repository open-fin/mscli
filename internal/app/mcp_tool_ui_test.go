package app

import (
	"testing"

	"gitcode.com/mindspore/mscli/agent/loop"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestConvertLoopEventMapsGenericToolResults(t *testing.T) {
	ev := convertLoopEvent(loop.Event{
		Type:       loop.EventToolStarted,
		ToolName:   "mcp__deepwiki__read_wiki_contents",
		ToolCallID: "call_1",
		Message:    "large result summary",
	})
	if ev == nil {
		t.Fatal("convertLoopEvent returned nil")
	}
	if ev.Type != model.ToolReplay {
		t.Fatalf("event type = %s, want ToolReplay", ev.Type)
	}
	if ev.ToolName != "mcp__deepwiki__read_wiki_contents" || ev.Message != "large result summary" {
		t.Fatalf("event = %#v", ev)
	}
}
