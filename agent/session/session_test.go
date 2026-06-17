package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/ui/model"
)

func TestCreateDefersDiskWritesUntilActivate(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	if _, err := os.Stat(s.Path()); !os.IsNotExist(err) {
		t.Fatalf("expected no trajectory before activate, got err=%v", err)
	}
	if _, err := os.Stat(snapshotPath(s.Path())); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot before activate, got err=%v", err)
	}

	if err := s.AppendSkillActivation("demo-skill"); err != nil {
		t.Fatalf("append skill activation: %v", err)
	}
	if err := s.AppendUserInput("hello"); err != nil {
		t.Fatalf("append user input: %v", err)
	}
	if err := s.AppendAssistant("hi"); err != nil {
		t.Fatalf("append assistant reply: %v", err)
	}
	if err := s.SaveSnapshotWithUsage("updated prompt", []llm.Message{
		llm.NewUserMessage("hello"),
		llm.NewAssistantMessage("hi"),
	}, &UsageSnapshot{
		Provider:   "anthropic",
		TokenScope: "total",
		Tokens:     1809,
		LocalDelta: 17,
		Usage: &llm.Usage{
			PromptTokens:     1660,
			CompletionTokens: 149,
			TotalTokens:      1809,
			Raw:              json.RawMessage(`{"prompt_tokens":1660,"completion_tokens":149,"total_tokens":1809,"cached_tokens":0}`),
		},
	}); err != nil {
		t.Fatalf("save buffered snapshot: %v", err)
	}

	if _, err := os.Stat(s.Path()); !os.IsNotExist(err) {
		t.Fatalf("expected no trajectory before activate after buffering, got err=%v", err)
	}
	if _, err := os.Stat(snapshotPath(s.Path())); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot before activate after buffering, got err=%v", err)
	}

	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close activated session: %v", err)
	}

	if _, err := os.Stat(s.Path()); err != nil {
		t.Fatalf("expected trajectory after activate: %v", err)
	}
	if _, err := os.Stat(snapshotPath(s.Path())); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot sidecar after activate, got err=%v", err)
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load activated session: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	if !loaded.HasPersistedDialogue() {
		t.Fatal("expected persisted dialogue after activation")
	}
	if got := loaded.Meta().SystemPrompt; got != "updated prompt" {
		t.Fatalf("meta system prompt = %q, want %q", got, "updated prompt")
	}

	systemPrompt, restored := loaded.RestoreContext()
	if systemPrompt != "updated prompt" {
		t.Fatalf("restored system prompt = %q, want %q", systemPrompt, "updated prompt")
	}
	if len(restored) != 2 {
		t.Fatalf("restored message count = %d, want 2", len(restored))
	}
	usage := loaded.UsageSnapshot()
	if usage == nil {
		t.Fatal("UsageSnapshot() = nil, want snapshot")
	}
	if got, want := usage.Provider, "anthropic"; got != want {
		t.Fatalf("usage.Provider = %q, want %q", got, want)
	}
	if got, want := usage.TokenScope, "total"; got != want {
		t.Fatalf("usage.TokenScope = %q, want %q", got, want)
	}
	if got, want := usage.Tokens, 1809; got != want {
		t.Fatalf("usage.Tokens = %d, want %d", got, want)
	}
	if got, want := usage.LocalDelta, 17; got != want {
		t.Fatalf("usage.LocalDelta = %d, want %d", got, want)
	}
	if usage.Usage == nil {
		t.Fatal("usage.Usage = nil, want canonical and raw usage")
	}
	if got, want := usage.Usage.PromptTokens, 1660; got != want {
		t.Fatalf("usage.Usage.PromptTokens = %d, want %d", got, want)
	}
	if got, want := usage.Usage.CompletionTokens, 149; got != want {
		t.Fatalf("usage.Usage.CompletionTokens = %d, want %d", got, want)
	}
	if !jsonEqual(t, usage.Usage.Raw, json.RawMessage(`{"prompt_tokens":1660,"completion_tokens":149,"total_tokens":1809,"cached_tokens":0}`)) {
		t.Fatalf("usage.Usage.Raw = %s, want semantic match", string(usage.Usage.Raw))
	}

	replay := loaded.ReplayEvents()
	if len(replay) != 3 {
		t.Fatalf("replay event count = %d, want 3", len(replay))
	}
}

func TestAppendToolResultPersistsAndReplaysMeta(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}

	meta := map[string]any{
		"status":      "completed",
		"duration_ms": int64(25),
		"exit_code":   0,
		"bytes":       int64(123),
		"edit_diff": map[string]any{
			"path":  "a.txt",
			"lines": []any{"-old", "+new"},
		},
	}
	if err := s.AppendToolResult("call-1", "edit", "edited", meta); err != nil {
		t.Fatalf("append tool result: %v", err)
	}
	meta["status"] = "mutated"
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	data, err := os.ReadFile(s.Path())
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	if !strings.Contains(string(data), `"meta"`) || !strings.Contains(string(data), `"edit_diff"`) {
		t.Fatalf("trajectory missing metadata:\n%s", string(data))
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load session: %v", err)
	}
	t.Cleanup(func() { _ = loaded.Close() })

	replay := loaded.ReplayEvents()
	if len(replay) != 1 {
		t.Fatalf("replay len = %d, want 1 (%#v)", len(replay), replay)
	}
	got := replay[0].Meta
	if got["status"] != "completed" {
		t.Fatalf("status = %#v, want completed", got["status"])
	}
	if _, ok := got["duration_ms"].(int64); !ok {
		t.Fatalf("duration_ms = %#v, want int64", got["duration_ms"])
	}
	if _, ok := got["exit_code"].(int); !ok {
		t.Fatalf("exit_code = %#v, want int", got["exit_code"])
	}
	if _, ok := got["bytes"].(int64); !ok {
		t.Fatalf("bytes = %#v, want int64", got["bytes"])
	}
	if got["edit_diff"] == nil {
		t.Fatalf("edit_diff missing after replay: %#v", got)
	}
}

func TestLoadByIDFallsBackToLegacySnapshotSidecar(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.AppendUserInput("legacy user"); err != nil {
		t.Fatalf("append user input: %v", err)
	}
	if err := s.AppendAssistant("legacy assistant"); err != nil {
		t.Fatalf("append assistant: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	data, err := os.ReadFile(s.Path())
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) < 2 {
		t.Fatalf("legacy trajectory lines = %d, want at least 2", len(lines))
	}
	if err := os.WriteFile(s.Path(), []byte(strings.Join([]string{lines[0], lines[len(lines)-1]}, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("rewrite legacy trajectory: %v", err)
	}

	legacySnapshot := Snapshot{
		SessionID:    s.ID(),
		WorkDir:      workDir,
		SystemPrompt: "legacy prompt",
		UpdatedAt:    time.Now(),
		Messages: []llm.Message{
			llm.NewUserMessage("legacy user"),
			llm.NewAssistantMessage("legacy assistant"),
		},
		ProviderUsage: &UsageSnapshot{
			Provider:   "anthropic",
			TokenScope: "total",
			Tokens:     42,
		},
	}
	snapshotData, err := json.MarshalIndent(legacySnapshot, "", "  ")
	if err != nil {
		t.Fatalf("marshal legacy snapshot: %v", err)
	}
	if err := os.WriteFile(snapshotPath(s.Path()), snapshotData, 0o600); err != nil {
		t.Fatalf("write legacy snapshot: %v", err)
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load legacy session: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	systemPrompt, messages := loaded.RestoreContext()
	if got, want := systemPrompt, "legacy prompt"; got != want {
		t.Fatalf("restored legacy system prompt = %q, want %q", got, want)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("restored legacy message count = %d, want %d", got, want)
	}
	if usage := loaded.UsageSnapshot(); usage == nil || usage.Tokens != 42 {
		t.Fatalf("legacy usage snapshot = %#v, want 42 tokens", usage)
	}
}

func TestSaveSnapshotWithUsageOnlyWritesContextBoundaryAfterCompaction(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}

	usage := &UsageSnapshot{
		Provider:   "anthropic",
		TokenScope: "total",
		Tokens:     99,
	}

	if err := s.AppendUserInput("first request"); err != nil {
		t.Fatalf("append first user input: %v", err)
	}
	if err := s.AppendAssistant("first reply"); err != nil {
		t.Fatalf("append first assistant reply: %v", err)
	}
	if err := s.SaveSnapshotWithUsage("system prompt", []llm.Message{
		llm.NewUserMessage("first request"),
		llm.NewAssistantMessage("first reply"),
	}, usage); err != nil {
		t.Fatalf("save first snapshot: %v", err)
	}

	if err := s.AppendUserInput("second request"); err != nil {
		t.Fatalf("append second user input: %v", err)
	}
	if err := s.AppendAssistant("second reply"); err != nil {
		t.Fatalf("append second assistant reply: %v", err)
	}
	if err := s.SaveSnapshotWithUsage("system prompt", []llm.Message{
		llm.NewUserMessage("first request"),
		llm.NewAssistantMessage("first reply"),
		llm.NewUserMessage("second request"),
		llm.NewAssistantMessage("second reply"),
	}, usage); err != nil {
		t.Fatalf("save second snapshot: %v", err)
	}

	if err := s.SaveSnapshotWithUsage("system prompt", []llm.Message{
		llm.NewAssistantMessage("Summary:\nkeep working from here."),
	}, usage); err != nil {
		t.Fatalf("save compact boundary snapshot: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	data, err := os.ReadFile(s.Path())
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	text := string(data)
	if got, want := strings.Count(text, `"type":"resume_state"`), 2; got != want {
		t.Fatalf("resume_state record count = %d, want %d\n%s", got, want, text)
	}
	if got, want := strings.Count(text, `"messages"`), 1; got != want {
		t.Fatalf("resume_state messages field count = %d, want %d\n%s", got, want, text)
	}
	if got, want := strings.Count(text, `"context_boundary":true`), 1; got != want {
		t.Fatalf("context boundary count = %d, want %d\n%s", got, want, text)
	}
}

func TestLoadManualCheckpointTrajectorySkipsCheckpointForReplay(t *testing.T) {
	homeDir := t.TempDir()
	t.Setenv("HOME", homeDir)

	workDir := t.TempDir()
	absWorkDir, err := filepath.Abs(workDir)
	if err != nil {
		t.Fatalf("abs work dir: %v", err)
	}
	sessionID := "sess_manual_checkpoint"
	path := trajectoryPath(workDirKey(absWorkDir), sessionID)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir trajectory dir: %v", err)
	}

	now := time.Date(2026, 5, 15, 12, 0, 0, 0, time.UTC)
	records := []any{
		Meta{Type: recordTypeMeta, Version: formatVersion, SessionID: sessionID, WorkDir: absWorkDir, WorkDirKey: workDirKey(absWorkDir), SystemPrompt: "system prompt", CreatedAt: now, UpdatedAt: now},
		CheckpointRecord{Type: recordTypeCheckpoint, Timestamp: now.Add(time.Second), MessageID: "msg-checkpoint", Preview: "checkpoint preview should not replay", ContextEntryCount: 1},
		MessageRecord{Type: recordTypeUser, Timestamp: now.Add(2 * time.Second), MessageID: "msg-user", Content: "user request"},
		MessageRecord{Type: recordTypeAssistant, Timestamp: now.Add(3 * time.Second), MessageID: "msg-assistant", Content: "assistant reply"},
	}
	var lines []string
	for _, record := range records {
		data, err := json.Marshal(record)
		if err != nil {
			t.Fatalf("marshal record: %v", err)
		}
		lines = append(lines, string(data))
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("write trajectory: %v", err)
	}

	loaded, err := LoadByID(workDir, sessionID)
	if err != nil {
		t.Fatalf("load manual checkpoint trajectory: %v", err)
	}
	t.Cleanup(func() { _ = loaded.Close() })

	_, messages := loaded.RestoreContext()
	if got, want := len(messages), 2; got != want {
		t.Fatalf("restored message count = %d, want %d", got, want)
	}
	if messages[0].Role != "user" || messages[0].Content != "user request" {
		t.Fatalf("first message = %#v, want user request", messages[0])
	}
	if messages[1].Role != "assistant" || messages[1].Content != "assistant reply" {
		t.Fatalf("second message = %#v, want assistant reply", messages[1])
	}
	for _, message := range messages {
		if strings.Contains(message.Content, "checkpoint") {
			t.Fatalf("checkpoint content replayed as message: %#v", message)
		}
	}
}

func TestLoadTrajectorySkipsCheckpointRecordsForReplay(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.AppendUserInput("user request"); err != nil {
		t.Fatalf("append user input: %v", err)
	}
	if err := s.AppendAssistant("assistant reply"); err != nil {
		t.Fatalf("append assistant reply: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load session with checkpoint: %v", err)
	}
	t.Cleanup(func() { _ = loaded.Close() })

	_, messages := loaded.RestoreContext()
	if got, want := len(messages), 2; got != want {
		t.Fatalf("restored message count = %d, want %d", got, want)
	}
	for _, message := range messages {
		if strings.Contains(message.Content, "checkpoint") {
			t.Fatalf("checkpoint content replayed as message: %#v", message)
		}
	}
}

func TestRestoreContextReconstructsFromCompactBoundaryAndLaterTrajectory(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.AppendUserInput("before compact"); err != nil {
		t.Fatalf("append pre-compact user input: %v", err)
	}
	if err := s.AppendAssistant("before compact reply"); err != nil {
		t.Fatalf("append pre-compact assistant reply: %v", err)
	}
	if err := s.SaveSnapshot("system prompt", []llm.Message{
		llm.NewAssistantMessage("Summary:\ncontinue from the compacted state."),
	}); err != nil {
		t.Fatalf("save compacted state: %v", err)
	}
	if err := s.AppendContextCompaction("manual", 120, 40, "Context compacted: 120 -> 40 tokens."); err != nil {
		t.Fatalf("append compaction notice: %v", err)
	}
	if err := s.AppendUserInput("after compact"); err != nil {
		t.Fatalf("append post-compact user input: %v", err)
	}
	if err := s.AppendAssistant("after compact reply"); err != nil {
		t.Fatalf("append post-compact assistant reply: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load session: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	systemPrompt, messages := loaded.RestoreContext()
	if got, want := systemPrompt, "system prompt"; got != want {
		t.Fatalf("restored system prompt = %q, want %q", got, want)
	}
	if got, want := len(messages), 3; got != want {
		t.Fatalf("restored message count = %d, want %d", got, want)
	}
	if got, want := messages[0].Content, "Summary:\ncontinue from the compacted state."; got != want {
		t.Fatalf("restored compact summary = %q, want %q", got, want)
	}
	if got, want := messages[1].Content, "after compact"; got != want {
		t.Fatalf("restored post-compact user = %q, want %q", got, want)
	}
	if got, want := messages[2].Content, "after compact reply"; got != want {
		t.Fatalf("restored post-compact assistant = %q, want %q", got, want)
	}
}

func TestMessagesFromCheckpointFallsBackToTrajectoryWhenCompactBoundaryReplacedPrefix(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.AppendUserInput("first request"); err != nil {
		t.Fatalf("append first user input: %v", err)
	}
	if err := s.AppendAssistant("first reply"); err != nil {
		t.Fatalf("append first assistant reply: %v", err)
	}
	if err := s.AppendUserInput("second request"); err != nil {
		t.Fatalf("append second user input: %v", err)
	}
	secondID := s.ListCheckpoints()[0].MessageID
	if err := s.AppendAssistant("second reply"); err != nil {
		t.Fatalf("append second assistant reply: %v", err)
	}
	if err := s.SaveSnapshot("system prompt", []llm.Message{
		llm.NewUserMessage("Summary:\nfirst and second turns."),
	}); err != nil {
		t.Fatalf("save compacted state: %v", err)
	}
	if err := s.AppendContextCompaction("manual", 100, 20, "Context compacted."); err != nil {
		t.Fatalf("append compaction notice: %v", err)
	}
	if err := s.AppendUserInput("third request"); err != nil {
		t.Fatalf("append third user input: %v", err)
	}

	segment, err := s.MessagesFromCheckpoint(secondID)
	if err != nil {
		t.Fatalf("MessagesFromCheckpoint() error = %v", err)
	}
	if got, want := len(segment), 3; got != want {
		t.Fatalf("segment message count = %d, want %d", got, want)
	}
	if got, want := segment[0].Content, "second request"; got != want {
		t.Fatalf("segment first message = %q, want %q", got, want)
	}
	if got, want := segment[2].Content, "third request"; got != want {
		t.Fatalf("segment latest message = %q, want %q", got, want)
	}
}

func TestCheckpointsRestoreConversationStateFromBeforeUserTurn(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.AppendUserInput("first request"); err != nil {
		t.Fatalf("append first user input: %v", err)
	}
	if err := s.AppendAssistant("first reply"); err != nil {
		t.Fatalf("append first assistant reply: %v", err)
	}
	if err := s.SaveSnapshot("system prompt", []llm.Message{
		llm.NewUserMessage("first request"),
		llm.NewAssistantMessage("first reply"),
	}); err != nil {
		t.Fatalf("save first snapshot: %v", err)
	}
	if err := s.AppendUserInput("second request"); err != nil {
		t.Fatalf("append second user input: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load session: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	checkpoints := loaded.ListCheckpoints()
	if got, want := len(checkpoints), 2; got != want {
		t.Fatalf("checkpoint count = %d, want %d", got, want)
	}
	if got, want := checkpoints[0].Preview, "second request"; got != want {
		t.Fatalf("latest checkpoint preview = %q, want %q", got, want)
	}
	if got, want := checkpoints[0].LastUserInput, "first request"; got != want {
		t.Fatalf("latest checkpoint last user input = %q, want %q", got, want)
	}
	if got, want := checkpoints[0].TurnCount, 1; got != want {
		t.Fatalf("latest checkpoint turn count = %d, want %d", got, want)
	}
	if got, want := checkpoints[1].TurnCount, 0; got != want {
		t.Fatalf("first checkpoint turn count = %d, want %d", got, want)
	}

	systemPrompt, messages, usage, err := loaded.RestoreCheckpointContext(checkpoints[0].MessageID)
	if err != nil {
		t.Fatalf("RestoreCheckpointContext() error = %v", err)
	}
	if got, want := systemPrompt, "system prompt"; got != want {
		t.Fatalf("checkpoint system prompt = %q, want %q", got, want)
	}
	if got, want := len(messages), 2; got != want {
		t.Fatalf("checkpoint message count = %d, want %d", got, want)
	}
	if got, want := messages[0].Content, "first request"; got != want {
		t.Fatalf("checkpoint first message = %q, want %q", got, want)
	}
	if usage != nil {
		t.Fatalf("checkpoint usage = %#v, want nil", usage)
	}

	segment, err := loaded.MessagesFromCheckpoint(checkpoints[0].MessageID)
	if err != nil {
		t.Fatalf("MessagesFromCheckpoint() error = %v", err)
	}
	if got, want := len(segment), 1; got != want {
		t.Fatalf("checkpoint segment message count = %d, want %d", got, want)
	}
	if got, want := segment[0].Content, "second request"; got != want {
		t.Fatalf("checkpoint segment first message = %q, want %q", got, want)
	}
}

func TestRestoreCheckpointFilesRevertsTrackedWriteAndCreate(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	trackedPath := filepath.Join(workDir, "tracked.txt")
	if err := os.WriteFile(trackedPath, []byte("before"), 0o644); err != nil {
		t.Fatalf("write tracked seed: %v", err)
	}

	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.AppendUserInput("seed"); err != nil {
		t.Fatalf("append seed user input: %v", err)
	}
	if err := s.AppendAssistant("seed reply"); err != nil {
		t.Fatalf("append seed assistant reply: %v", err)
	}
	if err := s.SaveSnapshot("system prompt", []llm.Message{
		llm.NewUserMessage("seed"),
		llm.NewAssistantMessage("seed reply"),
	}); err != nil {
		t.Fatalf("save seed snapshot: %v", err)
	}
	if err := s.AppendUserInput("change files"); err != nil {
		t.Fatalf("append mutation user input: %v", err)
	}
	if err := s.RecordFileMutation("tracked.txt", trackedPath); err != nil {
		t.Fatalf("RecordFileMutation(existing) error = %v", err)
	}
	if err := os.WriteFile(trackedPath, []byte("after"), 0o644); err != nil {
		t.Fatalf("write tracked mutation: %v", err)
	}

	createdPath := filepath.Join(workDir, "created.txt")
	if err := s.RecordFileMutation("created.txt", createdPath); err != nil {
		t.Fatalf("RecordFileMutation(new) error = %v", err)
	}
	if err := os.WriteFile(createdPath, []byte("created"), 0o644); err != nil {
		t.Fatalf("write created file: %v", err)
	}

	checkpoint := s.ListCheckpoints()[0]
	if !checkpoint.HasCodeRestore {
		t.Fatal("expected code restore to be available after tracked file mutations")
	}
	if err := s.RestoreCheckpointFiles(checkpoint.MessageID); err != nil {
		t.Fatalf("RestoreCheckpointFiles() error = %v", err)
	}

	if got, err := os.ReadFile(trackedPath); err != nil {
		t.Fatalf("read restored tracked file: %v", err)
	} else if string(got) != "before" {
		t.Fatalf("tracked file content = %q, want %q", string(got), "before")
	}
	if _, err := os.Stat(createdPath); !os.IsNotExist(err) {
		t.Fatalf("expected created file removed by restore, got %v", err)
	}
}

func TestForkFromCheckpointCopiesPrefixAndCheckpointBackups(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	trackedPath := filepath.Join(workDir, "tracked.txt")
	if err := os.WriteFile(trackedPath, []byte("base"), 0o644); err != nil {
		t.Fatalf("write tracked seed: %v", err)
	}

	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}

	if err := s.AppendUserInput("first request"); err != nil {
		t.Fatalf("append first user input: %v", err)
	}
	if err := s.AppendAssistant("first reply"); err != nil {
		t.Fatalf("append first assistant reply: %v", err)
	}
	if err := s.SaveSnapshot("system prompt", []llm.Message{
		llm.NewUserMessage("first request"),
		llm.NewAssistantMessage("first reply"),
	}); err != nil {
		t.Fatalf("save first snapshot: %v", err)
	}

	if err := s.AppendUserInput("second request"); err != nil {
		t.Fatalf("append second user input: %v", err)
	}
	if err := s.RecordFileMutation("tracked.txt", trackedPath); err != nil {
		t.Fatalf("record second-turn file mutation: %v", err)
	}
	if err := os.WriteFile(trackedPath, []byte("second turn"), 0o644); err != nil {
		t.Fatalf("write second-turn content: %v", err)
	}
	if err := s.AppendAssistant("second reply"); err != nil {
		t.Fatalf("append second assistant reply: %v", err)
	}
	if err := s.SaveSnapshot("system prompt", []llm.Message{
		llm.NewUserMessage("first request"),
		llm.NewAssistantMessage("first reply"),
		llm.NewUserMessage("second request"),
		llm.NewAssistantMessage("second reply"),
	}); err != nil {
		t.Fatalf("save second snapshot: %v", err)
	}

	if err := s.AppendUserInput("third request"); err != nil {
		t.Fatalf("append third user input: %v", err)
	}

	checkpoints := s.ListCheckpoints()
	if got, want := len(checkpoints), 3; got != want {
		t.Fatalf("checkpoint count = %d, want %d", got, want)
	}
	thirdID := checkpoints[0].MessageID
	secondID := checkpoints[1].MessageID

	fork, err := s.ForkFromCheckpoint(thirdID)
	if err != nil {
		t.Fatalf("ForkFromCheckpoint() error = %v", err)
	}
	t.Cleanup(func() {
		_ = fork.Close()
	})
	if err := fork.Activate(); err != nil {
		t.Fatalf("activate fork: %v", err)
	}

	systemPrompt, messages := fork.RestoreContext()
	if got, want := systemPrompt, "system prompt"; got != want {
		t.Fatalf("fork system prompt = %q, want %q", got, want)
	}
	if got, want := len(messages), 4; got != want {
		t.Fatalf("fork message count = %d, want %d", got, want)
	}

	forkCheckpoints := fork.ListCheckpoints()
	if got, want := len(forkCheckpoints), 2; got != want {
		t.Fatalf("fork checkpoint count = %d, want %d", got, want)
	}
	if got, want := forkCheckpoints[0].MessageID, secondID; got != want {
		t.Fatalf("fork latest checkpoint id = %q, want %q", got, want)
	}
	if !forkCheckpoints[0].HasCodeRestore {
		t.Fatal("expected copied checkpoint backups to stay available in fork")
	}

	if err := os.WriteFile(trackedPath, []byte("broken"), 0o644); err != nil {
		t.Fatalf("write broken content: %v", err)
	}
	if err := fork.RestoreCheckpointFiles(secondID); err != nil {
		t.Fatalf("fork RestoreCheckpointFiles() error = %v", err)
	}
	if got, err := os.ReadFile(trackedPath); err != nil {
		t.Fatalf("read rewound tracked file: %v", err)
	} else if string(got) != "base" {
		t.Fatalf("fork rewound tracked file = %q, want %q", string(got), "base")
	}
}

func TestForkCurrentCopiesFullConversationAndCheckpointBackups(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	trackedPath := filepath.Join(workDir, "tracked.txt")
	if err := os.WriteFile(trackedPath, []byte("base"), 0o644); err != nil {
		t.Fatalf("write tracked seed: %v", err)
	}

	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}

	if err := s.AppendUserInput("first request"); err != nil {
		t.Fatalf("append first user input: %v", err)
	}
	if err := s.AppendAssistant("first reply"); err != nil {
		t.Fatalf("append first assistant reply: %v", err)
	}
	if err := s.AppendUserInput("second request"); err != nil {
		t.Fatalf("append second user input: %v", err)
	}
	if err := s.RecordFileMutation("tracked.txt", trackedPath); err != nil {
		t.Fatalf("record second-turn file mutation: %v", err)
	}
	if err := os.WriteFile(trackedPath, []byte("second turn"), 0o644); err != nil {
		t.Fatalf("write second-turn content: %v", err)
	}
	if err := s.AppendAssistant("second reply"); err != nil {
		t.Fatalf("append second assistant reply: %v", err)
	}

	fork, err := s.ForkCurrent()
	if err != nil {
		t.Fatalf("ForkCurrent() error = %v", err)
	}
	t.Cleanup(func() {
		_ = fork.Close()
	})
	if err := fork.Activate(); err != nil {
		t.Fatalf("activate fork: %v", err)
	}

	systemPrompt, messages := fork.RestoreContext()
	if got, want := systemPrompt, "system prompt"; got != want {
		t.Fatalf("fork system prompt = %q, want %q", got, want)
	}
	if got, want := len(messages), 4; got != want {
		t.Fatalf("fork message count = %d, want %d", got, want)
	}
	if got, want := messages[3].Content, "second reply"; got != want {
		t.Fatalf("fork latest message = %q, want %q", got, want)
	}

	forkCheckpoints := fork.ListCheckpoints()
	if got, want := len(forkCheckpoints), 2; got != want {
		t.Fatalf("fork checkpoint count = %d, want %d", got, want)
	}
	if !forkCheckpoints[0].HasCodeRestore {
		t.Fatal("expected copied checkpoint backups to stay available in current fork")
	}
}

func TestWorkDirKeySanitizesWindowsInvalidFilenameChars(t *testing.T) {
	key := workDirKey(`C:\Users\alice\work\mscli`)

	for _, invalid := range []string{`\\`, ":", "*", "?", `"`, "<", ">", "|", "/"} {
		if strings.Contains(key, invalid) {
			t.Fatalf("workDirKey(%q) = %q, contains invalid filename char %q", `C:\Users\alice\work\mscli`, key, invalid)
		}
	}
	if strings.Trim(key, ".- ") == "" {
		t.Fatalf("workDirKey(%q) = %q, want non-empty safe key", `C:\Users\alice\work\mscli`, key)
	}
}

func TestContextCompactionRecordReplaysAsContextNotice(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.AppendUserInput("hello"); err != nil {
		t.Fatalf("append user input: %v", err)
	}
	const compactMessage = "Context compacted automatically: 120 -> 60 tokens."
	if err := s.AppendContextCompaction("auto", 120, 60, compactMessage); err != nil {
		t.Fatalf("append context compaction: %v", err)
	}
	if err := s.AppendAssistant("done"); err != nil {
		t.Fatalf("append assistant: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load activated session: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	replay := loaded.ReplayEvents()
	if len(replay) != 3 {
		t.Fatalf("replay event count = %d, want 3", len(replay))
	}
	if got, want := replay[1].Type, model.ContextNotice; got != want {
		t.Fatalf("compact replay type = %q, want %q", got, want)
	}
	if got := replay[1].Message; got != compactMessage {
		t.Fatalf("compact replay message = %q, want %q", got, compactMessage)
	}
	if got := replay[1].CtxUsed; got != 60 {
		t.Fatalf("compact replay CtxUsed = %d, want 60", got)
	}
	if got := replay[1].Meta["trigger"]; got != "auto" {
		t.Fatalf("compact replay trigger meta = %#v, want auto", got)
	}
}

func jsonEqual(t *testing.T, got, want json.RawMessage) bool {
	t.Helper()

	var gotValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("unmarshal got json: %v", err)
	}

	var wantValue any
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("unmarshal want json: %v", err)
	}

	return reflect.DeepEqual(gotValue, wantValue)
}

func TestReplayTimelinePreservesRecordTimestamps(t *testing.T) {
	t0 := time.Date(2026, time.March, 27, 10, 0, 0, 0, time.UTC)
	t1 := t0.Add(150 * time.Millisecond)
	t2 := t1.Add(250 * time.Millisecond)

	s := &Session{
		records: []MessageRecord{
			{Type: recordTypeUser, Timestamp: t0, Content: "hello"},
			{Type: recordTypeToolCall, Timestamp: t1, ToolName: "shell", Arguments: []byte(`{"command":"pwd"}`)},
			{Type: recordTypeAssistant, Timestamp: t2, Content: "done"},
		},
	}

	timeline := s.ReplayTimeline()
	if len(timeline) != 3 {
		t.Fatalf("timeline length = %d, want 3", len(timeline))
	}
	if !timeline[0].Timestamp.Equal(t0) {
		t.Fatalf("first timestamp = %v, want %v", timeline[0].Timestamp, t0)
	}
	if timeline[0].Event.Type != "UserInput" {
		t.Fatalf("first event type = %q, want %q", timeline[0].Event.Type, "UserInput")
	}
	if !timeline[1].Timestamp.Equal(t1) {
		t.Fatalf("second timestamp = %v, want %v", timeline[1].Timestamp, t1)
	}
	if timeline[1].Event.Type != "ToolCallStart" {
		t.Fatalf("second event type = %q, want %q", timeline[1].Event.Type, "ToolCallStart")
	}
	if !timeline[2].Timestamp.Equal(t2) {
		t.Fatalf("third timestamp = %v, want %v", timeline[2].Timestamp, t2)
	}
	if timeline[2].Event.Type != "AgentReply" {
		t.Fatalf("third event type = %q, want %q", timeline[2].Event.Type, "AgentReply")
	}
}

func TestReplayTimelineAssignsLoadSkillActivationToOriginalToolCall(t *testing.T) {
	t0 := time.Date(2026, time.March, 27, 10, 0, 0, 0, time.UTC)
	t1 := t0.Add(10 * time.Millisecond)
	t2 := t1.Add(10 * time.Millisecond)
	t3 := t2.Add(10 * time.Millisecond)

	s := &Session{
		records: []MessageRecord{
			{Type: recordTypeToolCall, Timestamp: t0, ToolName: "load_skill", ToolCallID: "call_skill", Arguments: []byte(`{"name":"model-agent"}`)},
			{Type: recordTypeToolCall, Timestamp: t1, ToolName: "glob", ToolCallID: "call_glob", Arguments: []byte(`{"pattern":"**/*.py"}`)},
			{Type: recordTypeSkill, Timestamp: t2, SkillName: "model-agent"},
			{Type: recordTypeToolResult, Timestamp: t3, ToolName: "glob", ToolCallID: "call_glob", Content: "a.py"},
		},
	}

	timeline := s.ReplayTimeline()
	if len(timeline) != 4 {
		t.Fatalf("timeline length = %d, want 4", len(timeline))
	}
	if got, want := timeline[2].Event.Type, model.ToolSkill; got != want {
		t.Fatalf("third event type = %q, want %q", got, want)
	}
	if got, want := timeline[2].Event.ToolCallID, "call_skill"; got != want {
		t.Fatalf("skill replay tool call id = %q, want %q", got, want)
	}
	if got, want := timeline[3].Event.ToolCallID, "call_glob"; got != want {
		t.Fatalf("glob replay tool call id = %q, want %q", got, want)
	}
}

func TestLoadReplayPathAcceptsTrajectoryJSONFilename(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	if err := s.AppendUserInput("hello"); err != nil {
		t.Fatalf("append user input: %v", err)
	}
	if err := s.AppendAssistant("hi"); err != nil {
		t.Fatalf("append assistant: %v", err)
	}
	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}

	replayDir := t.TempDir()
	replayPath := filepath.Join(replayDir, "trajectory.json")
	data, err := os.ReadFile(s.Path())
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	if err := os.WriteFile(replayPath, data, 0600); err != nil {
		t.Fatalf("write replay trajectory: %v", err)
	}

	loaded, err := LoadReplayPath(replayPath)
	if err != nil {
		t.Fatalf("load replay path: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	if got := filepath.Base(loaded.Path()); got != "trajectory.json" {
		t.Fatalf("loaded path base = %q, want %q", got, "trajectory.json")
	}
	replay := loaded.ReplayEvents()
	if len(replay) != 2 {
		t.Fatalf("replay event count = %d, want 2", len(replay))
	}
	if got := replay[0].Type; got != "UserInput" {
		t.Fatalf("first event type = %q, want %q", got, "UserInput")
	}
	if got := replay[1].Type; got != "AgentReply" {
		t.Fatalf("second event type = %q, want %q", got, "AgentReply")
	}
}

func TestListForWorkDirReturnsRecentDialogueSummaries(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()

	empty, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create empty session: %v", err)
	}
	if err := empty.Activate(); err != nil {
		t.Fatalf("activate empty session: %v", err)
	}
	if err := empty.Close(); err != nil {
		t.Fatalf("close empty session: %v", err)
	}

	first, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create first session: %v", err)
	}
	if err := first.AppendUserInput("first prompt line\nextra detail"); err != nil {
		t.Fatalf("append first user input: %v", err)
	}
	if err := first.AppendAssistant("first reply"); err != nil {
		t.Fatalf("append first assistant reply: %v", err)
	}
	if err := first.Activate(); err != nil {
		t.Fatalf("activate first session: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close first session: %v", err)
	}

	time.Sleep(20 * time.Millisecond)

	second, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create second session: %v", err)
	}
	if err := second.AppendUserInput("second prompt"); err != nil {
		t.Fatalf("append second user input: %v", err)
	}
	if err := second.AppendAssistant("second reply"); err != nil {
		t.Fatalf("append second assistant reply: %v", err)
	}
	if err := second.Activate(); err != nil {
		t.Fatalf("activate second session: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("close second session: %v", err)
	}

	summaries, err := ListForWorkDir(workDir)
	if err != nil {
		t.Fatalf("list sessions: %v", err)
	}
	if got, want := len(summaries), 2; got != want {
		t.Fatalf("summary count = %d, want %d", got, want)
	}
	if got, want := summaries[0].SessionID, second.ID(); got != want {
		t.Fatalf("latest session id = %q, want %q", got, want)
	}
	if got, want := summaries[0].FirstUserInput, "second prompt"; got != want {
		t.Fatalf("latest first user input = %q, want %q", got, want)
	}
	if got, want := summaries[0].LastUserInput, "second prompt"; got != want {
		t.Fatalf("latest last user input = %q, want %q", got, want)
	}
	if got, want := summaries[0].TurnCount, 1; got != want {
		t.Fatalf("latest turn count = %d, want %d", got, want)
	}
	if got, want := summaries[1].SessionID, first.ID(); got != want {
		t.Fatalf("older session id = %q, want %q", got, want)
	}
	if got, want := summaries[1].FirstUserInput, "first prompt line"; got != want {
		t.Fatalf("older first user input = %q, want %q", got, want)
	}
	if got, want := summaries[1].LastUserInput, "first prompt line"; got != want {
		t.Fatalf("older last user input = %q, want %q", got, want)
	}
	if got, want := summaries[1].TurnCount, 1; got != want {
		t.Fatalf("older turn count = %d, want %d", got, want)
	}
}

func TestCleanupExpiredRemovesOnlyStaleSessions(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()

	stale, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create stale session: %v", err)
	}
	if err := stale.AppendUserInput("stale prompt"); err != nil {
		t.Fatalf("append stale user input: %v", err)
	}
	if err := stale.Activate(); err != nil {
		t.Fatalf("activate stale session: %v", err)
	}
	if err := stale.Close(); err != nil {
		t.Fatalf("close stale session: %v", err)
	}

	staleDir := filepath.Dir(stale.Path())
	staleTime := time.Now().Add(-45 * 24 * time.Hour)
	for _, path := range []string{staleDir, stale.Path()} {
		if err := os.Chtimes(path, staleTime, staleTime); err != nil {
			t.Fatalf("chtimes stale path %s: %v", path, err)
		}
	}

	fresh, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create fresh session: %v", err)
	}
	if err := fresh.AppendUserInput("fresh prompt"); err != nil {
		t.Fatalf("append fresh user input: %v", err)
	}
	if err := fresh.Activate(); err != nil {
		t.Fatalf("activate fresh session: %v", err)
	}
	if err := fresh.Close(); err != nil {
		t.Fatalf("close fresh session: %v", err)
	}

	removed, err := CleanupExpired(30 * 24 * time.Hour)
	if err != nil {
		t.Fatalf("CleanupExpired() error = %v", err)
	}
	if got, want := removed, 1; got != want {
		t.Fatalf("removed session count = %d, want %d", got, want)
	}
	if _, err := os.Stat(staleDir); !os.IsNotExist(err) {
		t.Fatalf("expected stale session dir removed, got %v", err)
	}
	if _, err := os.Stat(filepath.Dir(fresh.Path())); err != nil {
		t.Fatalf("expected fresh session dir kept: %v", err)
	}
}

func TestCleanupExpiredPreservesMemoryAndNonSessionDirs(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	stale, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create stale session: %v", err)
	}
	if err := stale.AppendUserInput("stale prompt"); err != nil {
		t.Fatalf("append stale user input: %v", err)
	}
	if err := stale.Activate(); err != nil {
		t.Fatalf("activate stale session: %v", err)
	}
	if err := stale.Close(); err != nil {
		t.Fatalf("close stale session: %v", err)
	}

	bucketDir, err := BucketDirForWorkDir(workDir)
	if err != nil {
		t.Fatalf("BucketDirForWorkDir() error = %v", err)
	}
	memoryDir := filepath.Join(bucketDir, "memory")
	otherDir := filepath.Join(bucketDir, "not_a_session")
	for _, dir := range []string{memoryDir, otherDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("MkdirAll(%s) error = %v", dir, err)
		}
	}

	staleTime := time.Now().Add(-45 * 24 * time.Hour)
	for _, path := range []string{filepath.Dir(stale.Path()), stale.Path(), memoryDir, otherDir} {
		if err := os.Chtimes(path, staleTime, staleTime); err != nil {
			t.Fatalf("chtimes stale path %s: %v", path, err)
		}
	}

	removed, err := CleanupExpired(30 * 24 * time.Hour)
	if err != nil {
		t.Fatalf("CleanupExpired() error = %v", err)
	}
	if got, want := removed, 1; got != want {
		t.Fatalf("removed session count = %d, want %d", got, want)
	}
	if _, err := os.Stat(filepath.Dir(stale.Path())); !os.IsNotExist(err) {
		t.Fatalf("expected stale session dir removed, got %v", err)
	}
	if _, err := os.Stat(memoryDir); err != nil {
		t.Fatalf("expected memory dir kept: %v", err)
	}
	if _, err := os.Stat(otherDir); err != nil {
		t.Fatalf("expected non-session dir kept: %v", err)
	}
}

func TestPlaybackTimelineInsertsThinkingBetweenUserAndLLMResponse(t *testing.T) {
	t0 := time.Date(2026, time.March, 27, 10, 0, 0, 0, time.UTC)
	t1 := t0.Add(2 * time.Second)
	t2 := t1.Add(3 * time.Second)
	t3 := t2.Add(1 * time.Second)

	s := &Session{
		records: []MessageRecord{
			{Type: recordTypeUser, Timestamp: t0, Content: "hello"},
			{Type: recordTypeToolCall, Timestamp: t1, ToolName: "shell", Arguments: []byte(`{"command":"pwd"}`)},
			{Type: recordTypeToolResult, Timestamp: t2, ToolName: "shell", Content: "/tmp"},
			{Type: recordTypeAssistant, Timestamp: t3, Content: "done"},
		},
	}

	playback := s.PlaybackTimeline()
	if len(playback) != 8 {
		t.Fatalf("playback timeline length = %d, want 8", len(playback))
	}
	if playback[0].Event.Type != "UserInput" {
		t.Fatalf("first event type = %q, want %q", playback[0].Event.Type, "UserInput")
	}
	if playback[1].Event.Type != "AgentThinking" {
		t.Fatalf("second event type = %q, want %q", playback[1].Event.Type, "AgentThinking")
	}
	if !playback[1].Timestamp.Equal(t0) {
		t.Fatalf("thinking timestamp after user = %v, want %v", playback[1].Timestamp, t0)
	}
	if playback[2].Event.Type != "ToolCallStart" {
		t.Fatalf("third event type = %q, want %q", playback[2].Event.Type, "ToolCallStart")
	}
	if playback[3].Event.Type != "ToolReplay" {
		t.Fatalf("fourth event type = %q, want %q", playback[3].Event.Type, "ToolReplay")
	}
	if playback[4].Event.Type != "AgentThinking" {
		t.Fatalf("fifth event type = %q, want %q", playback[4].Event.Type, "AgentThinking")
	}
	if !playback[4].Timestamp.Equal(t2) {
		t.Fatalf("thinking timestamp after tool result = %v, want %v", playback[4].Timestamp, t2)
	}
	if playback[5].Event.Type != "AgentReplyDelta" {
		t.Fatalf("sixth event type = %q, want %q", playback[5].Event.Type, "AgentReplyDelta")
	}
	if playback[6].Event.Type != "AgentReplyDelta" {
		t.Fatalf("seventh event type = %q, want %q", playback[6].Event.Type, "AgentReplyDelta")
	}
	if playback[7].Event.Type != "AgentReply" {
		t.Fatalf("eighth event type = %q, want %q", playback[7].Event.Type, "AgentReply")
	}
	if got := playback[5].Event.Message + playback[6].Event.Message; got != "done" {
		t.Fatalf("delta content = %q, want %q", got, "done")
	}
}

func TestPlaybackTimelineCapsLongShellReplayToFiveSeconds(t *testing.T) {
	t0 := time.Date(2026, time.March, 27, 11, 0, 0, 0, time.UTC)
	t1 := t0.Add(12 * time.Second)
	t2 := t1.Add(8 * time.Second)

	s := &Session{
		records: []MessageRecord{
			{Type: recordTypeToolCall, Timestamp: t0, ToolName: "shell", Arguments: []byte(`{"command":"sleep 12"}`)},
			{Type: recordTypeToolResult, Timestamp: t1, ToolName: "shell", Content: "done"},
			{Type: recordTypeUser, Timestamp: t2, Content: "next"},
		},
	}

	playback := s.PlaybackTimeline()
	if len(playback) != 3 {
		t.Fatalf("playback timeline length = %d, want 3", len(playback))
	}
	if got, want := playback[1].Timestamp.Sub(playback[0].Timestamp), 5*time.Second; got != want {
		t.Fatalf("compressed shell duration = %v, want %v", got, want)
	}
	if got, want := playback[2].Timestamp.Sub(playback[1].Timestamp), 8*time.Second; got != want {
		t.Fatalf("post-shell gap = %v, want %v", got, want)
	}
	if playback[0].Event.ReplayWait == nil {
		t.Fatal("expected replay wait metadata on shell tool call")
	}
	if got, want := playback[0].Event.ReplayWait.OriginalDuration, 12*time.Second; got != want {
		t.Fatalf("shell original wait = %v, want %v", got, want)
	}
	if got, want := playback[0].Event.ReplayWait.SimulatedDuration, 5*time.Second; got != want {
		t.Fatalf("shell simulated wait = %v, want %v", got, want)
	}
}

func TestPlaybackTimelineCapsLongAssistantReplayToFiveSeconds(t *testing.T) {
	t0 := time.Date(2026, time.March, 27, 12, 0, 0, 0, time.UTC)
	t1 := t0.Add(12 * time.Second)

	s := &Session{
		records: []MessageRecord{
			{Type: recordTypeUser, Timestamp: t0, Content: "hello"},
			{Type: recordTypeAssistant, Timestamp: t1, Content: "done"},
		},
	}

	playback := s.PlaybackTimeline()
	if len(playback) != 5 {
		t.Fatalf("playback timeline length = %d, want 5", len(playback))
	}
	if got, want := playback[4].Timestamp.Sub(playback[1].Timestamp), 5*time.Second; got != want {
		t.Fatalf("compressed assistant duration = %v, want %v", got, want)
	}
	if playback[1].Event.ReplayWait == nil {
		t.Fatal("expected replay wait metadata on thinking event")
	}
	if got, want := playback[1].Event.ReplayWait.OriginalDuration, 4*time.Second; got != want {
		t.Fatalf("assistant original wait = %v, want %v", got, want)
	}
	if got, want := playback[1].Event.ReplayWait.SimulatedDuration, scaleReplayDuration(4*time.Second, 12*time.Second, 5*time.Second); got != want {
		t.Fatalf("assistant simulated wait = %v, want %v", got, want)
	}
}

func TestPlaybackTimelineCapsOverlappingToolCallsInSingleFiveSecondWindow(t *testing.T) {
	t0 := time.Date(2026, time.March, 27, 13, 0, 0, 0, time.UTC)
	t1 := t0.Add(12 * time.Second)
	t2 := t0.Add(12500 * time.Millisecond)

	s := &Session{
		records: []MessageRecord{
			{Type: recordTypeToolCall, Timestamp: t0, ToolName: "glob", ToolCallID: "call_glob_1", Arguments: []byte(`{"pattern":"**/*.log"}`)},
			{Type: recordTypeToolCall, Timestamp: t0, ToolName: "glob", ToolCallID: "call_glob_2", Arguments: []byte(`{"pattern":"**/*.py"}`)},
			{Type: recordTypeToolResult, Timestamp: t1, ToolName: "glob", ToolCallID: "call_glob_1", Content: "a.log"},
			{Type: recordTypeToolResult, Timestamp: t2, ToolName: "glob", ToolCallID: "call_glob_2", Content: "b.py"},
		},
	}

	playback := s.PlaybackTimeline()
	if len(playback) != 4 {
		t.Fatalf("playback timeline length = %d, want 4", len(playback))
	}
	if got, want := playback[3].Timestamp.Sub(playback[0].Timestamp), 5*time.Second; got != want {
		t.Fatalf("compressed tool cluster duration = %v, want %v", got, want)
	}
	if got, want := playback[3].Timestamp.Sub(playback[2].Timestamp), scaleReplayDuration(500*time.Millisecond, 12500*time.Millisecond, 5*time.Second); got != want {
		t.Fatalf("compressed result gap = %v, want %v", got, want)
	}
	if playback[0].Event.ReplayWait == nil {
		t.Fatal("expected replay wait metadata on first overlapping tool call")
	}
	if got, want := playback[0].Event.ReplayWait.OriginalDuration, 12500*time.Millisecond; got != want {
		t.Fatalf("tool cluster original wait = %v, want %v", got, want)
	}
	if got, want := playback[0].Event.ReplayWait.SimulatedDuration, 5*time.Second; got != want {
		t.Fatalf("tool cluster simulated wait = %v, want %v", got, want)
	}
}
