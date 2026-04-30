package loop

import (
	"time"

	"gitcode.com/mindspore/mscli/integrations/llm"
)

// Task represents a user task.
type Task struct {
	ID                   string
	Description          string
	// InitialMessages are inserted into the LLM context before the task's user
	// message, but are not recorded as user-entered prompts.
	InitialMessages     []llm.Message
	// UserMessage overrides the context message while Description remains the visible task label.
	UserMessage          string
	Context              map[string]string
	DisableResearchTools bool
}

// Event represents an engine event.
type Event struct {
	Type       string
	Task       string
	Message    string
	ToolName   string
	ToolCallID string
	Summary    string
	Meta       map[string]any
	CtxUsed    int
	CtxMax     int
	TokensUsed int
	Usage      llm.Usage
	Timestamp  time.Time
}

// NewEvent creates a new event.
func NewEvent(eventType, message string) Event {
	return Event{
		Type:      eventType,
		Message:   message,
		Timestamp: time.Now(),
	}
}

// Event types.
const (
	// Task lifecycle
	EventTaskStarted   = "TaskStarted"
	EventTaskCompleted = "TaskCompleted"
	EventTaskFailed    = "TaskFailed"

	// LLM events
	EventLLMThinking   = "LLMThinking"
	EventLLMResponse   = "LLMResponse"
	EventToolCallStart = "ToolCallStart"

	// Tool events
	EventToolStarted     = "ToolStarted"
	EventToolCompleted   = "ToolCompleted"
	EventToolError       = "ToolError"
	EventToolInterrupted = "ToolInterrupted"

	// UI compatible events
	EventCmdStarted          = "CmdStarted"
	EventCmdOutput           = "CmdOutput"
	EventCmdFinished         = "CmdFinished"
	EventAgentReply          = "AgentReply"
	EventAgentReplyDelta     = "AgentReplyDelta"
	EventAgentBackgroundWork = "AgentBackgroundWork"
	EventAgentThinking       = "AgentThinking"
	EventContextCompactStart = "ContextCompactStarted"
	EventContextCompacted    = "ContextCompacted"
	EventTokenUpdate         = "TokenUpdate"
	EventToolRead            = "ToolRead"
	EventToolGrep            = "ToolGrep"
	EventToolGlob            = "ToolGlob"
	EventToolEdit            = "ToolEdit"
	EventToolWrite           = "ToolWrite"
	EventToolSkill           = "ToolSkill"
	EventToolAskUserQuestion = "ToolAskUserQuestion"
	EventAnalysisReady       = "AnalysisReady"
	EventDone                = "Done"
)
