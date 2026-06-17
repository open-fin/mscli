// Package llm provides a unified interface for LLM providers.
package llm

import (
	"context"
	"encoding/json"
	"strings"
)

// Provider is the unified interface for LLM services.
type Provider interface {
	// Name returns the provider name.
	Name() string

	// Complete performs a non-streaming completion request.
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// CompleteStream performs a streaming completion request.
	CompleteStream(ctx context.Context, req *CompletionRequest) (StreamIterator, error)

	// SupportsTools returns whether the provider supports tool calls.
	SupportsTools() bool

	// AvailableModels returns the list of available models.
	AvailableModels() []ModelInfo
}

// CompletionRequest represents a completion request.
type CompletionRequest struct {
	Model       string
	Messages    []Message
	Tools       []Tool
	Temperature *float32
	MaxTokens   *int
	TopP        float32
	Stop        []string
	Effort      string
}

// CompletionResponse represents a completion response.
type CompletionResponse struct {
	ID           string
	Model        string
	Content      string
	ToolCalls    []ToolCall
	FinishReason FinishReason
	Usage        Usage
}

// FinishReason represents why the completion finished.
type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
)

// Message represents a chat message.
type Message struct {
	Role         string               `json:"role"`
	Content      string               `json:"content"`
	ContentParts []MessageContentPart `json:"content_parts,omitempty"`
	ToolCalls    []ToolCall           `json:"tool_calls,omitempty"`
	ToolCallID   string               `json:"tool_call_id,omitempty"`
}

// MessageContentPart represents one structured text content part.
type MessageContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// Tool represents a tool definition.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents a tool function definition.
type ToolFunction struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  ToolSchema `json:"parameters"`
}

// ToolSchema represents the JSON schema for tool parameters.
type ToolSchema struct {
	Type                 string              `json:"type"`
	Properties           map[string]Property `json:"properties,omitempty"`
	Required             []string            `json:"required,omitempty"`
	AdditionalProperties any                 `json:"additionalProperties,omitempty"`
}

// Property represents a property in the tool schema.
type Property struct {
	Type                 string              `json:"type"`
	Description          string              `json:"description"`
	Enum                 []string            `json:"enum,omitempty"`
	Items                *Property           `json:"items,omitempty"`
	Properties           map[string]Property `json:"properties,omitempty"`
	Required             []string            `json:"required,omitempty"`
	AdditionalProperties any                 `json:"additionalProperties,omitempty"`
}

// ToolCall represents a tool call request from the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolCallFunc `json:"function"`
}

// ToolCallFunc contains the function information for a tool call.
type ToolCallFunc struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResult represents the result of a tool execution.
type ToolResult struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	Raw              json.RawMessage
}

// Clone returns a deep copy of the usage payload.
func (u Usage) Clone() Usage {
	u.Raw = cloneRawJSON(u.Raw)
	return u
}

// IsZero reports whether the usage has no canonical counts and no raw payload.
func (u Usage) IsZero() bool {
	return u.PromptTokens == 0 && u.CompletionTokens == 0 && u.TotalTokens == 0 && len(u.Raw) == 0
}

func cloneRawJSON(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	cloned := make(json.RawMessage, len(raw))
	copy(cloned, raw)
	return cloned
}

func ptrUsage(u Usage) *Usage {
	copy := u.Clone()
	return &copy
}

// ModelInfo represents information about an available model.
type ModelInfo struct {
	ID        string
	Provider  string
	MaxTokens int
}

// StreamIterator is the interface for streaming responses.
type StreamIterator interface {
	// Next returns the next chunk. Returns io.EOF when stream ends.
	Next() (*StreamChunk, error)
	// Close closes the iterator.
	Close() error
}

// StreamChunk represents a chunk in a streaming response.
type StreamChunk struct {
	ID             string
	Model          string
	Content        string
	BackgroundWork bool
	ToolCalls      []ToolCall
	FinishReason   FinishReason
	Usage          *Usage
}

// NewUserMessage creates a new user message.
func NewUserMessage(content string) Message {
	return Message{Role: "user", Content: content}
}

// NewUserMessageParts creates a user message with structured text parts.
func NewUserMessageParts(parts []MessageContentPart) Message {
	return Message{
		Role:         "user",
		Content:      MessageContentPartsText(parts),
		ContentParts: normalizeMessageContentParts(parts),
	}
}

// NewSystemMessage creates a new system message.
func NewSystemMessage(content string) Message {
	return Message{Role: "system", Content: content}
}

// NewAssistantMessage creates a new assistant message.
func NewAssistantMessage(content string) Message {
	return Message{Role: "assistant", Content: content}
}

// NewToolMessage creates a new tool message.
func NewToolMessage(toolCallID, content string) Message {
	return Message{Role: "tool", Content: content, ToolCallID: toolCallID}
}

// NewTextContentPart creates a text content part.
func NewTextContentPart(text string) MessageContentPart {
	return MessageContentPart{Type: "text", Text: text}
}

// MessageContentPartsText returns a plain-text rendering of structured parts.
func MessageContentPartsText(parts []MessageContentPart) string {
	if len(parts) == 0 {
		return ""
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type != "" && part.Type != "text" {
			continue
		}
		if part.Text == "" {
			continue
		}
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, "\n\n")
}

func normalizeMessageContentParts(parts []MessageContentPart) []MessageContentPart {
	if len(parts) == 0 {
		return nil
	}
	normalized := make([]MessageContentPart, 0, len(parts))
	for _, part := range parts {
		if part.Type == "" {
			part.Type = "text"
		}
		if part.Type != "text" || part.Text == "" {
			continue
		}
		normalized = append(normalized, part)
	}
	return normalized
}
