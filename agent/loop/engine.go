package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	ctxmanager "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/configs"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/internal/pathpolicy"
	"gitcode.com/mindspore/mscli/permission"
	"gitcode.com/mindspore/mscli/tools"
)

// EngineConfig holds engine configuration.
type EngineConfig struct {
	MaxIterations  int
	ContextWindow  int
	MaxTokens      *int
	Temperature    *float32
	Effort         string
	TimeoutPerTurn time.Duration
	SystemPrompt   string
}

var ErrMaxIterations = errors.New("maximum iterations exceeded")

// Engine runs the ReAct loop: LLM → tool call → LLM → done.
type Engine struct {
	config         EngineConfig
	provider       llm.Provider
	tools          *tools.Registry
	ctxManager     *ctxmanager.Manager
	permission     permission.PermissionService
	pathAuthorizer PathAuthorizer
	recorder       *TrajectoryRecorder
	debugDumper    *llm.DebugDumper
}

// TrajectoryRecorder records runtime conversation events for persistence.
type TrajectoryRecorder struct {
	RecordUserInput         func(string) error
	RecordAssistant         func(string) error
	RecordToolCall          func(llm.ToolCall) error
	RecordToolResult        func(llm.ToolCall, string, map[string]any) error
	RecordSkillActivate     func(string) error
	RecordContextCompaction func(trigger string, beforeTokens, afterTokens int, message string) error
	PrepareFileMutation     func(context.Context, llm.ToolCall) error
	PersistSnapshot         func() error
}

// NewEngine creates a new engine.
func NewEngine(cfg EngineConfig, provider llm.Provider, tools *tools.Registry) *Engine {
	if cfg.SystemPrompt == "" {
		cfg.SystemPrompt = DefaultSystemPrompt()
	}

	engine := &Engine{
		config:   cfg,
		provider: provider,
		tools:    tools,
	}

	managerCfg := ctxmanager.DefaultManagerConfig()
	if cfg.ContextWindow > 0 {
		managerCfg.ContextWindow = cfg.ContextWindow
		managerCfg.ReserveTokens = configs.DefaultReserveTokens(managerCfg.ContextWindow)
	}
	managerCfg.CompactProvider = provider
	engine.ctxManager = ctxmanager.NewManager(managerCfg)
	engine.ctxManager.SetSystemPrompt(cfg.SystemPrompt)
	engine.permission = permission.NewNoOpPermissionService()

	return engine
}

// SetContextManager sets the context manager.
func (e *Engine) SetContextManager(cm *ctxmanager.Manager) {
	if cm == nil {
		return
	}
	if cm.GetSystemPrompt() == nil {
		switch {
		case e.ctxManager != nil && e.ctxManager.GetSystemPrompt() != nil:
			cm.SetSystemPrompt(e.ctxManager.GetSystemPrompt().Content)
		case e.config.SystemPrompt != "":
			cm.SetSystemPrompt(e.config.SystemPrompt)
		}
	}
	cm.SetCompactProvider(e.provider)
	cm.SetDebugDumper(e.debugDumper)
	e.ctxManager = cm
}

// SetPermissionService sets the permission service.
func (e *Engine) SetPermissionService(ps permission.PermissionService) {
	e.permission = ps
}

func (e *Engine) SetPathAuthorizer(authorizer PathAuthorizer) {
	e.pathAuthorizer = authorizer
}

// SetTrajectoryRecorder records runtime conversation events for persistence.
func (e *Engine) SetTrajectoryRecorder(recorder *TrajectoryRecorder) {
	e.recorder = recorder
}

// SetLLMDebugDumper enables raw request/response dumping for LLM calls.
func (e *Engine) SetLLMDebugDumper(dumper *llm.DebugDumper) {
	e.debugDumper = dumper
	if e.ctxManager != nil {
		e.ctxManager.SetDebugDumper(dumper)
	}
}

// SetEffort updates the reasoning effort sent with future LLM requests.
func (e *Engine) SetEffort(effort string) {
	e.config.Effort = strings.ToLower(strings.TrimSpace(effort))
}

// ToolNames returns the names of registered tools.
func (e *Engine) ToolNames() []string {
	toolList := e.tools.List()
	names := make([]string, len(toolList))
	for i, t := range toolList {
		names[i] = t.Name()
	}
	return names
}

// Run executes a task and returns events.
func (e *Engine) Run(task Task) ([]Event, error) {
	return e.RunWithContext(context.Background(), task)
}

// RunWithContext executes the ReAct loop for a task.
func (e *Engine) RunWithContext(ctx context.Context, task Task) ([]Event, error) {
	return e.runWithContext(ctx, task, nil)
}

// RunWithContextStream executes the ReAct loop and emits events as they occur.
func (e *Engine) RunWithContextStream(ctx context.Context, task Task, sink func(Event)) error {
	_, err := e.runWithContext(ctx, task, sink)
	return err
}

func (e *Engine) runWithContext(ctx context.Context, task Task, sink func(Event)) ([]Event, error) {
	exec := &executor{
		engine:    e,
		task:      task,
		events:    make([]Event, 0),
		sink:      sink,
		startTime: time.Now(),
	}
	return exec.run(ctx)
}

// executor manages a single ReAct loop run.
type executor struct {
	engine     *Engine
	task       Task
	events     []Event
	iterCount  int
	startTime  time.Time
	totalUsage llm.Usage
	sink       func(Event)

	responsesPreviousID string
	responsesFollowup   []llm.Message
}

type contextCompactionNotice struct {
	BeforeTokens int
	AfterTokens  int
}

const ContextCompactStartMessage = "compacting conversation..."

func (ex *executor) run(ctx context.Context) ([]Event, error) {
	for _, msg := range ex.task.InitialMessages {
		notice, err := ex.addContextMessage(ctx, msg)
		if err != nil {
			ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist message error: %v", err)))
			return ex.events, err
		}
		if err := ex.persistSnapshot(); err != nil {
			ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist snapshot error: %v", err)))
			return ex.events, err
		}
		if err := ex.emitContextCompactionNotice(notice); err != nil {
			ex.addEvent(NewEvent(EventTaskFailed, err.Error()))
			return ex.events, err
		}
	}

	userMessage := ex.task.userMessageContent()
	notice, err := ex.addContextMessage(ctx, llm.NewUserMessage(userMessage))
	if err != nil {
		ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist message error: %v", err)))
		return ex.events, err
	}
	if ex.engine.recorder != nil && ex.engine.recorder.RecordUserInput != nil {
		if err := ex.engine.recorder.RecordUserInput(userMessage); err != nil {
			ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist message error: %v", err)))
			return ex.events, err
		}
	}
	if err := ex.persistSnapshot(); err != nil {
		ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist snapshot error: %v", err)))
		return ex.events, err
	}
	if err := ex.emitContextCompactionNotice(notice); err != nil {
		ex.addEvent(NewEvent(EventTaskFailed, err.Error()))
		return ex.events, err
	}
	ex.addEvent(NewEvent(EventTaskStarted, fmt.Sprintf("Task: %s", ex.task.Description)))

	completed := false
	for ex.engine.config.MaxIterations == 0 || ex.iterCount < ex.engine.config.MaxIterations {
		ex.iterCount++
		ex.addEvent(NewEvent(EventAgentThinking, ""))

		if err := ctx.Err(); err != nil {
			return ex.events, err
		}

		resp, err := ex.callLLM(ctx)
		if err != nil {
			return ex.events, err
		}

		ex.trackUsage(resp.Usage)

		continueLoop, err := ex.handleResponse(ctx, resp)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return ex.events, err
			}
			ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Handle response error: %v", err)))
			return ex.events, err
		}
		if !continueLoop {
			completed = true
			break
		}
	}

	if completed {
		ex.addEvent(NewEvent(EventTaskCompleted, "Task completed successfully"))
	} else if ex.engine.config.MaxIterations > 0 && ex.iterCount >= ex.engine.config.MaxIterations {
		ex.addEvent(NewEvent(EventTaskFailed, "Task exceeded maximum iterations."))
		return ex.events, ErrMaxIterations
	} else {
		ex.addEvent(NewEvent(EventTaskCompleted, "Task completed successfully"))
	}

	return ex.events, nil
}

func (t Task) userMessageContent() string {
	if strings.TrimSpace(t.UserMessage) != "" {
		return t.UserMessage
	}
	return t.Description
}

func (ex *executor) callLLM(ctx context.Context) (*llm.CompletionResponse, error) {
	timeout := ex.engine.config.TimeoutPerTurn
	if timeout == 0 {
		timeout = 5 * time.Minute
	}

	llmCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ex.sanitizeToolPairsBeforeRequest()

	req := &llm.CompletionRequest{
		Messages:    ex.requestMessages(),
		Tools:       ex.filteredTools(),
		Temperature: ex.engine.config.Temperature,
		MaxTokens:   ex.engine.config.MaxTokens,
		Effort:      ex.engine.config.Effort,
	}

	if ex.usesResponsesChain() && ex.responsesPreviousID != "" {
		llmCtx = llm.WithPreviousResponseID(llmCtx, ex.responsesPreviousID)
	}
	if ex.engine.debugDumper != nil {
		llmCtx = llm.WithDebugDumper(llmCtx, ex.engine.debugDumper)
	}

	resp, err := ex.streamCompletion(llmCtx, req)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) || errors.Is(llmCtx.Err(), context.Canceled) {
			return nil, context.Canceled
		}
		if ctx.Err() == context.DeadlineExceeded || llmCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("request timeout: %w", err)
		}
		errMsg := fmt.Sprintf("LLM error: %v", err)
		ex.addEvent(NewEvent(EventTaskFailed, errMsg))
		return nil, fmt.Errorf("LLM completion: %w", err)
	}

	return resp, nil
}

func (ex *executor) filteredTools() []llm.Tool {
	if !ex.task.DisableResearchTools {
		return ex.engine.tools.ToLLMTools()
	}

	filtered := ex.engine.tools.ToLLMToolsFiltered(func(_ tools.Tool, metadata tools.ToolMetadata) bool {
		if len(metadata.Classes) == 0 {
			return false
		}
		return !hasToolClass(metadata, tools.ToolClassExploration) && !hasToolClass(metadata, tools.ToolClassContextExpansion)
	})
	if len(filtered) == 0 {
		return nil
	}
	return filtered
}

func hasToolClass(metadata tools.ToolMetadata, class tools.ToolClass) bool {
	for _, current := range metadata.Classes {
		if current == class {
			return true
		}
	}
	return false
}

func (ex *executor) sanitizeToolPairsBeforeRequest() {
	if ex.engine == nil || ex.engine.ctxManager == nil {
		return
	}

	messages := ex.engine.ctxManager.GetNonSystemMessages()
	valid := validToolCallIDs(messages)
	sanitized, report := sanitizeMessagesForValidToolCallIDs(messages, valid)
	if report.changed() {
		ex.engine.ctxManager.SetNonSystemMessages(sanitized)
		valid = validToolCallIDs(sanitized)
	}

	if ex.usesResponsesChain() && ex.responsesPreviousID != "" && len(ex.responsesFollowup) > 0 {
		sanitizedFollowup, followupReport := sanitizeMessagesForValidToolCallIDs(ex.responsesFollowup, valid)
		ex.responsesFollowup = sanitizedFollowup
		if len(ex.responsesFollowup) == 0 {
			ex.responsesPreviousID = ""
		}
		if !report.removedPairs() && followupReport.removedPairs() {
			report = followupReport
		} else if followupReport.normalizedToolResults > 0 {
			report.normalizedToolResults += followupReport.normalizedToolResults
		}
	}

	if !report.removedPairs() {
		return
	}

	ev := NewEvent(EventToolError, report.warningMessage())
	ev.ToolName = "context"
	ex.addEvent(ev)
}

func (ex *executor) requestMessages() []llm.Message {
	msgs := ex.baseRequestMessages()
	if !ex.task.DisableResearchTools {
		return msgs
	}

	guidance := llm.NewSystemMessage("Research tools are disabled for this turn. Produce the requested answer or artifact from the existing context. You may still use any available non-research tools.")
	out := make([]llm.Message, 0, len(msgs)+1)
	if len(msgs) > 0 && msgs[0].Role == "system" {
		out = append(out, msgs[0], guidance)
		out = append(out, msgs[1:]...)
		return out
	}
	out = append(out, guidance)
	out = append(out, msgs...)
	return out
}

func (ex *executor) baseRequestMessages() []llm.Message {
	if !ex.usesResponsesChain() || ex.responsesPreviousID == "" || len(ex.responsesFollowup) == 0 {
		return ex.engine.ctxManager.GetMessages()
	}

	msgs := make([]llm.Message, 0, len(ex.responsesFollowup)+1)
	if system := ex.engine.ctxManager.GetSystemPrompt(); system != nil {
		msgs = append(msgs, *system)
	}
	msgs = append(msgs, ex.responsesFollowup...)
	return msgs
}

func (ex *executor) usesResponsesChain() bool {
	return ex.engine.provider != nil && ex.engine.provider.Name() == string(llm.ProviderOpenAIResponses)
}

func (ex *executor) streamCompletion(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	iter, err := ex.engine.provider.CompleteStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("stream completion: %w", err)
	}
	defer iter.Close()

	resp := &llm.CompletionResponse{}
	for {
		chunk, nextErr := iter.Next()
		if chunk != nil {
			ex.applyStreamChunk(resp, chunk)
		}
		if nextErr != nil {
			if nextErr == io.EOF {
				break
			}
			return nil, nextErr
		}
	}

	if resp.FinishReason == "" {
		if len(resp.ToolCalls) > 0 {
			resp.FinishReason = llm.FinishToolCalls
		} else {
			resp.FinishReason = llm.FinishStop
		}
	}

	return resp, nil
}

func (ex *executor) applyStreamChunk(resp *llm.CompletionResponse, chunk *llm.StreamChunk) {
	if chunk.ID != "" {
		resp.ID = chunk.ID
	}
	if chunk.Model != "" {
		resp.Model = chunk.Model
	}
	if chunk.Content != "" {
		resp.Content += chunk.Content
		ex.addEvent(NewEvent(EventAgentReplyDelta, chunk.Content))
	}
	if chunk.BackgroundWork {
		ex.addEvent(NewEvent(EventAgentBackgroundWork, ""))
	}
	if len(chunk.ToolCalls) > 0 {
		resp.ToolCalls = make([]llm.ToolCall, len(chunk.ToolCalls))
		copy(resp.ToolCalls, chunk.ToolCalls)
	}
	if chunk.FinishReason != "" {
		resp.FinishReason = chunk.FinishReason
	}
	if chunk.Usage != nil {
		resp.Usage = *chunk.Usage
	}
}

func (ex *executor) handleResponse(ctx context.Context, resp *llm.CompletionResponse) (bool, error) {
	notice, err := ex.addContextMessage(ctx, llm.Message{
		Role:      "assistant",
		Content:   resp.Content,
		ToolCalls: resp.ToolCalls,
	})
	if err != nil {
		return false, err
	}
	if notice == nil {
		ex.syncContextTokenUsage(resp.Usage)
	}
	if ex.engine.recorder != nil {
		if strings.TrimSpace(resp.Content) != "" && ex.engine.recorder.RecordAssistant != nil {
			if err := ex.engine.recorder.RecordAssistant(resp.Content); err != nil {
				return false, err
			}
		}
		for _, tc := range resp.ToolCalls {
			if ex.engine.recorder.RecordToolCall != nil {
				if err := ex.engine.recorder.RecordToolCall(tc); err != nil {
					return false, err
				}
			}
		}
	}
	if err := ex.persistSnapshot(); err != nil {
		return false, err
	}
	if err := ex.emitContextCompactionNotice(notice); err != nil {
		return false, err
	}

	if ex.usesResponsesChain() && strings.TrimSpace(resp.ID) != "" {
		ex.responsesPreviousID = strings.TrimSpace(resp.ID)
		ex.responsesFollowup = nil
	}

	if resp.Content != "" {
		ex.addEvent(NewEvent(EventAgentReply, resp.Content))
	}

	if len(resp.ToolCalls) > 0 {
		for _, tc := range resp.ToolCalls {
			if err := ex.executeToolCall(ctx, tc); err != nil {
				return false, err
			}
		}
		return true, nil
	}

	return false, nil
}

func (ex *executor) executeToolCall(ctx context.Context, tc llm.ToolCall) error {
	toolName := tc.Function.Name

	tool, ok := ex.engine.tools.Get(toolName)
	if !ok {
		errMsg := fmt.Sprintf("Tool not found: %s", toolName)
		meta := toolResultMeta(tools.StatusFailed, tools.SourceLoop)
		write, err := ex.addToolResultWithFallback(ctx, tc.ID, errMsg, meta)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
			return err
		}
		ex.addToolErrorEvent(toolName, tc.ID, errMsg, write.Meta)
		return nil
	}

	// Check permission before emitting ToolCallStart so the UI does not
	// show "running command..." while actually waiting for user approval.
	action := extractAction(toolName, tc.Function.Arguments)
	path := extractPathArg(tc.Function.Arguments)
	granted, err := ex.engine.permission.Request(ctx, toolName, action, path)
	if err != nil {
		return err
	}
	if !granted {
		errMsg := fmt.Sprintf("Permission denied for tool: %s", toolName)
		meta := toolResultMeta(tools.StatusDeclined, tools.SourcePermission)
		write, err := ex.addToolResultWithFallback(ctx, tc.ID, errMsg, meta)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
			return err
		}
		ex.addToolErrorEvent(toolName, tc.ID, errMsg, write.Meta)
		return nil
	}

	if err := ex.prepareFileMutation(ctx, tc); err != nil {
		return err
	}

	startEv := NewEvent(EventToolCallStart, describeToolCall(toolName, tc.Function.Arguments))
	startEv.ToolName = toolName
	startEv.ToolCallID = tc.ID
	ex.addEvent(startEv)

	// Execute
	var result *tools.Result
	toolStart := time.Now()
	result, err = ex.executeTool(ctx, tool, toolName, tc.ID, tc.Function.Arguments)
	toolDuration := time.Since(toolStart)
	ex.addReturnedResultMeta(result, toolDuration)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			meta := terminalToolResultMeta(tools.StatusInterrupted, sourceForTool(tool), toolDuration)
			if interruptErr := ex.handleInterruptedToolCall(tc, "", meta); interruptErr != nil {
				return interruptErr
			}
			return context.Canceled
		}
		errMsg := fmt.Sprintf("Tool execution error: %v", err)
		meta := terminalToolResultMeta(tools.StatusFailed, sourceForTool(tool), toolDuration)
		write, err := ex.addToolResultWithFallback(ctx, tc.ID, errMsg, meta)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
			return err
		}
		ex.addToolErrorEvent(toolName, tc.ID, errMsg, write.Meta)
		return nil
	}

	if result == nil {
		errMsg := fmt.Sprintf("Tool %s returned no result", toolName)
		meta := terminalToolResultMeta(tools.StatusFailed, sourceForTool(tool), toolDuration)
		write, err := ex.addToolResultWithFallback(ctx, tc.ID, errMsg, meta)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
			return err
		}
		ex.addToolErrorEvent(toolName, tc.ID, errMsg, write.Meta)
		return nil
	}

	if result != nil && result.Error != nil {
		if handled, err := ex.handlePathDenial(ctx, tc, tool, toolName, result); err != nil {
			return err
		} else if handled {
			return nil
		}
		if result.Error == nil {
			goto toolSuccess
		}
		if errors.Is(result.Error, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			meta := interruptedToolMeta(result.Meta, sourceForTool(tool), toolDuration)
			if interruptErr := ex.handleInterruptedToolCall(tc, result.Content, meta); interruptErr != nil {
				return interruptErr
			}
			return context.Canceled
		}
		errMsg := result.Error.Error()
		write, err := ex.addToolResultWithFallback(ctx, tc.ID, errMsg, result.Meta)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
			return err
		}
		ex.addToolErrorEvent(toolName, tc.ID, fmt.Sprintf("Tool %s failed: %s", toolName, errMsg), write.Meta)
		return nil
	}

toolSuccess:
	if result == nil {
		result = tools.StringResult("")
	}
	if errors.Is(ctx.Err(), context.Canceled) {
		meta := interruptedToolMeta(result.Meta, sourceForTool(tool), toolDuration)
		if interruptErr := ex.handleInterruptedToolCall(tc, result.Content, meta); interruptErr != nil {
			return interruptErr
		}
		return context.Canceled
	}

	write, err := ex.addToolResultWithFallback(ctx, tc.ID, result.Content, result.Meta)
	if err != nil {
		return err
	}
	if toolName == "load_skill" && ex.engine.recorder != nil && ex.engine.recorder.RecordSkillActivate != nil {
		if skillName := skillNameFromToolCall(tc); skillName != "" {
			if err := ex.engine.recorder.RecordSkillActivate(skillName); err != nil {
				return err
			}
		}
	}
	if err := ex.persistSnapshot(); err != nil {
		return err
	}
	if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
		return err
	}
	eventResult := *result
	eventResult.Content = write.Content
	eventResult.Meta = write.Meta
	ex.addToolEvent(toolName, tc.ID, &eventResult)
	return nil
}

func (ex *executor) prepareFileMutation(ctx context.Context, tc llm.ToolCall) error {
	if ex.engine.recorder == nil || ex.engine.recorder.PrepareFileMutation == nil {
		return nil
	}
	return ex.engine.recorder.PrepareFileMutation(ctx, tc)
}

func (ex *executor) executeTool(ctx context.Context, tool tools.Tool, toolName, toolCallID string, args json.RawMessage) (*tools.Result, error) {
	if streamingTool, canStream := tool.(tools.StreamingTool); canStream {
		return streamingTool.ExecuteStream(ctx, args, func(update tools.StreamEvent) {
			ex.addStreamingToolEvent(toolName, toolCallID, update)
		})
	}
	return tool.Execute(ctx, args)
}

func (ex *executor) handlePathDenial(ctx context.Context, tc llm.ToolCall, tool tools.Tool, toolName string, result *tools.Result) (bool, error) {
	denial, ok := pathpolicy.ExtractPathDenial(result)
	if !ok || ex.engine.pathAuthorizer == nil {
		return false, nil
	}
	decision, err := ex.engine.pathAuthorizer.RequestPathAuthorization(ctx, denial)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			return false, context.Canceled
		}
		return false, err
	}
	if decision.Scope == PathAuthorizationDeny {
		return true, ex.addPathAuthorizationDeniedResult(ctx, tc, toolName, denial)
	}
	root := strings.TrimSpace(decision.Root)
	if root == "" {
		return true, ex.addPathAuthorizationDeniedResult(ctx, tc, toolName, denial)
	}
	mode := decision.Mode
	if mode == "" {
		mode = PathAuthorizationModeRead
	}
	if mode != PathAuthorizationModeRead && mode != PathAuthorizationModeWrite {
		return true, ex.addPathAuthorizationDeniedResult(ctx, tc, toolName, denial)
	}

	opts := pathpolicy.ResolveOptions{TemporaryReadRoots: []string{root}}
	if mode == PathAuthorizationModeWrite {
		opts = pathpolicy.ResolveOptions{TemporaryWriteRoots: []string{root}}
	}
	retryCtx := pathpolicy.ContextWithResolveOptions(ctx, opts)
	if err := ex.prepareFileMutation(retryCtx, tc); err != nil {
		return true, err
	}
	retryResult, err := ex.executeTool(retryCtx, tool, toolName, tc.ID, tc.Function.Arguments)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			if interruptErr := ex.handleInterruptedToolCall(tc, "", nil); interruptErr != nil {
				return true, interruptErr
			}
			return true, context.Canceled
		}
		errMsg := fmt.Sprintf("Tool execution error: %v", err)
		write, addErr := ex.addToolResultWithFallback(ctx, tc.ID, errMsg, nil)
		if addErr != nil {
			return true, addErr
		}
		if persistErr := ex.persistSnapshot(); persistErr != nil {
			return true, persistErr
		}
		if noticeErr := ex.emitContextCompactionNotice(write.Notice); noticeErr != nil {
			return true, noticeErr
		}
		ex.addEvent(NewEvent(EventToolError, errMsg))
		return true, nil
	}
	if retryResult == nil {
		retryResult = tools.StringResult("")
	}
	if retryResult.Error != nil {
		*result = *retryResult
		return false, nil
	}
	*result = *retryResult
	return false, nil
}

func (ex *executor) addPathAuthorizationDeniedResult(ctx context.Context, tc llm.ToolCall, toolName string, denial *pathpolicy.PathDenial) error {
	message := "Path authorization denied."
	if denial != nil {
		message = denial.ErrorMessage()
	}
	write, err := ex.addToolResultWithFallback(ctx, tc.ID, message, nil)
	if err != nil {
		return err
	}
	if err := ex.persistSnapshot(); err != nil {
		return err
	}
	if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
		return err
	}
	ev := NewEvent(EventToolError, fmt.Sprintf("Tool %s failed: %s", toolName, message))
	ev.ToolName = toolName
	ev.ToolCallID = tc.ID
	ex.addEvent(ev)
	return nil
}

func (ex *executor) addReturnedResultMeta(result *tools.Result, duration time.Duration) {
	if result == nil {
		return
	}
	if result.Meta == nil {
		result.Meta = make(map[string]any)
	}
	if _, ok := result.Meta[tools.MetaDurationMS]; !ok {
		tools.SetResultDuration(result, duration.Milliseconds())
	}
	if _, ok := result.Meta[tools.MetaStatus]; !ok {
		status := tools.StatusCompleted
		if result.Error != nil {
			status = tools.StatusFailed
		}
		tools.SetResultStatus(result, status)
	}
}

func toolResultMeta(status, source string) map[string]any {
	meta := map[string]any{}
	if status != "" {
		meta[tools.MetaStatus] = status
	}
	if source != "" {
		meta[tools.MetaSource] = source
	}
	return meta
}

func terminalToolResultMeta(status, source string, duration time.Duration) map[string]any {
	meta := toolResultMeta(status, source)
	meta[tools.MetaDurationMS] = duration.Milliseconds()
	return meta
}

func interruptedToolMeta(existing map[string]any, source string, duration time.Duration) map[string]any {
	meta := cloneToolMeta(existing)
	if meta == nil {
		meta = map[string]any{}
	}
	meta[tools.MetaStatus] = tools.StatusInterrupted
	if source != "" {
		meta[tools.MetaSource] = source
	}
	meta[tools.MetaDurationMS] = duration.Milliseconds()
	return meta
}

func sourceForTool(tool tools.Tool) string {
	switch tools.CapabilitiesForTool(tool).Kind {
	case tools.KindFilesystem:
		return tools.SourceFS
	case tools.KindShell:
		return tools.SourceShell
	case tools.KindSkill:
		return tools.SourceSkill
	case tools.KindMCP:
		return tools.SourceMCP
	default:
		return tools.SourceLoop
	}
}

func (ex *executor) handleInterruptedToolCall(tc llm.ToolCall, partialOutput string, meta map[string]any) error {
	content := interruptedToolResultContent(partialOutput)
	if meta == nil {
		meta = toolResultMeta(tools.StatusInterrupted, tools.SourceLoop)
	}
	write, err := ex.addToolResultWithFallback(context.Background(), tc.ID, content, meta)
	if err != nil {
		return err
	}
	if err := ex.persistSnapshot(); err != nil {
		return err
	}
	if err := ex.emitContextCompactionNotice(write.Notice); err != nil {
		return err
	}

	eventMessage := strings.TrimSpace(partialOutput)
	if write.Fallback {
		eventMessage = write.Content
	}
	ev := NewEvent(EventToolInterrupted, eventMessage)
	ev.ToolName = tc.Function.Name
	ev.ToolCallID = tc.ID
	ev.Summary = "interrupted"
	ev.Meta = write.Meta
	ex.addEvent(ev)
	return nil
}

func interruptedToolResultContent(partialOutput string) string {
	lines := []string{
		"tool interrupted",
		"status: interrupted",
		"reason: user requested cancellation",
		"result: incomplete; do not treat this tool call as successful",
	}
	partialOutput = strings.TrimSpace(partialOutput)
	if partialOutput == "" {
		lines = append(lines, "partial_output: none")
		return strings.Join(lines, "\n")
	}
	lines = append(lines, "partial_output:")
	lines = append(lines, partialOutput)
	return strings.Join(lines, "\n")
}

var toolEventMap = map[string]string{
	"read":            EventToolRead,
	"grep":            EventToolGrep,
	"glob":            EventToolGlob,
	"edit":            EventToolEdit,
	"write":           EventToolWrite,
	"shell":           EventCmdFinished,
	"load_skill":      EventToolSkill,
	"AskUserQuestion": EventToolAskUserQuestion,
}

func (ex *executor) addStreamingToolEvent(toolName, toolCallID string, update tools.StreamEvent) {
	var eventType string
	switch update.Type {
	case tools.StreamEventStarted:
		if toolName != "shell" {
			return
		}
		eventType = EventCmdStarted
	case tools.StreamEventOutput:
		if toolName != "shell" {
			return
		}
		eventType = EventCmdOutput
	default:
		return
	}

	ev := NewEvent(eventType, update.Message)
	ev.ToolName = toolName
	ev.ToolCallID = toolCallID
	ev.Summary = update.Summary
	ex.addEvent(ev)
}

func (ex *executor) addToolErrorEvent(toolName, toolCallID, message string, meta map[string]any) {
	ev := NewEvent(EventToolError, message)
	ev.ToolName = toolName
	ev.ToolCallID = toolCallID
	ev.Meta = cloneToolMeta(meta)
	ex.addEvent(ev)
}

func (ex *executor) addToolEvent(toolName, toolCallID string, result *tools.Result) {
	eventType := EventToolStarted
	if t, ok := toolEventMap[toolName]; ok {
		eventType = t
	}
	ev := NewEvent(eventType, result.Content)
	ev.ToolName = toolName
	ev.ToolCallID = toolCallID
	ev.Summary = result.Summary
	ev.Meta = result.Meta
	ex.addEvent(ev)
}

func (ex *executor) addEvent(ev Event) {
	usage := ex.engine.ctxManager.TokenUsage()
	ev.CtxUsed = usage.Current
	ev.CtxMax = usage.ContextWindow
	ev.TokensUsed = ex.totalUsage.TotalTokens
	ex.events = append(ex.events, ev)
	if ex.sink != nil {
		ex.sink(ev)
	}
}

func (ex *executor) trackUsage(u llm.Usage) {
	ex.totalUsage.PromptTokens += u.PromptTokens
	ex.totalUsage.CompletionTokens += u.CompletionTokens
	ex.totalUsage.TotalTokens += u.TotalTokens
}

func (ex *executor) syncContextTokenUsage(u llm.Usage) {
	if ex.engine == nil || ex.engine.ctxManager == nil {
		return
	}

	providerName := ""
	if ex.engine.provider != nil {
		providerName = ex.engine.provider.Name()
	}

	ex.engine.ctxManager.SetProviderTokenUsage(providerName, u)
}

func (ex *executor) persistSnapshot() error {
	if ex.engine.recorder == nil || ex.engine.recorder.PersistSnapshot == nil {
		return nil
	}
	return ex.engine.recorder.PersistSnapshot()
}

func (ex *executor) addContextMessage(ctx context.Context, msg llm.Message) (*contextCompactionNotice, error) {
	if strings.TrimSpace(msg.Role) != "user" && ex.engine.ctxManager.ShouldCompactAfterAdding(msg) {
		ex.addEvent(NewEvent(EventContextCompactStart, ContextCompactStartMessage))
	}
	beforeUsage := ex.engine.ctxManager.TokenUsage()
	beforeCompactCount := ex.engine.ctxManager.CompactCount()
	if err := ex.engine.ctxManager.AddMessageWithContext(ctx, msg); err != nil {
		return nil, err
	}
	afterCompactCount := ex.engine.ctxManager.CompactCount()
	if afterCompactCount <= beforeCompactCount {
		return nil, nil
	}
	if ex.usesResponsesChain() {
		ex.responsesPreviousID = ""
		ex.responsesFollowup = nil
	}
	afterUsage := ex.engine.ctxManager.TokenUsage()
	return &contextCompactionNotice{
		BeforeTokens: beforeUsage.Current,
		AfterTokens:  afterUsage.Current,
	}, nil
}

func (ex *executor) emitContextCompactionNotice(notice *contextCompactionNotice) error {
	if notice == nil {
		return nil
	}
	message := fmt.Sprintf("Context compacted automatically: %d -> %d tokens.", notice.BeforeTokens, notice.AfterTokens)
	if ex.engine.recorder != nil && ex.engine.recorder.RecordContextCompaction != nil {
		if err := ex.engine.recorder.RecordContextCompaction("auto", notice.BeforeTokens, notice.AfterTokens, message); err != nil {
			return fmt.Errorf("persist context compaction event: %w", err)
		}
	}
	ex.addEvent(NewEvent(EventContextCompacted, message))
	return nil
}

const emptyToolResultPlaceholder = "(tool completed with empty output)"

func normalizeToolResultContent(content string) string {
	if strings.TrimSpace(content) == "" {
		return emptyToolResultPlaceholder
	}
	return content
}

func (ex *executor) addToolResult(ctx context.Context, callID, content string, meta map[string]any) (*contextCompactionNotice, error) {
	content = normalizeToolResultContent(content)
	msg := llm.NewToolMessage(callID, content)
	notice, err := ex.addContextMessage(ctx, msg)
	if err != nil {
		return nil, err
	}
	if ex.usesResponsesChain() && ex.responsesPreviousID != "" {
		ex.responsesFollowup = append(ex.responsesFollowup, msg)
	}
	if ex.engine.recorder != nil && ex.engine.recorder.RecordToolResult != nil {
		var toolCall llm.ToolCall
		toolCall.ID = callID
		if tc := ex.findToolCall(callID); tc != nil {
			toolCall = *tc
		}
		if err := ex.engine.recorder.RecordToolResult(toolCall, content, cloneToolMeta(meta)); err != nil {
			return nil, err
		}
	}
	return notice, nil
}

type toolResultWrite struct {
	Notice   *contextCompactionNotice
	Content  string
	Meta     map[string]any
	Fallback bool
}

func (ex *executor) addToolResultWithFallback(ctx context.Context, callID, content string, meta map[string]any) (*toolResultWrite, error) {
	notice, err := ex.addToolResult(ctx, callID, content, meta)
	if err != nil {
		fallback := fmt.Sprintf("tool result replaced due to context limit: %v", err)
		fallbackMeta := cloneToolMeta(meta)
		if fallbackMeta == nil {
			fallbackMeta = map[string]any{}
		}
		fallbackMeta[tools.MetaFallback] = true
		fallbackNotice, fallbackErr := ex.addToolResult(ctx, callID, fallback, fallbackMeta)
		if fallbackErr != nil {
			return nil, fmt.Errorf("persist tool result fallback: %w (original error: %v)", fallbackErr, err)
		}
		if err := ex.persistSnapshot(); err != nil {
			return nil, err
		}
		return &toolResultWrite{
			Notice:   fallbackNotice,
			Content:  fallback,
			Meta:     fallbackMeta,
			Fallback: true,
		}, nil
	}
	return &toolResultWrite{
		Notice:  notice,
		Content: content,
		Meta:    cloneToolMeta(meta),
	}, nil
}

func cloneToolMeta(meta map[string]any) map[string]any {
	if len(meta) == 0 {
		return nil
	}
	out := make(map[string]any, len(meta))
	for key, value := range meta {
		out[key] = value
	}
	return out
}

func (ex *executor) findToolCall(callID string) *llm.ToolCall {
	for _, msg := range ex.engine.ctxManager.GetNonSystemMessages() {
		if msg.Role != "assistant" {
			continue
		}
		for _, tc := range msg.ToolCalls {
			if tc.ID == callID {
				copy := tc
				return &copy
			}
		}
	}
	return nil
}

func skillNameFromToolCall(tc llm.ToolCall) string {
	if tc.Function.Name != "load_skill" {
		return ""
	}

	var args struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(tc.Function.Arguments, &args); err != nil {
		return ""
	}
	return strings.TrimSpace(args.Name)
}

func extractAction(toolName string, raw json.RawMessage) string {
	if toolName == "shell" {
		var args struct {
			Command string `json:"command"`
		}
		if err := json.Unmarshal(raw, &args); err == nil {
			if cmd := strings.TrimSpace(args.Command); cmd != "" {
				return cmd
			}
		}
	}
	return string(raw)
}

func extractPathArg(raw json.RawMessage) string {
	var params map[string]any
	if err := json.Unmarshal(raw, &params); err != nil {
		return ""
	}
	for _, key := range []string{"path", "file_path"} {
		if v, ok := params[key].(string); ok {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func describeToolCall(toolName string, raw json.RawMessage) string {
	var params map[string]any
	_ = json.Unmarshal(raw, &params)

	getString := func(keys ...string) string {
		for _, key := range keys {
			if v, ok := params[key].(string); ok {
				v = strings.TrimSpace(v)
				if v != "" {
					return v
				}
			}
		}
		return ""
	}

	switch toolName {
	case "shell":
		return getString("command")
	case "read", "edit", "write":
		return getString("path", "file_path")
	case "grep":
		pattern := getString("pattern")
		path := getString("path")
		switch {
		case pattern != "" && path != "":
			return fmt.Sprintf("%q in %s", pattern, path)
		case pattern != "":
			return pattern
		default:
			return path
		}
	case "glob":
		pattern := getString("pattern")
		path := getString("path")
		switch {
		case pattern != "" && path != "":
			return fmt.Sprintf("%s in %s", pattern, path)
		case pattern != "":
			return pattern
		default:
			return path
		}
	case "load_skill":
		return getString("name")
	case "AskUserQuestion":
		type questionArg struct {
			Header   string `json:"header"`
			Question string `json:"question"`
		}
		var args struct {
			Questions []questionArg `json:"questions"`
		}
		if err := json.Unmarshal(raw, &args); err == nil {
			parts := make([]string, 0, len(args.Questions))
			for _, question := range args.Questions {
				if header := strings.TrimSpace(question.Header); header != "" {
					parts = append(parts, header)
					continue
				}
				if text := strings.TrimSpace(question.Question); text != "" {
					parts = append(parts, text)
				}
			}
			switch len(parts) {
			case 0:
				switch count := len(args.Questions); count {
				case 1:
					return "1 question"
				case 0:
				default:
					return fmt.Sprintf("%d questions", count)
				}
			case 1:
				return parts[0]
			default:
				return fmt.Sprintf("%s (+%d more)", parts[0], len(parts)-1)
			}
		}
	}

	preview := strings.TrimSpace(string(raw))
	if preview == "" {
		return toolName
	}
	return preview
}

func DefaultSystemPrompt() string {
	return `You are MindSpore CLI, an AI agent for AI infrastructure and model training workflows.

You help ML engineers and AI infra developers get training jobs running, diagnose failures, align results, migrate model code, and improve performance. You focus on training-task-oriented workflows rather than general-purpose code generation.

You have access to the following tools:
- read: Read file contents
- write: Create or overwrite files
- edit: Edit files by replacing text
- grep: Search for patterns in files
- glob: Find files matching patterns
- shell: Execute shell commands
- AskUserQuestion: Ask the user clarifying multiple-choice questions
- load_skill: Load a skill's detailed instructions. Call this when the user's task matches an available skill listed below.
- mcp__<server>__<tool>: Additional MCP tools may be available when configured and approved.

Guidelines:
1. Use tools to gather information before making changes
2. Always read files before editing them
3. Make minimal, focused changes
4. Use grep and glob to explore the codebase
5. Run tests with shell to verify changes
6. Before any write call, verify arguments contain BOTH "path" and "content"; if either is missing, do not call write yet.
7. Never call write with empty JSON arguments ({}).
8. When a user describes a training problem (failure, accuracy, performance), load the appropriate diagnosis skill.
9. When a user asks to migrate or port a model, load migrate-agent.
10. If you are blocked on user preferences, ambiguous requirements, or implementation choices, use AskUserQuestion instead of guessing.
11. When using AskUserQuestion, pass one to four concrete options and never add an explicit Other or manual-input option because the UI already provides a built-in custom-input path.
12. Additional MCP tools may be available with names beginning mcp__; use them only when their descriptions match the task.

IMPORTANT: When you have gathered enough information to answer the user's question, you MUST provide your final answer directly WITHOUT using any more tools. Do not keep calling tools indefinitely - provide a clear, concise response once you have the information needed.

When making edits, ensure the old_string matches exactly (including whitespace and newlines).`
}

func defaultSystemPrompt() string {
	return DefaultSystemPrompt()
}
