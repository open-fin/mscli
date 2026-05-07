package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	agentctx "gitcode.com/mindspore/mscli/agent/context"
	"gitcode.com/mindspore/mscli/agent/loop"
	"gitcode.com/mindspore/mscli/agent/session"
	"gitcode.com/mindspore/mscli/configs"
	"gitcode.com/mindspore/mscli/integrations/llm"
	"gitcode.com/mindspore/mscli/integrations/skills"
	factoryruntime "gitcode.com/mindspore/mscli/internal/factory/runtime"
	itrain "gitcode.com/mindspore/mscli/internal/train"
	"gitcode.com/mindspore/mscli/internal/version"
	"gitcode.com/mindspore/mscli/permission"
	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
	rshell "gitcode.com/mindspore/mscli/runtime/shell"
	"gitcode.com/mindspore/mscli/tools"
	askuserquestion "gitcode.com/mindspore/mscli/tools/ask_user_question"
	"gitcode.com/mindspore/mscli/tools/fs"
	"gitcode.com/mindspore/mscli/tools/shell"
	skillstool "gitcode.com/mindspore/mscli/tools/skills"
	"gitcode.com/mindspore/mscli/ui/model"
	wtrain "gitcode.com/mindspore/mscli/workflow/train"
)

var errAPIKeyNotFound = errors.New("api key not found")

var buildProvider = func(resolved llm.ResolvedConfig) (llm.Provider, error) {
	return llm.DefaultManager().Build(resolved)
}

var Version = "MindSpore CLI. " + version.Version

// Application is the top-level composition container.
type Application struct {
	Engine                  *loop.Engine
	EventCh                 chan model.Event
	llmReady                bool
	llmDebugDumper          *llm.DebugDumper
	WorkDir                 string
	RepoURL                 string
	Config                  *configs.Config
	provider                llm.Provider
	toolRegistry            *tools.Registry
	ctxManager              *agentctx.Manager
	permService             permission.PermissionService
	permissionUI            *PermissionPromptUI
	questionUI              *AskUserQuestionPromptUI
	permissionSettingsIssue *permissionSettingsIssue
	mcpManager              runtimemcp.Manager
	session                 *session.Session
	memoryConfig            autoMemoryConfig
	replayBacklog           []model.Event
	replayTimeline          []session.ReplayFrame
	deferHistoryReplay      bool
	historyReplayStarted    atomic.Bool
	replayOnly              bool
	replaySpeed             float64
	sessionLLMActivity      atomic.Bool
	resumeHintOnExit        atomic.Bool
	sessionStoreReady       atomic.Bool
	initialInput            string

	// Skills
	skillLoader   *skills.Loader
	skillsHomeDir string
	startupOnce   sync.Once

	// Local diagnosis summaries.
	latestDiagnoseSummary *factoryruntime.DiagnoseRunSummary
	latestFixSummary      *factoryruntime.FixRunSummary
	latestRunKind         string

	// Foreground chat task state
	pendingMaxIterationDecision bool
	taskRunID                   uint64
	taskCancels                 map[uint64]context.CancelFunc
	replayCancel                context.CancelFunc
	taskMu                      sync.Mutex

	needsSetupPopup      bool
	startupSessionPicker *sessionPickerRequest

	// Train mode state
	trainMode       bool
	trainPhase      string // "setup","ready","running","failed","analyzing","fixing","evaluating","drift_detected","completed","stopped"
	trainReq        *itrain.Request
	trainReqs       map[string]itrain.Request
	trainBootstrap  map[string]*bootstrapRunState
	trainCurrentRun string
	trainCancel     context.CancelFunc
	trainIssueType  string // "runtime", "accuracy", or ""
	trainRunID      uint64
	trainTasks      map[uint64]struct{}
	trainController *wtrain.Controller
	pendingTrain    *pendingTrainStart
	trainMu         sync.RWMutex
}

// BootstrapConfig holds bootstrap configuration.
type BootstrapConfig struct {
	URL                 string
	Model               string
	Key                 string
	Debug               bool
	InitialInput        string
	HeadlessCommand     string
	Resume              bool
	ResumeSessionID     string
	Replay              bool
	ReplaySessionID     string
	ReplaySpeed         float64
	MCPApprovalPrompter MCPApprovalPrompter
}

// Wire builds and returns the Application.
func Wire(cfg BootstrapConfig) (*Application, error) {
	workDir, err := os.Getwd()
	if err != nil {
		workDir = "."
	}
	workDir, _ = filepath.Abs(workDir)

	eventCh := make(chan model.Event, 64)

	sessionRetentionDays := defaultSessionRetentionDays
	if appCfg, err := loadAppConfig(); err == nil {
		sessionRetentionDays = appCfg.sessionRetentionDays()
	}
	if _, err := session.CleanupExpired(time.Duration(sessionRetentionDays) * 24 * time.Hour); err != nil {
		// Session cleanup is best-effort and should not block startup.
	}

	config, err := configs.LoadWithEnv()
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	if cfg.URL != "" {
		config.Model.URL = cfg.URL
	}
	previousModel := config.Model.Model
	if cfg.Model != "" {
		config.Model.Model = cfg.Model
	}
	if cfg.Key != "" {
		config.Model.Key = cfg.Key
	}
	configs.RefreshModelTokenDefaults(config, previousModel)

	var provider llm.Provider
	llmReady := true
	resolveOpts := llm.ResolveOptions{
		PreferConfigAPIKey:  strings.TrimSpace(cfg.Key) != "",
		PreferConfigBaseURL: strings.TrimSpace(cfg.URL) != "",
	}
	provider, err = initProvider(config.Model, resolveOpts)
	if err != nil {
		if errors.Is(err, errAPIKeyNotFound) {
			llmReady = false
			provider = nil
		} else {
			return nil, fmt.Errorf("init provider: %w", err)
		}
	}

	// If LLM is not ready (missing API key), show the BYO model setup prompt.
	var needsSetupPopup bool
	if !llmReady {
		mode, appCfg := detectModelMode()
		_ = appCfg
		switch mode {
		case modelModeOwnEnv:
			// Env vars were applied but init still failed for another reason.
		default:
			needsSetupPopup = true
		}
	}

	memoryCfg, err := resolveAutoMemoryConfig(workDir)
	if err != nil {
		return nil, fmt.Errorf("init memory: %w", err)
	}

	toolRegistry := initTools(config, workDir, memoryCfg)
	var (
		mcpManager       runtimemcp.Manager
		mcpStartupEvents []model.Event
	)
	if !cfg.Replay {
		var mcpErr error
		mcpManager, mcpStartupEvents, mcpErr = initMCPTools(context.Background(), toolRegistry, workDir, defaultMCPStartupDiscoveryTimeout, cfg.MCPApprovalPrompter)
		if mcpErr != nil {
			mcpStartupEvents = append(mcpStartupEvents, model.Event{Type: model.ToolWarning, ToolName: "mcp", Message: mcpErr.Error()})
		}
	}

	// Skills: embedded skills are extracted next to the executable,
	// user-installed skills in ~/.mscli/skills/ override them,
	// project-local skills in .mscli/skills/ override both.
	homeDir, _ := os.UserHomeDir()
	execSkillsDir := ""
	if ep, err := os.Executable(); err == nil {
		execSkillsDir = filepath.Join(filepath.Dir(ep), ".mscli", "skills")
		_ = skills.ExtractBuiltin(execSkillsDir)
	}
	skillLoader := skills.NewLoader(
		execSkillsDir,
		filepath.Join(homeDir, ".mscli", "skills"),
		filepath.Join(workDir, ".mscli", "skills"),
	)
	toolRegistry.MustRegister(skillstool.NewLoadSkillTool(skillLoader))

	registerSkillCommands(skillLoader.List())

	managerCfg := agentctx.DefaultManagerConfig()
	managerCfg.ContextWindow = config.Context.Window
	managerCfg.ReserveTokens = config.Context.ReserveTokens
	managerCfg.CompactionThreshold = config.Context.CompactionThreshold
	managerCfg.CompactProvider = provider
	ctxManager := agentctx.NewManager(managerCfg)

	// Build system prompt: base + skill summaries + auto-memory operating
	// instructions. Loaded MSCLI.md and MEMORY.md contents are injected into a
	// separate initial user-role context message.
	systemPrompt := buildEffectiveSystemPromptFromSummariesWithMemory(skillLoader.List(), memoryCfg)

	var (
		runtimeSession       *session.Session
		replayBacklog        []model.Event
		replayTimeline       []session.ReplayFrame
		startupSessionPicker *sessionPickerRequest
	)
	if cfg.Resume && strings.TrimSpace(cfg.ResumeSessionID) == "" {
		startupSessionPicker = &sessionPickerRequest{Mode: model.SessionPickerResume}
	}
	if cfg.Replay && strings.TrimSpace(cfg.ReplaySessionID) == "" {
		startupSessionPicker = &sessionPickerRequest{
			Mode:        model.SessionPickerReplay,
			ReplaySpeed: cfg.ReplaySpeed,
		}
	}
	if cfg.Resume || cfg.Replay {
		if startupSessionPicker == nil {
			sessionID := cfg.ResumeSessionID
			if cfg.Replay {
				sessionID = cfg.ReplaySessionID
			}
			if strings.TrimSpace(sessionID) != "" {
				targetLabel := "session"
				if cfg.Replay && looksLikeTrajectoryPath(sessionID) {
					targetLabel = "trajectory"
				}
				if cfg.Replay && looksLikeTrajectoryPath(sessionID) {
					sourceSession, loadErr := session.LoadReplayPath(sessionID)
					if loadErr != nil {
						return nil, fmt.Errorf("load %s %s: %w", targetLabel, sessionID, loadErr)
					}
					systemPrompt, restoredMessages := sourceSession.RestoreContext()
					ctxManager.SetSystemPrompt(systemPrompt)
					ctxManager.SetNonSystemMessages(restoredMessages)
					restoreProviderUsageSnapshot(ctxManager, sourceSession.UsageSnapshot())
					replayTimeline = sourceSession.PlaybackTimeline()
					if metaWorkDir := strings.TrimSpace(sourceSession.Meta().WorkDir); metaWorkDir != "" {
						workDir = metaWorkDir
					}
					runtimeSession, err = session.Create(workDir, systemPrompt)
					if err != nil {
						return nil, fmt.Errorf("create replay session: %w", err)
					}
				} else {
					runtimeSession, err = session.LoadByID(workDir, sessionID)
					if err != nil {
						return nil, fmt.Errorf("load %s %s: %w", targetLabel, sessionID, err)
					}
					systemPrompt, restoredMessages := runtimeSession.RestoreContext()
					if !cfg.Replay {
						systemPrompt = buildEffectiveSystemPromptFromSummariesWithMemory(skillLoader.List(), memoryCfg)
					}
					ctxManager.SetSystemPrompt(systemPrompt)
					ctxManager.SetNonSystemMessages(restoredMessages)
					restoreProviderUsageSnapshot(ctxManager, runtimeSession.UsageSnapshot())
					if cfg.Replay {
						replayTimeline = runtimeSession.PlaybackTimeline()
					} else {
						replayBacklog = runtimeSession.ReplayEvents()
					}
				}
			} else {
				runtimeSession, err = session.LoadLatest(workDir)
				if err != nil {
					return nil, fmt.Errorf("load latest session: %w", err)
				}
				_, restoredMessages := runtimeSession.RestoreContext()
				systemPrompt = buildEffectiveSystemPromptFromSummariesWithMemory(skillLoader.List(), memoryCfg)
				ctxManager.SetSystemPrompt(systemPrompt)
				ctxManager.SetNonSystemMessages(restoredMessages)
				restoreProviderUsageSnapshot(ctxManager, runtimeSession.UsageSnapshot())
				replayBacklog = runtimeSession.ReplayEvents()
			}
		} else {
			runtimeSession, err = session.Create(workDir, systemPrompt)
			if err != nil {
				return nil, fmt.Errorf("create session: %w", err)
			}
			ctxManager.SetSystemPrompt(systemPrompt)
		}
	} else {
		runtimeSession, err = session.Create(workDir, systemPrompt)
		if err != nil {
			return nil, fmt.Errorf("create session: %w", err)
		}
		ctxManager.SetSystemPrompt(systemPrompt)
	}
	if len(mcpStartupEvents) > 0 {
		replayBacklog = append(replayBacklog, mcpStartupEvents...)
	}

	var llmDebugDumper *llm.DebugDumper
	if cfg.Debug && runtimeSession != nil {
		llmDebugDumper = llm.NewDebugDumper(filepath.Dir(runtimeSession.Path()))
	}

	engineCfg := newEngineConfig(config, systemPrompt)
	engine := loop.NewEngine(engineCfg, provider, toolRegistry)
	engine.SetContextManager(ctxManager)
	engine.SetLLMDebugDumper(llmDebugDumper)
	permService := permission.NewDefaultPermissionService(config.Permissions)
	permissionUI := NewPermissionPromptUI(eventCh)
	questionUI := NewAskUserQuestionPromptUI(eventCh)
	permService.SetUI(permissionUI)
	toolRegistry.MustRegister(askuserquestion.NewTool(questionUI))
	var (
		permSettingsIssue *permissionSettingsIssue
		sessionStoreReady bool
	)
	if issue := preloadScopedPermissionRules(permService, workDir); issue != nil {
		permSettingsIssue = issue
	}
	if cfg.Resume && startupSessionPicker == nil {
		storeCfg := sessionPermissionStoreConfig(runtimeSession)
		if store, err := permission.NewPermissionStore(storeCfg); err == nil {
			permService.SetStore(store)
			sessionStoreReady = true
		} else {
			if permSettingsIssue == nil {
				storePath := storeCfg.Path
				permSettingsIssue = &permissionSettingsIssue{
					FilePath: normalizePermissionSettingsPath(storePath, workDir),
					Detail:   err.Error(),
				}
			}
		}
	}
	engine.SetPermissionService(permService)

	app := &Application{
		Engine:                  engine,
		EventCh:                 eventCh,
		WorkDir:                 workDir,
		RepoURL:                 "gitcode.com/mindspore/mscli",
		Config:                  config,
		llmDebugDumper:          llmDebugDumper,
		provider:                provider,
		toolRegistry:            toolRegistry,
		ctxManager:              ctxManager,
		permService:             permService,
		permissionUI:            permissionUI,
		questionUI:              questionUI,
		permissionSettingsIssue: permSettingsIssue,
		mcpManager:              mcpManager,
		session:                 runtimeSession,
		memoryConfig:            memoryCfg,
		replayBacklog:           replayBacklog,
		replayTimeline:          replayTimeline,
		deferHistoryReplay:      cfg.Resume && !cfg.Replay && startupSessionPicker == nil,
		replayOnly:              cfg.Replay && startupSessionPicker == nil,
		replaySpeed:             replaySpeedOrDefault(cfg.ReplaySpeed),
		initialInput:            strings.TrimSpace(cfg.InitialInput),
		llmReady:                llmReady,
		skillLoader:             skillLoader,
		skillsHomeDir:           strings.TrimSpace(homeDir),
		needsSetupPopup:         needsSetupPopup,
		startupSessionPicker:    startupSessionPicker,
	}
	app.resumeHintOnExit.Store(cfg.Resume && startupSessionPicker == nil)
	permissionUI.SetYOLOCallbacks(
		func() bool {
			if svc, ok := app.permService.(*permission.DefaultPermissionService); ok {
				return svc.Check("shell", "") == permission.PermissionAllowAlways
			}
			return false
		},
		func() {
			app.cmdYolo()
		},
	)

	if sessionStoreReady {
		app.sessionStoreReady.Store(true)
	}
	app.refreshEngineSessionBindings()

	return app, nil
}

func looksLikeTrajectoryPath(target string) bool {
	target = strings.TrimSpace(target)
	if target == "" {
		return false
	}
	if strings.HasSuffix(target, ".json") || strings.HasSuffix(target, ".jsonl") {
		return true
	}
	if strings.HasPrefix(target, ".") || strings.ContainsAny(target, `/\`) {
		return true
	}
	return false
}

func replaySpeedOrDefault(speed float64) float64 {
	if speed <= 0 {
		return 1
	}
	return speed
}

func sessionPermissionStoreConfig(runtimeSession *session.Session) permission.PermissionStoreConfig {
	cfg := permission.DefaultPermissionStoreConfig()
	if runtimeSession == nil {
		return cfg
	}
	sessionDir := filepath.Dir(runtimeSession.Path())
	if strings.TrimSpace(sessionDir) == "" {
		return cfg
	}
	cfg.Path = filepath.Join(sessionDir, "permissions.json")
	return cfg
}

func (a *Application) ensureSessionPermissionStore() {
	if a == nil || a.permService == nil || a.session == nil || a.replayOnly {
		return
	}
	if a.sessionStoreReady.Load() {
		return
	}
	storeSetter, ok := a.permService.(interface {
		SetStore(permission.PermissionStore)
	})
	if !ok {
		return
	}

	storeCfg := sessionPermissionStoreConfig(a.session)
	store, err := permission.NewPermissionStore(storeCfg)
	if err != nil {
		if a.permissionSettingsIssue == nil {
			a.permissionSettingsIssue = &permissionSettingsIssue{
				FilePath: normalizePermissionSettingsPath(storeCfg.Path, a.WorkDir),
				Detail:   err.Error(),
			}
		}
		return
	}

	storeSetter.SetStore(store)
	a.sessionStoreReady.Store(true)
}

func (a *Application) refreshEngineSessionBindings() {
	if a == nil || a.Engine == nil {
		return
	}
	if a.ctxManager != nil {
		a.ctxManager.SetCompactProvider(a.provider)
		a.ctxManager.SetDebugDumper(a.llmDebugDumper)
		if a.llmDebugDumper != nil {
			a.ctxManager.SetPreCompactSnapshotHook(a.dumpPreCompactSnapshot)
		} else {
			a.ctxManager.SetPreCompactSnapshotHook(nil)
		}
		trajectoryPath := ""
		if a.session != nil {
			trajectoryPath = a.session.Path()
		}
		a.ctxManager.SetTrajectoryPath(trajectoryPath)
	}
	a.Engine.SetLLMDebugDumper(a.llmDebugDumper)
	a.Engine.SetTrajectoryRecorder(newTrajectoryRecorder(a.session, a.ctxManager, a.WorkDir, a.memoryConfig, a.noteLiveLLMActivity))
}

func (a *Application) dumpPreCompactSnapshot(snapshot agentctx.CompactSnapshot) error {
	if a == nil || a.llmDebugDumper == nil || a.session == nil {
		return nil
	}

	sessionDir := filepath.Dir(a.session.Path())
	if strings.TrimSpace(sessionDir) == "" || sessionDir == "." {
		return nil
	}

	meta := a.session.Meta()
	workDir := strings.TrimSpace(meta.WorkDir)
	if workDir == "" {
		workDir = a.WorkDir
	}

	messages := make([]llm.Message, len(snapshot.Messages))
	copy(messages, snapshot.Messages)
	payload := session.Snapshot{
		SessionID:     a.session.ID(),
		WorkDir:       workDir,
		SystemPrompt:  snapshot.SystemPrompt,
		UpdatedAt:     time.Now(),
		Messages:      messages,
		ProviderUsage: providerUsageSnapshotFromDetails(snapshot.Usage),
	}

	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal pre-compact snapshot: %w", err)
	}
	if err := os.MkdirAll(sessionDir, 0o755); err != nil {
		return fmt.Errorf("create session directory: %w", err)
	}

	trigger := strings.TrimSpace(string(snapshot.Trigger))
	if trigger == "" {
		trigger = "compact"
	}
	stamp := time.Now().UTC().Format("20060102-150405-000000000")
	path := filepath.Join(sessionDir, fmt.Sprintf("snapshot.compact-pre-%s-%s.json", trigger, stamp))
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write pre-compact snapshot: %w", err)
	}
	return nil
}

func (a *Application) rotateSession() error {
	if a == nil || a.replayOnly {
		return nil
	}

	systemPrompt, err := a.rebuildSystemPrompt()
	if err != nil {
		return fmt.Errorf("build system prompt: %w", err)
	}
	nextSession, err := session.Create(a.WorkDir, systemPrompt)
	if err != nil {
		return fmt.Errorf("create session: %w", err)
	}
	if err := nextSession.Activate(); err != nil {
		return fmt.Errorf("activate session: %w", err)
	}

	previous := a.session
	a.session = nextSession
	a.sessionLLMActivity.Store(false)
	a.resumeHintOnExit.Store(false)
	a.sessionStoreReady.Store(false)
	if a.ctxManager != nil {
		a.ctxManager.SetSystemPrompt(systemPrompt)
	}

	if permSvc, ok := a.permService.(*permission.DefaultPermissionService); ok {
		permSvc.ResetSessionState()
	}

	if a.llmDebugDumper != nil {
		a.llmDebugDumper = llm.NewDebugDumper(filepath.Dir(nextSession.Path()))
	}
	a.refreshEngineSessionBindings()

	if previous != nil {
		_ = previous.Close()
	}
	return nil
}

// SetProvider updates model/key and reinitializes the engine.
func (a *Application) SetProvider(providerName, modelName, apiKey string) error {
	normalizedProvider := llm.NormalizeProvider(providerName)
	if normalizedProvider != "" && !llm.IsSupportedProvider(normalizedProvider) {
		return fmt.Errorf("unsupported provider: %s", providerName)
	}

	previousModel := a.Config.Model.Model

	if normalizedProvider != "" {
		a.Config.Model.Provider = normalizedProvider
	}

	if modelName != "" {
		a.Config.Model.Model = modelName
	}
	if apiKey != "" {
		a.Config.Model.Key = apiKey
	}
	configs.RefreshModelTokenDefaults(a.Config, previousModel)

	resolveOpts := llm.ResolveOptions{
		PreferConfigAPIKey: strings.TrimSpace(apiKey) != "",
	}
	provider, err := initProvider(a.Config.Model, resolveOpts)
	if err != nil {
		if err == errAPIKeyNotFound {
			a.llmReady = false
			provider = nil
		} else {
			return fmt.Errorf("init provider: %w", err)
		}
	} else {
		a.llmReady = true
	}

	systemPrompt := ""
	if a.ctxManager != nil {
		if msg := a.ctxManager.GetSystemPrompt(); msg != nil {
			systemPrompt = msg.Content
		}
	}

	engineCfg := newEngineConfig(a.Config, systemPrompt)
	newEngine := loop.NewEngine(engineCfg, provider, a.toolRegistry)
	if a.ctxManager != nil {
		if err := a.ctxManager.SetContextWindowLimits(a.Config.Context.Window, a.Config.Context.ReserveTokens); err != nil {
			return fmt.Errorf("update context limits: %w", err)
		}
		a.ctxManager.SetCompactProvider(provider)
	}
	newEngine.SetContextManager(a.ctxManager)
	newEngine.SetPermissionService(a.permService)

	a.Engine = newEngine
	a.provider = provider
	a.refreshEngineSessionBindings()

	return nil
}

func initProvider(cfg configs.ModelConfig, opts llm.ResolveOptions) (llm.Provider, error) {
	resolved, err := llm.ResolveConfigWithOptions(cfg, opts)
	if err != nil {
		if errors.Is(err, llm.ErrMissingAPIKey) {
			return nil, errAPIKeyNotFound
		}
		return nil, fmt.Errorf("resolve provider config: %w", err)
	}

	client, err := buildProvider(resolved)
	if err != nil {
		return nil, fmt.Errorf("build provider: %w", err)
	}
	return client, nil
}

func newTrajectoryRecorder(s *session.Session, cm *agentctx.Manager, workDir string, memoryCfg autoMemoryConfig, noteLiveLLMActivity func() error) *loop.TrajectoryRecorder {
	ensureSessionActive := func() error {
		if noteLiveLLMActivity == nil {
			return nil
		}
		return noteLiveLLMActivity()
	}

	return &loop.TrajectoryRecorder{
		RecordUserInput: func(content string) error {
			if s == nil {
				return nil
			}
			return s.AppendUserInput(content)
		},
		RecordAssistant: func(content string) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendAssistant(content)
		},
		RecordToolCall: func(tc llm.ToolCall) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendToolCall(tc)
		},
		RecordToolResult: func(tc llm.ToolCall, content string, meta map[string]any) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendToolResult(tc.ID, tc.Function.Name, content, meta)
		},
		RecordSkillActivate: func(skillName string) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendSkillActivation(skillName)
		},
		RecordContextCompaction: func(trigger string, beforeTokens, afterTokens int, message string) error {
			if s == nil {
				return nil
			}
			if err := ensureSessionActive(); err != nil {
				return err
			}
			return s.AppendContextCompaction(trigger, beforeTokens, afterTokens, message)
		},
		PrepareFileMutation: func(tc llm.ToolCall) error {
			if s == nil {
				return nil
			}
			switch strings.TrimSpace(tc.Function.Name) {
			case "write", "edit":
			default:
				return nil
			}

			path, err := trackedFileMutationPath(tc)
			if err != nil {
				return err
			}
			if strings.TrimSpace(path) == "" {
				return nil
			}
			fullPath, err := fs.ResolveSafePathWithOptions(workDir, path, fsPathOptionsForMemory(memoryCfg))
			if err != nil {
				return err
			}
			if memoryCfg.Enabled && pathWithinAppRoot(memoryCfg.Dir, fullPath) {
				return nil
			}
			return s.RecordFileMutation(path, fullPath)
		},
		PersistSnapshot: func() error {
			if s == nil || cm == nil {
				return nil
			}
			systemPrompt := ""
			if msg := cm.GetSystemPrompt(); msg != nil {
				systemPrompt = msg.Content
			}
			return s.SaveSnapshotWithUsage(
				systemPrompt,
				cm.GetNonSystemMessages(),
				providerUsageSnapshotFromDetails(cm.TokenUsageDetails()),
			)
		},
	}
}

func trackedFileMutationPath(tc llm.ToolCall) (string, error) {
	var params struct {
		Path     string `json:"path"`
		FilePath string `json:"file_path"`
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal(tc.Function.Arguments, &params); err != nil {
		return "", fmt.Errorf("parse %s path: %w", tc.Function.Name, err)
	}
	for _, path := range []string{params.Path, params.FilePath, params.Filename} {
		if trimmed := strings.TrimSpace(path); trimmed != "" {
			return trimmed, nil
		}
	}
	return "", nil
}

func requestMaxTokensPtr(v *int) *int {
	if v == nil {
		return nil
	}
	copy := *v
	return &copy
}

func requestTemperaturePtr(v *float64) *float32 {
	if v == nil {
		return nil
	}
	copy := float32(*v)
	return &copy
}

func requestMaxIterations(v *int) int {
	if v == nil {
		return configs.DefaultRequestMaxIterations
	}
	return *v
}

func newEngineConfig(cfg *configs.Config, systemPrompt string) loop.EngineConfig {
	return loop.EngineConfig{
		MaxIterations:  requestMaxIterations(cfg.Request.MaxIterations),
		ContextWindow:  cfg.Context.Window,
		MaxTokens:      requestMaxTokensPtr(cfg.Request.MaxTokens),
		Temperature:    requestTemperaturePtr(cfg.Request.Temperature),
		Effort:         configs.NormalizeEffort(cfg.Request.Effort),
		TimeoutPerTurn: time.Duration(cfg.Model.TimeoutSec) * time.Second,
		SystemPrompt:   systemPrompt,
	}
}

// detectModelMode checks whether model config is already available.
// Mode is modelModeOwnEnv if BYO model env vars are complete.
func detectModelMode() (string, *appConfig) {
	provider := strings.TrimSpace(os.Getenv("MSCLI_PROVIDER"))
	apiKey := strings.TrimSpace(os.Getenv("MSCLI_API_KEY"))
	modelName := strings.TrimSpace(os.Getenv("MSCLI_MODEL"))
	if provider != "" && apiKey != "" && modelName != "" {
		return modelModeOwnEnv, nil
	}
	return "", nil
}

func (a *Application) emitModelSetupPopup(canEscape bool) {
	currentMode := ""
	if a.llmReady {
		currentMode = modelModeOwn
	}

	providerName, baseURL, modelName := "", "", ""
	configured := false
	if a.Config != nil {
		providerName = strings.TrimSpace(a.Config.Model.Provider)
		baseURL = strings.TrimSpace(a.Config.Model.URL)
		modelName = strings.TrimSpace(a.Config.Model.Model)
		configured = providerName != "" && baseURL != "" && modelName != ""
	}

	popup := &model.SetupPopup{
		Screen:       model.SetupScreenEnvInfo,
		CanEscape:    canEscape,
		CurrentMode:  currentMode,
		ModeSelected: 0,
		Provider:     providerName,
		BaseURL:      baseURL,
		ModelName:    modelName,
		Configured:   configured,
	}

	a.EventCh <- model.Event{
		Type:       model.ModelSetupOpen,
		SetupPopup: popup,
	}
}

func initTools(cfg *configs.Config, workDir string, memoryCfg autoMemoryConfig) *tools.Registry {
	registry := tools.NewRegistry()

	pathOptions := fsPathOptionsForMemory(memoryCfg)
	registry.MustRegister(fs.NewReadToolWithOptions(workDir, pathOptions))
	registry.MustRegister(fs.NewWriteToolWithOptions(workDir, pathOptions))
	registry.MustRegister(fs.NewEditToolWithOptions(workDir, pathOptions))
	registry.MustRegister(fs.NewGrepToolWithOptions(workDir, pathOptions))
	registry.MustRegister(fs.NewGlobToolWithOptions(workDir, pathOptions))

	shellRunner := rshell.NewRunner(rshell.Config{
		WorkDir:        workDir,
		Timeout:        time.Duration(cfg.Execution.TimeoutSec) * time.Second,
		AllowedCmds:    cfg.Permissions.AllowedTools,
		BlockedCmds:    cfg.Permissions.BlockedTools,
		RequireConfirm: []string{"rm", "mv", "cp"},
		// Python switches to block buffering when stdout is piped, which
		// prevents line-oriented command output from streaming live in the UI.
		Env: map[string]string{
			"PYTHONUNBUFFERED": "1",
		},
	})
	registry.MustRegister(shell.NewShellTool(shellRunner, workDir))

	return registry
}

func fsPathOptionsForMemory(memoryCfg autoMemoryConfig) fs.PathOptions {
	if !memoryCfg.Enabled || strings.TrimSpace(memoryCfg.Dir) == "" {
		return fs.PathOptions{}
	}
	return fs.PathOptions{AllowedAbsoluteRoots: []string{memoryCfg.Dir}}
}

func pathWithinAppRoot(root, target string) bool {
	root = strings.TrimSpace(root)
	target = strings.TrimSpace(target)
	if root == "" || target == "" {
		return false
	}
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return false
	}
	targetAbs, err := filepath.Abs(target)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(filepath.Clean(rootAbs), filepath.Clean(targetAbs))
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)))
}
