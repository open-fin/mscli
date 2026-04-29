package tools

import "time"

const (
	MetaStatus               = "status"
	MetaDurationMS           = "duration_ms"
	MetaExitCode             = "exit_code"
	MetaTruncated            = "truncated"
	MetaBytes                = "bytes"
	MetaArtifactPath         = "artifact_path"
	MetaArtifactRelativePath = "artifact_relative_path"
	MetaContentType          = "content_type"
	MetaSource               = "source"
	MetaServer               = "server"
	MetaTool                 = "tool"
	MetaFallback             = "fallback"
)

const (
	StatusCompleted   = "completed"
	StatusFailed      = "failed"
	StatusInterrupted = "interrupted"
	StatusDeclined    = "declined"
)

const (
	SourceFS         = "fs"
	SourceShell      = "shell"
	SourceSkill      = "skill"
	SourceMCP        = "mcp"
	SourceLoop       = "loop"
	SourcePermission = "permission"
)

const (
	ContentTypeText = "text/plain"
	ContentTypeJSON = "application/json"
)

func MergeResultMeta(result *Result, meta map[string]any) {
	if result == nil || len(meta) == 0 {
		return
	}
	if result.Meta == nil {
		result.Meta = make(map[string]any, len(meta))
	}
	for key, value := range meta {
		if _, exists := result.Meta[key]; exists {
			continue
		}
		result.Meta[key] = value
	}
}

func SetResultStatus(result *Result, status string) {
	setResultMeta(result, MetaStatus, status)
}

func SetResultDuration(result *Result, durationMS int64) {
	setResultMeta(result, MetaDurationMS, durationMS)
}

func SetResultDurationValue(result *Result, duration time.Duration) {
	SetResultDuration(result, duration.Milliseconds())
}

func SetResultExitCode(result *Result, exitCode int) {
	setResultMeta(result, MetaExitCode, exitCode)
}

func SetResultTruncated(result *Result, truncated bool) {
	setResultMeta(result, MetaTruncated, truncated)
}

func SetResultBytes(result *Result, bytes int64) {
	setResultMeta(result, MetaBytes, bytes)
}

func SetResultArtifact(result *Result, path, relativePath string) {
	if path != "" {
		setResultMeta(result, MetaArtifactPath, path)
	}
	if relativePath != "" {
		setResultMeta(result, MetaArtifactRelativePath, relativePath)
	}
}

func SetResultContentType(result *Result, contentType string) {
	setResultMeta(result, MetaContentType, contentType)
}

func SetResultSource(result *Result, source string) {
	setResultMeta(result, MetaSource, source)
}

func SetResultServer(result *Result, server string) {
	setResultMeta(result, MetaServer, server)
}

func SetResultTool(result *Result, tool string) {
	setResultMeta(result, MetaTool, tool)
}

func setResultMeta(result *Result, key string, value any) {
	if result == nil || key == "" {
		return
	}
	if result.Meta == nil {
		result.Meta = make(map[string]any)
	}
	result.Meta[key] = value
}
