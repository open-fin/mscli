// Package mcp implements Model Context Protocol runtime support.
package mcp

import (
	"encoding/json"
	"time"
)

// Scope identifies where an MCP server config was loaded from.
type Scope string

const (
	ScopeUser    Scope = "user"
	ScopeProject Scope = "project"
	ScopeLocal   Scope = "local"
)

// ServerConfig is the supported MCP server config shape.
type ServerConfig struct {
	Type    string            `json:"type,omitempty"`
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	URL     string            `json:"url,omitempty"`
	Raw     map[string]any    `json:"-"`
}

// TransportType returns the explicit transport type, defaulting to stdio.
func (c ServerConfig) TransportType() string {
	if c.Type == "" {
		return "stdio"
	}
	return c.Type
}

// ScopedServer is a named server config with source scope metadata.
type ScopedServer struct {
	Name   string
	Scope  Scope
	Config ServerConfig
	Hash   string
}

// ToolDefinition is a discovered MCP tool adapted for the local registry.
type ToolDefinition struct {
	ServerName       string
	OriginalToolName string
	Name             string
	Description      string
	InputSchema      map[string]any
	Raw              map[string]any
}

// CallResult is the normalized result of an MCP tools/call request.
type CallResult struct {
	Content           []ContentBlock
	StructuredContent any
	IsError           bool
	Raw               json.RawMessage
}

// ContentBlock is a normalized MCP content block.
type ContentBlock struct {
	Type string
	Text string
	Data map[string]any
}

// Config controls MCP runtime behavior.
type Config struct {
	WorkDir        string
	ConnectTimeout time.Duration
	CallTimeout    time.Duration
}
