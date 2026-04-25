package mcp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// ApprovalDecision is a persisted project MCP server decision.
type ApprovalDecision string

const (
	DecisionApproved ApprovalDecision = "approved"
	DecisionRejected ApprovalDecision = "rejected"
)

// ApprovalEntry is one persisted project-server approval decision.
type ApprovalEntry struct {
	Workspace string           `json:"workspace"`
	Server    string           `json:"server"`
	Hash      string           `json:"hash"`
	Decision  ApprovalDecision `json:"decision"`
}

type approvalFile struct {
	Entries []ApprovalEntry `json:"entries"`
}

// ApprovalStore persists MCP project-server decisions.
type ApprovalStore struct {
	path string
	mu   sync.Mutex
}

// DefaultApprovalStorePath returns the user-global MCP approval file path.
func DefaultApprovalStorePath(home string) string {
	if home == "" {
		return filepath.Join(".mscli", "mcp_approvals.json")
	}
	return filepath.Join(home, ".mscli", "mcp_approvals.json")
}

// NewApprovalStore creates a file-backed approval store.
func NewApprovalStore(path string) *ApprovalStore {
	return &ApprovalStore{path: path}
}

// CanonicalConfigHash returns a stable hash for the explicit server config.
func CanonicalConfigHash(cfg ServerConfig) string {
	raw := cfg.Raw
	if raw == nil {
		raw = rawFromConfig(cfg)
	}
	data, err := json.Marshal(canonicalize(raw))
	if err != nil {
		data = []byte("{}")
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// Lookup returns a stored decision for the exact workspace/server/hash tuple.
func (s *ApprovalStore) Lookup(workspace, server, hash string) (ApprovalDecision, bool, error) {
	if s == nil || s.path == "" {
		return "", false, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	file, err := s.loadLocked()
	if err != nil {
		return "", false, err
	}
	for _, entry := range file.Entries {
		if entry.Workspace == workspace && entry.Server == server && entry.Hash == hash {
			return entry.Decision, true, nil
		}
	}
	return "", false, nil
}

// Record persists a decision for the exact workspace/server/hash tuple.
func (s *ApprovalStore) Record(workspace, server, hash string, decision ApprovalDecision) error {
	if s == nil || s.path == "" {
		return fmt.Errorf("approval store path is empty")
	}
	if decision != DecisionApproved && decision != DecisionRejected {
		return fmt.Errorf("unsupported approval decision %q", decision)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	file, err := s.loadLocked()
	if err != nil {
		return err
	}
	entry := ApprovalEntry{
		Workspace: workspace,
		Server:    server,
		Hash:      hash,
		Decision:  decision,
	}
	for i, existing := range file.Entries {
		if existing.Workspace == workspace && existing.Server == server && existing.Hash == hash {
			file.Entries[i] = entry
			return s.saveLocked(file)
		}
	}
	file.Entries = append(file.Entries, entry)
	return s.saveLocked(file)
}

// ResetWorkspace removes all approval decisions for a workspace.
func (s *ApprovalStore) ResetWorkspace(workspace string) (int, error) {
	if s == nil || s.path == "" {
		return 0, fmt.Errorf("approval store path is empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	file, err := s.loadLocked()
	if err != nil {
		return 0, err
	}
	kept := make([]ApprovalEntry, 0, len(file.Entries))
	removed := 0
	for _, entry := range file.Entries {
		if entry.Workspace == workspace {
			removed++
			continue
		}
		kept = append(kept, entry)
	}
	file.Entries = kept
	if err := s.saveLocked(file); err != nil {
		return 0, err
	}
	return removed, nil
}

func (s *ApprovalStore) loadLocked() (approvalFile, error) {
	var file approvalFile
	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return file, nil
		}
		return file, fmt.Errorf("read mcp approvals: %w", err)
	}
	if len(data) == 0 {
		return file, nil
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return file, fmt.Errorf("decode mcp approvals: %w", err)
	}
	return file, nil
}

func (s *ApprovalStore) saveLocked(file approvalFile) error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0755); err != nil {
		return fmt.Errorf("create mcp approvals directory: %w", err)
	}
	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return fmt.Errorf("encode mcp approvals: %w", err)
	}
	if err := os.WriteFile(s.path, data, 0644); err != nil {
		return fmt.Errorf("write mcp approvals: %w", err)
	}
	return nil
}

func rawFromConfig(cfg ServerConfig) map[string]any {
	raw := make(map[string]any)
	if cfg.Type != "" {
		raw["type"] = cfg.Type
	}
	if cfg.Command != "" {
		raw["command"] = cfg.Command
	}
	if len(cfg.Args) > 0 {
		raw["args"] = cfg.Args
	}
	if len(cfg.Env) > 0 {
		raw["env"] = cfg.Env
	}
	if cfg.URL != "" {
		raw["url"] = cfg.URL
	}
	return raw
}

func canonicalize(v any) any {
	switch x := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for key := range x {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		out := make(map[string]any, len(x))
		for _, key := range keys {
			out[key] = canonicalize(x[key])
		}
		return out
	case map[string]string:
		keys := make([]string, 0, len(x))
		for key := range x {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		out := make(map[string]any, len(x))
		for _, key := range keys {
			out[key] = x[key]
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = canonicalize(item)
		}
		return out
	case []string:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = item
		}
		return out
	default:
		return x
	}
}
