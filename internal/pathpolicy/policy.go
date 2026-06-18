package pathpolicy

import (
	"path/filepath"
	"strings"
	"sync"
)

type RootSource string

const (
	RootSourceBuiltin RootSource = "builtin"
	RootSourceConfig  RootSource = "config"
	RootSourceSession RootSource = "session"
)

type RootEntry struct {
	Path   string
	Source RootSource
}

type RootSet struct {
	mu    sync.RWMutex
	roots []RootEntry
}

func NewRootSet(entries ...RootEntry) *RootSet {
	rs := &RootSet{}
	for _, entry := range entries {
		rs.Add(entry.Path, entry.Source)
	}
	return rs
}

func (r *RootSet) Add(path string, source RootSource) {
	if r == nil {
		return
	}
	cleaned := strings.TrimSpace(path)
	if cleaned == "" {
		return
	}
	cleaned = filepath.Clean(cleaned)

	r.mu.Lock()
	defer r.mu.Unlock()
	for _, existing := range r.roots {
		if samePath(existing.Path, cleaned) && existing.Source == source {
			return
		}
	}
	r.roots = append(r.roots, RootEntry{Path: cleaned, Source: source})
}

func (r *RootSet) Snapshot() []RootEntry {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]RootEntry, len(r.roots))
	copy(out, r.roots)
	return out
}

type PathPolicy struct {
	WorkDir string

	ReadRoots  *RootSet
	WriteRoots *RootSet

	BuiltinReadRoots []string
}

func NewPathPolicy(workDir string, readRoots []string, builtinReadRoots []string) *PathPolicy {
	return NewPathPolicyWithWriteRoots(workDir, readRoots, nil, builtinReadRoots)
}

func NewPathPolicyWithWriteRoots(workDir string, readRoots []string, writeRoots []string, builtinReadRoots []string) *PathPolicy {
	p := &PathPolicy{
		WorkDir:          workDir,
		ReadRoots:        NewRootSet(),
		WriteRoots:       NewRootSet(),
		BuiltinReadRoots: append([]string(nil), builtinReadRoots...),
	}
	for _, root := range readRoots {
		p.ReadRoots.Add(root, RootSourceConfig)
	}
	for _, root := range writeRoots {
		p.WriteRoots.Add(root, RootSourceConfig)
	}
	for _, root := range builtinReadRoots {
		p.ReadRoots.Add(root, RootSourceBuiltin)
	}
	return p
}

func (p *PathPolicy) AddSessionReadRoot(root string) {
	if p == nil {
		return
	}
	if p.ReadRoots == nil {
		p.ReadRoots = NewRootSet()
	}
	p.ReadRoots.Add(root, RootSourceSession)
}

func (p *PathPolicy) AddSessionWriteRoot(root string) {
	if p == nil {
		return
	}
	if p.WriteRoots == nil {
		p.WriteRoots = NewRootSet()
	}
	p.WriteRoots.Add(root, RootSourceSession)
}

func samePath(a, b string) bool {
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}
