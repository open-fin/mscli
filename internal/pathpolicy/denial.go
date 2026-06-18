package pathpolicy

import (
	"encoding/json"
	"errors"
	"fmt"

	"gitcode.com/mindspore/mscli/tools"
)

const PathDenialMetaKey = "path_denial"

var ErrExternalPathDenied = errors.New("external path access denied")

type DenialKind string

const (
	DenialKindExternalRead  DenialKind = "external_read_denied"
	DenialKindExternalWrite DenialKind = "external_write_denied"
)

type PathDenial struct {
	Kind          string `json:"kind"`
	Operation     string `json:"operation"`
	InputPath     string `json:"input_path"`
	ResolvedPath  string `json:"resolved_path"`
	WorkDir       string `json:"work_dir"`
	SuggestedRoot string `json:"suggested_root"`
	RootKind      string `json:"root_kind"`
	Reason        string `json:"reason"`
}

func NewPathDenialResult(denial *PathDenial) *tools.Result {
	return &tools.Result{
		Error: ErrExternalPathDenied,
		Meta: map[string]any{
			PathDenialMetaKey: denial,
		},
	}
}

type PathDenialError struct {
	Denial *PathDenial
}

func (e *PathDenialError) Error() string {
	if e == nil || e.Denial == nil {
		return ErrExternalPathDenied.Error()
	}
	return e.Denial.ErrorMessage()
}

func NewPathDenialError(denial *PathDenial) error {
	return &PathDenialError{Denial: denial}
}

func ExtractPathDenialFromError(err error) (*PathDenial, bool) {
	var denialErr *PathDenialError
	if !errors.As(err, &denialErr) || denialErr == nil || denialErr.Denial == nil {
		return nil, false
	}
	return denialErr.Denial, true
}

func ExtractPathDenial(result *tools.Result) (*PathDenial, bool) {
	if result == nil || result.Meta == nil {
		return nil, false
	}
	raw, ok := result.Meta[PathDenialMetaKey]
	if !ok || raw == nil {
		return nil, false
	}
	return coercePathDenial(raw)
}

func coercePathDenial(raw any) (*PathDenial, bool) {
	switch v := raw.(type) {
	case *PathDenial:
		if v == nil {
			return nil, false
		}
		return v, true
	case PathDenial:
		return &v, true
	case map[string]any:
		data, err := json.Marshal(v)
		if err != nil {
			return nil, false
		}
		var denial PathDenial
		if err := json.Unmarshal(data, &denial); err != nil {
			return nil, false
		}
		return &denial, true
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return nil, false
		}
		var denial PathDenial
		if err := json.Unmarshal(data, &denial); err != nil {
			return nil, false
		}
		if denial.Kind == "" && denial.Operation == "" && denial.InputPath == "" {
			return nil, false
		}
		return &denial, true
	}
}

func (d *PathDenial) ErrorMessage() string {
	if d == nil {
		return ErrExternalPathDenied.Error()
	}
	msg := "External path access denied."
	if d.Operation != "" {
		msg += fmt.Sprintf("\nOperation: %s", d.Operation)
	}
	if d.WorkDir != "" {
		msg += fmt.Sprintf("\nCurrent workspace: %s", d.WorkDir)
	}
	if d.InputPath != "" {
		msg += fmt.Sprintf("\nRequested path: %s", d.InputPath)
	}
	if d.ResolvedPath != "" && d.ResolvedPath != d.InputPath {
		msg += fmt.Sprintf("\nResolved path: %s", d.ResolvedPath)
	}
	if d.SuggestedRoot != "" {
		msg += fmt.Sprintf("\nSuggested root: %s", d.SuggestedRoot)
	}
	if d.Reason != "" {
		msg += fmt.Sprintf("\nReason: %s", d.Reason)
	}
	return msg
}
