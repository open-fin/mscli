package tools

// Kind identifies the broad runtime category of a tool.
type Kind string

const (
	KindUnknown    Kind = "unknown"
	KindFilesystem Kind = "filesystem"
	KindShell      Kind = "shell"
	KindSkill      Kind = "skill"
	KindMCP        Kind = "mcp"
)

const (
	ResultTypeText     = "text"
	ResultTypeJSON     = "json"
	ResultTypeArtifact = "artifact"
)

// Capabilities describes runtime behavior relevant to policy, UI, and replay.
type Capabilities struct {
	Kind              Kind
	ReadOnly          bool
	MutatesWorkspace  bool
	MutatesContext    bool
	NetworkAccess     bool
	LongRunning       bool
	SupportsStreaming bool
	ResultTypes       []string
	Risk              string
}

// CapabilityProvider is implemented by tools that provide explicit metadata.
type CapabilityProvider interface {
	Capabilities() Capabilities
}

// ToolCapability is an ordered registry entry with runtime metadata.
type ToolCapability struct {
	Name         string
	Capabilities Capabilities
}

// CapabilitiesForTool returns explicit tool capabilities or conservative
// defaults for tools that have not adopted the optional metadata interface.
func CapabilitiesForTool(t Tool) Capabilities {
	if provider, ok := t.(CapabilityProvider); ok {
		return normalizeCapabilities(provider.Capabilities())
	}
	return defaultCapabilities()
}

func defaultCapabilities() Capabilities {
	return Capabilities{
		Kind:             KindUnknown,
		MutatesWorkspace: true,
		LongRunning:      true,
		ResultTypes:      []string{ResultTypeText},
		Risk:             "unknown",
	}
}

func normalizeCapabilities(c Capabilities) Capabilities {
	if c.Kind == "" {
		c.Kind = KindUnknown
	}
	if c.Risk == "" {
		c.Risk = "unknown"
	}
	if len(c.ResultTypes) == 0 {
		c.ResultTypes = []string{ResultTypeText}
	}
	return c
}
