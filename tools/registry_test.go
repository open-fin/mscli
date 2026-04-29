package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"gitcode.com/mindspore/mscli/integrations/llm"
)

func TestRegistryMetadataHelpers(t *testing.T) {
	registry := NewRegistry()
	registry.MustRegister(plainStubTool{name: "read"})
	registry.MustRegister(plainStubTool{name: "grep"})
	registry.MustRegister(plainStubTool{name: "glob"})
	registry.MustRegister(plainStubTool{name: "shell"})
	registry.MustRegister(plainStubTool{name: "load_skill"})
	registry.MustRegister(plainStubTool{name: "write"})
	registry.MustRegister(plainStubTool{name: "edit"})
	registry.MustRegister(plainStubTool{name: "plain"})

	metadata := registry.Metadata("read")
	if !sameClasses(metadata.Classes, []ToolClass{ToolClassExploration, ToolClassVerification}) {
		t.Fatalf("Metadata(read) = %v, want exploration and verification", metadata.Classes)
	}
	metadata.Classes[0] = ToolClassMutation
	if !registry.HasClass("read", ToolClassExploration) {
		t.Fatal("mutating returned metadata changed registry metadata")
	}
	if registry.HasClass("read", ToolClassMutation) {
		t.Fatal("HasClass(read, mutation) = true, want false")
	}
	for _, name := range []string{"read", "grep", "glob", "shell", "load_skill"} {
		if !registry.HasClass(name, ToolClassExploration) && !registry.HasClass(name, ToolClassContextExpansion) {
			t.Fatalf("registry classification for %q = non-research, want research", name)
		}
	}
	for _, name := range []string{"write", "edit"} {
		if registry.HasClass(name, ToolClassExploration) || registry.HasClass(name, ToolClassContextExpansion) {
			t.Fatalf("registry classification for %q = research, want non-research", name)
		}
	}
	if registry.HasClass("plain", ToolClassExploration) {
		t.Fatal("HasClass(plain, exploration) = true, want false")
	}
	if got := registry.Metadata("missing").Classes; len(got) != 0 {
		t.Fatalf("Metadata(missing) = %v, want empty", got)
	}
}

func TestRegistryToLLMToolsFilteredUsesRuntimeMetadata(t *testing.T) {
	registry := NewRegistry()
	registry.MustRegister(plainStubTool{name: "read"})
	registry.MustRegister(plainStubTool{name: "write"})
	registry.MustRegister(plainStubTool{name: "plain"})

	allTools := registry.ToLLMTools()
	if got, want := llmToolNames(allTools), []string{"read", "write", "plain"}; !sameStrings(got, want) {
		t.Fatalf("ToLLMTools names = %v, want %v", got, want)
	}

	filtered := registry.ToLLMToolsFiltered(func(tool Tool, metadata ToolMetadata) bool {
		if len(metadata.Classes) == 0 {
			return false
		}
		for _, class := range metadata.Classes {
			if class == ToolClassExploration {
				return false
			}
		}
		return true
	})
	if got, want := llmToolNames(filtered), []string{"write"}; !sameStrings(got, want) {
		t.Fatalf("filtered tool names = %v, want %v", got, want)
	}
}

func TestToolMetadataDoesNotEnterLLMToolSchema(t *testing.T) {
	registry := NewRegistry()
	registry.MustRegister(plainStubTool{name: "read"})

	llmTools := registry.ToLLMTools()
	if len(llmTools) != 1 {
		t.Fatalf("ToLLMTools len = %d, want 1", len(llmTools))
	}
	encoded, err := json.Marshal(llmTools[0])
	if err != nil {
		t.Fatalf("marshal llm tool: %v", err)
	}
	text := string(encoded)
	for _, internal := range []string{"exploration", "verification", "context_expansion", "external_effect"} {
		if strings.Contains(text, internal) {
			t.Fatalf("LLM tool schema contains runtime metadata %q: %s", internal, text)
		}
	}
}

func TestRegistryUnregisterRemovesToolAndOrder(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(plainStubTool{name: "one"}); err != nil {
		t.Fatalf("Register one: %v", err)
	}
	if err := registry.Register(plainStubTool{name: "two"}); err != nil {
		t.Fatalf("Register two: %v", err)
	}

	if !registry.Unregister("one") {
		t.Fatal("Unregister(one) = false, want true")
	}
	if _, ok := registry.Get("one"); ok {
		t.Fatal("Get(one) ok after unregister")
	}
	if got := registry.Names(); len(got) != 1 || got[0] != "two" {
		t.Fatalf("Names() = %#v, want [two]", got)
	}
	if registry.Unregister("missing") {
		t.Fatal("Unregister(missing) = true, want false")
	}
}

type plainStubTool struct {
	name string
}

func (t plainStubTool) Name() string { return t.name }

func (t plainStubTool) Description() string { return "plain stub" }

func (t plainStubTool) Schema() llm.ToolSchema { return llm.ToolSchema{Type: "object"} }

func (t plainStubTool) Execute(context.Context, json.RawMessage) (*Result, error) {
	return StringResult("ok"), nil
}

func TestCapabilitiesForToolUsesConservativeFallback(t *testing.T) {
	got := CapabilitiesForTool(plainStubTool{name: "plain"})
	if got.Kind != KindUnknown {
		t.Fatalf("Kind = %q, want %q", got.Kind, KindUnknown)
	}
	if got.ReadOnly {
		t.Fatal("ReadOnly = true, want false for unknown tool")
	}
	if !got.MutatesWorkspace {
		t.Fatal("MutatesWorkspace = false, want true for conservative fallback")
	}
	if !got.LongRunning {
		t.Fatal("LongRunning = false, want true for conservative fallback")
	}
	if got.Risk != "unknown" {
		t.Fatalf("Risk = %q, want unknown", got.Risk)
	}
}

func TestRegistryCapabilityListPreservesRegistrationOrder(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(capabilityTool{name: "one", kind: KindFilesystem}); err != nil {
		t.Fatalf("Register one: %v", err)
	}
	if err := registry.Register(capabilityTool{name: "two", kind: KindShell}); err != nil {
		t.Fatalf("Register two: %v", err)
	}

	got := registry.CapabilityList()
	if len(got) != 2 {
		t.Fatalf("CapabilityList len = %d, want 2", len(got))
	}
	if got[0].Name != "one" || got[0].Capabilities.Kind != KindFilesystem {
		t.Fatalf("CapabilityList[0] = %#v, want filesystem one", got[0])
	}
	if got[1].Name != "two" || got[1].Capabilities.Kind != KindShell {
		t.Fatalf("CapabilityList[1] = %#v, want shell two", got[1])
	}
}

type capabilityTool struct {
	plainStubTool
	name string
	kind Kind
}

func (c capabilityTool) Name() string { return c.name }

func (c capabilityTool) Capabilities() Capabilities {
	return Capabilities{Kind: c.kind, ReadOnly: true, Risk: "low"}
}

func llmToolNames(tools []llm.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Function.Name)
	}
	return names
}

func sameClasses(got, want []ToolClass) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func sameStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
