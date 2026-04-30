package slash

import (
	"strings"
	"testing"
)

func TestDefaultRegistryIncludesExpectedVisibleCommands(t *testing.T) {
	registry := NewRegistry()

	for _, name := range []string{"/branch", "/effort", "/fork", "/init"} {
		cmd, ok := registry.Get(name)
		if !ok {
			t.Fatalf("default registry missing %s", name)
		}
		if cmd.Hidden {
			t.Fatalf("%s should be visible in slash suggestions", name)
		}
	}
}

func TestDefaultRegistryIncludesFactory(t *testing.T) {
	registry := NewRegistry()
	cmd, ok := registry.Get("/factory")
	if !ok {
		t.Fatal("default registry missing /factory")
	}
	if cmd.Hidden {
		t.Fatal("/factory should be visible in slash suggestions")
	}
	if !strings.Contains(cmd.Description, "Factory") {
		t.Fatalf("Description = %q, want Factory-related text", cmd.Description)
	}
}

func TestSuggestionsIncludeFactoryForFPrefix(t *testing.T) {
	suggestions := NewRegistry().Suggestions("/f")
	for _, suggestion := range suggestions {
		if suggestion == "/factory" {
			return
		}
	}
	t.Fatalf("Suggestions(/f) = %#v, want /factory", suggestions)
}
