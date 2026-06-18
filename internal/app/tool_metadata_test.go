package app

import (
	"testing"

	"gitcode.com/mindspore/mscli/configs"
)

func TestInitToolsRegisteredToolsHaveMetadata(t *testing.T) {
	registry := initTools(configs.DefaultConfig(), t.TempDir(), autoMemoryConfig{})
	for _, name := range registry.Names() {
		if got := registry.Metadata(name).Classes; len(got) == 0 {
			t.Fatalf("registered tool %q has no runtime metadata", name)
		}
	}
}
