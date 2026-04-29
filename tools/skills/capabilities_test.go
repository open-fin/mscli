package skills

import (
	"testing"

	skillslib "github.com/mindspore-lab/mindspore-cli/integrations/skills"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

func TestLoadSkillCapabilities(t *testing.T) {
	got := tools.CapabilitiesForTool(NewLoadSkillTool(skillslib.NewLoader(t.TempDir())))
	if got.Kind != tools.KindSkill {
		t.Fatalf("Kind = %q, want %q", got.Kind, tools.KindSkill)
	}
	if !got.ReadOnly {
		t.Fatal("ReadOnly = false, want true")
	}
	if got.MutatesWorkspace {
		t.Fatal("MutatesWorkspace = true, want false")
	}
	if !got.MutatesContext {
		t.Fatal("MutatesContext = false, want true")
	}
}
