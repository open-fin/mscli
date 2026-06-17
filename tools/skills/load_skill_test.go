package skills

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	skillslib "gitcode.com/mindspore/mscli/integrations/skills"
	"gitcode.com/mindspore/mscli/tools"
)

func TestLoadSkillToolSetsSourceMeta(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "demo")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("mkdir skill: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte("---\nname: demo\ndescription: demo skill\n---\nbody\n"), 0644); err != nil {
		t.Fatalf("write skill: %v", err)
	}

	result, err := NewLoadSkillTool(skillslib.NewLoader(root)).Execute(context.Background(), []byte(`{"name":"demo"}`))
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Result error: %v", result.Error)
	}
	if got := result.Meta[tools.MetaSource]; got != tools.SourceSkill {
		t.Fatalf("source meta = %#v, want %q", got, tools.SourceSkill)
	}
}
