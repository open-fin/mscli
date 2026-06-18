package fs

import "gitcode.com/mindspore/mscli/internal/pathpolicy"

func newWorkspaceResolver(workDir string) *pathpolicy.Resolver {
	policy := pathpolicy.NewPathPolicy(workDir, nil, BuiltinReadRoots())
	return pathpolicy.NewResolver(policy)
}

func BuiltinReadRoots() []string {
	return []string{
		"~/.mscli/skills",
		"~/.mscli/mindspore-skills",
	}
}
