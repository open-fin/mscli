package fs

import "gitcode.com/mindspore/mscli/tools"

// Tool wraps fs read/write/patch operations.
type Tool struct{}

func withSourceMeta(result *tools.Result) *tools.Result {
	tools.SetResultSource(result, tools.SourceFS)
	return result
}
