package tools

import "testing"

func TestResultMetaHelpersMergeAndPreserveExistingMeta(t *testing.T) {
	result := &Result{
		Meta: map[string]any{
			"edit_diff": map[string]any{"path": "a.txt"},
		},
	}

	SetResultStatus(result, StatusCompleted)
	SetResultDuration(result, 42)
	SetResultExitCode(result, 7)
	SetResultTruncated(result, true)
	SetResultArtifact(result, "/tmp/full.txt", "projects/ws/tool-results/full.txt")
	SetResultBytes(result, 123)
	SetResultContentType(result, ContentTypeText)
	SetResultSource(result, SourceFS)

	if result.Meta["edit_diff"] == nil {
		t.Fatalf("edit_diff metadata was not preserved: %#v", result.Meta)
	}
	for key, want := range map[string]any{
		MetaStatus:               StatusCompleted,
		MetaDurationMS:           int64(42),
		MetaExitCode:             7,
		MetaTruncated:            true,
		MetaArtifactPath:         "/tmp/full.txt",
		MetaArtifactRelativePath: "projects/ws/tool-results/full.txt",
		MetaBytes:                int64(123),
		MetaContentType:          ContentTypeText,
		MetaSource:               SourceFS,
	} {
		if got := result.Meta[key]; got != want {
			t.Fatalf("Meta[%s] = %#v, want %#v (all meta %#v)", key, got, want, result.Meta)
		}
	}
}

func TestMergeResultMetaDoesNotOverwriteExistingKeys(t *testing.T) {
	result := &Result{Meta: map[string]any{MetaStatus: StatusFailed}}

	MergeResultMeta(result, map[string]any{
		MetaStatus: StatusCompleted,
		MetaSource: SourceLoop,
	})

	if got := result.Meta[MetaStatus]; got != StatusFailed {
		t.Fatalf("status = %v, want preserved failed", got)
	}
	if got := result.Meta[MetaSource]; got != SourceLoop {
		t.Fatalf("source = %v, want loop", got)
	}
}
