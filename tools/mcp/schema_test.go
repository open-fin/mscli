package mcp

import "testing"

func TestConvertSchemaPreservesObjectProperties(t *testing.T) {
	schema := ConvertSchema(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"mode": map[string]any{
				"type":        "string",
				"description": "execution mode",
				"enum":        []any{"fast", "safe"},
			},
			"config": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{"type": "string"},
				},
				"required": []any{"path"},
			},
			"items": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
			},
		},
		"required": []any{"mode"},
	})

	if schema.Type != "object" {
		t.Fatalf("schema.Type = %q", schema.Type)
	}
	if got := schema.Properties["mode"].Enum; len(got) != 2 || got[0] != "fast" || got[1] != "safe" {
		t.Fatalf("mode enum = %#v", got)
	}
	if got := schema.Properties["config"].Properties["path"].Type; got != "string" {
		t.Fatalf("nested path type = %q", got)
	}
	if got := schema.Properties["config"].Required; len(got) != 1 || got[0] != "path" {
		t.Fatalf("nested required = %#v", got)
	}
	if schema.Properties["items"].Items == nil || schema.Properties["items"].Items.Type != "string" {
		t.Fatalf("array items = %#v", schema.Properties["items"].Items)
	}
}

func TestConvertSchemaDefaultsToEmptyObject(t *testing.T) {
	schema := ConvertSchema(nil)
	if schema.Type != "object" {
		t.Fatalf("schema.Type = %q, want object", schema.Type)
	}
	if len(schema.Properties) != 0 {
		t.Fatalf("properties = %#v, want empty", schema.Properties)
	}
}

func TestConvertSchemaInfersObjectFromProperties(t *testing.T) {
	schema := ConvertSchema(map[string]any{
		"properties": map[string]any{
			"text": map[string]any{"type": "string"},
		},
	})
	if schema.Type != "object" {
		t.Fatalf("schema.Type = %q, want object", schema.Type)
	}
	if schema.Properties["text"].Type != "string" {
		t.Fatalf("text type = %q", schema.Properties["text"].Type)
	}
}
