package mcp

import (
	"gitcode.com/mindspore/mscli/integrations/llm"
)

// ConvertSchema converts an MCP inputSchema into the local LLM tool schema.
func ConvertSchema(raw map[string]any) llm.ToolSchema {
	if raw == nil {
		return emptyObjectSchema()
	}
	schemaType, _ := raw["type"].(string)
	if schemaType == "" {
		if _, ok := raw["properties"]; ok {
			schemaType = "object"
		}
	}
	if schemaType != "object" {
		return emptyObjectSchema()
	}
	schema := llm.ToolSchema{
		Type:       "object",
		Properties: convertProperties(raw["properties"]),
		Required:   stringSlice(raw["required"]),
	}
	if value, ok := raw["additionalProperties"]; ok {
		schema.AdditionalProperties = value
	}
	return schema
}

func emptyObjectSchema() llm.ToolSchema {
	return llm.ToolSchema{
		Type:       "object",
		Properties: map[string]llm.Property{},
	}
}

func convertProperties(raw any) map[string]llm.Property {
	rawMap, ok := raw.(map[string]any)
	if !ok || len(rawMap) == 0 {
		return map[string]llm.Property{}
	}
	out := make(map[string]llm.Property, len(rawMap))
	for name, value := range rawMap {
		propMap, ok := value.(map[string]any)
		if !ok {
			continue
		}
		out[name] = convertProperty(propMap)
	}
	return out
}

func convertProperty(raw map[string]any) llm.Property {
	prop := llm.Property{}
	if value, ok := raw["type"].(string); ok {
		prop.Type = value
	}
	if value, ok := raw["description"].(string); ok {
		prop.Description = value
	}
	prop.Enum = stringSlice(raw["enum"])
	if items, ok := raw["items"].(map[string]any); ok {
		converted := convertProperty(items)
		prop.Items = &converted
	}
	prop.Properties = convertProperties(raw["properties"])
	prop.Required = stringSlice(raw["required"])
	if value, ok := raw["additionalProperties"]; ok {
		prop.AdditionalProperties = value
	}
	return prop
}

func stringSlice(raw any) []string {
	switch values := raw.(type) {
	case []string:
		out := make([]string, len(values))
		copy(out, values)
		return out
	case []any:
		out := make([]string, 0, len(values))
		for _, value := range values {
			s, ok := value.(string)
			if !ok {
				continue
			}
			out = append(out, s)
		}
		return out
	default:
		return nil
	}
}
