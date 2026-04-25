package mcp

import "strings"

// NormalizeName converts a server or tool name to the provider-safe MCP form.
func NormalizeName(name string) string {
	name = strings.TrimSpace(name)
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return strings.Trim(b.String(), "_")
}

// BuildToolName builds a fully qualified MCP tool name.
func BuildToolName(serverName, toolName string) string {
	return "mcp__" + NormalizeName(serverName) + "__" + NormalizeName(toolName)
}
