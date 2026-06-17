package mcp

const mcpProtocolVersion = "2025-11-25"

var supportedMCPProtocolVersions = map[string]bool{
	"2025-11-25": true,
	"2025-06-18": true,
	"2025-03-26": true,
}

func supportedMCPProtocolVersion(version string) bool {
	return supportedMCPProtocolVersions[version]
}
