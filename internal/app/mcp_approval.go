package app

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"

	runtimemcp "gitcode.com/mindspore/mscli/runtime/mcp"
)

type terminalMCPApprovalPrompter struct {
	in  io.Reader
	out io.Writer
}

func newTerminalMCPApprovalPrompter(in io.Reader, out io.Writer) *terminalMCPApprovalPrompter {
	return &terminalMCPApprovalPrompter{in: in, out: out}
}

func (p *terminalMCPApprovalPrompter) PromptMCPApproval(ctx context.Context, req MCPApprovalRequest) (runtimemcp.ApprovalDecision, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if p.in == nil || p.out == nil {
		return "", nil
	}
	fmt.Fprintf(p.out, "\nProject MCP server requires approval before it can start.\n")
	fmt.Fprintf(p.out, "Workspace: %s\n", req.WorkspaceRoot)
	fmt.Fprintf(p.out, "Server: %s\n", req.ServerName)
	if req.Transport != "" {
		fmt.Fprintf(p.out, "Transport: %s\n", req.Transport)
	}
	if strings.TrimSpace(req.URL) != "" {
		fmt.Fprintf(p.out, "URL: %s\n", req.URL)
	}
	if strings.TrimSpace(req.Command) != "" {
		fmt.Fprintf(p.out, "Command: %s\n", req.Command)
	}
	if len(req.Args) > 0 {
		fmt.Fprintf(p.out, "Args: %s\n", strings.Join(req.Args, " "))
	}
	if len(req.EnvKeys) > 0 {
		fmt.Fprintf(p.out, "Env keys: %s\n", strings.Join(req.EnvKeys, ", "))
	}
	fmt.Fprintf(p.out, "Config hash: %s\n", req.ConfigHash)
	fmt.Fprintf(p.out, "Approve this project MCP server? [y]es/[n]o/[s]kip: ")

	line, err := bufio.NewReader(p.in).ReadString('\n')
	if err != nil && len(line) == 0 {
		if err == io.EOF {
			return "", nil
		}
		return "", err
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes", "1":
		return runtimemcp.DecisionApproved, nil
	case "n", "no", "2":
		return runtimemcp.DecisionRejected, nil
	default:
		return "", nil
	}
}
