# CLAUDE.md

@./AGENTS.md

## MCP Servers

This project uses the [julia-mcp](https://github.com/aplavin/julia-mcp) MCP server
for Julia language tooling. Before starting any work, verify it is active:

```bash
claude mcp list
```

If `julia-mcp` is not listed or shows as disconnected, add it:

```bash
claude mcp add --transport stdio julia-mcp -- <command-from-julia-mcp-readme>
```

Then use `/mcp` inside the session to confirm the tools are available before proceeding.
