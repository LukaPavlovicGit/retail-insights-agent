"""Retail Insights Agent: a chat assistant that answers retail questions from BigQuery for scoped executives.

Layering, top to bottom: cli -> graph (the LangGraph turn pipeline) -> guards / llm / golden / memory -> tools -> mcp_server.
The graph only reaches BigQuery and the saved-reports library through `tools.ToolBox`, which talks to the stdio MCP server in `mcp_server`.
`observability`, `persona`, `evals`, `config` and `db` are shared support modules used by the layers above.
"""
