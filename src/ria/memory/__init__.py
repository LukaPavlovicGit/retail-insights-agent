"""Per-user memory that outlives a conversation: answer preferences (format, depth, charts).

`prefs` extracts preferences from user messages and stores them in SQLite. Saved reports are not here; they live in
`mcp_server.reports`. The graph reads this memory to shape prompts and rendering.
"""
