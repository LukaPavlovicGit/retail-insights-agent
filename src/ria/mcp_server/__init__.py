"""The stdio MCP server that owns all external side effects: BigQuery access and the saved-reports library.

`server` defines the tools, `bigquery` runs capped queries with a circuit breaker, `reports` stores saved reports.
The agent reaches it only through `ria.tools`; this is the bottom layer and imports nothing from `graph`.
"""
