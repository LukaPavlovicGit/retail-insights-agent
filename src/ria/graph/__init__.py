"""The agent itself: a LangGraph state machine that handles one conversational turn.

`build` wires the nodes, `state` defines what flows between them, `run` executes a turn, and `analysis`, `library`,
`router`, `schema_qa` and `lifecycle` hold the nodes by purpose. It calls down into `guards`, `llm`, `golden`,
`memory` and `tools`; the CLI and the evals are its callers.
"""
