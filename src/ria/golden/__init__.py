"""Golden Trios: analyst-approved (question, SQL, report) examples used to steer the model.

`loader` reads seed trios from YAML, `embed` makes vectors, `store` keeps trios in SQLite with a vector index, and
`retrieve` finds the closest ones for a question. The graph injects them into the SQL and analysis prompts.
"""
