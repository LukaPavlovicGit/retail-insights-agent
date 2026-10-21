"""The assistant's persona: voice and tone text loaded from a Markdown file in the data directory.

`loader.load_persona` falls back to a built-in default. `graph.deps.Deps.persona` is the only caller of the loader.
`gate` guards changes to the file: `lint_persona` (offline check), `live_gate` (eval set with the candidate) and
`publish_persona` (versioned, atomic swap).
"""
