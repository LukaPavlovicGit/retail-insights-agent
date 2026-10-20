"""Safety rules enforced in code, not in prompts.

`sql_guard` validates and rewrites model SQL (single SELECT, allowed tables, no personal columns, per-user category
scope, row cap); `schema` holds the table schemas and the personal-column list; `pii_mask` masks personal data in
text that is shown or stored. Used by the graph before every query and before every reply is returned.
"""
