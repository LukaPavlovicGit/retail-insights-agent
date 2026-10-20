"""Evaluation harness for `ria eval`.

`checks` defines the YAML case format and deterministic pass/fail rules; `runner` runs each case through the live graph
and adds an LLM judge score. Nothing else in the package imports this; it sits on top of `graph` and `observability`.
"""
