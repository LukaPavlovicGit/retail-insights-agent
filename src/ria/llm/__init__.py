"""LLM access behind one interface.

`client` defines `LLMClient`, `LLMResult` and `LLMUnavailable`; `gemini` is the real implementation (retries, fallback
model); `fake` is the scripted test double. The graph depends on the interface only, never on a provider SDK.
"""
