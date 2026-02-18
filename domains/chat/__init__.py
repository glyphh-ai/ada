"""Generic per-model chat service.

Every deployed model gets a chat endpoint that does:
  Glyphh HDC match → optional LLM formatting → response

The runtime owns LLM selection (local or external).
"""
