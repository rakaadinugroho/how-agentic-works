#!/usr/bin/env python3
"""
Ollama API client — simple wrapper for chatting with local models.
Supports function/tool calling via Ollama's native API.
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:latest"
EMBED_MODEL = "bge-m3:latest"


def chat(messages, tools=None, temperature=0.1):
    """
    Send a chat request to Ollama with optional tool definitions.
    Returns the full response with content and/or tool_calls.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]


def embed(texts):
    """
    Generate embeddings using bge-m3.
    Returns list of float vectors.
    """
    if isinstance(texts, str):
        texts = [texts]
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def list_tools_schema():
    """Return Ollama-compatible tool definitions for all available tools."""
    from tools import TOOL_DEFINITIONS
    return TOOL_DEFINITIONS
