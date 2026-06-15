#!/usr/bin/env python3
"""
Tool definitions and execution engine.
Each tool has a name, description, JSON parameter schema, and a Python function.
"""
import json
import urllib.request
import urllib.parse
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    """Extract readable text from HTML, stripping tags and scripts."""
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip = False
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div", "tr"):
            self.text.append(" ")

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data.strip())

    def get_text(self):
        return " ".join(t for t in self.text if t)


def web_search(query):
    """Search the web using DuckDuckGo via ddgs library."""
    from ddgs import DDGS

    results = list(DDGS().text(query, max_results=5))
    if not results:
        return f"No results found for: {query}"

    output = f"Search results for '{query}':\n"
    for i, r in enumerate(results):
        output += f"\n{i+1}. {r['title']}\n   URL: {r['href']}\n   {r['body']}\n"
    return output


def read_webpage(url):
    """Fetch and extract text content from a webpage."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    extractor = TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()

    # Trim to reasonable length
    if len(text) > 4000:
        text = text[:4000] + "...[truncated]"

    return f"Content from {url}:\n\n{text}"


# ---- Working memory (shared across tools) ----
_working_memory = {}


def save_note(key, value):
    """Save a fact to working memory."""
    _working_memory[key] = value
    return f"Saved: {key} = {value[:100]}{'...' if len(value) > 100 else ''}"


def recall_note(key):
    """Recall a fact from working memory."""
    if key in _working_memory:
        return f"{key}: {_working_memory[key]}"
    return f"No note found for key: {key}"


def list_notes():
    """List all keys in working memory."""
    if not _working_memory:
        return "Working memory is empty."
    return "Working memory keys:\n" + "\n".join(f"  - {k}" for k in _working_memory)


def think(reflection):
    """Think internally — a scratchpad for reasoning. The agent can write
    observations, plan next steps, or reflect on what it has learned.
    This tool does nothing except acknowledge the thought was recorded."""
    return f"Thought recorded: {reflection[:200]}"


# ---- Tool registry ----
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use this when you need to find current facts, definitions, or anything not in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Fetch and read the text content of a webpage. Use this after web_search to get full article content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the webpage to read"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a key-value fact to working memory for later recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "A short key for the fact"},
                    "value": {"type": "string", "description": "The fact or information to save"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_note",
            "description": "Recall a previously saved fact from working memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to recall"},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Record an internal thought or reflection. Use this to plan next steps, synthesize findings, or note observations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reflection": {"type": "string", "description": "Your thought or observation"},
                },
                "required": ["reflection"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "web_search": web_search,
    "read_webpage": read_webpage,
    "save_note": save_note,
    "recall_note": recall_note,
    "think": think,
}


def execute_tool(name, arguments):
    """Execute a tool by name with given arguments. Returns the result string."""
    if name not in TOOL_HANDLERS:
        return f"Error: unknown tool '{name}'"
    try:
        return TOOL_HANDLERS[name](**arguments)
    except Exception as e:
        return f"Tool error: {e}"
