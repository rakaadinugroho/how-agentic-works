#!/usr/bin/env python3
"""
Tool definitions and execution engine.
Each tool has a name, description, JSON parameter schema, and a Python function.
"""
import json
import re
import urllib.request
import urllib.parse
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# ---- Garuda (Garba Rujukan Digital) ----
GARUDA_BASE = "https://garuda.kemdiktisaintek.go.id"


def _fetch_garuda(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_articles(html):
    articles = []
    blocks = re.split(r'<div class="article-item">', html)[1:]
    for block in blocks:
        end = block.find('<div class="ui divider">')
        if end != -1:
            block = block[:end]
        article = {}

        title_m = re.search(r'<a class="title-article"[^>]*>\s*<xmp>(.*?)</xmp>\s*</a>', block, re.DOTALL)
        if title_m:
            article["title"] = title_m.group(1).strip()

        detail_m = re.search(r'href="/documents/detail/(\d+)"', block)
        if detail_m:
            article["garuda_id"] = detail_m.group(1)
            article["url"] = f"{GARUDA_BASE}/documents/detail/{detail_m.group(1)}"

        authors = re.findall(r'<a class="author-article"[^>]*>\s*<xmp>(.*?)</xmp>\s*</a>', block, re.DOTALL)
        article["authors"] = [a.strip() for a in authors]

        journal_m = re.search(r'<xmp class="subtitle-article">(.*?)</xmp>', block, re.DOTALL)
        if journal_m:
            article["journal"] = journal_m.group(1).strip()

        abstract_m = re.search(r'<xmp class="abstract-article">(.*?)</xmp>', block, re.DOTALL)
        if abstract_m:
            article["abstract"] = abstract_m.group(1).strip()

        doi_m = re.search(r'https://doi\.org/([^\s"\'<]+)', block)
        if doi_m:
            article["doi"] = f"https://doi.org/{doi_m.group(1)}"

        dl_m = re.search(r'<a class="title-citation" href="(http[^"]*download[^"]*|http[^"]*article/download[^"]*)"', block)
        if dl_m:
            article["download_url"] = dl_m.group(1)

        gs_m = re.search(r'href="(http://scholar\.google\.com/scholar\?[^"]*)"', block)
        if gs_m:
            article["google_scholar"] = gs_m.group(1)

        if article.get("title"):
            articles.append(article)
    return articles


def garuda_search(query, page=1):
    """Search Indonesian academic publications via Garuda (Garba Rujukan Digital)."""
    encoded = urllib.parse.quote(query)
    url = f"{GARUDA_BASE}/documents?page={page}&q={encoded}&select=title"
    html = _fetch_garuda(url)

    total_m = re.search(r'Found\s+(\d+)\s+documents', html)
    total = int(total_m.group(1)) if total_m else 0

    articles = _parse_articles(html)

    if not articles:
        return f"No Indonesian publications found for: {query}"

    per_page = len(articles)
    total_pages = (total + per_page - 1) // per_page if per_page else 1

    output = f"Garuda search results for '{query}' (page {page}/{total_pages}, total: {total} documents):\n"
    for i, a in enumerate(articles):
        idx = (page - 1) * per_page + i + 1
        output += f"\n{idx}. {a['title']}"
        if a.get("authors"):
            output += f"\n   Authors: {', '.join(a['authors'][:5])}"
            if len(a["authors"]) > 5:
                output += f" (+{len(a['authors']) - 5} more)"
        if a.get("journal"):
            output += f"\n   Journal: {a['journal']}"
        if a.get("doi"):
            output += f"\n   DOI: {a['doi']}"
        output += f"\n   Garuda ID: {a.get('garuda_id', 'N/A')}"
        if a.get("abstract"):
            abstract = a["abstract"][:400] + "..." if len(a["abstract"]) > 400 else a["abstract"]
            output += f"\n   Abstract: {abstract}"
        output += "\n"
    return output


def _search_single(query):
    """Search Garuda for a single query, return list of article dicts."""
    encoded = urllib.parse.quote(query)
    url = f"{GARUDA_BASE}/documents?page=1&q={encoded}&select=title"
    try:
        html = _fetch_garuda(url)
        return _parse_articles(html)
    except Exception:
        return []


def garuda_multi_search(queries):
    """Search Garuda with multiple keywords in parallel. Merges and deduplicates results."""
    if isinstance(queries, str):
        queries = [q.strip() for q in queries.split(",") if q.strip()]
    if not queries:
        return "Error: no queries provided."

    seen = set()
    all_articles = []

    with ThreadPoolExecutor(max_workers=min(5, len(queries))) as ex:
        futures = {ex.submit(_search_single, q): q for q in queries}
        for f in as_completed(futures):
            query = futures[f]
            try:
                articles = f.result()
            except Exception:
                articles = []
            for a in articles:
                gid = a.get("garuda_id")
                if gid and gid not in seen:
                    seen.add(gid)
                    a["matched_query"] = query
                    all_articles.append(a)

    if not all_articles:
        return f"No results across {len(queries)} queries: {', '.join(queries)}"

    output = f"Garuda multi-search ({len(queries)} queries, {len(all_articles)} unique results):\n"
    for i, a in enumerate(all_articles[:15]):
        output += f"\n{i+1}. {a['title']}"
        if a.get("authors"):
            output += f"\n   Authors: {', '.join(a['authors'][:5])}"
            if len(a["authors"]) > 5:
                output += f" (+{len(a['authors']) - 5} more)"
        if a.get("journal"):
            output += f"\n   Journal: {a['journal']}"
        if a.get("doi"):
            output += f"\n   DOI: {a['doi']}"
        output += f"\n   Garuda ID: {a.get('garuda_id', 'N/A')}  [matched: {a.get('matched_query', '?')}]"
        if a.get("abstract"):
            abstract = a["abstract"][:300] + "..." if len(a["abstract"]) > 300 else a["abstract"]
            output += f"\n   Abstract: {abstract}"
        output += "\n"

    if len(all_articles) > 15:
        output += f"\n... and {len(all_articles) - 15} more results.\n"

    return output


def garuda_detail(garuda_id):
    """Fetch full details of a Garuda document by its ID."""
    url = f"{GARUDA_BASE}/documents/detail/{garuda_id}"
    html = _fetch_garuda(url)

    title_m = re.search(r'<h3 class="ui header">\s*<xmp>(.*?)</xmp>', html, re.DOTALL)
    title = title_m.group(1).strip() if title_m else "N/A"

    authors_m = re.findall(r'<a href="/author/view/\d+">\s*<xmp>(.*?)</xmp>\s*</a>', html, re.DOTALL)
    authors = [a.strip() for a in authors_m]

    journal_m = re.search(r'<div class="ui segment article-display"[^>]*>.*?<xmp>(.*?)</xmp>', html, re.DOTALL)
    journal = journal_m.group(1).strip() if journal_m else None

    vol_m = re.search(r'<hr>\s*<xmp>(.*?)</xmp>', html, re.DOTALL)
    volume_issue = vol_m.group(1).strip() if vol_m else None

    date_m = re.search(r'Publish Date\s*<br>\s*(\d+\s+\w+\s+\d+)', html)
    pub_date = date_m.group(1) if date_m else None

    doi_m = re.search(r'https://doi\.org/([^\s"\'<]+)', html)
    doi = f"https://doi.org/{doi_m.group(1)}" if doi_m else None

    original_m = re.search(r'<a[^>]*href="(https?://[^"]*)"[^>]*>\s*<div[^>]*>\s*Original Source', html, re.DOTALL)
    original_url = original_m.group(1) if original_m else None

    download_m = re.search(r'<a[^>]*href="(https?://[^"]*)"[^>]*>\s*<div[^>]*>\s*Download Original', html, re.DOTALL)
    download_url = download_m.group(1) if download_m else None

    abstract_m = re.search(r'<xmp class="abstract-article">(.*?)</xmp>', html, re.DOTALL)
    abstract = abstract_m.group(1).strip() if abstract_m else None

    output = f"Garuda Document #{garuda_id}:\n"
    output += f"Title: {title}\n"
    if journal:
        output += f"Journal: {journal}\n"
    if volume_issue:
        output += f"Volume/Issue: {volume_issue}\n"
    if authors:
        output += f"Authors: {', '.join(authors)}\n"
    if pub_date:
        output += f"Published: {pub_date}\n"
    if doi:
        output += f"DOI: {doi}\n"
    if original_url:
        output += f"Original Source: {original_url}\n"
    if download_url:
        output += f"Download: {download_url}\n"
    if abstract:
        output += f"\nAbstract:\n{abstract[:2000]}"
        if len(abstract) > 2000:
            output += "...[truncated]"
    return output


def garuda_read_pdf(url):
    """Download and extract text + references from a PDF (Garuda original/download link)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Error: pypdf not installed. Run: pip install pypdf"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        pdf_bytes = resp.read()

    if not pdf_bytes or len(pdf_bytes) < 100:
        return f"Error: could not download PDF from {url}"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        pages = reader.pages
        full_text = []
        for page in pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

        text = "\n".join(full_text)
        if not text.strip():
            return f"PDF downloaded ({len(pdf_bytes)} bytes) but no extractable text found. It may be a scanned document."

        references = _extract_references(text)
        body = text
        if references:
            ref_start = text.lower().find(references.lower()[:80]) if len(references) >= 80 else text.lower().find(references.lower())
            if ref_start > 0:
                body = text[:ref_start].strip()

        output = f"PDF from: {url}\n"
        output += f"Pages: {len(pages)}\n"
        output += f"Total text: {len(text)} chars\n\n"

        output += "=== BODY TEXT ===\n"
        body = body[:3000] + "...[truncated]" if len(body) > 3000 else body
        output += body + "\n"

        if references:
            refs = references[:2000] + "...[truncated]" if len(references) > 2000 else references
            output += f"\n=== REFERENCES / DAFTAR PUSTAKA ===\n{refs}"

        return output
    finally:
        os.unlink(tmp_path)


def _extract_references(text):
    """Extract the references/bibliography section from a paper."""
    ref_headers = [
        r'(?i)^\s*DAFTAR\s+PUSTAKA\s*$',
        r'(?i)^\s*DAFTAR\s+RUJUKAN\s*$',
        r'(?i)^\s*REFERENSI\s*$',
        r'(?i)^\s*REFERENCES\s*$',
        r'(?i)^\s*BIBLIOGRAPHY\s*$',
        r'(?i)^\s*BIBLIOGRAFI\s*$',
        r'(?i)^\s*DAFTAR\s+ACUAN\s*$',
        r'(?i)^\s*RUJUKAN\s*$',
        r'(?i)^\s*LITERATURE\s+CITED\s*$',
        r'(?i)^\s*WORKS?\s+CITED\s*$',
    ]

    for pattern in ref_headers:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            after = text[m.end():]
            return m.group(0) + "\n" + after.strip()
    return ""

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
    {
        "type": "function",
        "function": {
            "name": "garuda_search",
            "description": "Search Indonesian academic publications on Garuda (Garba Rujukan Digital) — the national research database. Returns titles, authors, journals, abstracts, and DOIs. Use this for Indonesian-language research or finding publications from Indonesian institutions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords (supports Indonesian or English)"},
                    "page": {"type": "integer", "description": "Page number (default: 1)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "garuda_multi_search",
            "description": "Search Garuda with MULTIPLE keyword variations in PARALLEL for broader coverage. Break complex research questions into 3-5 keyword variations (synonyms, related angles, narrower/broader terms). Merges results, removes duplicates. MUCH more effective than a single search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keyword variations to search in parallel (e.g. ['renewable energy Indonesia', 'geothermal Dieng impact', 'energi terbarukan dataran tinggi'])",
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "garuda_detail",
            "description": "Fetch the full details of a Garuda publication by its document ID (garuda_id from garuda_search results).",
            "parameters": {
                "type": "object",
                "properties": {
                    "garuda_id": {"type": "string", "description": "The Garuda document ID from search results"},
                },
                "required": ["garuda_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "garuda_read_pdf",
            "description": "Download and read the full PDF of a publication from Garuda's original source or download link. Extracts body text and references (daftar pustaka). Use this after garuda_detail to get the complete paper content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The PDF URL from garuda_detail (original_source or download link)"},
                },
                "required": ["url"],
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
    "garuda_search": garuda_search,
    "garuda_multi_search": garuda_multi_search,
    "garuda_detail": garuda_detail,
    "garuda_read_pdf": garuda_read_pdf,
}


def execute_tool(name, arguments):
    """Execute a tool by name with given arguments. Returns the result string."""
    if name not in TOOL_HANDLERS:
        return f"Error: unknown tool '{name}'"
    try:
        return TOOL_HANDLERS[name](**arguments)
    except Exception as e:
        return f"Tool error: {e}"
