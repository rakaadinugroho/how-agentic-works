#!/bin/bash
# =============================================================================
# Agentic AI — Message Composition Step-by-Step via Ollama API (curl)
# =============================================================================
# Demonstrates how messages accumulate across turns — the "memory" mechanism.
# Run: bash ollama-messages-demo.sh
# =============================================================================

OLLAMA="http://localhost:11434/api/chat"
MODEL="gemma4:latest"

echo "═══════════════════════════════════════════════════════════"
echo "  AGENTIC AI · MESSAGE COMPOSITION DEMO"
echo "  Model: $MODEL"
echo "═══════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────
# TURN 1 — Simple prompt (no memory yet)
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TURN 1 · First message — no history"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Request body:"
cat << 'EOF'
{
  "model": "gemma4:latest",
  "messages": [
    {
      "role": "system",
      "content": "You are a research assistant. Answer concisely."
    },
    {
      "role": "user",
      "content": "What is an LLM?"
    }
  ],
  "stream": false
}
EOF

echo ""
echo "---"
curl -s "$OLLAMA" -d '{
  "model": "'$MODEL'",
  "messages": [
    {"role": "system", "content": "You are a research assistant. Answer concisely."},
    {"role": "user", "content": "What is an LLM?"}
  ],
  "stream": false
}' | python3 -c "
import sys, json
r = json.load(sys.stdin)
print('Response:', r['message']['content'][:200])
print()
print('[ MEMORY: 2 messages stored (system + user) ]')
"

# ─────────────────────────────────────────────────────────────
# TURN 2 — With memory (previous messages included)
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TURN 2 · Memory in action — prev messages + new question"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Request body:"
cat << 'EOF'
{
  "model": "gemma4:latest",
  "messages": [
    { "role": "system",  "content": "You are a research assistant..." },
    { "role": "user",    "content": "What is an LLM?" },
    { "role": "assistant","content": "An LLM (Large Language Model) is..." },
    { "role": "user",    "content": "How is it different from traditional ML?" }
  ],
  "stream": false
}
EOF

echo ""
echo "---"
curl -s "$OLLAMA" -d '{
  "model": "'$MODEL'",
  "messages": [
    {"role": "system",   "content": "You are a research assistant. Answer concisely."},
    {"role": "user",     "content": "What is an LLM?"},
    {"role": "assistant","content": "An LLM (Large Language Model) is a neural network trained on massive text corpora to predict the next token, enabling it to generate coherent text, reason, and follow instructions."},
    {"role": "user",     "content": "How is it different from traditional ML?"}
  ],
  "stream": false
}' | python3 -c "
import sys, json
r = json.load(sys.stdin)
print('Response:', r['message']['content'][:200])
print()
print('[ MEMORY: 4 messages — full conversation history sent every turn ]')
"

# ─────────────────────────────────────────────────────────────
# TURN 3 — WITH TOOL CALLING
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TURN 3 · Adding tools — LLM decides to call web_search"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Request body:"
cat << 'EOF'
{
  "model": "gemma4:latest",
  "messages": [
    { "role": "system",  "content": "You have access to tools: web_search(query), think(note). Use them before answering research questions." },
    { "role": "user",    "content": "Who won the 2024 Indonesian general election?" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": { "type": "string", "description": "Search query" }
          },
          "required": ["query"]
        }
      }
    }
  ],
  "stream": false
}
EOF

echo ""
echo "---"
RESP=$(curl -s "$OLLAMA" -d '{
  "model": "'$MODEL'",
  "messages": [
    {"role": "system", "content": "You have access to tools: web_search(query), think(note). Use tools before answering research questions. Always search the web for current facts."},
    {"role": "user",   "content": "Who won the 2024 Indonesian general election?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "stream": false
}')
echo "$RESP" | python3 -c "
import sys, json
r = json.load(sys.stdin)
msg = r['message']
if 'tool_calls' in msg:
    for tc in msg['tool_calls']:
        f = tc['function']
        print(f'LLM called: {f[\"name\"]}({f[\"arguments\"]})')
else:
    print('Content:', msg.get('content', '')[:200])
print()
print('[ MEMORY: LLM output is stored — next turn adds tool result ]')
"

# ─────────────────────────────────────────────────────────────
# TURN 4 — Tool result fed back, full memory included
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TURN 4 · Tool result appended — LLM sees full history"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Request body (5 messages — system + user + tool_call + tool_result + context):"
cat << 'EOF'
{
  "model": "gemma4:latest",
  "messages": [
    { "role": "system",  "content": "You are a research agent with tools..." },
    { "role": "user",    "content": "Who won the 2024 Indonesian election?" },
    { "role": "assistant","content": "Calling web_search(...)",
      "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "2024 Indonesian election winner"}}}] },
    { "role": "tool",    "content": "Results: 1. Prabowo Subianto won the 2024 Indonesian presidential election with 58% of the vote. 2. He defeated Anies Baswedan and Ganjar Pranowo..." },
    { "role": "user",    "content": "Who was his running mate?" }
  ],
  "stream": false
}
EOF

echo ""
echo "---"
curl -s "$OLLAMA" -d '{
  "model": "'$MODEL'",
  "messages": [
    {"role": "system",   "content": "You are a research agent. Answer based on the tool results provided."},
    {"role": "user",     "content": "Who won the 2024 Indonesian presidential election?"},
    {"role": "assistant","content": "Calling web_search(2024 Indonesian election winner)",
     "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "2024 Indonesian election winner"}}}]},
    {"role": "tool",     "content": "Results: 1. Prabowo Subianto won the 2024 Indonesian presidential election with 58% of the vote, defeating Anies Baswedan and Ganjar Pranowo."},
    {"role": "user",     "content": "Who was his running mate?"}
  ],
  "stream": false
}' | python3 -c "
import sys, json
r = json.load(sys.stdin)
print('Response:', r['message']['content'][:250])
print()
print('[ MEMORY: 5 messages — every turn sees FULL history ]')
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  KEY INSIGHT"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  The 'memory' is simply the messages[] array."
echo "  Every turn sends the ENTIRE accumulated history:"
echo ""
echo "    Turn 1: [system, user]                        = 2 msgs"
echo "    Turn 2: [system, user, assistant, user]        = 4 msgs"
echo "    Turn 3: [system, user, assistant(tool_call)]   = 3 msgs  "
echo "    Turn 4: [system, user, asst, tool, user]        = 5 msgs"
echo ""
echo "  No external DB. No vector store. Just the array."
echo "═══════════════════════════════════════════════════════════"
