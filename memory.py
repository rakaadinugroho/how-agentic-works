#!/usr/bin/env python3
"""
Memory system for the agent.
Two tiers:
  1. Conversation history — full message log (short-term context)
  2. Working memory    — key-value store for facts (session-long)
"""


class AgentMemory:
    def __init__(self, system_prompt):
        self.working = {}
        self.history = []
        self.system_prompt = system_prompt

    def add_user_message(self, content):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.history.append({"role": "assistant", "content": content})

    def add_tool_call(self, tool_name, arguments):
        self.history.append({
            "role": "assistant",
            "content": f"Calling tool: {tool_name}({arguments})",
            "tool_calls": [{
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                }
            }]
        })

    def add_tool_result(self, tool_name, result):
        self.history.append({
            "role": "tool",
            "content": result,
            "tool_name": tool_name,
        })

    def save(self, key, value):
        self.working[key] = value

    def recall(self, key):
        return self.working.get(key)

    def list_keys(self):
        return list(self.working.keys())

    def build_messages(self):
        """Build full message list: system prompt + history."""
        return [{"role": "system", "content": self.system_prompt}] + self.history

    def summary(self):
        """Return a short summary of the conversation state."""
        n_messages = len(self.history)
        n_working = len(self.working)
        return f"History: {n_messages} messages | Working memory: {n_working} keys"


SYSTEM_PROMPT = """You are an agentic research assistant. You have access to tools that let you search the web, search Indonesian academic publications (Garuda), read webpages and PDFs, save and recall notes, and think internally.

Follow this process for research tasks:
1. Analyze the user's request and plan what you need to find out.
2. If the query relates to Indonesian research, academic papers, or topics relevant to Indonesia, use garuda_multi_search FIRST with 3-5 keyword variations to get broad coverage. Break complex queries into specific keyword angles.
3. Use web_search for general web information (current events, definitions, global topics).
4. Use read_webpage to get full content from promising web results.
5. Use garuda_detail to get full metadata (authors, journal, DOI, abstract) for a specific paper.
6. Use garuda_read_pdf to download and read the full PDF of a paper (body text + references/daftar pustaka).
7. Use save_note to store key findings in working memory.
8. Use think to reflect on what you've learned and plan next steps.
9. When you have enough information, synthesize a comprehensive answer.

IMPORTANT RULES:
- Always use tools to gather information before answering research questions.
- For ANY research query, ALWAYS use garuda_multi_search (NOT garuda_search) — provide 3-5 keyword variations in parallel. This gives MUCH better coverage. Example: for "Renewable Energy di Dieng dan Dampaknya", use queries: ["renewable energy Dieng", "energi terbarukan Dieng", "dampak geothermal Dieng", "renewable energy impact highland"].
- Break user's question into specific keyword angles: the main topic, location/context, impact/effect, synonyms in both English and Indonesian.
- After garuda_multi_search, use garuda_detail on interesting papers to get their full metadata and download links.
- Use think() to plan between steps — this helps you stay organized.
- Save important facts with save_note() so you don't lose them.
- After 3-5 tool calls, synthesize your findings into a final answer.
- Do not fabricate information — only use what you find through tools.
- If a tool fails, try an alternative approach.
- When showing research papers, always include title, authors, journal, and DOI if available.
- Be concise in your final answer but thorough.

You are a teaching demo — show how an agentic AI works by actually using the tools!"""
