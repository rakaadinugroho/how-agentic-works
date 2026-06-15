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


SYSTEM_PROMPT = """You are an agentic research assistant. You have access to tools that let you search the web, read webpages, save and recall notes, and think internally.

Follow this process for research tasks:
1. Analyze the user's request and plan what you need to find out.
2. Use web_search to find relevant information.
3. Use read_webpage to get full content from promising results.
4. Use save_note to store key findings in working memory.
5. Use think to reflect on what you've learned and plan next steps.
6. When you have enough information, synthesize a comprehensive answer.

IMPORTANT RULES:
- Always use tools to gather information before answering research questions.
- Use think() to plan between steps — this helps you stay organized.
- Save important facts with save_note() so you don't lose them.
- After 3-5 tool calls, synthesize your findings into a final answer.
- Do not fabricate information — only use what you find through tools.
- If a tool fails, try an alternative approach.
- Be concise in your final answer but thorough.

You are a teaching demo — show how an agentic AI works by actually using the tools!"""
