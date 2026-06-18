#!/usr/bin/env python3
"""
Agentic Research AI — ReAct Agent Loop.

This is the heart of the system. It implements the ReAct (Reason + Act) pattern:
  1. THINK: Send context to LLM, let it reason about what to do
  2. ACT: If LLM requests a tool, execute it
  3. OBSERVE: Feed tool result back to LLM
  4. LOOP: Repeat until the LLM decides to respond (or max steps reached)
"""
import json
import re
import textwrap
from ollama_client import chat, list_tools_schema
from tools import execute_tool
from memory import AgentMemory, SYSTEM_PROMPT
from rag import RAG


class AgenticResearch:
    def __init__(self, max_steps=8, verbose=True):
        self.memory = AgentMemory(SYSTEM_PROMPT)
        self.rag = RAG()
        self.max_steps = max_steps
        self.verbose = verbose
        self.tools = list_tools_schema()

    def _log(self, icon, text, color="default"):
        """Print a formatted step to the console."""
        if not self.verbose:
            return
        colors = {
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }
        c = colors.get(color, "")
        r = colors["reset"]
        print(f"\n{c}{icon} {text}{r}")

    def research(self, query):
        """
        Run the agentic research loop for a given query.
        Returns the final answer string.
        """
        self._log("[USER]", f'"{query}"', "bold")
        self.memory.add_user_message(query)

        for step in range(1, self.max_steps + 1):
            self._log(f"[STEP {step}/{self.max_steps}]", "LLM is thinking...", "blue")

            messages = self.memory.build_messages()
            response = chat(messages, tools=self.tools)

            if response.get("tool_calls"):
                for tc in response["tool_calls"]:
                    func = tc["function"]
                    tool_name = func["name"]
                    tool_args = func.get("arguments", {})

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    self._log("[TOOL]", f"{tool_name}({json.dumps(tool_args)})", "green")
                    result = execute_tool(tool_name, tool_args)

                    result_preview = result[:300] + "..." if len(result) > 300 else result
                    self._log("[RESULT]", result_preview, "yellow")

                    self.memory.add_tool_call(tool_name, tool_args)
                    self.memory.add_tool_result(tool_name, result)

                    if tool_name == "save_note" and "value" in tool_args:
                        self.rag.add(tool_args["value"])

                continue

            answer = response.get("content", "")
            self.memory.add_assistant_message(answer)
            self._log("[ANSWER]", "", "bold")
            print(textwrap.fill(answer, width=80))
            return answer

        self._log("[MAX STEPS]", "Reached limit, forcing final answer...", "red")
        messages = self.memory.build_messages()
        messages.append({
            "role": "system",
            "content": "You have reached the maximum number of steps. Synthesize a final answer now from everything you've learned. Do not call any more tools — just respond.",
        })
        response = chat(messages)
        answer = response.get("content", "Unable to complete research within step limit.")
        self.memory.add_assistant_message(answer)
        print(textwrap.fill(answer, width=80))
        return answer

    def research_stream(self, query):
        """Streaming version — yields dicts: {type, ...} for real-time UI updates."""
        yield {"type": "user", "content": query}
        self.memory.add_user_message(query)

        for step in range(1, self.max_steps + 1):
            yield {"type": "step", "step": step, "max": self.max_steps}
            messages = self.memory.build_messages()
            response = chat(messages, tools=self.tools)

            if response.get("tool_calls"):
                for tc in response["tool_calls"]:
                    func = tc["function"]
                    tool_name = func["name"]
                    tool_args = func.get("arguments", {})

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    yield {"type": "tool_call", "name": tool_name, "args": tool_args}
                    result = execute_tool(tool_name, tool_args)

                    papers = _parse_papers_from_result(tool_name, result)

                    yield {
                        "type": "tool_result",
                        "name": tool_name,
                        "result": result[:800] + ("..." if len(result) > 800 else ""),
                        "papers": papers,
                    }

                    self.memory.add_tool_call(tool_name, tool_args)
                    self.memory.add_tool_result(tool_name, result)

                    if tool_name == "save_note" and "value" in tool_args:
                        self.rag.add(tool_args["value"])

                continue

            answer = (response.get("content") or "").strip()
            if not answer:
                messages = self.memory.build_messages()
                messages.append({
                    "role": "system",
                    "content": "Your last response had no content. Please provide a substantive answer based on everything you've gathered.",
                })
                response = chat(messages)
                answer = (response.get("content") or "").strip()
            if not answer:
                answer = "Maaf, tidak dapat menghasilkan jawaban. Silakan coba pertanyaan yang lebih spesifik."
            self.memory.add_assistant_message(answer)
            yield {"type": "answer", "content": answer}
            return

        messages = self.memory.build_messages()
        messages.append({
            "role": "system",
            "content": "You have reached the maximum number of steps. Synthesize a final answer now from everything you've learned. Do not call any more tools — just respond.",
        })
        response = chat(messages)
        answer = (response.get("content") or "Unable to complete research within step limit.").strip()
        if not answer:
            answer = "Maaf, proses research tidak dapat diselesaikan. Silakan coba lagi."
        self.memory.add_assistant_message(answer)
        yield {"type": "answer", "content": answer}

    def rag_query(self, query, top_k=3):
        """Direct RAG query (also used inside the agent via query_kb tool)."""
        return self.rag.query(query, top_k)

    def get_memory_summary(self):
        return self.memory.summary()


def _parse_papers_from_result(tool_name, result):
    """Parse garuda_search/multi_search/detail results into structured paper objects."""
    if tool_name not in ("garuda_search", "garuda_multi_search", "garuda_detail"):
        return []
    papers = []
    pattern = r'(\d+)\.\s+(.+?)\n\s+Authors:\s+(.+?)\n(?:\s+Journal:\s+(.+?)\n)?(?:\s+DOI:\s+(.+?)\n)?\s+Garuda ID:\s+(\d+)'
    for m in re.finditer(pattern, result):
        papers.append({
            "index": m.group(1),
            "title": m.group(2).strip(),
            "authors": m.group(3).strip(),
            "journal": (m.group(4) or "").strip(),
            "doi": (m.group(5) or "").strip(),
            "garuda_id": m.group(6).strip(),
        })
    return papers
