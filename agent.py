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

            # Build message context
            messages = self.memory.build_messages()

            # Call LLM with tools
            response = chat(messages, tools=self.tools)

            # Check for tool calls
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

                    # Execute the tool
                    result = execute_tool(tool_name, tool_args)

                    # Show result (truncated)
                    result_preview = result[:300] + "..." if len(result) > 300 else result
                    self._log("[RESULT]", result_preview, "yellow")

                    # Add to conversation history
                    self.memory.add_tool_call(tool_name, tool_args)
                    self.memory.add_tool_result(tool_name, result)

                    # If it's a save_note, also add to RAG
                    if tool_name == "save_note" and "value" in tool_args:
                        self.rag.add(tool_args["value"])

                # Continue loop — LLM will process tool results
                continue

            # No tool calls — this is the final answer
            answer = response.get("content", "")
            self.memory.add_assistant_message(answer)
            self._log("[ANSWER]", "", "bold")
            print(textwrap.fill(answer, width=80))
            return answer

        # Max steps reached
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

    def rag_query(self, query, top_k=3):
        """Direct RAG query (also used inside the agent via query_kb tool)."""
        return self.rag.query(query, top_k)

    def get_memory_summary(self):
        return self.memory.summary()
