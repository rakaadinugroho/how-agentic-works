#!/usr/bin/env python3
"""
Agentic Research AI — Interactive Demo.

This demo shows how an agentic AI works by walking through:
  1. LLM Reasoning (gemma4 via Ollama)
  2. Tool Use / Function Calling (web search, fetch, notes)
  3. Memory (conversation history + working memory)
  4. RAG (bge-m3 embeddings + vector search)

Usage:
    python3 demo.py
"""
import sys
import textwrap
from agent import AgenticResearch


def banner():
    print("\033[94m" + "=" * 60 + "\033[0m")
    print("\033[1m\033[94m  AGENTIC RESEARCH AI\033[0m")
    print("  Demonstrating: LLM · Tools · Memory · RAG")
    print("\033[94m" + "=" * 60 + "\033[0m")
    print()
    print("This agent can:")
    print("  1. Search the web for information")
    print("  2. Read full webpages")
    print("  3. Save findings to memory")
    print("  4. Embed and search knowledge via RAG (bge-m3)")
    print("  5. Think and plan between steps")
    print()
    print("Try a research query like:")
    print('  → "Explain what AI agents are and how they differ from traditional ML"')
    print('  → "Research the latest trends in agentic AI in 2025"')
    print('  → "What is RAG and why is it important for AI agents?"')
    print()
    print("Type 'quit' to exit, 'memory' to see working memory state.")
    print()


def main():
    banner()

    agent = AgenticResearch(max_steps=6, verbose=True)

    while True:
        try:
            query = input("\n\033[1mYou >\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if query.lower() == "memory":
            print(f"\n  {agent.get_memory_summary()}")
            if agent.memory.working:
                print("  Working memory keys:")
                for k, v in agent.memory.working.items():
                    print(f"    - {k}: {v[:80]}...")
            continue

        try:
            agent.research(query)
        except Exception as e:
            print(f"\n\033[91mError: {e}\033[0m")
            print("Make sure Ollama is running with: ollama serve")


if __name__ == "__main__":
    main()
