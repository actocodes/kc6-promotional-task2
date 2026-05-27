"""
agent.py
--------
AI Agent (MCP Client) — LangChain create_agent + MultiServerMCPClient

Responsibilities:
  1. Connect to the FastMCP server via streamable-http transport.
  2. Fetch remote tools; wrap them as LangChain @tool objects.
  3. Register a sampling handler so the server's Reflection tool can
     delegate LLM calls back to THIS process.
  4. Forward server log notifications into agent_system.log.
  5. Run an interactive multi-turn session (or batch demo).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any imports that need API keys
load_dotenv()

from langchain.agents import create_agent
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import LoggingMessageNotificationParams

from .logger import client_log, ingest_server_log, log_separator
from .mcp_tools import make_knowledge_tool, make_reflect_tool
from .sampling_handler import handle_sampling

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# Choose model: prefer Anthropic, fall back to OpenAI
def _pick_model() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-5"
    if os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
    )


# ── Server log callback ────────────────────────────────────────────────────────

async def _on_server_log(
    params: LoggingMessageNotificationParams,
    context: CallbackContext,
) -> None:
    """Forward MCP server log notifications into agent_system.log."""
    level = params.level if isinstance(params.level, str) else str(params.level)
    data  = params.data if isinstance(params.data, str) else str(params.data)
    ingest_server_log(level, f"[{context.server_name}] {data}")


# ── Core agent runner ──────────────────────────────────────────────────────────

async def run_agent(user_query: str) -> str:
    """
    Run a single agent turn against the MCP server.

    1. Open a MultiServerMCPClient connection with callbacks and sampling handler.
    2. Fetch MCP tools; wrap as LangChain tools.
    3. create_agent and ainvoke.
    4. Return final response text.
    """
    client_log.info("Connecting to MCP server at %s", MCP_SERVER_URL)

    callbacks = Callbacks(on_logging_message=_on_server_log)

    # FastMCP's built-in Anthropic sampling handler wires ctx.sample() → our LLM
    from fastmcp.client.sampling.handlers.anthropic import AnthropicSamplingHandler
    from fastmcp import Client as FastMCPClient

    sampling_handler = None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if anthropic_key:
        sampling_handler = AnthropicSamplingHandler(default_model="claude-sonnet-4-5")
        client_log.info("Sampling handler: AnthropicSamplingHandler (claude-sonnet-4-5)")
    elif openai_key:
        from fastmcp.client.sampling.handlers.openai import OpenAISamplingHandler
        sampling_handler = OpenAISamplingHandler(default_model="gpt-4o-mini")
        client_log.info("Sampling handler: OpenAISamplingHandler (gpt-4o-mini)")
    else:
        client_log.warning("No sampling handler configured — Reflection tool will error")

    async with MultiServerMCPClient(
        connections={
            "thinking_agent_server": {
                "transport": "http",
                "url": MCP_SERVER_URL,
            }
        },
        callbacks=callbacks,
    ) as mcp_client:
        client_log.info("MCP client connected")

        # Fetch remote tools
        mcp_tools = await mcp_client.get_tools()
        client_log.info("Fetched %d tools from MCP server", len(mcp_tools))
        for t in mcp_tools:
            client_log.debug("  tool: %s", t.name)

        # Build wrapped LangChain tool list
        reflect  = make_reflect_tool(mcp_tools)
        knowledge = make_knowledge_tool(mcp_tools)
        agent_tools = [reflect, knowledge]

        # Also pass through any other MCP tools not explicitly wrapped
        for mt in mcp_tools:
            if mt.name not in ("reflect",):
                agent_tools.append(mt)

        model = _pick_model()
        client_log.info("Creating agent with model=%s tools=%d", model, len(agent_tools))

        agent = create_agent(model, agent_tools)

        client_log.info("Sending query to agent: %r", user_query[:80])
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]}
        )

        # Extract final text
        messages = response.get("messages", [])
        final = messages[-1].content if messages else str(response)
        if isinstance(final, list):
            final = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in final
            )

        client_log.info("Agent response received (%d chars)", len(str(final)))
        return str(final)


# ── Multi-turn interactive session ─────────────────────────────────────────────

async def interactive_session() -> None:
    log_separator("INTERACTIVE SESSION START")
    client_log.info("Thinking Agent Stage 2 — interactive mode")
    client_log.info("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            query = input("\n🤔 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        try:
            answer = await run_agent(query)
            print(f"\n🤖 Agent:\n{answer}\n")
        except Exception as exc:  # noqa: BLE001
            client_log.error("Agent error: %s", exc, exc_info=True)
            print(f"\n❌ Error: {exc}\n")

    log_separator("INTERACTIVE SESSION END")


# ── Demo batch run ─────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "What is Corrective RAG and how does hierarchical indexing improve retrieval?",
    "Explain how MCP Sampling works and why the server doesn't need an LLM API key.",
    "Give me a code example of creating a FastMCP server with a custom tool.",
    "What are the key differences between LangChain agents and LangGraph state machines?",
]


async def demo_run() -> None:
    log_separator("DEMO RUN START")
    client_log.info(
        "Running demo batch with %d queries", len(DEMO_QUERIES)
    )

    for i, query in enumerate(DEMO_QUERIES, 1):
        client_log.info("─── Demo query %d/%d ───", i, len(DEMO_QUERIES))
        try:
            answer = await run_agent(query)
            client_log.info("Demo query %d complete (%d chars)", i, len(answer))
            print(f"\n{'='*60}")
            print(f"Q{i}: {query}")
            print(f"{'─'*60}")
            print(answer[:800] + ("…" if len(answer) > 800 else ""))
        except Exception as exc:  # noqa: BLE001
            client_log.error("Demo query %d failed: %s", i, exc, exc_info=True)

    log_separator("DEMO RUN END")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"
    if mode == "demo":
        asyncio.run(demo_run())
    else:
        asyncio.run(interactive_session())


if __name__ == "__main__":
    main()
