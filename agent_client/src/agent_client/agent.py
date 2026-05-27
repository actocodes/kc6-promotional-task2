from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import LoggingMessageNotificationParams

from .logger import client_log, ingest_server_log, log_separator
from .mcp_tools import make_knowledge_tool, make_reflect_tool
from .sampling_handler import handle_sampling

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

def _pick_model() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-5"
    if os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    
    client_log.info("No cloud API keys detected. Falling back to local Ollama engine.")
    return "ollama"

def _on_server_log(
    params: LoggingMessageNotificationParams,
    server_name: str,
) -> None:
    level = params.level if isinstance(params.level, str) else str(params.level)
    data  = params.data if isinstance(params.data, str) else str(params.data)
    ingest_server_log(level, f"[{server_name}] {data}")

async def run_agent(user_query: str) -> str:
    client_log.info("Initializing agent run for query: %r", user_query)

    try:
        model_name = _pick_model()
        client_log.info("Selected model: %s", model_name)
    except RuntimeError as err:
        client_log.critical("Model selection failed: %s", err)
        raise

    client_log.info("Connecting to MCP server at %s", MCP_SERVER_URL)

    mcp_client = MultiServerMCPClient(
        connections={
            "thinking_agent_server": {
                "transport": "http",
                "url": MCP_SERVER_URL,
            }
        },
    )
    client_log.info("MCP Client instance initialized directly.")

    mcp_tools = await mcp_client.get_tools()
    client_log.info("Successfully retrieved %d tools from MCP server", len(mcp_tools))

    knowledge_tool = make_knowledge_tool(mcp_tools)
    reflect_tool = make_reflect_tool(mcp_tools)
    
    base_tools = list(mcp_tools)
    all_tools = [knowledge_tool]
    
    if reflect_tool not in base_tools:
        all_tools.append(reflect_tool)
    all_tools.extend(base_tools)

    client_log.info("Total active tools available to agent: %d", len(all_tools))

    if model_name.startswith("openai:"):
        from langchain_openai import ChatOpenAI
        real_model = model_name.split(":")[1]
        llm = ChatOpenAI(model=real_model, temperature=0.2)
    elif model_name == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model="llama3.1",
            temperature=0.2,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model_name, temperature=0.2)

    from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an experienced software engineering assistant. Use the available tools to verify documentation and critique your answers for high accuracy."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools, verbose=False)

    client_log.info("Executing agent workflow invoke pipeline")
    response = await executor.ainvoke({"input": user_query})
    
    output = response.get("output", "")
    client_log.info("Agent execution finished successfully (%d chars)", len(output))
    return str(output)

async def interactive_session() -> None:
    log_separator("INTERACTIVE SESSION START")
    client_log.info("\nThinking Agent System Initialization Successful.")
    client_log.info("Type your question below. Enter 'quit' or 'exit' to terminate.\n")

    while True:
        try:
            query = await asyncio.to_thread(input, "\nYou: ")
            query = query.strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("\nThinking...")
        try:
            answer = await run_agent(query)
            print(f"\nAgent > {answer}\n")
        except Exception as exc:
            client_log.error("Agent error: %s", exc, exc_info=True)
            print(f"\nError: {exc}\n")

    log_separator("INTERACTIVE SESSION END")

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
        except Exception as exc:
            client_log.error("Demo query %d failed: %s", i, exc, exc_info=True)

    log_separator("DEMO RUN END")

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"
    if mode == "demo":
        asyncio.run(demo_run())
    else:
        asyncio.run(interactive_session())


if __name__ == "__main__":
    main()