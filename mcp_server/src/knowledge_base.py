"""
knowledge_base.py
-----------------
Hierarchical enterprise knowledge base for the CRAG resource.

Structure:
  Level 0 – Domain summaries   (coarse-grain)
  Level 1 – Topic overviews    (mid-grain)
  Level 2 – Granular code/doc chunks (fine-grain)
"""

from __future__ import annotations

KNOWLEDGE_BASE: list[dict] = [
    # ── LEVEL 0 ── Domain summary ─────────────────────────────────────────────
    {
        "id": "L0-AI",
        "level": 0,
        "title": "Artificial Intelligence Overview",
        "text": (
            "Artificial Intelligence (AI) is the simulation of human intelligence in machines. "
            "Modern AI encompasses machine learning, deep learning, natural language processing, "
            "computer vision, and reinforcement learning. Large Language Models (LLMs) such as "
            "GPT-4 and Claude represent the frontier of generative AI, capable of reasoning, "
            "code generation, and multi-step problem solving."
        ),
        "children": ["L1-LLM", "L1-RAG"],
    },
    {
        "id": "L0-ARCH",
        "level": 0,
        "title": "Software Architecture Patterns",
        "text": (
            "Enterprise software architecture relies on proven patterns: microservices, event-driven "
            "systems, CQRS, domain-driven design, and the emerging Agent-as-a-Service paradigm. "
            "The Model Context Protocol (MCP) enables LLMs to interact with external tools and data "
            "sources through a standardised JSON-RPC transport layer."
        ),
        "children": ["L1-MCP", "L1-AGENTS"],
    },
    # ── LEVEL 1 ── Topic overviews ─────────────────────────────────────────────
    {
        "id": "L1-LLM",
        "level": 1,
        "title": "Large Language Models",
        "parent": "L0-AI",
        "text": (
            "LLMs are transformer-based neural networks trained on vast text corpora using "
            "self-supervised learning. Key capabilities include in-context learning, chain-of-thought "
            "reasoning, and tool use. Prominent models: OpenAI GPT-4o, Anthropic Claude 3.5, "
            "Google Gemini 2.0. Deployment typically uses APIs over HTTPS with streaming support."
        ),
        "children": ["L2-SAMPLING", "L2-PROMPTING"],
    },
    {
        "id": "L1-RAG",
        "level": 1,
        "title": "Retrieval-Augmented Generation",
        "parent": "L0-AI",
        "text": (
            "RAG combines dense retrieval with generative models to ground responses in external "
            "knowledge. A retriever fetches relevant documents; the generator conditions its output "
            "on these documents. Variants include Corrective RAG (CRAG), which validates retrieved "
            "content before generation, and hierarchical indexing for multi-level document search."
        ),
        "children": ["L2-CRAG", "L2-VECTORDB"],
    },
    {
        "id": "L1-MCP",
        "level": 1,
        "title": "Model Context Protocol (MCP)",
        "parent": "L0-ARCH",
        "text": (
            "MCP is an open standard for connecting LLMs to tools and data sources. The protocol "
            "defines three primitives: Tools (executable functions), Resources (data sources), and "
            "Prompts (reusable templates). Transport options include stdio, SSE, and Streamable HTTP. "
            "FastMCP is the leading Python framework for building MCP servers."
        ),
        "children": ["L2-FASTMCP", "L2-TRANSPORT"],
    },
    {
        "id": "L1-AGENTS",
        "level": 1,
        "title": "AI Agent Architecture",
        "parent": "L0-ARCH",
        "text": (
            "AI agents perceive, reason, and act autonomously. A ReAct agent interleaves reasoning "
            "traces with tool actions. LangChain's create_agent factory builds agents that accept "
            "a list of LangChain-compatible tools. LangGraph adds stateful, multi-step graph "
            "execution on top of LangChain."
        ),
        "children": ["L2-LANGCHAIN", "L2-LANGGRAPH"],
    },
    # ── LEVEL 2 ── Granular code/doc chunks ────────────────────────────────────
    {
        "id": "L2-SAMPLING",
        "level": 2,
        "title": "MCP Sampling Mechanism",
        "parent": "L1-LLM",
        "text": (
            "MCP Sampling allows a server to request LLM completions from the connected client. "
            "The server calls ctx.sample() or ctx.request_sampling(); the client executes the "
            "LLM call and returns the result. This keeps API keys and model selection on the client "
            "side, making the server stateless with respect to model credentials. "
            "Sample payload structure:\n"
            "  messages: [{role: 'user', content: [{type: 'text', text: '...'}]}]\n"
            "  maxTokens: int\n"
            "  systemPrompt: str (optional)"
        ),
    },
    {
        "id": "L2-PROMPTING",
        "level": 2,
        "title": "Advanced Prompting Techniques",
        "parent": "L1-LLM",
        "text": (
            "Chain-of-Thought (CoT): instruct the model to think step-by-step before answering.\n"
            "Tree-of-Thought (ToT): branch multiple reasoning paths and evaluate each.\n"
            "ReAct: alternate Thought → Action → Observation cycles.\n"
            "Self-Critique / Reflection: ask the model to review its own output and correct errors.\n"
            "Few-shot prompting: include worked examples in the context window."
        ),
    },
    {
        "id": "L2-CRAG",
        "level": 2,
        "title": "Corrective RAG Implementation",
        "parent": "L1-RAG",
        "text": (
            "CRAG pipeline:\n"
            "1. Multi-query expansion – generate N semantic variations of the user query.\n"
            "2. Hierarchical retrieval – search level-0 summaries, then drill into level-1/2 chunks.\n"
            "3. ToT evaluation – score each retrieved chunk on relevance (0-10); prune scores < 5.\n"
            "4. Fallback – if all scores < 5, call external search (Tavily) to augment context.\n"
            "5. Synthesis – concatenate surviving chunks and return as the resource payload.\n"
            "Python skeleton:\n"
            "  expanded = expand_queries(query, n=3)\n"
            "  chunks = hierarchical_search(expanded)\n"
            "  scored = tot_evaluate(chunks)\n"
            "  if max(scored) < 5: chunks += tavily_search(query)\n"
            "  return synthesize(chunks)"
        ),
    },
    {
        "id": "L2-VECTORDB",
        "level": 2,
        "title": "Vector Database Patterns",
        "parent": "L1-RAG",
        "text": (
            "Popular vector stores: Chroma, Pinecone, Weaviate, pgvector.\n"
            "Embedding models: text-embedding-3-small (OpenAI), all-MiniLM-L6-v2 (sentence-transformers).\n"
            "Similarity metrics: cosine, dot-product, euclidean.\n"
            "Hybrid search combines dense vectors with BM25 sparse retrieval for better coverage.\n"
            "For this project we use BM25 (rank-bm25) plus keyword overlap scoring as a lightweight "
            "alternative that requires no external service."
        ),
    },
    {
        "id": "L2-FASTMCP",
        "level": 2,
        "title": "FastMCP Server Code Patterns",
        "parent": "L1-MCP",
        "text": (
            "```python\n"
            "from fastmcp import FastMCP, Context\n\n"
            "mcp = FastMCP('MyServer')\n\n"
            "@mcp.tool\n"
            "async def my_tool(query: str, ctx: Context) -> str:\n"
            "    result = await ctx.sample(f'Answer: {query}')\n"
            "    return result.text\n\n"
            "@mcp.resource('knowledge://domain/docs')\n"
            "async def domain_docs(ctx: Context) -> str:\n"
            "    return 'domain knowledge here'\n\n"
            "if __name__ == '__main__':\n"
            "    mcp.run(transport='http', host='0.0.0.0', port=8000)\n"
            "```\n"
            "The Context object provides: ctx.sample(), ctx.log(), ctx.report_progress()."
        ),
    },
    {
        "id": "L2-TRANSPORT",
        "level": 2,
        "title": "MCP Transport Configuration",
        "parent": "L1-MCP",
        "text": (
            "Streamable HTTP transport:\n"
            "  - Server: mcp.run(transport='http', host='0.0.0.0', port=8000)\n"
            "  - Client URL: http://localhost:8000/mcp\n"
            "  - Supports SSE streaming and resumable connections via EventStore.\n"
            "stdio transport:\n"
            "  - Server: mcp.run(transport='stdio')\n"
            "  - Client: subprocess communication via stdin/stdout.\n"
            "MultiServerMCPClient (LangChain adapter):\n"
            "  {'server': {'transport': 'http', 'url': 'http://localhost:8000/mcp'}}"
        ),
    },
    {
        "id": "L2-LANGCHAIN",
        "level": 2,
        "title": "LangChain Agent Creation",
        "parent": "L1-AGENTS",
        "text": (
            "```python\n"
            "from langchain.agents import create_agent\n"
            "from langchain_anthropic import ChatAnthropic\n\n"
            "llm = ChatAnthropic(model='claude-sonnet-4-5')\n"
            "tools = [tool_a, tool_b]  # LangChain Tool objects\n"
            "agent = create_agent('claude-sonnet-4-5', tools)\n"
            "response = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Hello'}]})\n"
            "```\n"
            "LangChain @tool decorator:\n"
            "from langchain_core.tools import tool\n"
            "@tool\n"
            "def my_tool(query: str) -> str:\n"
            "    '''Tool docstring becomes description.'''\n"
            "    return query.upper()"
        ),
    },
    {
        "id": "L2-LANGGRAPH",
        "level": 2,
        "title": "LangGraph State Machines",
        "parent": "L1-AGENTS",
        "text": (
            "LangGraph builds stateful agent graphs:\n"
            "  - Nodes: Python functions that read/write state.\n"
            "  - Edges: conditional routing between nodes.\n"
            "  - State: TypedDict with Annotated fields using add_messages reducer.\n"
            "  - create_react_agent(llm, tools) returns a pre-built ReAct graph.\n"
            "Persistence: use MemorySaver() or SqliteSaver() checkpointers for multi-turn memory.\n"
            "Streaming: graph.astream_events() yields token-level and node-level events."
        ),
    },
]

# ── Index helpers ──────────────────────────────────────────────────────────────

_by_id: dict[str, dict] = {doc["id"]: doc for doc in KNOWLEDGE_BASE}


def get_doc(doc_id: str) -> dict | None:
    return _by_id.get(doc_id)


def get_children(doc_id: str) -> list[dict]:
    parent = _by_id.get(doc_id)
    if not parent or "children" not in parent:
        return []
    return [_by_id[cid] for cid in parent["children"] if cid in _by_id]


def get_all_docs() -> list[dict]:
    return KNOWLEDGE_BASE


def get_by_level(level: int) -> list[dict]:
    return [d for d in KNOWLEDGE_BASE if d["level"] == level]
