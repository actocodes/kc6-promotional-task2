# Thinking Agent - Stage 2: Distributed MCP Architecture

## Overview
This project breaks the Stage 1 monolithic Thinking Agent into a production-grade, distributed system:

| Component | Technology | Role |
|-----------|------------|------|
| `mcp_server` | FastMCP + Streamable HTTP | Exposes `reflect` tool + CRAG resource |
| `agent_client` | LangChain `create_agent` + `MultiServerMCPClient` | Routes queries, handles MCP Sampling |

**Key architectural features:**
- **MCP Sampling:** the server delegates all LLM calls back to the client; no API keys on the server
- **Hierarchical CRAG:** multi-query expansion -> BM25 hierarchical search -> Tree-of-Thought evaluation -> Tavily fallback
- **Dual-stream logging:** `[CLIENT]` and `[SERVER]` log streams merged into `agent_system.log`

## Repository Structure
```
kc6-promotional-task2/
├── pyproject.toml
├── .env
├── agent_system.log
├── README.md
├── REFLECTION.md
│
├── mcp_server/
│   ├── pyproject.toml
│   └── src/mcp_server/
│       ├── __init__.py
│       ├── server.py
│       ├── crag_engine.py
│       └── knowledge_base.py
│
└── agent_client/
    ├── pyproject.toml
    └── src/agent_client/
        ├── __init__.py
        ├── agent.py
        ├── mcp_tools.py
        ├── sampling_handler.py
        └── logger.py
```

## Prerequisites
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/),** install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- At least one LLM API key or Ollama: `ANTHROPIC_API_KEY` **or** `OPENAI_API_KEY` 
- Optional: `TAVILY_API_KEY` for CRAG web fallback

## Quick Start
### 1. Clone and set up environment
```bash
git clone git@github.com:actocodes/kc6-promotional-task2.git
cd kc6-promotional-task2

# Copy and fill in your API keys
cp .env.example .env
# Edit .env - add ANTHROPIC_API_KEY and optionally TAVILY_API_KEY
```

### 2. Install all workspace dependencies
```bash
# Install both packages in one command
uv sync
```

### 3. Terminal A - Start the MCP Server
```bash
uv run --package mcp_server python -m mcp_server.server
```

Expected output:
```
[2026-05-16 10:14:17] [SERVER] [INFO] ThinkingAgentServer starting on http://0.0.0.0:8000/mcp
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Terminal B - Start the Agent Client
**Interactive mode** (default):
```bash
uv run --package agent_client python -m agent_client.agent
```

**Demo batch mode** (runs 4 pre-set queries, generates log):
```bash
uv run --package agent_client python -m agent_client.agent demo
```

## Configuration
All configuration is via environment variables in `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | No | Anthropic API key for LLM calls + MCP Sampling |
| `OPENAI_API_KEY` | No | OpenAI fallback for LLM calls + MCP Sampling |
| `TAVILY_API_KEY` | No | Enables web fallback in CRAG when internal docs score low |
| `MCP_SERVER_URL` | No | Override server URL (default: `http://localhost:8000/mcp`) |

## Architecture Deep-Dive
### MCP Server (`mcp_server`)
#### Reflection Tool - `@mcp.tool reflect()`
```
Agent invokes reflect() -> Server constructs critic prompt -> ctx.sample() -> MCP protocol -> Client sampling handler -> ctx.sample() -> LLM response (critique) -> ctx.sample() -> Client (correction pass) -> Return corrected answer -> Agent
```

The server **never holds an API key**. All LLM calls are delegated via MCP Sampling.

#### CRAG Resource - `@mcp.resource knowledge://domain/docs/{query}`
```
1. Multi-Query Expansion    query → [q1, q2, q3]  (deterministic heuristics)
2. Hierarchical BM25        L0 domains → top-2 → L1 topics → L2 chunks
3. Tree-of-Thought (ToT)    3 scoring paths per chunk → avg score
4. Threshold filter         keep chunks with avg ≥ 5.0/10
5. Tavily Fallback          if nothing passes → web search augmentation
6. Synthesis                format + return as resource text
```

Knowledge base hierarchy:
```
L0: Domain Summaries (AI, Software Architecture)
 └─ L1: Topic Overviews (LLMs, RAG, MCP, Agents)
     └─ L2: Granular Chunks (Sampling, CRAG, FastMCP, LangChain, ...)
```

### Agent Client (`agent_client`)
#### Connection Flow
```python
mcp_client = MultiServerMCPClient(
    connections={
        "thinking_agent_server": {
            "transport": "http",
            "url": MCP_SERVER_URL,
        }
    },
)
```

#### Sampling Handler
When the server's `reflect()` calls `ctx.sample()`, the MCP protocol routes the request to `FastMCP`'s `AnthropicSamplingHandler` registered on the client. The handler calls the Anthropic API and returns the text, the server never sees the API key.

### Dual-Stream Logging
```
agent_system.log
├── [CLIENT] [INFO]  — local agent orchestration events
├── [SERVER] [DEBUG] — forwarded MCP server log notifications
├── [CLIENT] [INFO]  — tool call initiated
├── [SERVER] [INFO]  — ToT evaluation scores
└── ...
```
