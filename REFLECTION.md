# REFLECTION.md

## Stage 2 — Architectural Reflections

---

### 1. The Architecture Shift: Local LangChain Tool → Decoupled MCP Tool/Resource

**Operational advantages**

Moving from a monolithic LangChain tool (everything in one process) to a network-decoupled MCP Server/Client architecture delivers several concrete benefits:

- **Independent scalability.** The server exposes domain logic (CRAG retrieval, reflection) that any number of clients can consume simultaneously. In Stage 1, each agent instance carried its own copy of the tool. Now one server instance serves many.
- **Technology decoupling.** The server is a FastMCP Python service; the client is a LangChain agent. They can evolve independently, be deployed in separate containers, and be maintained by different teams. The only contract is the MCP protocol.
- **Security boundary.** LLM API keys live exclusively on the client. The server has zero model credentials; it delegates generation back to the client via MCP Sampling. This mirrors how enterprise microservices separate data-processing logic from credential management.
- **Reusability.** The `reflect` tool and the CRAG resource can be surfaced to any MCP-compatible host — Claude Desktop, Cursor, a custom chatbot — without changing the server code.

**Performance bottlenecks**

- **Network round-trip overhead.** Every tool call and resource read now crosses a network boundary (even if localhost). The Reflection tool alone requires two LLM sampling calls, each of which travels: agent → MCP server → sampling request → client LLM → response → server → agent. This adds 2–4× the latency of a purely local call.
- **Session initialisation cost.** The Streamable HTTP transport negotiates a session on every connection. For short-lived agents that create a fresh connection per query (as in our demo), this adds ~50–200 ms overhead. A persistent session context would eliminate this.
- **Serialisation.** All tool arguments and results must be JSON-serialisable and transmitted over HTTP. Large CRAG payloads (>10 KB) incur meaningful serialisation and transport cost.
- **CRAG pipeline latency.** Our hierarchical search + ToT evaluation is synchronous CPU-bound work inside the server's async event loop. Under high concurrency this would block other requests. Production should offload retrieval to a thread pool.

---

### 2. The Sampling Paradox: Server Requests LLM Completion from Client

MCP Sampling inverts the typical LLM call-stack: instead of the client orchestrating a model, the **server asks the client to run an LLM on its behalf**.

**Security implications**

- *Client retains full credential control.* The server never sees an API key. It sends a structured `SamplingMessage` payload and receives only text. This is excellent for multi-tenant scenarios: a shared MCP server can leverage each connected client's own model budget without any secret sharing.
- *Prompt injection risk shifts to the client.* Because the server constructs the prompt that the client's LLM will process, a compromised or malicious server could craft adversarial prompts. Clients should treat sampling requests as untrusted input and consider prompt sanitisation or sandboxing.
- *Auditability.* Since all sampling passes through the client, the client can log, rate-limit, or reject requests. The server cannot bypass this — a strong audit boundary by design.

**Structural implications**

- The server must not assume which model the client uses. Our `reflect` tool crafts prompts that are model-agnostic.
- `ctx.sample()` blocks the server-side tool coroutine until the client responds. Long client-side LLM calls (>30 s) can exhaust server request timeouts.
- Implementing the sampling handler requires the client to be an async process capable of running its own LLM calls — this is a non-trivial capability requirement that rules out thin REST clients.
- In practice, this is a **trust delegation** pattern: the server trusts the client's model to produce coherent critique and correction. If the client swaps to a weaker model, the Reflection tool silently degrades.

---

### 3. State & Context Management: Hierarchical CRAG Inside an MCP Resource

**How chunking altered data context flow**

In Stage 1, the retrieval tool returned a flat list of documents that the LangChain agent appended directly to its context window. In Stage 2:

1. The resource **pre-filters** with multi-query expansion and BM25 to produce a candidate set.
2. A Tree-of-Thought pass **prunes** candidates that score below a relevance threshold.
3. Only the surviving, high-signal chunks reach the client as the resource payload.

This means the agent's context window receives **curated, hierarchically-grounded text** rather than a raw retrieval dump. The immediate effect is a smaller, higher-quality context — which reduces LLM hallucination and token cost.

**Challenges introduced**

- *State is request-scoped, not persistent.* Each call to `knowledge://domain/docs/{query}` runs the full CRAG pipeline from scratch. There is no caching layer; a repeated query rebuilds the BM25 scores and ToT evaluations every time. Adding an in-memory cache keyed on query hash would cut repeated-query latency by ~80%.
- *Context truncation at the transport layer.* MCP resource payloads are returned as a single text string. If the CRAG output exceeds the client's context window, the agent will silently truncate. A streaming resource or chunked pagination would be the production-grade solution.
- *Query intent is lost at the resource boundary.* The agent passes a query string; the server cannot see the agent's conversation history or the previous tool results. This means the CRAG engine cannot perform conversational query clarification. A richer resource protocol (e.g., passing session history as a parameter) would enable follow-up refinement.
- *ToT scoring is heuristic, not LLM-based.* The three reasoning paths (literal keyword overlap, bigram semantic match, structural level preference) are deterministic proxies for true semantic relevance. In production, each ToT path should call `ctx.sample()` to get an LLM relevance score — this would make the CRAG evaluation far more accurate but would triple the resource latency.

---

*Generated as part of KodeCamp Thinking Agent Stage 2 submission.*
