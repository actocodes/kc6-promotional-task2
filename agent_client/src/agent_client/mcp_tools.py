from __future__ import annotations

import urllib.parse

import httpx
from langchain_core.tools import tool

from .logger import client_log

MCP_SERVER_URL = "http://localhost:8000/mcp"

async def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    import uuid

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        init_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"sampling": {}},
                "clientInfo": {"name": "agent_client", "version": "0.1.0"},
            },
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        init_resp = await client.post(MCP_SERVER_URL, json=init_payload, headers=headers)
        session_id = init_resp.headers.get("mcp-session-id", "")

        if session_id:
            headers["mcp-session-id"] = session_id

        tool_resp = await client.post(MCP_SERVER_URL, json=payload, headers=headers)
        tool_resp.raise_for_status()

        raw = tool_resp.text
        result_text = _parse_mcp_response(raw)
        return result_text


def _parse_mcp_response(raw: str) -> str:
    import json

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
                result = data.get("result", {})
                contents = result.get("content", [])
                texts = [c.get("text", "") for c in contents if c.get("type") == "text"]
                if texts:
                    return "\n".join(texts)
            except json.JSONDecodeError:
                pass
    try:
        data = json.loads(raw)
        result = data.get("result", {})
        contents = result.get("content", [])
        texts = [c.get("text", "") for c in contents if c.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    return raw

def make_reflect_tool(mcp_client_tools: list):
    mcp_reflect = next((t for t in mcp_client_tools if t.name == "reflect"), None)

    if mcp_reflect is not None:
        @tool
        async def reflect_tool(
            draft_answer: str,
            original_query: str,
            constraints: str = "accurate, concise, well-structured",
        ) -> str:
            """
            Critique and correct a draft answer using the server-side Reflection tool.
            The server delegates LLM critique and correction back to this client via MCP Sampling.

            Args:
                draft_answer: The initial answer text to be improved.
                original_query: The user's original question.
                constraints: Comma-separated quality constraints.
            Returns:
                Critique + corrected answer string.
            """
            client_log.info("Calling MCP reflect tool via adapter")
            result = await mcp_reflect.ainvoke({
                "draft_answer": draft_answer,
                "original_query": original_query,
                "constraints": constraints,
            })
            client_log.info("reflect tool returned %d chars", len(str(result)))
            return str(result)

        return reflect_tool

    @tool
    async def reflect_tool(
        draft_answer: str,
        original_query: str,
        constraints: str = "accurate, concise, well-structured",
    ) -> str:
        """
        Critique and correct a draft answer using the server-side Reflection tool.
        Uses MCP Sampling to delegate LLM calls back to this client.
        """
        client_log.info("Calling MCP reflect tool via direct HTTP")
        result = await _call_mcp_tool("reflect", {
            "draft_answer": draft_answer,
            "original_query": original_query,
            "constraints": constraints,
        })
        client_log.info("reflect tool returned %d chars", len(result))
        return result

    return reflect_tool


def make_knowledge_tool(mcp_client_tools: list):
    @tool
    async def query_knowledge_tool(query: str) -> str:
        client_log.info("Querying CRAG knowledge resource | query=%r", query)

        try:
            encoded = urllib.parse.quote(query, safe="")
            import json, uuid, httpx as _httpx

            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "resources/read",
                "params": {"uri": f"knowledge://domain/docs/{encoded}"},
            }
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }

            init_payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"sampling": {}},
                    "clientInfo": {"name": "agent_client", "version": "0.1.0"},
                },
            }
            async with _httpx.AsyncClient(timeout=60) as http:
                init_resp = await http.post(MCP_SERVER_URL, json=init_payload, headers=headers)
                sid = init_resp.headers.get("mcp-session-id", "")
                if sid:
                    headers["mcp-session-id"] = sid
                resp = await http.post(MCP_SERVER_URL, json=payload, headers=headers)
                resp.raise_for_status()
                raw = resp.text

            result = _parse_mcp_response_resource(raw)
            client_log.info("CRAG resource returned %d chars", len(result))
            return result

        except Exception as exc:  # noqa: BLE001
            client_log.warning("CRAG resource call failed (%s); returning empty", exc)
            return f"Knowledge base temporarily unavailable: {exc}"

    return query_knowledge_tool


def _parse_mcp_response_resource(raw: str) -> str:
    """Parse resources/read response from SSE or JSON."""
    import json

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
                result = data.get("result", {})
                contents = result.get("contents", [])
                texts = [c.get("text", "") for c in contents if c.get("type") == "text" or "text" in c]
                if texts:
                    return "\n".join(texts)
            except json.JSONDecodeError:
                pass
    try:
        data = json.loads(raw)
        result = data.get("result", {})
        contents = result.get("contents", [])
        texts = [c.get("text", "") for c in contents]
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    return raw
