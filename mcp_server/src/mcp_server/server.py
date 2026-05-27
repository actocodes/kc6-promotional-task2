from __future__ import annotations

import logging
import sys

from fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent

from .crag_engine import run_crag

def _make_logger() -> logging.Logger:
    fmt = "[%(asctime)s] [SERVER] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    log = logging.getLogger("mcp_server")
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    log.propagate = False
    return log


logger = _make_logger()

mcp = FastMCP(
    name="ThinkingAgentServer",
    instructions=(
        "This server exposes a Reflection/self-critique tool that uses MCP Sampling "
        "to delegate LLM calls back to the client, and a Hierarchical CRAG resource "
        "containing enterprise domain knowledge."
    ),
)

@mcp.tool
async def reflect(
    draft_answer: str,
    original_query: str,
    constraints: str,
    ctx: Context,
) -> str:
    logger.info("Reflection tool invoked | query=%r", original_query[:80])
    await ctx.log("info", f"Starting reflection for query: {original_query[:60]}...")

    logger.debug("Initiating Critic sampling pass")
    await ctx.log("debug", "Sampling: Critic pass initiated")

    critic_messages = [
        SamplingMessage(
            role="user",
            content=TextContent(
                type="text",
                text=(
                    f"You are a rigorous AI critic. Evaluate the following draft answer "
                    f"against the user's original query and the stated quality constraints.\n\n"
                    f"## Original Query\n{original_query}\n\n"
                    f"## Quality Constraints\n{constraints}\n\n"
                    f"## Draft Answer\n{draft_answer}\n\n"
                    f"Identify specific flaws, gaps, inaccuracies, or violations of the "
                    f"constraints. Be concise and structured. Format as a bullet list."
                ),
            ),
        )
    ]

    critic_result = await ctx.sample(
        messages=critic_messages,
        system_prompt=(
            "You are an expert AI critic. Your job is to find weaknesses in answers. "
            "Be precise, actionable, and constructive."
        ),
        max_tokens=512,
        temperature=0.3,
    )
    critique = critic_result.text or "No critique generated."
    logger.info("Critic pass complete | critique_length=%d chars", len(critique))
    await ctx.log("info", f"Critic pass complete ({len(critique)} chars)")

    logger.debug("Initiating Correction sampling pass")
    await ctx.log("debug", "Sampling: Correction pass initiated")

    correction_messages = [
        SamplingMessage(
            role="user",
            content=TextContent(
                type="text",
                text=(
                    f"You are an expert writer. Rewrite the draft answer below, "
                    f"addressing every point in the critique while satisfying all constraints.\n\n"
                    f"## Original Query\n{original_query}\n\n"
                    f"## Quality Constraints\n{constraints}\n\n"
                    f"## Draft Answer\n{draft_answer}\n\n"
                    f"## Critique\n{critique}\n\n"
                    f"Produce a corrected, improved answer. Do NOT include any preamble — "
                    f"output only the improved answer."
                ),
            ),
        )
    ]

    correction_result = await ctx.sample(
        messages=correction_messages,
        system_prompt=(
            "You are an expert technical writer. Produce clear, accurate, well-structured "
            "answers that fully address the user's question."
        ),
        max_tokens=1024,
        temperature=0.4,
    )
    corrected = correction_result.text or draft_answer
    logger.info(
        "Correction pass complete | corrected_length=%d chars", len(corrected)
    )
    await ctx.log("info", "Reflection complete, corrected answer produced")

    return (
        f"## Reflection Result\n\n"
        f"### Critique\n{critique}\n\n"
        f"### Corrected Answer\n{corrected}"
    )

@mcp.resource("knowledge://domain/docs")
async def domain_knowledge_resource(ctx: Context) -> str:
    logger.info("Resource 'knowledge://domain/docs' accessed (index request)")
    await ctx.log("info", "Knowledge base index resource accessed")

    lines = ["# Enterprise Knowledge Base. Domain Index\n"]
    lines.append("Use the `query_knowledge` tool to search specific topics.\n\n")
    lines.append("## Available Domains\n")

    from .knowledge_base import get_by_level
    for doc in get_by_level(0):
        lines.append(f"- **{doc['title']}** (`{doc['id']}`): {doc['text'][:120]}…\n")

    return "\n".join(lines)


@mcp.resource("knowledge://domain/docs/{query}")
async def domain_knowledge_search(query: str, ctx: Context) -> str:
    logger.info("CRAG resource called | query=%r", query)
    await ctx.log("info", f"CRAG resource: processing query '{query}'")
    await ctx.log("debug", "Initiating ToT Evaluation on Resource...")

    result = await run_crag(query)

    await ctx.log("info", "CRAG pipeline complete")
    return result

def main() -> None:
    logger.info("ThinkingAgentServer starting on http://0.0.0.0:8000/mcp")
    mcp.run(transport="http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
