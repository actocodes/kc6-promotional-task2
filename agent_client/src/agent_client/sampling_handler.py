from __future__ import annotations

import os

from fastmcp.client.sampling import RequestContext, SamplingMessage, SamplingParams

from .logger import client_log, ingest_server_log


async def handle_sampling(
    messages: list[SamplingMessage],
    params: SamplingParams,
    context: RequestContext,
) -> str:
    client_log.info(
        "Sampling request received from server | messages=%d maxTokens=%s",
        len(messages),
        params.maxTokens,
    )
    
    system_prompt = params.systemPrompt or "You are a helpful AI assistant."
    conversation: list[dict] = []
    for msg in messages:
        content = (
            msg.content.text
            if hasattr(msg.content, "text")
            else str(msg.content)
        )
        conversation.append({"role": msg.role, "content": content})

    model_used = "unknown"
    try:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            result = await _call_anthropic(
                conversation, system_prompt, params, anthropic_key
            )
            model_used = "claude-sonnet-4-5"
        else:
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if not openai_key:
                raise RuntimeError(
                    "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set."
                )
            result = await _call_openai(
                conversation, system_prompt, params, openai_key
            )
            model_used = "gpt-4o-mini"

    except Exception as exc:  # noqa: BLE001
        client_log.error("Sampling LLM call failed: %s", exc)
        result = f"[Sampling error: {exc}]"

    client_log.info(
        "Sampling complete | model=%s result_length=%d chars",
        model_used, len(result),
    )
    return result

async def _call_anthropic(
    conversation: list[dict],
    system: str,
    params: SamplingParams,
    api_key: str,
) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=params.maxTokens or 1024,
        system=system,
        messages=conversation,
        temperature=params.temperature or 0.4,
    )
    return response.content[0].text


async def _call_openai(
    conversation: list[dict],
    system: str,
    params: SamplingParams,
    api_key: str,
) -> str:
    import openai

    client = openai.AsyncOpenAI(api_key=api_key)
    messages = [{"role": "system", "content": system}] + conversation
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=params.maxTokens or 1024,
        messages=messages,
        temperature=params.temperature or 0.4,
    )
    return response.choices[0].message.content or ""
