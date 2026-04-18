from __future__ import annotations

"""
Shared LLM utilities: model capability detection, retry wrapper, API call helpers.

TODO (Option A) — Responses API migration:
  Newer OpenAI models (gpt-5+) support reasoning_effort + function tools
  simultaneously on the /v1/responses endpoint but NOT on /v1/chat/completions.
  When migrating Garak back to function-calling, switch to:

      client.responses.create(
          model=...,
          input=messages,          # same format as chat messages
          tools=tool_schemas,      # same OpenAI function schema format
          reasoning={"effort": effort},  # top-level, not extra_body
      )

  The response object differs: choices[0].message → output[0].content,
  tool calls surface under output[i].type == "function_call" rather than
  message.tool_calls.  Stream handling also differs (output deltas vs
  choice deltas).  The openai Python SDK >= 1.x exposes this as
  client.responses (AsyncResponses).  Check the API reference at
  https://platform.openai.com/docs/api-reference/responses for the
  current schema before implementing.
"""

import asyncio
import logging
import random

logger = logging.getLogger(__name__)

# Models that support the reasoning_effort parameter (top-level, chat completions)
_REASONING_EFFORT_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def supports_reasoning_effort(model: str) -> bool:
    """Return True if the model supports the reasoning_effort parameter."""
    m = model.lower()
    return any(m.startswith(p) for p in _REASONING_EFFORT_PREFIXES)


def reasoning_extra_body(model: str, effort: str) -> dict:
    """
    Return the extra_body dict for reasoning_effort, or {} if not supported.

    The chat completions API accepts `reasoning_effort` as a top-level
    parameter (passed via extra_body) for o1/o3/o4/gpt-5-series models.

    Args:
        model:  model name string (e.g. "gpt-4o", "gpt-5.4", "o3")
        effort: "none" | "low" | "medium" | "high"
    """
    if effort == "none" or not supports_reasoning_effort(model):
        return {}
    return {"reasoning_effort": effort}


async def chat_completion_with_retry(
    client,
    *,
    max_retries: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs,
):
    """
    Call client.chat.completions.create(**kwargs) with exponential backoff.

    Retries on:
      - openai.RateLimitError          (429) — back off and retry
      - openai.APIStatusError 5xx      — transient server error, retry
      - openai.APIConnectionError      — network blip, retry

    Raises immediately on:
      - 4xx errors other than 429 (bad request, auth failure, etc.)
      - Exhausted all retries

    Backoff formula: min(base * 2^attempt, max_delay) + jitter(0..1s)
    """
    import openai

    _RETRYABLE = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(**kwargs)
        except _RETRYABLE as exc:
            if attempt == max_retries:
                logger.error(
                    "LLM call failed after %d retries: %s", max_retries, exc
                )
                raise
            delay = min(base_delay * (2 ** attempt), max_delay) + random.random()
            # Surface the retry-after header when present (rate limit response)
            retry_after = getattr(exc, "response", None)
            if retry_after is not None:
                try:
                    ra = float(retry_after.headers.get("retry-after", delay))
                    delay = max(delay, ra)
                except Exception:
                    pass
            logger.warning(
                "LLM call hit %s (attempt %d/%d) — retrying in %.1fs",
                type(exc).__name__, attempt + 1, max_retries, delay,
            )
            await asyncio.sleep(delay)
        except openai.APIStatusError as exc:
            # Retry 5xx, raise immediately on 4xx (except 429 caught above)
            if exc.status_code >= 500:
                if attempt == max_retries:
                    raise
                delay = min(base_delay * (2 ** attempt), max_delay) + random.random()
                logger.warning(
                    "LLM call got HTTP %d (attempt %d/%d) — retrying in %.1fs",
                    exc.status_code, attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
            else:
                raise
