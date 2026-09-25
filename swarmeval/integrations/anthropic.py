"""Optional Anthropic SDK integration.

* ``instrument_client(client)`` -- wraps ``client.messages.create`` so that any
  call made inside an active ``ctx.agent(...)`` span is recorded automatically
  with real token usage.
* ``anthropic_llm(client, model)`` -- returns ``fn(prompt) -> LLMResult`` for
  use with ``AgentSpan.llm``.

The ``anthropic`` package is only imported when a client is not supplied.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ..recorder import LLMResult, _current_span

DEFAULT_MODEL = "claude-opus-5"


def _text_of(response: Any) -> str:
    if getattr(response, "stop_reason", None) == "refusal":
        return ""
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _client(client: Any) -> Any:
    if client is not None:
        return client
    import anthropic  # local import: optional dependency
    return anthropic.Anthropic()


def anthropic_llm(client: Any = None, model: str = DEFAULT_MODEL, max_tokens: int = 16000,
                  system: Optional[str] = None, fallbacks: bool = True, **extra: Any) -> Callable[[str], LLMResult]:
    """Build a prompt -> LLMResult callable backed by the Messages API.

    With ``fallbacks=True`` (default) requests use the beta server-side
    fallback so a safety refusal is retried on a fallback model."""
    c = _client(client)

    def call(prompt: str) -> LLMResult:
        kwargs: dict = {"model": model, "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}], **extra}
        if system:
            kwargs["system"] = system
        if fallbacks:
            resp = c.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = c.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return LLMResult(_text_of(resp), int(getattr(usage, "input_tokens", 0) or 0),
                         int(getattr(usage, "output_tokens", 0) or 0), getattr(resp, "model", model), resp)

    return call


def instrument_client(client: Any) -> Any:
    """Monkey-patch ``client.messages.create`` to record calls made inside an
    active agent span.  Returns the same client."""
    original = client.messages.create

    def create(*args: Any, **kwargs: Any) -> Any:
        span = _current_span.get()
        if span is None:
            return original(*args, **kwargs)
        with span.llm_call(model=kwargs.get("model"), prompt=kwargs.get("messages")) as call:
            resp = original(*args, **kwargs)
            usage = getattr(resp, "usage", None)
            call.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            call.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            call.model = getattr(resp, "model", call.model)
            call.output = _text_of(resp)
            return resp

    client.messages.create = create
    return client
