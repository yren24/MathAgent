from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping


DEFAULT_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"


def request_structured_output(
    input_text: str,
    *,
    instructions: str,
    schema: Mapping[str, Any],
    schema_name: str,
    model: str,
    api_key: str,
    endpoint: str = DEFAULT_RESPONSES_ENDPOINT,
    timeout: float = 60.0,
    opener: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not input_text.strip():
        raise ValueError("input_text must not be empty")
    if not instructions.strip():
        raise ValueError("instructions must not be empty")
    if not schema_name.strip():
        raise ValueError("schema_name must not be empty")
    if not model.strip():
        raise ValueError("model must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    body = {
        "model": model,
        "store": False,
        "instructions": instructions,
        "input": input_text,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": dict(schema),
            }
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI Responses API failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"OpenAI Responses API connection failed: {exc.reason}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("OpenAI Responses API returned a non-object payload")
    if payload.get("error"):
        raise RuntimeError(f"OpenAI Responses API returned an error: {payload['error']}")
    if payload.get("status") in {"failed", "cancelled", "incomplete"}:
        raise RuntimeError(
            "OpenAI Responses API did not complete successfully: "
            f"{payload.get('status')}"
        )
    output_text = response_output_text(payload)
    parsed = json.loads(output_text)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI structured output must contain a JSON object")
    provenance = {
        "used": True,
        "provider": "openai",
        "endpoint": "responses",
        "model_requested": model,
        "model_returned": payload.get("model"),
        "response_id": payload.get("id"),
        "store": False,
        "used_for_numeric_decisions": False,
    }
    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        provenance["usage"] = {
            key: usage.get(key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if usage.get(key) is not None
        }
    return parsed, provenance, body


def response_output_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for item in payload.get("output", ()):
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content", ()):
            if not isinstance(content, Mapping):
                continue
            if content.get("type") == "refusal":
                raise ValueError(
                    f"OpenAI request was refused: {content.get('refusal')}"
                )
            text = content.get("text")
            if content.get("type") == "output_text" and isinstance(text, str):
                return text
    raise ValueError("OpenAI response did not contain structured output text")
