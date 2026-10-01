from __future__ import annotations

import json

import pytest

from mint_scout.openai_responses import request_structured_output


class _FakeResponse:
    def __init__(self, payload: object):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_structured_response_keeps_api_key_out_of_body_and_provenance():
    captured: dict = {}

    def opener(request, *, timeout):
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "id": "resp_1",
                "model": "test-model-2026",
                "status": "completed",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "total_tokens": 16,
                },
                "output_text": json.dumps({"value": "ok"}),
            }
        )

    parsed, provenance, body = request_structured_output(
        "input",
        instructions="Return the requested value.",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
        },
        schema_name="test_schema",
        model="test-model",
        api_key="top-secret",
        opener=opener,
    )

    assert parsed == {"value": "ok"}
    assert captured["authorization"] == "Bearer top-secret"
    assert "top-secret" not in json.dumps(body)
    assert "top-secret" not in json.dumps(provenance)
    assert body["store"] is False
    assert body["text"]["format"]["strict"] is True
    assert provenance["usage"]["total_tokens"] == 16


def test_structured_response_rejects_incomplete_api_result():
    def opener(_request, *, timeout):
        assert timeout == 60.0
        return _FakeResponse({"status": "incomplete"})

    with pytest.raises(RuntimeError, match="did not complete"):
        request_structured_output(
            "input",
            instructions="Return JSON.",
            schema={"type": "object"},
            schema_name="test_schema",
            model="test-model",
            api_key="secret",
            opener=opener,
        )
