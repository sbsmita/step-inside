"""Structured-output LLM calls on Amazon Bedrock.

LLM_PROVIDER=nova (default) uses Amazon Nova Pro via the Converse API, available on every Bedrock account.
LLM_PROVIDER=claude uses Claude via the Bedrock Messages API endpoint (needs the Anthropic Marketplace agreement)."""
import base64
import json
import logging
import os
from functools import lru_cache

import boto3
from anthropic import AnthropicBedrockMantle, APIError, BetaRefusalFallbackMiddleware
from botocore.exceptions import BotoCoreError, ClientError

log = logging.getLogger(__name__)

PROVIDER = os.environ.get("LLM_PROVIDER", "nova")
REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_MODEL = os.environ.get("NOVA_MODEL", "amazon.nova-pro-v1:0")
MODEL = os.environ.get("BEDROCK_MODEL", "anthropic.claude-opus-5-5")
FALLBACK_MODEL = os.environ.get("BEDROCK_FALLBACK_MODEL", "anthropic.claude-opus-4-8")
LABELS = {"nova": "Amazon Nova Pro on Bedrock", "claude": "Claude on Amazon Bedrock"}


class LLMUnavailable(Exception):
    pass


def enabled() -> bool:
    return os.environ.get("STEP_INSIDE_LLM", "bedrock") == "bedrock"


def label() -> str:
    return LABELS.get(PROVIDER, PROVIDER)


def structured(system: str, payload: dict, schema: dict, max_tokens: int = 4000, effort: str = "low",
               images: list[bytes] | None = None) -> dict:
    """One model call that must return JSON matching `schema`. `images` are JPEG bytes the model can look at."""
    if not enabled():
        raise LLMUnavailable("LLM disabled by STEP_INSIDE_LLM")
    images = images or []
    if PROVIDER == "claude":
        out = _claude(system, payload, schema, max_tokens, effort, images)
    else:
        out = _nova(system, payload, schema, max_tokens, images)
    missing = [k for k in schema.get("required", []) if k not in out]
    if missing:
        raise LLMUnavailable(f"response missing {missing}")
    return out


# ---------- Amazon Nova (Converse API) ----------
@lru_cache(maxsize=1)
def _runtime():
    return boto3.client("bedrock-runtime", region_name=REGION)


def _nova(system: str, payload: dict, schema: dict, max_tokens: int, images: list[bytes]) -> dict:
    """Structured output via a forced tool call: the tool's input schema is the response schema."""
    content = [{"image": {"format": "jpeg", "source": {"bytes": b}}} for b in images]
    content.append({"text": json.dumps(payload)})
    try:
        resp = _runtime().converse(
            modelId=NOVA_MODEL,
            system=[{"text": system}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": min(max_tokens, 5000)},
            toolConfig={
                "tools": [{"toolSpec": {"name": "respond", "description": "Return the answer in this exact structure.",
                                        "inputSchema": {"json": schema}}}],
                "toolChoice": {"tool": {"name": "respond"}},
            },
        )
    except (BotoCoreError, ClientError) as e:
        log.warning("nova call failed: %s", e)
        raise LLMUnavailable(str(e)) from e
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"]
    raise LLMUnavailable(f"no structured output (stopReason={resp.get('stopReason')})")


# ---------- Claude (Bedrock Messages API) ----------
@lru_cache(maxsize=1)
def _claude_client() -> AnthropicBedrockMantle:
    return AnthropicBedrockMantle(
        aws_region=REGION,
        middleware=[BetaRefusalFallbackMiddleware([{"model": FALLBACK_MODEL}])],
        timeout=20.0,
        max_retries=1,
    )


def _claude(system: str, payload: dict, schema: dict, max_tokens: int, effort: str, images: list[bytes]) -> dict:
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(b).decode()}} for b in images]
    content.append({"type": "text", "text": json.dumps(payload)})
    try:
        resp = _claude_client().beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
    except APIError as e:
        log.warning("claude call failed: %s", e)
        raise LLMUnavailable(str(e)) from e
    except Exception as e:  # missing AWS credentials surface as botocore errors, not APIError
        log.warning("claude unreachable: %s", e)
        raise LLMUnavailable(str(e)) from e
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise LLMUnavailable(f"stop_reason={resp.stop_reason}")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))
