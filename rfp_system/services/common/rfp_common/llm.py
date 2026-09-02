"""The inference gateway.

Callers name a capability. This maps it to a backend, retries, and fails over
to the other backend when the primary is unavailable. No service anywhere else
knows which model served a request.

Two backends that share no capacity, no AZ constraint, and no scaling behavior:
Bedrock (managed, no cold start) and a SageMaker Async endpoint running
Qwen3.8-27B at MinCapacity 0. Losing one makes the system slower, not stopped.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Literal

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import Config
from .contracts import Capability, InferenceRequest, InferenceResponse

log = logging.getLogger(__name__)

Backend = Literal["bedrock", "sagemaker_async"]

# Which backend serves which capability, and where it falls back.
#
# LONG_CONTEXT_SHRED and VISION_PARSE need the 262K window and the vision
# encoder, so they need Qwen3.8-27B, which Bedrock does not offer. Their
# fallback is degraded on purpose: a chunked shred on Bedrock loses the
# Section L to Section M cross-references but beats producing nothing.
ROUTING: dict[Capability, tuple[Backend, Backend | None]] = {
    Capability.DRAFT: ("bedrock", "sagemaker_async"),
    Capability.ANSWER: ("bedrock", "sagemaker_async"),
    Capability.CLASSIFY: ("bedrock", None),
    Capability.REWRITE_QUERY: ("bedrock", None),
    Capability.LONG_CONTEXT_SHRED: ("sagemaker_async", "bedrock"),
    Capability.VISION_PARSE: ("sagemaker_async", None),
}

_RETRYABLE = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "InternalServerException",
    "ModelTimeoutException",
}


class AllBackendsFailed(RuntimeError):
    pass


class InferenceGateway:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        boto = BotoConfig(
            region_name=cfg.region,
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=cfg.primary_backend_timeout_s,
        )
        self._bedrock = boto3.client("bedrock-runtime", config=boto)
        self._sagemaker = boto3.client("sagemaker-runtime", config=boto)
        self._s3 = boto3.client("s3", region_name=cfg.region)

    def invoke(self, req: InferenceRequest) -> InferenceResponse:
        primary, fallback = ROUTING[req.capability]

        try:
            return self._dispatch(primary, req, degraded=False)
        except (ClientError, TimeoutError) as exc:
            if fallback is None:
                raise AllBackendsFailed(
                    f"{req.capability} has no fallback and {primary} failed"
                ) from exc
            log.warning(
                "primary backend %s failed for %s, failing over to %s: %s",
                primary,
                req.capability,
                fallback,
                exc,
            )

        try:
            return self._dispatch(fallback, req, degraded=True)
        except (ClientError, TimeoutError) as exc:
            raise AllBackendsFailed(
                f"both {primary} and {fallback} failed for {req.capability}"
            ) from exc

    # -- backends ----------------------------------------------------------

    def _dispatch(
        self, backend: Backend, req: InferenceRequest, *, degraded: bool
    ) -> InferenceResponse:
        if backend == "bedrock":
            return self._bedrock_converse(req, degraded=degraded)
        return self._sagemaker_async(req, degraded=degraded)

    def _bedrock_converse(
        self, req: InferenceRequest, *, degraded: bool
    ) -> InferenceResponse:
        response = self._bedrock.converse(
            modelId=self._cfg.bedrock_model_id,
            system=[{"text": req.system}],
            messages=[{"role": "user", "content": [{"text": req.user}]}],
            inferenceConfig={
                "maxTokens": req.max_tokens,
                "temperature": req.temperature,
            },
        )
        usage = response["usage"]
        return InferenceResponse(
            text=response["output"]["message"]["content"][0]["text"],
            model_used=self._cfg.bedrock_model_id,
            backend="bedrock",
            input_tokens=usage["inputTokens"],
            output_tokens=usage["outputTokens"],
            degraded=degraded,
        )

    def _sagemaker_async(
        self, req: InferenceRequest, *, degraded: bool
    ) -> InferenceResponse:
        """Submit to the async endpoint and poll for the result.

        Async rather than real-time because the endpoint sits at MinCapacity 0
        and a scale-from-zero on a 30GB model takes 4 to 8 minutes. Real-time
        would time out; async holds the request while the instance comes up.
        """
        key = f"async-input/{uuid.uuid4()}.json"
        payload = {
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "image_s3_keys": list(req.image_s3_keys),
        }
        bucket = self._cfg.async_output_s3_uri.split("/")[2]
        self._s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode())

        submission = self._sagemaker.invoke_endpoint_async(
            EndpointName=self._cfg.sagemaker_endpoint,
            InputLocation=f"s3://{bucket}/{key}",
            ContentType="application/json",
        )
        return self._await_async_result(submission["OutputLocation"], degraded=degraded)

    def _await_async_result(
        self, output_location: str, *, degraded: bool, timeout_s: int = 900
    ) -> InferenceResponse:
        bucket, _, key = output_location.removeprefix("s3://").partition("/")
        deadline = time.monotonic() + timeout_s
        delay = 5.0

        while time.monotonic() < deadline:
            try:
                body = self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                    raise
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)
                continue

            result = json.loads(body)
            return InferenceResponse(
                text=result["text"],
                model_used=result.get("model", "qwen3.8-27b"),
                backend="sagemaker_async",
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                degraded=degraded,
            )

        raise TimeoutError(f"async inference did not complete within {timeout_s}s")
