"""SageMaker adapter for vLLM.

SageMaker calls GET /ping for health and POST /invocations for work. vLLM
serves an OpenAI-compatible API on a different contract, so this starts vLLM as
a subprocess and translates.

Two things worth knowing about the settings this launches with:

  --kv-cache-dtype fp8 is required, not a tuning choice. A 262K-token shred at
  bf16 KV cache needs roughly 17GB, which does not fit alongside 30GB of FP8
  weights on a 48GB L40S. At fp8 the cache is about 8.5GB and it does.

  That 8.5GB is an estimate from inferred head counts. Measure it on the real
  model before committing to ml.g6e.xlarge. If it does not fit, drop
  MAX_MODEL_LEN to 180000 (still a 450-page solicitation) before reaching for
  a four-GPU instance.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("serve")

VLLM_PORT = 8000
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"
MODEL_ID = os.environ["MODEL_ID"]

app = FastAPI()
_client = httpx.Client(base_url=VLLM_URL, timeout=900.0)


def _launch_vllm() -> subprocess.Popen:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--port", str(VLLM_PORT),
        "--quantization", os.environ["QUANTIZATION"],
        "--kv-cache-dtype", os.environ["KV_CACHE_DTYPE"],
        "--max-model-len", os.environ["MAX_MODEL_LEN"],
        "--gpu-memory-utilization", os.environ["GPU_MEMORY_UTILIZATION"],
        "--served-model-name", "qwen3.8-27b",
        # Every request here is batch size 1: one shred, or one page to parse.
        # Concurrency would only compete for the KV cache the long context needs.
        "--max-num-seqs", "1",
    ]
    log.info("starting vllm: %s", " ".join(cmd))
    return subprocess.Popen(cmd)


@app.get("/ping")
def ping() -> JSONResponse:
    """SageMaker polls this during scale-from-zero.

    Returning 200 before the model is loaded would have SageMaker route a
    request into a container that cannot serve it, so this reports vLLM's real
    readiness and lets the 4-to-8-minute startup take as long as it takes."""
    try:
        response = _client.get("/health", timeout=2.0)
        if response.status_code == 200:
            return JSONResponse({"status": "ok"}, status_code=200)
    except httpx.HTTPError:
        pass
    return JSONResponse({"status": "loading"}, status_code=503)


@app.post("/invocations")
async def invocations(request: Request) -> dict:
    payload = await request.json()

    messages = payload["messages"]
    if payload.get("image_s3_keys"):
        messages = _attach_images(messages, payload["image_s3_keys"])

    try:
        response = _client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.8-27b",
                "messages": messages,
                "max_tokens": payload.get("max_tokens", 4096),
                "temperature": payload.get("temperature", 0.2),
            },
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        log.exception("vllm call failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    body = response.json()
    usage = body.get("usage", {})
    return {
        "text": body["choices"][0]["message"]["content"],
        "model": "qwen3.8-27b",
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


def _attach_images(messages: list[dict], s3_keys: list[str]) -> list[dict]:
    """Presign the page images so vLLM's vision path can fetch them.

    Scanned bid packages arrive as page images. Passing them here is what
    replaces a Textract plus OCR cleanup pipeline."""
    import boto3

    s3 = boto3.client("s3")
    parts: list[dict] = []
    for key in s3_keys:
        bucket, _, obj = key.removeprefix("s3://").partition("/")
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": obj}, ExpiresIn=3600
        )
        parts.append({"type": "image_url", "image_url": {"url": url}})

    out = list(messages)
    last = dict(out[-1])
    last["content"] = [{"type": "text", "text": last["content"]}, *parts]
    out[-1] = last
    return out


if __name__ == "__main__":
    import uvicorn

    proc = _launch_vllm()
    # Fail fast if vLLM dies during load rather than serving 503s until
    # SageMaker's startup timeout expires with no useful log line.
    for _ in range(60):
        if proc.poll() is not None:
            sys.exit(f"vllm exited during startup with code {proc.returncode}")
        time.sleep(5)
        try:
            if _client.get("/health", timeout=2.0).status_code == 200:
                log.info("vllm ready")
                break
        except httpx.HTTPError:
            continue

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]), log_level="info")
