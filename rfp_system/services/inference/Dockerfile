# Qwen3.8-27B on SageMaker Async, for the two jobs Bedrock cannot serve:
# the 262K-token RFP shred and vision parsing of scanned pages.
#
# Built on vllm/vllm-openai rather than the AWS LMI container because Qwen3.8's
# gated DeltaNet hybrid attention is new enough that pinning the vLLM version
# matters more than staying on AWS's upgrade cadence.
#
# Pin the exact tag. "latest" here means a silent model-serving change on a
# rebuild, discovered during a proposal.
ARG VLLM_VERSION=v0.11.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV MODEL_ID=Qwen/Qwen3.8-27B \
    QUANTIZATION=fp8 \
    KV_CACHE_DTYPE=fp8 \
    MAX_MODEL_LEN=262144 \
    GPU_MEMORY_UTILIZATION=0.92 \
    PORT=8080

# SageMaker requires /ping and /invocations. vLLM serves neither, so a thin
# adapter fronts it.
RUN python -m pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.32" "boto3>=1.35"
COPY services/inference/serve.py /opt/serve.py

EXPOSE 8080
ENTRYPOINT ["python", "/opt/serve.py"]
