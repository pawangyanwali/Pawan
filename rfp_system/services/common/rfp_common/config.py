"""Configuration from the environment. No defaults that would silently work.

Every value a service needs is read once at import. A missing variable fails
the cold start, which surfaces in the canary window rather than on the first
request a writer makes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _req(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    region: str
    table_name: str
    vector_bucket: str
    vector_index: str
    raw_bucket: str
    parsed_bucket: str
    output_bucket: str
    bedrock_model_id: str
    sagemaker_endpoint: str
    async_output_s3_uri: str
    draft_queue_url: str
    shred_queue_url: str
    index_queue_url: str
    parse_queue_url: str
    ingest_queue_url: str
    entra_tenant_id: str
    entra_client_id: str
    entra_cert_secret_arn: str
    # Failover budget. A draft that has waited this long on the primary backend
    # reroutes rather than continuing to retry.
    primary_backend_timeout_s: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            table_name=_req("TABLE_NAME"),
            vector_bucket=_req("VECTOR_BUCKET"),
            vector_index=_req("VECTOR_INDEX"),
            raw_bucket=_req("RAW_BUCKET"),
            parsed_bucket=_req("PARSED_BUCKET"),
            output_bucket=_req("OUTPUT_BUCKET"),
            bedrock_model_id=_req("BEDROCK_MODEL_ID"),
            sagemaker_endpoint=_req("SAGEMAKER_ENDPOINT"),
            async_output_s3_uri=_req("ASYNC_OUTPUT_S3_URI"),
            draft_queue_url=_req("DRAFT_QUEUE_URL"),
            shred_queue_url=_req("SHRED_QUEUE_URL"),
            index_queue_url=_req("INDEX_QUEUE_URL"),
            parse_queue_url=_req("PARSE_QUEUE_URL"),
            ingest_queue_url=_req("INGEST_QUEUE_URL"),
            entra_tenant_id=_req("ENTRA_TENANT_ID"),
            entra_client_id=_req("ENTRA_CLIENT_ID"),
            entra_cert_secret_arn=_req("ENTRA_CERT_SECRET_ARN"),
            primary_backend_timeout_s=_int("PRIMARY_BACKEND_TIMEOUT_S", 45),
        )
