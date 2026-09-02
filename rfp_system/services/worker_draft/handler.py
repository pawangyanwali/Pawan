"""Draft one proposal section.

One SQS message is one requirement. Keeping the unit that small means a spot
interruption, a throttle, or a bad response costs one section and not a volume,
and the writer sees a slower queue rather than a failure.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import boto3

from rfp_common.config import Config
from rfp_common.contracts import (
    Block,
    Capability,
    Citation,
    DraftRequest,
    DraftSection,
    InferenceRequest,
    RetrievedChunk,
)
from rfp_common.llm import AllBackendsFailed, InferenceGateway
from rfp_common.store import Store

log = logging.getLogger()
log.setLevel(logging.INFO)

CFG = Config.from_env()
GATEWAY = InferenceGateway(CFG)
STORE = Store(CFG.table_name, CFG.region)
LAMBDA = boto3.client("lambda", region_name=CFG.region)

SYSTEM = """\
You draft one section of a federal proposal.

Rules you do not break:
- Every factual claim cites a source ID from the provided context. Contract
  numbers, period of performance, CPARS ratings, staff names, dollar values,
  and places of performance come from the context or they do not appear.
- If the context does not support a claim the requirement asks for, say so in
  a note rather than inventing it.
- Answer the requirement as written. Do not restate it back.
- Write the way the retrieved winning sections are written.

Return JSON matching the provided schema. No prose outside the JSON."""


def handler(event: dict, _context: object) -> dict:
    failures: list[dict] = []

    for record in event["Records"]:
        try:
            _draft_one(DraftRequest.model_validate_json(record["body"]))
        except Exception:
            # Report per-message so one poisoned requirement does not send the
            # whole batch back to the queue and re-run the ones that worked.
            log.exception("draft failed for message %s", record["messageId"])
            failures.append({"itemIdentifier": record["messageId"]})

    return {"batchItemFailures": failures}


def _draft_one(req: DraftRequest) -> None:
    requirement = next(
        r for r in STORE.requirements(req.rfp_id) if r.requirement_id == req.requirement_id
    )

    chunks = _retrieve(
        query=requirement.verbatim_text,
        rfp_id=req.rfp_id,
        requested_by=req.requested_by,
    )

    try:
        response = GATEWAY.invoke(
            InferenceRequest(
                capability=Capability.DRAFT,
                system=SYSTEM,
                user=_prompt(requirement, chunks, req),
                max_tokens=6000,
                temperature=0.3,
                response_schema=_SCHEMA,
            )
        )
    except AllBackendsFailed:
        log.exception("no inference backend available for %s", req.requirement_id)
        raise

    blocks = _parse_blocks(response.text)
    section = DraftSection(
        job_id=req.job_id,
        rfp_id=req.rfp_id,
        requirement_id=req.requirement_id,
        blocks=blocks,
        uncited_spans=_uncited(blocks),
        model_used=response.model_used,
        generated_at=datetime.now(UTC),
    )

    # Persist before returning. A section that was generated but not stored is
    # a section the writer waited for and never got.
    STORE.put_section(section)

    if response.degraded:
        log.warning(
            "section %s served by the failover backend", req.requirement_id
        )


def _retrieve(*, query: str, rfp_id: str, requested_by: str) -> list[RetrievedChunk]:
    response = LAMBDA.invoke(
        FunctionName="rfp-retrieval",
        Payload=json.dumps(
            {
                "query": query,
                "rfp_id": rfp_id,
                "requested_by": requested_by,
                "top_k": 8,
            }
        ).encode(),
    )
    payload = json.loads(response["Payload"].read())
    return [RetrievedChunk.model_validate(c) for c in payload["chunks"]]


def _prompt(requirement, chunks: list[RetrievedChunk], req: DraftRequest) -> str:
    context = "\n\n".join(
        f"[{c.chunk.chunk_id}] ({' > '.join(c.chunk.heading_path)}, "
        f"{c.chunk.metadata.agency or 'unknown agency'}, "
        f"{c.chunk.metadata.outcome.value})\n{c.chunk.text}"
        for c in chunks
    )
    limit = f"\nPage limit: {req.page_limit}." if req.page_limit else ""
    evaluation = (
        f"\nEvaluated against: {', '.join(requirement.evaluation_refs)}."
        if requirement.evaluation_refs
        else ""
    )
    return (
        f"Requirement {requirement.source_ref}:\n{requirement.verbatim_text}\n"
        f"{evaluation}{limit}\n"
        f"Outline slot: {req.outline_slot}\n\n"
        f"Context from past bids:\n{context}"
    )


def _parse_blocks(text: str) -> tuple[Block, ...]:
    payload = json.loads(text)
    return tuple(
        Block(
            kind=b["kind"],
            level=b.get("level"),
            text=b.get("text"),
            rows=tuple(tuple(r) for r in b["rows"]) if b.get("rows") else None,
            citations=tuple(
                Citation(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    bid_id=c["bid_id"],
                    quoted_span=c["quoted_span"],
                )
                for c in b.get("citations", [])
            ),
        )
        for b in payload["blocks"]
    )


def _uncited(blocks: tuple[Block, ...]) -> tuple[str, ...]:
    """Prose blocks with no citation.

    Not an error. The UI marks these so a writer looks at them before accepting.
    A fluent past-performance paragraph with an invented contract number passes
    casual review and fails at the customer."""
    return tuple(
        b.text
        for b in blocks
        if b.kind in ("paragraph", "bullet", "numbered") and b.text and not b.citations
    )


_SCHEMA = {
    "type": "object",
    "required": ["blocks"],
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind"],
                "properties": {
                    "kind": {
                        "enum": [
                            "heading",
                            "paragraph",
                            "bullet",
                            "numbered",
                            "table",
                            "figure_ref",
                        ]
                    },
                    "level": {"type": "integer"},
                    "text": {"type": "string"},
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "chunk_id",
                                "doc_id",
                                "bid_id",
                                "quoted_span",
                            ],
                            "properties": {
                                "chunk_id": {"type": "string"},
                                "doc_id": {"type": "string"},
                                "bid_id": {"type": "string"},
                                "quoted_span": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
    },
}
