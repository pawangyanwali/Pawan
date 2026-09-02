"""Versioned contracts between services.

These models are the module boundary. A service may change anything about how
it works as long as it still accepts and produces these shapes.

Adding an optional field is backward compatible. Removing a field, renaming
one, or narrowing a type is not: bump SCHEMA_VERSION and support both shapes
for one deploy cycle.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


class Outcome(StrEnum):
    WIN = "win"
    LOSS = "loss"
    NO_BID = "no_bid"
    PENDING = "pending"
    UNKNOWN = "unknown"


class DocKind(StrEnum):
    SOLICITATION = "solicitation"
    PROPOSAL_VOLUME = "proposal_volume"
    PAST_PERFORMANCE = "past_performance"
    PRICING = "pricing"
    DEBRIEF = "debrief"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"


class BidMetadata(Base):
    """What we know about the pursuit a document belongs to.

    `outcome` drives retrieval ranking. Leaving it UNKNOWN means the chunk
    competes with winning content on equal footing, which is why the labeling
    backlog matters more than any model choice.
    """

    bid_id: str
    agency: str | None = None
    solicitation_number: str | None = None
    naics: str | None = None
    contract_vehicle: str | None = None
    submitted_on: date | None = None
    outcome: Outcome = Outcome.UNKNOWN
    award_value_usd: Decimal | None = None
    incumbent: str | None = None
    evaluation_type: Literal["lpta", "tradeoff", "other"] | None = None


class SourceDocument(Base):
    doc_id: str
    bid_id: str
    kind: DocKind
    sharepoint_item_id: str
    sharepoint_drive_id: str
    filename: str
    etag: str
    modified_at: datetime
    s3_raw_key: str
    # Entra object IDs permitted to read the source file. Mirrored onto every
    # chunk so the vector query can filter before anything is fetched.
    acl_object_ids: tuple[str, ...] = ()


class Chunk(Base):
    """One retrievable unit.

    `heading_path` is prepended to the embedded text. Without it a chunk about
    "our transition approach" is indistinguishable from twelve others.
    """

    chunk_id: str
    doc_id: str
    bid_id: str
    text: str
    heading_path: tuple[str, ...]
    page_anchor: int | None = None
    token_count: int
    metadata: BidMetadata
    acl_object_ids: tuple[str, ...] = ()


class RetrievedChunk(Base):
    chunk: Chunk
    vector_score: float
    rerank_score: float | None = None


# --------------------------------------------------------------------------
# Compliance
# --------------------------------------------------------------------------


class RequirementType(StrEnum):
    FORMAT = "format"
    CONTENT = "content"
    SUBMISSION = "submission"
    EVALUATION = "evaluation"


class Requirement(Base):
    """One row of the compliance matrix.

    `verbatim_text` is never paraphrased. Evaluators check against the words in
    the solicitation, so the matrix has to carry them exactly.
    """

    requirement_id: str
    rfp_id: str
    source_ref: str  # "L.3.2.1"
    verbatim_text: str
    requirement_type: RequirementType
    volume: str | None = None
    page_limit: int | None = None
    evaluation_refs: tuple[str, ...] = ()  # Section M criteria this maps to
    owner: str | None = None
    response_location: str | None = None
    status: Literal["open", "drafted", "reviewed", "complete"] = "open"


class ComplianceGap(Base):
    requirement_id: str
    reason: Literal["no_response", "unaddressed", "over_page_limit", "format_violation"]
    detail: str


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


class Citation(Base):
    chunk_id: str
    doc_id: str
    bid_id: str
    quoted_span: str


class Block(Base):
    """A unit of rendered content.

    Generation emits these, never markdown. The renderer maps each to a
    python-docx call against the customer template, so numbering, styles, and
    section breaks come from the .dotx rather than from a converter's guess.
    """

    kind: Literal["heading", "paragraph", "bullet", "numbered", "table", "figure_ref"]
    level: int | None = None
    text: str | None = None
    rows: tuple[tuple[str, ...], ...] | None = None
    citations: tuple[Citation, ...] = ()


class DraftRequest(Base):
    job_id: str
    rfp_id: str
    requirement_id: str
    outline_slot: str
    page_limit: int | None = None
    style_notes: str | None = None
    requested_by: str  # Entra object ID; drives ACL-trimmed retrieval


class DraftSection(Base):
    job_id: str
    rfp_id: str
    requirement_id: str
    blocks: tuple[Block, ...]
    # Sentences with no supporting citation. The UI marks these; a writer may
    # accept one, and has to look at it first.
    uncited_spans: tuple[str, ...] = ()
    model_used: str
    generated_at: datetime


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


class LaborRate(Base):
    card_version: str
    labor_category: str
    direct_rate_usd: Decimal
    escalation_pct_by_year: tuple[Decimal, ...]


class WrapRates(Base):
    card_version: str
    fringe_pct: Decimal
    overhead_pct: Decimal
    ga_pct: Decimal
    fee_pct: Decimal


class StaffingLine(Base):
    """Proposed by the model. Priced by `compute`. Never both."""

    labor_category: str
    hours_by_year: tuple[int, ...]
    rationale: str


class PricedLine(Base):
    labor_category: str
    hours_by_year: tuple[int, ...]
    cost_by_year_usd: tuple[Decimal, ...]
    card_version: str


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


class Capability(StrEnum):
    """What a call needs, not which model serves it.

    Callers name a capability. The gateway maps it to a backend. Moving a
    workload between Bedrock and the self-hosted endpoint is a table entry.
    """

    DRAFT = "draft"
    ANSWER = "answer"
    CLASSIFY = "classify"
    REWRITE_QUERY = "rewrite_query"
    LONG_CONTEXT_SHRED = "long_context_shred"
    VISION_PARSE = "vision_parse"


class InferenceRequest(Base):
    capability: Capability
    system: str
    user: str
    max_tokens: int = 4096
    temperature: float = 0.2
    image_s3_keys: tuple[str, ...] = ()
    response_schema: dict | None = None


class InferenceResponse(Base):
    text: str
    model_used: str
    backend: Literal["bedrock", "sagemaker_async"]
    input_tokens: int
    output_tokens: int
    degraded: bool = False  # True when served by the failover path


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class JobRecord(Base):
    job_id: str
    job_type: Literal["ingest", "parse", "index", "draft", "shred", "backfill"]
    status: JobStatus
    rfp_id: str | None = None
    attempts: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    schema_version: int = Field(default=SCHEMA_VERSION)
