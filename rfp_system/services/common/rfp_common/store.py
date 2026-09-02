"""DynamoDB access.

One table, composite key. `pk` scopes an entity family, `sk` orders within it.
Everything this system persists is read by a known key, which is why there is
no relational database here.

    pk                    sk                        entity
    BID#<bid_id>          META                      BidMetadata
    BID#<bid_id>          DOC#<doc_id>              SourceDocument
    CHUNK#<chunk_id>      META                      Chunk (text + metadata)
    RFP#<rfp_id>          REQ#<requirement_id>      Requirement
    RFP#<rfp_id>          SECTION#<req_id>#<ts>     DraftSection (versioned)
    CARD#<version>        RATE#<labor_category>     LaborRate
    CARD#<version>        WRAP                      WrapRates
    JOB#<job_id>          META                      JobRecord
    SYNC#<drive_id>       DELTA                     Graph deltaLink
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Iterator

import boto3
from boto3.dynamodb.conditions import Key

from .contracts import (
    Chunk,
    DraftSection,
    JobRecord,
    Requirement,
    SourceDocument,
)


def _encode(model: Any) -> dict:
    """Pydantic to DynamoDB. Floats are rejected by DynamoDB, so round-trip
    through JSON with Decimal parsing rather than hand-converting each field."""
    return json.loads(model.model_dump_json(), parse_float=Decimal)


class Store:
    def __init__(self, table_name: str, region: str) -> None:
        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    # -- chunks ------------------------------------------------------------

    def put_chunk(self, chunk: Chunk) -> None:
        self._table.put_item(
            Item={"pk": f"CHUNK#{chunk.chunk_id}", "sk": "META", **_encode(chunk)}
        )

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Batch-get the text for IDs returned by a vector query.

        S3 Vectors returns IDs and metadata; the text lives here. Splitting them
        keeps the vector index small and the text cheap to update.
        """
        out: list[Chunk] = []
        for i in range(0, len(chunk_ids), 100):
            keys = [{"pk": f"CHUNK#{c}", "sk": "META"} for c in chunk_ids[i : i + 100]]
            response = self._table.meta.client.batch_get_item(
                RequestItems={self._table.name: {"Keys": keys}}
            )
            out.extend(
                Chunk.model_validate(item)
                for item in response["Responses"][self._table.name]
            )
        return out

    # -- documents ---------------------------------------------------------

    def put_document(self, doc: SourceDocument) -> None:
        self._table.put_item(
            Item={"pk": f"BID#{doc.bid_id}", "sk": f"DOC#{doc.doc_id}", **_encode(doc)}
        )

    def documents_for_bid(self, bid_id: str) -> Iterator[SourceDocument]:
        response = self._table.query(
            KeyConditionExpression=Key("pk").eq(f"BID#{bid_id}")
            & Key("sk").begins_with("DOC#")
        )
        for item in response["Items"]:
            yield SourceDocument.model_validate(item)

    # -- compliance --------------------------------------------------------

    def put_requirement(self, req: Requirement) -> None:
        self._table.put_item(
            Item={
                "pk": f"RFP#{req.rfp_id}",
                "sk": f"REQ#{req.requirement_id}",
                **_encode(req),
            }
        )

    def requirements(self, rfp_id: str) -> list[Requirement]:
        response = self._table.query(
            KeyConditionExpression=Key("pk").eq(f"RFP#{rfp_id}")
            & Key("sk").begins_with("REQ#")
        )
        return [Requirement.model_validate(i) for i in response["Items"]]

    # -- drafts ------------------------------------------------------------

    def put_section(self, section: DraftSection) -> None:
        """Every generation is a new version, never an overwrite.

        A writer who preferred yesterday's draft can get it back, and a bad
        prompt change is recoverable without a restore.
        """
        stamp = section.generated_at.isoformat()
        self._table.put_item(
            Item={
                "pk": f"RFP#{section.rfp_id}",
                "sk": f"SECTION#{section.requirement_id}#{stamp}",
                **_encode(section),
            }
        )

    def latest_section(self, rfp_id: str, requirement_id: str) -> DraftSection | None:
        response = self._table.query(
            KeyConditionExpression=Key("pk").eq(f"RFP#{rfp_id}")
            & Key("sk").begins_with(f"SECTION#{requirement_id}#"),
            ScanIndexForward=False,
            Limit=1,
        )
        items = response["Items"]
        return DraftSection.model_validate(items[0]) if items else None

    # -- jobs --------------------------------------------------------------

    def put_job(self, job: JobRecord) -> None:
        self._table.put_item(
            Item={"pk": f"JOB#{job.job_id}", "sk": "META", **_encode(job)}
        )

    # -- graph sync --------------------------------------------------------

    def get_delta_link(self, drive_id: str) -> str | None:
        response = self._table.get_item(
            Key={"pk": f"SYNC#{drive_id}", "sk": "DELTA"}
        )
        item = response.get("Item")
        return item["delta_link"] if item else None

    def put_delta_link(self, drive_id: str, delta_link: str) -> None:
        self._table.put_item(
            Item={"pk": f"SYNC#{drive_id}", "sk": "DELTA", "delta_link": delta_link}
        )
