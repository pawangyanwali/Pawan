"""S3 Vectors access.

The index holds embeddings plus filterable metadata. Chunk text lives in
DynamoDB, fetched by ID after a query returns. That split keeps the index small
and makes a text correction a DynamoDB write rather than a re-embed.

Verify operation and parameter names against the current boto3 s3vectors
client before the first deploy. The service reached GA in December 2025 and the
API is younger than most of what this codebase depends on.
"""

from __future__ import annotations

from typing import Any

import boto3

from .contracts import BidMetadata, Chunk, Outcome


class VectorIndex:
    def __init__(self, bucket: str, index: str, region: str) -> None:
        self._client = boto3.client("s3vectors", region_name=region)
        self._bucket = bucket
        self._index = index

    def put(self, chunk: Chunk, embedding: list[float]) -> None:
        self._client.put_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index,
            vectors=[
                {
                    "key": chunk.chunk_id,
                    "data": {"float32": embedding},
                    "metadata": _filterable(chunk),
                }
            ],
        )

    def put_batch(self, items: list[tuple[Chunk, list[float]]]) -> None:
        """Backfill path. One call per 500 chunks beats one call per chunk by
        roughly two orders of magnitude on the initial corpus load."""
        for i in range(0, len(items), 500):
            self._client.put_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                vectors=[
                    {
                        "key": chunk.chunk_id,
                        "data": {"float32": embedding},
                        "metadata": _filterable(chunk),
                    }
                    for chunk, embedding in items[i : i + 500]
                ],
            )

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 50,
        acl_object_ids: list[str],
        agency: str | None = None,
        outcomes: list[Outcome] | None = None,
        after_year: int | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Returns (chunk_id, distance, metadata).

        The ACL filter runs inside the query rather than after it. Filtering
        afterwards means fetching content the caller may not read, and a top_k
        that silently shrinks once the unreadable results are dropped.
        """
        conditions: list[dict[str, Any]] = [
            {"acl_object_ids": {"$in": acl_object_ids}}
        ]
        if agency:
            conditions.append({"agency": {"$eq": agency}})
        if outcomes:
            conditions.append({"outcome": {"$in": [o.value for o in outcomes]}})
        if after_year:
            conditions.append({"submitted_year": {"$gte": after_year}})

        response = self._client.query_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index,
            queryVector={"float32": embedding},
            topK=top_k,
            filter={"$and": conditions} if len(conditions) > 1 else conditions[0],
            returnMetadata=True,
            returnDistance=True,
        )
        return [
            (v["key"], v["distance"], v.get("metadata", {}))
            for v in response.get("vectors", [])
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        self._client.delete_vectors(
            vectorBucketName=self._bucket, indexName=self._index, keys=chunk_ids
        )


def _filterable(chunk: Chunk) -> dict[str, Any]:
    """Only what a query filters on. Metadata counts against index size, so the
    heading path, page anchor, and text stay in DynamoDB."""
    md: BidMetadata = chunk.metadata
    out: dict[str, Any] = {
        "bid_id": chunk.bid_id,
        "doc_id": chunk.doc_id,
        "outcome": md.outcome.value,
        "acl_object_ids": list(chunk.acl_object_ids),
    }
    if md.agency:
        out["agency"] = md.agency
    if md.naics:
        out["naics"] = md.naics
    if md.contract_vehicle:
        out["contract_vehicle"] = md.contract_vehicle
    if md.solicitation_number:
        out["solicitation_number"] = md.solicitation_number
    if md.submitted_on:
        out["submitted_year"] = md.submitted_on.year
    return out
