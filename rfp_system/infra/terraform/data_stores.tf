# ---------------------------------------------------------------------------
# DynamoDB: everything this system persists.
#
# One table, on-demand billing. Idle cost is storage. Multi-AZ by default, so
# there is nothing to fail over and nothing to patch.
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "main" {
  name         = "${var.name_prefix}-main"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  # Analytics run on Athena over scheduled exports, not on this table. Keeps a
  # win-rate query off the store the drafting path depends on.
  lifecycle {
    prevent_destroy = true
  }
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

locals {
  buckets = {
    raw    = "SharePoint mirror. Versioned; a file deleted in SharePoint stays recoverable here."
    parsed = "Extracted text and page images."
    output = "Generated .docx."
  }
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = "${var.name_prefix}-${each.key}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.main.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.this["raw"].id

  rule {
    id     = "archive-old-versions"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "STANDARD_IA"
    }
  }
}

resource "aws_kms_key" "main" {
  description             = "${var.name_prefix} data at rest"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "main" {
  name          = "alias/${var.name_prefix}"
  target_key_id = aws_kms_key.main.key_id
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# S3 Vectors
#
# GA December 2025. Verify that the AWS provider version pinned in versions.tf
# carries these resource types before the first apply; if it does not, create
# the bucket and index with the AWS CLI and import them, rather than reaching
# for OpenSearch Serverless and its ~$350/month floor.
#
# Dimension 1024 matches Qwen3-Embedding-0.6B. Changing it means re-embedding
# the whole corpus, so it is not a value to guess at.
# ---------------------------------------------------------------------------

resource "aws_s3vectors_vector_bucket" "main" {
  vector_bucket_name = "${var.name_prefix}-vectors"

  encryption_configuration {
    sse_type   = "aws:kms"
    kms_key_id = aws_kms_key.main.arn
  }
}

resource "aws_s3vectors_index" "chunks" {
  vector_bucket_name  = aws_s3vectors_vector_bucket.main.vector_bucket_name
  index_name          = "chunks"
  data_type           = "float32"
  dimension           = 1024
  distance_metric     = "cosine"

  metadata_configuration {
    # Text is not stored here. It lives in DynamoDB, fetched by chunk ID after
    # a query, so a text correction is a write rather than a re-embed.
    non_filterable_metadata_keys = ["heading_path", "page_anchor"]
  }
}
