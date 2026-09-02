# ---------------------------------------------------------------------------
# Every service is a Lambda container image. Idle cost across all of them is
# zero.
#
# Each deploys behind a weighted alias so CodeDeploy can shift 10% of traffic,
# watch an error alarm for five minutes, and roll back on its own. With no
# staging environment, that automatic rollback is what stands between a bad
# deploy and a writer four hours from submission.
# ---------------------------------------------------------------------------

locals {
  # memory is sized to the job, not guessed. retrieval carries a 0.6B ONNX
  # embedding model, so it needs the memory (and therefore the vCPU) to load it.
  functions = {
    api           = { memory = 1024, timeout = 30, queue = null }
    retrieval     = { memory = 3008, timeout = 60, queue = null }
    render        = { memory = 2048, timeout = 120, queue = null }
    compute       = { memory = 1024, timeout = 30, queue = null }
    worker_ingest = { memory = 1024, timeout = 300, queue = "ingest" }
    worker_parse  = { memory = 3008, timeout = 900, queue = "parse" }
    worker_index  = { memory = 3008, timeout = 300, queue = "index" }
    worker_draft  = { memory = 1024, timeout = 900, queue = "draft" }
  }

  common_env = {
    TABLE_NAME          = aws_dynamodb_table.main.name
    VECTOR_BUCKET       = aws_s3vectors_vector_bucket.main.vector_bucket_name
    VECTOR_INDEX        = aws_s3vectors_index.chunks.index_name
    RAW_BUCKET          = aws_s3_bucket.this["raw"].id
    PARSED_BUCKET       = aws_s3_bucket.this["parsed"].id
    OUTPUT_BUCKET       = aws_s3_bucket.this["output"].id
    BEDROCK_MODEL_ID    = var.bedrock_model_id
    SAGEMAKER_ENDPOINT  = aws_sagemaker_endpoint.inference.name
    ASYNC_OUTPUT_S3_URI = "s3://${aws_s3_bucket.this["output"].id}/async-output/"
    INGEST_QUEUE_URL    = aws_sqs_queue.main["ingest"].url
    PARSE_QUEUE_URL     = aws_sqs_queue.main["parse"].url
    INDEX_QUEUE_URL     = aws_sqs_queue.main["index"].url
    DRAFT_QUEUE_URL     = aws_sqs_queue.main["draft"].url
    SHRED_QUEUE_URL     = aws_sqs_queue.main["shred"].url
    ENTRA_TENANT_ID     = var.entra_tenant_id
    ENTRA_CLIENT_ID     = var.entra_client_id
    ENTRA_CERT_SECRET_ARN = aws_secretsmanager_secret.entra_cert.arn
  }
}

resource "aws_ecr_repository" "service" {
  for_each             = local.functions
  name                 = "${var.name_prefix}/${replace(each.key, "_", "-")}"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_lambda_function" "service" {
  for_each = local.functions

  function_name = "${var.name_prefix}-${replace(each.key, "_", "-")}"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.service[each.key].repository_url}:${var.image_tag}"
  memory_size   = each.value.memory
  timeout       = each.value.timeout
  publish       = true

  environment {
    variables = local.common_env
  }

  tracing_config {
    mode = "Active"
  }
}

resource "aws_lambda_alias" "live" {
  for_each         = local.functions
  name             = "live"
  function_name    = aws_lambda_function.service[each.key].function_name
  function_version = aws_lambda_function.service[each.key].version

  # CodeDeploy owns the weighting during a canary. Terraform must not fight it.
  lifecycle {
    ignore_changes = [routing_config, function_version]
  }
}

# Provisioned concurrency on retrieval only. The 0.6B ONNX embedding model
# takes 10 to 15 seconds to load cold, and at ~300 queries a month that is a
# cold start on most of them. $12/month removes it.
resource "aws_lambda_provisioned_concurrency_config" "retrieval" {
  function_name                     = aws_lambda_function.service["retrieval"].function_name
  qualifier                         = aws_lambda_alias.live["retrieval"].name
  provisioned_concurrent_executions = 1
}

resource "aws_lambda_event_source_mapping" "queue" {
  for_each = { for k, v in local.functions : k => v if v.queue != null }

  event_source_arn = aws_sqs_queue.main[each.value.queue].arn
  function_name    = aws_lambda_alias.live[each.key].arn
  batch_size       = 5

  # Per-message failure reporting. One poisoned requirement must not send the
  # whole batch back and re-run the sections that worked.
  function_response_types = ["ReportBatchItemFailures"]

  scaling_config {
    maximum_concurrency = 10
  }
}

# ---------------------------------------------------------------------------
# Canary deploys
# ---------------------------------------------------------------------------

resource "aws_codedeploy_app" "lambda" {
  name             = "${var.name_prefix}-lambda"
  compute_platform = "Lambda"
}

resource "aws_codedeploy_deployment_group" "service" {
  for_each = local.functions

  app_name               = aws_codedeploy_app.lambda.name
  deployment_group_name  = replace(each.key, "_", "-")
  service_role_arn       = aws_iam_role.codedeploy.arn
  deployment_config_name = "CodeDeployDefault.LambdaCanary10Percent5Minutes"

  deployment_style {
    deployment_option = "WITH_TRAFFIC_CONTROL"
    deployment_type   = "BLUE_GREEN"
  }

  auto_rollback_configuration {
    enabled = true
    events  = ["DEPLOYMENT_FAILURE", "DEPLOYMENT_STOP_ON_ALARM"]
  }

  alarm_configuration {
    enabled = true
    alarms  = [aws_cloudwatch_metric_alarm.function_errors[each.key].alarm_name]
  }
}

resource "aws_secretsmanager_secret" "entra_cert" {
  description = "Certificate for Entra app-only auth to Microsoft Graph. Expiry is the most common cause of this integration going dark, and it fails at the worst moment because nobody was watching a date."
  name        = "${var.name_prefix}/entra-cert"
  kms_key_id  = aws_kms_key.main.arn
}

variable "image_tag" {
  description = "Immutable image tag, normally the git SHA. Never 'latest'."
  type        = string
}
