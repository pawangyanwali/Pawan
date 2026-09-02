# Five queues, each with a dead-letter queue. A job that fails three times is
# visible in a DLQ rather than gone; DLQ depth above zero is an alarm, because
# it means a writer's section never appeared.

locals {
  queues = {
    ingest = { visibility = 300, retries = 3 }
    parse  = { visibility = 900, retries = 3 }
    index  = { visibility = 300, retries = 3 }
    draft  = { visibility = 900, retries = 3 }
    shred  = { visibility = 900, retries = 2 }
  }
}

resource "aws_sqs_queue" "dlq" {
  for_each                  = local.queues
  name                      = "${var.name_prefix}-${each.key}-dlq"
  message_retention_seconds = 1209600 # 14 days
  kms_master_key_id         = aws_kms_key.main.id
}

resource "aws_sqs_queue" "main" {
  for_each = local.queues
  name     = "${var.name_prefix}-${each.key}"

  # Long enough that a message being worked on stays invisible through a
  # SageMaker scale-from-zero. Shorter and a slow job gets delivered twice.
  visibility_timeout_seconds = each.value.visibility
  kms_master_key_id          = aws_kms_key.main.id

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq[each.key].arn
    maxReceiveCount     = each.value.retries
  })
}
