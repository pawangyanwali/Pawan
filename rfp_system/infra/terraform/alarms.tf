# Seven alarms. Everything else is a dashboard.
#
# An alarm nobody acts on trains people to ignore the ones that matter, so this
# list is deliberately short.

resource "aws_sns_topic" "alerts" {
  name              = "${var.name_prefix}-alerts"
  kms_master_key_id = aws_kms_key.main.id
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# Gates the canary. A new version that raises the error rate rolls itself back.
resource "aws_cloudwatch_metric_alarm" "function_errors" {
  for_each = local.functions

  alarm_name          = "${var.name_prefix}-${replace(each.key, "_", "-")}-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Sum"
  threshold           = 2
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    FunctionName = aws_lambda_function.service[each.key].function_name
    Resource     = "${aws_lambda_function.service[each.key].function_name}:live"
  }
}

# A job died. Somebody's section never appeared.
resource "aws_cloudwatch_metric_alarm" "dlq_depth" {
  for_each = local.queues

  alarm_name          = "${var.name_prefix}-${each.key}-dlq"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Maximum"
  threshold           = 0
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    QueueName = aws_sqs_queue.dlq[each.key].name
  }
}

# Writers are waiting past the SLA.
resource "aws_cloudwatch_metric_alarm" "draft_backlog_age" {
  alarm_name          = "${var.name_prefix}-draft-stalled"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApproximateAgeOfOldestMessage"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Maximum"
  threshold           = 600
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    QueueName = aws_sqs_queue.main["draft"].name
  }
}

# Approaching an account quota. Drafting is about to fail over.
resource "aws_cloudwatch_metric_alarm" "bedrock_throttle" {
  alarm_name          = "${var.name_prefix}-bedrock-throttled"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "InvocationThrottles"
  namespace           = "AWS/Bedrock"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Shred is degraded to the chunked Bedrock fallback.
resource "aws_cloudwatch_metric_alarm" "endpoint_errors" {
  alarm_name          = "${var.name_prefix}-inference-endpoint-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocation4XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.inference.name
  }
}

# Quality regressions do not throw errors. With no staging environment, the
# nightly evaluation run is what a staging environment would otherwise catch.
resource "aws_cloudwatch_metric_alarm" "citation_accuracy" {
  alarm_name          = "${var.name_prefix}-citation-accuracy"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "CitationAccuracy"
  namespace           = "RFP/Eval"
  period              = 86400
  statistic           = "Average"
  threshold           = 0.95
  treat_missing_data  = "breaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "compliance_recall" {
  alarm_name          = "${var.name_prefix}-compliance-recall"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ComplianceRecall"
  namespace           = "RFP/Eval"
  period              = 86400
  statistic           = "Average"
  threshold           = 0.98
  treat_missing_data  = "breaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
