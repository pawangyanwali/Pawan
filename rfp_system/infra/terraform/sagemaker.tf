# ---------------------------------------------------------------------------
# Qwen3.8-27B on an async endpoint at MinCapacity 0.
#
# Serves two jobs Bedrock cannot: the 262K-token RFP shred and vision parsing
# of scanned pages. Costs nothing when nothing is queued.
#
# Async rather than real-time because scale-from-zero on a 30GB model takes 4
# to 8 minutes. A real-time endpoint would time out; async holds the request
# while the instance comes up. That is acceptable here because the shred is a
# background job nobody watches, and drafting runs on Bedrock.
# ---------------------------------------------------------------------------

resource "aws_sagemaker_model" "inference" {
  name               = "${var.name_prefix}-qwen38-27b"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = var.inference_image_uri
    model_data_url = var.model_artifact_s3_uri

    environment = {
      MODEL_ID               = "Qwen/Qwen3.8-27B"
      QUANTIZATION           = "fp8"
      KV_CACHE_DTYPE         = "fp8"
      MAX_MODEL_LEN          = "262144"
      GPU_MEMORY_UTILIZATION = "0.92"
      PORT                   = "8080"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "inference" {
  name = "${var.name_prefix}-inference"

  production_variants {
    variant_name           = "default"
    model_name             = aws_sagemaker_model.inference.name
    instance_type          = var.inference_instance_type
    initial_instance_count = 1
  }

  async_inference_config {
    output_config {
      s3_output_path  = "s3://${aws_s3_bucket.this["output"].id}/async-output/"
      s3_failure_path = "s3://${aws_s3_bucket.this["output"].id}/async-failure/"
      kms_key_id      = aws_kms_key.main.arn
    }

    client_config {
      # Two pursuits shredding at once. Raise it if that becomes three.
      max_concurrent_invocations_per_instance = 1
    }
  }
}

resource "aws_sagemaker_endpoint" "inference" {
  name                 = "${var.name_prefix}-inference"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.inference.name
}

resource "aws_appautoscaling_target" "inference" {
  service_namespace  = "sagemaker"
  resource_id        = "endpoint/${aws_sagemaker_endpoint.inference.name}/variant/default"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  min_capacity       = 0
  max_capacity       = 2
}

resource "aws_appautoscaling_policy" "inference" {
  name               = "${var.name_prefix}-inference-backlog"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.inference.service_namespace
  resource_id        = aws_appautoscaling_target.inference.resource_id
  scalable_dimension = aws_appautoscaling_target.inference.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value = 1.0

    customized_metric_specification {
      metric_name = "ApproximateBacklogSizePerInstance"
      namespace   = "AWS/SageMaker"
      statistic   = "Average"

      dimensions {
        name  = "EndpointName"
        value = aws_sagemaker_endpoint.inference.name
      }
    }

    scale_out_cooldown = 60
    # Ten minutes. Parsing arrives in bursts, and paying for ten idle minutes
    # beats paying an 8-minute cold start twice.
    scale_in_cooldown = 600
  }
}
