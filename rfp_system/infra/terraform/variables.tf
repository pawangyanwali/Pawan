variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name_prefix" {
  type    = string
  default = "rfp"
}

variable "bedrock_model_id" {
  description = "Managed Bedrock model for drafting, Q&A, and classification. Revisit at each Bedrock release: if Qwen3.8-27B lands as a managed model, this replaces the SageMaker endpoint entirely."
  type        = string
  default     = "qwen.qwen3-32b-v1:0"
}

variable "inference_image_uri" {
  description = "ECR URI for the vLLM container serving Qwen3.8-27B."
  type        = string
}

variable "model_artifact_s3_uri" {
  description = "S3 location of the FP8 checkpoint. About 30GB, and the bulk of a 4-to-8 minute cold start."
  type        = string
}

variable "inference_instance_type" {
  description = "One L40S at 48GB. Holds 30GB of FP8 weights plus roughly 8.5GB of KV cache at 262K context. Measure the cache before trusting that."
  type        = string
  default     = "ml.g6e.xlarge"
}

variable "entra_tenant_id" {
  type = string
}

variable "entra_client_id" {
  type = string
}

variable "alarm_email" {
  description = "Where DLQ depth, stalled drafts, and Graph auth failures go."
  type        = string
}
