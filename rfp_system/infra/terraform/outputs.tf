output "table_name" {
  value = aws_dynamodb_table.main.name
}

output "vector_bucket" {
  value = aws_s3vectors_vector_bucket.main.vector_bucket_name
}

output "inference_endpoint" {
  value       = aws_sagemaker_endpoint.inference.name
  description = "Sits at zero instances until something queues. Cold start is 4 to 8 minutes."
}

output "ecr_repositories" {
  value = { for k, v in aws_ecr_repository.service : k => v.repository_url }
}

output "entra_cert_secret_arn" {
  value       = aws_secretsmanager_secret.entra_cert.arn
  description = "Populate out of band. Terraform creates the secret, never its value."
}
