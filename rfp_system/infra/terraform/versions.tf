terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }

  # Remote state from day one. There is no dev environment, so the state file
  # is the only record of what production is.
  backend "s3" {
    key          = "rfp-system/prod.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      System      = "rfp-response"
      Environment = "prod"
      ManagedBy   = "terraform"
    }
  }
}
