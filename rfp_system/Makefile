# Build, test, deploy. Everything a handoff needs to run this system.

AWS_REGION  ?= us-east-1
ACCOUNT     := $(shell aws sts get-caller-identity --query Account --output text)
REGISTRY    := $(ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com
TAG         ?= $(shell git rev-parse --short HEAD)

SERVICES := api retrieval render compute worker_ingest worker_parse worker_index worker_draft

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

.PHONY: base
base: ## Build the shared Lambda base image
	docker build -f docker/lambda.base.Dockerfile -t rfp-base:$(TAG) .
	docker tag rfp-base:$(TAG) rfp-base:local

.PHONY: build
build: base ## Build every service image
	@for s in $(SERVICES); do \
		echo "building $$s"; \
		docker build --build-arg BASE=rfp-base:local \
			-f services/$$s/Dockerfile \
			-t rfp/$$(echo $$s | tr _ -):$(TAG) . || exit 1; \
	done

.PHONY: build-inference
build-inference: ## Build the vLLM container for the SageMaker endpoint
	docker build -f services/inference/Dockerfile -t rfp/inference:$(TAG) .

.PHONY: push
push: build ## Push service images to ECR
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(REGISTRY)
	@for s in $(SERVICES); do \
		name=$$(echo $$s | tr _ -); \
		docker tag rfp/$$name:$(TAG) $(REGISTRY)/rfp/$$name:$(TAG); \
		docker push $(REGISTRY)/rfp/$$name:$(TAG) || exit 1; \
	done

.PHONY: test
test: ## Contract tests. A change that breaks a contract fails here, not in production.
	python -m pytest tests/ -q

.PHONY: local
local: ## LocalStack plus a mocked inference gateway. Catches contract mismatches on a laptop.
	docker compose -f local/docker-compose.yml up

.PHONY: plan
plan: ## Terraform plan against production
	cd infra/terraform && terraform plan -var="image_tag=$(TAG)"

.PHONY: deploy
deploy: push ## Apply. CodeDeploy canaries at 10% for 5 minutes, then rolls forward or back on its own.
	cd infra/terraform && terraform apply -var="image_tag=$(TAG)" -auto-approve
	$(MAKE) smoke

.PHONY: smoke
smoke: ## Six checks against production. Under 30 seconds. Failure triggers the rollback.
	python scripts/smoke.py

.PHONY: eval
eval: ## Held-out evaluation. Publishes CitationAccuracy and ComplianceRecall to CloudWatch.
	python scripts/evaluate.py --held-out data/heldout.jsonl

.PHONY: backfill
backfill: ## One-time corpus load. Idempotent per document; you will run it more than once.
	python scripts/backfill.py --confirm
