# Base image for every Lambda service.
#
# One base, one dependency resolution, one place to patch. Each service adds
# its own layer with a handler and any extra dependency it alone needs.
#
# Build from the repo root:
#   docker build -f docker/lambda.base.Dockerfile -t rfp-base:local .

FROM public.ecr.aws/lambda/python:3.12

COPY services/common/ /tmp/common/
RUN python -m pip install --no-cache-dir /tmp/common && rm -rf /tmp/common

# boto3 ships in the Lambda runtime, but the version lags and this system uses
# the s3vectors client, which is newer than most runtime bundles. Pin it here.
RUN python -m pip install --no-cache-dir "boto3>=1.35" "pydantic>=2.9"
