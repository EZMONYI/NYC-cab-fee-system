#!/usr/bin/env bash

PROJECT_ID="ml-fare-prediction-371004"  # e.g., ml-fare-prediction-xxxxxx
BUCKET_ID="ml-fare-prediction-371004"    # e.g., my-bucket
# A name should start with a letter and contain only letters, numbers and underscores.
JOB_NAME="project_6_trial_1_$(date +"%m%d_%H%M")"

JOB_DIR="gs://${BUCKET_ID}/"
TRAINING_PACKAGE_PATH="$(pwd)/ai_platform_trainer"
MAIN_TRAINER_MODULE=ai_platform_trainer.train
REGION=us-east1
RUNTIME_VERSION=2.4
PYTHON_VERSION=3.7
CONFIG_YAML=config.yaml

# https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training
gcloud ai-platform jobs submit training "${JOB_NAME}" \
 --job-dir "${JOB_DIR}" \
 --package-path "${TRAINING_PACKAGE_PATH}" \
 --module-name "${MAIN_TRAINER_MODULE}" \
 --region "${REGION}" \
 --runtime-version="${RUNTIME_VERSION}" \
 --python-version="${PYTHON_VERSION}" \
 --config "${CONFIG_YAML}"
