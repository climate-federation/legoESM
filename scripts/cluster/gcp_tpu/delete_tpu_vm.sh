#!/usr/bin/env bash
# Delete the legoESM Cloud TPU VM.  A TPU VM bills for as long as it exists,
# so tear it down as soon as a run is finished.
#
# Usage:
#   GCP_PROJECT=my-proj TPU_ZONE=us-east5-a TPU_NAME=legoesm-tpu \
#       ./delete_tpu_vm.sh
set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}"
TPU_ZONE="${TPU_ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-legoesm-tpu}"

echo "Deleting TPU VM '$TPU_NAME' (zone=$TPU_ZONE, project=$GCP_PROJECT) ..."
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --quiet

echo "Deleted. Confirm none remain:"
gcloud compute tpus tpu-vm list --zone="$TPU_ZONE" --project="$GCP_PROJECT" || true