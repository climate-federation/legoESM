#!/usr/bin/env bash
# Provision a single-host Cloud TPU VM for legoESM scaling tests.
#
# Target (per the TPU plan): a single-host slice — v5e-8 (v5litepod-8) or
# v6e-8 — i.e. 8 chips in ONE host, pure SPMD (no multi-host, no MPI).  The
# cubed-sphere FV3 workload uses 6 of the 8 chips (true face sharding; 8 does
# not divide the 6-face layout — see docs/scaling/scaling_tpu.md).
#
# All settings are overridable via environment variables.  ACCELERATOR_TYPE
# and RUNTIME_VERSION change over TPU generations — verify current values:
#   gcloud compute tpus accelerator-types list --zone="$TPU_ZONE"
#   gcloud compute tpus tpu-vm versions list --zone="$TPU_ZONE"
#
# Usage:
#   GCP_PROJECT=my-proj TPU_ZONE=us-east5-a ./create_tpu_vm.sh
#   GCP_PROJECT=my-proj ACCELERATOR_TYPE=v6e-8 RUNTIME_VERSION=v2-alpha-tpuv6e \
#       ./create_tpu_vm.sh
#   # Cheap 4-chip smoke-test VM (same runtime as v5e-8); pair with run_smoke.sh:
#   GCP_PROJECT=my-proj ACCELERATOR_TYPE=v5litepod-4 USE_SPOT=1 ./create_tpu_vm.sh
set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}"
TPU_ZONE="${TPU_ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-legoesm-tpu}"
# v5e-8 single host.  For v6e (Trillium): ACCELERATOR_TYPE=v6e-8,
# RUNTIME_VERSION=v2-alpha-tpuv6e.
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-8}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"
# Set USE_SPOT=1 for a Spot (preemptible) VM — far cheaper, can be reclaimed.
USE_SPOT="${USE_SPOT:-0}"

spot_flag=()
if [[ "$USE_SPOT" == "1" ]]; then
  spot_flag=(--spot)
fi

echo "Creating TPU VM:"
echo "  project      = $GCP_PROJECT"
echo "  zone         = $TPU_ZONE"
echo "  name         = $TPU_NAME"
echo "  accelerator  = $ACCELERATOR_TYPE"
echo "  runtime      = $RUNTIME_VERSION"
echo "  spot         = $USE_SPOT"
echo

gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --version="$RUNTIME_VERSION" \
  "${spot_flag[@]}"

echo
echo "Created. Next steps:"
echo "  # 1. ssh in:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --project=$GCP_PROJECT"
echo "  # 2. on the VM, get the repo (git clone preferred — avoids copying"
echo "  #    the multi-GB .venv/.git/results that 'scp .' would drag along):"
echo "  git clone <your-legoESM-remote-url> ~/legoESM"
echo "  # 3. on the VM:  cd ~/legoESM && bash scripts/cluster/gcp_tpu/setup_env.sh"
echo "  # 4. on the VM:  bash scripts/cluster/gcp_tpu/run_bench.sh"
echo
echo "REMEMBER to delete the VM when done (it bills while it exists):"
echo "  GCP_PROJECT=$GCP_PROJECT TPU_ZONE=$TPU_ZONE TPU_NAME=$TPU_NAME \\"
echo "      bash scripts/cluster/gcp_tpu/delete_tpu_vm.sh"