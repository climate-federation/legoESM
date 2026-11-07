#!/bin/bash
# Build the ERF 26.09 SDM oracle driver (host, double precision).
# Usage: build.sh <ERF checkout @ 6d09aeb120eb4eac388106004bfaaa7f795a2e26> <build dir>
set -euo pipefail
ERF=$1; B=$2; H=$(cd "$(dirname "$0")" && pwd)
test "$(git -C "$ERF" rev-parse HEAD)" = 6d09aeb120eb4eac388106004bfaaa7f795a2e26
mkdir -p "$B"
bash "$H/extract.sh" "$ERF" "$B/extract"
S=$ERF/Source
g++ -std=c++17 -O0 -ffp-contract=off -I"$H/stub" -I"$S" -I"$S/Utils" -I"$S/MaterialProperties" \
    -I"$S/Particles" -I"$B/extract" -o "$B/sdm_driver" "$H/sdm_driver.cpp"
echo "BUILD_OK $B/sdm_driver"
