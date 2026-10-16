#!/bin/bash
# Pinned CAM6 reference source (read-only oracle for namelist/dycore questions).
# Tag = the CAM pinned by CESM release-cesm2.1.5 Externals.cfg (final CESM2.1,
# the CMIP6 release line).  Lives OUTSIDE the repo, like the NEMO/FV3 oracles.
# Light shallow clone, no build: fine on a login node.
#   bash scripts/cluster/cam6_ref/fetch_cam6_src.sh [DEST]
set -euo pipefail
TAG=cam_cesm2_1_rel_60
SHA=a03b84b7c4e34f965b115686f22a043b85739e56   # ${TAG}^{} commit
DEST=${1:-/burg-archive/glab/users/pg2328/cam6_ref/CAM_${TAG}}
[ -d "$DEST/.git" ] || git -c advice.detachedHead=false clone -q --depth 1 --branch "$TAG" https://github.com/ESCOMP/CAM.git "$DEST"
[ "$(git -C "$DEST" rev-parse HEAD)" = "$SHA" ] || { echo "SHA mismatch in $DEST"; exit 1; }
echo "CAM $TAG @ $SHA -> $DEST"
