#!/bin/bash
# Build the CLUBB-core oracle: larson-group/clubb_release at the pinned SHA,
# constants patched to legoESM's, f2py Python API (compile.py -python).
# Usage: build.sh DEST_DIR   (run on a compute node; needs gfortran, cmake, uv)
set -euo pipefail
SHA=54403a04390fc4a0d4865357556e77c1ed6ad513
DEST=${1:?usage: build.sh DEST_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none https://github.com/larson-group/clubb_release.git "$DEST"
fi
cd "$DEST"
git checkout -q "$SHA"
# Apply the constants patch unless it is already fully applied; a partial or
# conflicting state makes `git apply` fail and stops the build (set -e).
if ! git apply --reverse --check "$HERE/constants_legoesm.patch" 2>/dev/null; then
  git apply "$HERE/constants_legoesm.patch"
fi
git diff --stat
python3 compile.py -python
ls install/latest/python
