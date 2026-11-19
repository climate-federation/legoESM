#!/bin/bash
# Build the clubb_release oracle with qclvar exposed (patch_qclvar_out.py) on
# top of the clubb_core build (same SHA, legoESM constants patch).
# Usage: build_oracle.sh DEST_DIR   (compute node; gfortran, cmake, uv)
set -euo pipefail
DEST=${1:?usage: build_oracle.sh DEST_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(grep -oP '^SHA=\K\w+' "$HERE/../clubb_core/build.sh")
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none https://github.com/larson-group/clubb_release.git "$DEST"
fi
cd "$DEST"
git checkout -q "$SHA"
git diff --quiet || { echo "DEST has local edits; use a fresh directory" >&2; exit 1; }
git apply "$HERE/../clubb_core/constants_legoesm.patch"
python3 "$HERE/patch_qclvar_out.py" "$DEST"
git diff --stat
python3 compile.py -python
ls install/latest/python
