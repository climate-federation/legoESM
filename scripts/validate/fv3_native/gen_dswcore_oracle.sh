#!/usr/bin/env bash
# Regenerate the phase-4b d_sw one-step oracle fixture end-to-end.
# PRODUCTION config: no -DSW_DYNAMICS/ONE_SIDE/ROT3, sw_core big_number
# = 1.E30.  Fresh build dir (stale .mod mixing has produced -O2 segfaults).
# Usage: gen_dswcore_oracle.sh [workdir] [res] [ng]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
W=${1:-$(mktemp -d)}; RES=${2:-12}; NG=${3:-3}
FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none"
PY=${PYTHON:-python3}

mkdir -p "$W" && cd "$W"
PYTHONPATH="$REPO/packages/core:$REPO/src" "$PY" \
    "$HERE/export_dswcore_inputs.py" --res "$RES" --ng "$NG" --outdir "$W"
gfortran $FF -J "$W" -c "$HERE/fv3_swcore_shim.F90"    -o "$W/shim.o"
gfortran $FF -J "$W" -c "$HERE/fv3_swcore_extract.F90" -o "$W/ext.o"
gfortran $FF -J "$W" -c "$HERE/fv3_dswcore_extract.F90" -o "$W/dsw.o"
gfortran $FF -J "$W" "$HERE/fv3_dswcore_oracle_driver.F90" \
    "$W/shim.o" "$W/ext.o" "$W/dsw.o" -o "$W/oracle"
"$W/oracle"
# build the oracle npz on the python-return array origins
"$PY" "$REPO/tests/grids/fixtures/build_dsw_fixture.py" \
    "$W/dswcore_output.txt" "$W/dswcore_oracle_c12.npz" "$RES" "$NG"
echo "fixtures: $W/dswcore_input.npz + $W/dswcore_oracle_c12.npz"
echo "install into tests/grids/fixtures/ to update the committed pair"
