#!/usr/bin/env bash
# Regenerate the phase-4a c_sw one-step oracle fixture end-to-end.
# PRODUCTION configuration: no -DSW_DYNAMICS (ptc transports pt), no
# OVERLOAD_R4 (sw_core big_number = 1.E30).  Compile in a FRESH build dir
# (stale .mod mixing across shim revisions has produced -O2 segfaults).
#
# Usage: gen_swcore_oracle.sh [workdir] [res] [ng]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
W=${1:-$(mktemp -d)}; RES=${2:-12}; NG=${3:-3}
FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none"
PY=${PYTHON:-python3}

mkdir -p "$W" && cd "$W"
PYTHONPATH="$REPO/packages/core:$REPO/src" "$PY" \
    "$HERE/export_swcore_inputs.py" --res "$RES" --ng "$NG" --outdir "$W"
gfortran $FF -J "$W" -c "$HERE/fv3_swcore_shim.F90"    -o "$W/shim.o"
gfortran $FF -J "$W" -c "$HERE/fv3_swcore_extract.F90" -o "$W/ext.o"
gfortran $FF -J "$W" "$HERE/fv3_swcore_oracle_driver.F90" \
    "$W/shim.o" "$W/ext.o" -o "$W/oracle"
"$W/oracle"
PYTHONPATH="$REPO/packages/core:$REPO/src" "$PY" - "$W" "$RES" "$NG" <<'PYEOF'
import sys
import numpy as np
w, res, ng = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lo = 1 - ng
shapes = {"DELPC": (res+2*ng, res+2*ng), "PTC": (res+2*ng, res+2*ng),
          "UA": (res+2*ng, res+2*ng), "VA": (res+2*ng, res+2*ng),
          "UT": (res+2*ng, res+2*ng), "VT": (res+2*ng, res+2*ng),
          "UC": (res+2*ng+1, res+2*ng), "VC": (res+2*ng, res+2*ng+1),
          "DIVG_D": (res+2*ng+1, res+2*ng+1)}
out = {k: np.full(v, np.nan) for k, v in shapes.items()}
for line in open(f"{w}/swcore_output.txt"):
    if line.startswith("#"):
        continue
    p = line.split()
    out[p[0]][int(p[1]) - lo, int(p[2]) - lo] = float(p[3])
np.savez_compressed(f"{w}/swcore_oracle_c{res}.npz",
                    **{k.lower(): v for k, v in out.items()},
                    res=res, ng=ng, dt2=112.5, nord=1)
print(f"fixture: {w}/swcore_oracle_c{res}.npz  "
      f"(install into tests/grids/fixtures/ alongside swcore_input.npz)")
PYEOF
