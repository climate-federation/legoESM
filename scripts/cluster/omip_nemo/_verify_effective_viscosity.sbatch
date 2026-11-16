#!/bin/bash
#SBATCH --job-name=effvisc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/effvisc_%j.out
#
# WHAT LATERAL VISCOSITY IS THE RUNNING JOB ACTUALLY APPLYING?
#
# The production card ends in "| tail -40", so the driver's own
# [visc-schedule] line -- which now prints the effective A_h range -- is
# buffered and discarded: the schedule segments fire at days 0, 8 and 16,
# all far from the last forty lines of an eleven-hour leg.  The runtime
# evidence is therefore unreachable from the log, and "the code is in the
# tree" is a claim about the tree, not about the run.
#
# So reproduce step 1 exactly: take the config the RUN ITSELF recorded in its
# manifest, and push it through the renormalisation function from the RUN
# TREE's own driver (not this checkout's -- the two trees have diverged by
# 422 commits and only the run tree's copy can testify about the run).
#
# Deliberately NOT re-deriving the arithmetic here: a hand-rolled copy of the
# formula would prove only that I can multiply.  The point is to execute the
# function the job executes.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_omip_wt
cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithvisc_d30}

echo "=== run tree HEAD $(git -C $WT log --oneline -1) ==="
$PY -u - "$OUT" <<'PY'
import json, sys
from pathlib import Path

from legoesm.ocean.state import LatLonCGridOceanConfig
from scripts.run.run_omip_core2 import renormalise_ah_profile

out = Path(sys.argv[1])
man = json.loads((out / "run_manifest.json").read_text())
seg = json.loads((out / "visc_schedule.json").read_text())["segments_day_Ah_Csmaglap"]


def _find(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                yield v
            yield from _find(v, key)


lv_rec = next(_find(man, "lateral_viscosity"))
prof = tuple(lv_rec["A_h_lat_profile"])
lv = LatLonCGridOceanConfig().replace_flat(
    A_h=lv_rec["A_h"], A_h_lat_profile=prof,
    A_h_lat_profile_v=lv_rec["A_h_lat_profile_v"]).lateral_viscosity

print(f"manifest setup: A_h={lv.A_h:g}, profile {min(prof):.4f}-{max(prof):.4f}, "
      f"A_h_floor={lv_rec['A_h_floor']:g}, A_h_lat_scaling={lv_rec['A_h_lat_scaling']}")
eff = [lv.A_h * p for p in prof]
print(f"setup effective A_h {min(eff):9.4g} - {max(eff):9.4g} m2/s   "
      "(equals the file by construction)")

# Walk the segments the run will walk, through the run tree's own function.
cur = lv
for day, ah, cs in seg:
    cur = renormalise_ah_profile(cur, ah)._replace(C_smag_lap=cs)
    e = [cur.A_h * p for p in cur.A_h_lat_profile]
    print(f"day {day:>4}: A_h knob {ah:>8g}  C_smag {cs:<4g} -> "
          f"effective A_h {min(e):9.4g} - {max(e):9.4g} m2/s")

# NEMO's own field, read straight from the file, is the number to match.
import netCDF4 as nc4
import numpy as np
f = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/"
     "orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc")
ds = nc4.Dataset(f)
a = np.asarray(ds.variables["ahmf_3d"][:]).squeeze()[0]
ds.close()
a = a[a > 0]
print(f"NEMO eddy_viscosity_3D surface level: {a.min():9.4g} - {a.max():9.4g} m2/s")

final = [cur.A_h * p for p in cur.A_h_lat_profile]
ok = abs(max(final) - a.max()) / a.max() < 1e-6
print(f"VERDICT: final segment {'MATCHES' if ok else 'DOES NOT MATCH'} NEMO's "
      f"field at the profile maximum ({max(final):.6g} vs {a.max():.6g})")
PY
echo "=== rc=$? ==="
