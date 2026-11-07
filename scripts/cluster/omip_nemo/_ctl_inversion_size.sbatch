#!/bin/bash
#SBATCH --job-name=ctlinv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ctlinv_%j.out
#
# HOW BIG ARE THE INSTABILITIES THE TRIGGER FIRES ON?  (GLM CRITICAL-2.)
# The trigger fires wherever N2 <= -1e-12, and at dz ~ 1 m that threshold is an
# inversion of order 1e-10 K: delivering 100 m2/s to an already-neutral column
# moves no heat, and the firing fraction would then be a flattering instrument
# rather than evidence the mechanism was exercised.  Convert the firing points'
# N2 to the temperature inversion they represent, dT = |N2| dz / (g alpha), and
# report the distribution.  p99 well under ~0.05 K means the arm tested
# DELIVERY, not EROSION.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
$PY - <<'PY2'
import numpy as np
from legoesm import constants
from legoesm.ocean.eos import (compute_buoyancy_frequency_nemo_bn2,
                               nemo_bn2_depth_ladders)
E = ("/burg-archive/glab/users/pg2328/_base2_wt/results/omip_nemo/"
     "trp_base2_d30/snapshot_day0015.npz")
z = np.load(E)
T = np.asarray(z["T"])[..., :4]; S = np.asarray(z["S"])[..., :4]
wet = np.asarray(z["land_mask"]) > 0.5
lat = np.asarray(z["lat_T"]); lon = np.asarray(z["lon_T"]) % 360.0


class _Z:                      # the reference ladders, as the model uses them
    z_full_ref = np.asarray(z["z_center_ref"]) if "z_center_ref" in z else None
    z_half_ref = None


zc = np.abs(np.asarray(z["z_interface_ref"]))       # interior interfaces
gdept = np.abs(np.asarray(z["z_center_ref"])) if "z_center_ref" in z else None
if gdept is None:
    raise SystemExit("snapshot carries no t-point ladder")
N2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
    T, S, gdept[:4], zc[:3], eos_form="teos10", e3w_source="depth_difference"))
dz = np.diff(gdept[:4])                              # centre-to-centre spacing
alpha = 3.0e-4                                       # K^-1, tropical surface
for k in range(3):
    n2 = N2[..., k][wet]
    fire = n2 <= -1e-12
    dT = np.abs(n2[fire]) * dz[k] / (constants.g * alpha)
    print(f"interface {k} ({gdept[k]:.2f}->{gdept[k+1]:.2f} m): "
          f"firing {fire.mean():.4f} of wet columns, n={fire.sum()}")
    if fire.any():
        q = np.percentile(dT, [50, 90, 99, 100])
        print(f"   inversion dT [K]  median {q[0]:.2e}  p90 {q[1]:.2e}  "
              f"p99 {q[2]:.2e}  max {q[3]:.2e}")
        print(f"   fraction of firing points with dT > 0.01 K: "
              f"{(dT > 0.01).mean():.4f};  > 0.05 K: {(dT > 0.05).mean():.4f}")
box = wet & (np.abs(lat) <= 2) & (lon >= 220) & (lon < 240)
n2b = N2[..., 0][box]; fb = n2b <= -1e-12
print(f"\ncold-tongue box at this hour (00Z = mid-afternoon local): "
      f"firing {fb.mean():.4f} of {box.sum()} columns")
if fb.any():
    dTb = np.abs(n2b[fb]) * dz[0] / (constants.g * alpha)
    print(f"   inversion dT [K]  median {np.median(dTb):.2e}  max {dTb.max():.2e}")
PY2
echo DONE
