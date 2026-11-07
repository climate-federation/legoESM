#!/bin/bash
#SBATCH --job-name=sfctke
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sfctke_%j.out
#
# WHERE DOES THE ENERGY STOP ARRIVING?  Our turbulent layer ends at 17.9 m
# where the oracle's reaches 73.2 m, and the momentum/tracer partition is
# faithful wherever we have any turbulence -- so what is missing is ENERGY,
# not the way it is divided.  Energy enters this column at the surface, as
# NEMO's Dirichlet condition en(1) = MAX(rn_emin0, rn_ebb*|tau|/rho0).
#
# PRE-REGISTERED, one question, two outcomes:
#   * our surface TKE MATCHES rn_ebb*|tau|/rho0 for our own stress (within
#     ~20%) -> the source is faithful and the loss is downstream, in transport
#     or dissipation.  The carried-coefficient arm acts downstream, so this
#     outcome SUPPORTS it.
#   * our surface TKE is far below that value -> the boundary condition itself
#     is the starvation point and no downstream lever can fix it.
# Either way this is read off a saved snapshot; it costs no simulation.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
SNAP=${SNAP:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_base2_d30/snapshot_day0030.npz}
export SNAP
$PY - <<'PY2'
import os
import numpy as np
from legoesm.ocean.physics.vertical_mixing.tke import _NEMO_TKE_EBB
from legoesm import constants

SNAP = os.environ["SNAP"]
z = np.load(SNAP)
print(f"[input] {SNAP}")
keys = set(z.files)
print("[keys] tke present:", "tke" in keys,
      "| stress keys:", sorted(k for k in keys if "tau" in k.lower()
                               or "stress" in k.lower()))
lat = np.asarray(z["lat_T"]); lon = np.asarray(z["lon_T"]) % 360.0
wet = np.asarray(z["land_mask"]) > 0.5
box = wet & (np.abs(lat) <= 2.0) & (lon >= 220.0) & (lon < 240.0)
print(f"[box] {box.sum()} cold-tongue columns")

if "tke" not in keys:
    raise SystemExit("SNAPSHOT CARRIES NO TKE FIELD: this question cannot be "
                     "answered from it. Re-run the arm with the prognostic "
                     "TKE state saved, or read it from a restart file.")

e = np.asarray(z["tke"])[box]           # (ncol, nlev-1), interior W interfaces
zi = np.abs(np.asarray(z["z_interface_ref"]))
print(f"[tke] top interior interface at {zi[0]:.2f} m; "
      f"median e there {np.nanmedian(e[:, 0]):.4e} m2/s2")
for d0 in (5.0, 20.0, 60.0):
    k = int(np.argmin(np.abs(zi - d0)))
    print(f"   at {zi[k]:6.2f} m : median e {np.nanmedian(e[:, k]):.4e} m2/s2")

# NEMO's Dirichlet surface value from OUR OWN stress, same formula, same rho0.
tau_key = next((k for k in ("tau_mod", "taum", "tau_magnitude", "wind_stress_mod")
                if k in keys), None)
if tau_key is None:
    tx = next((k for k in ("tau_x", "taux", "wind_stress_x") if k in keys), None)
    ty = next((k for k in ("tau_y", "tauy", "wind_stress_y") if k in keys), None)
    if tx is None or ty is None:
        raise SystemExit(
            "no wind-stress field in the snapshot; the analytic Dirichlet "
            f"value cannot be formed. keys={sorted(keys)}")
    taum = np.hypot(np.asarray(z[tx]), np.asarray(z[ty]))[box]
    print(f"[stress] |tau| from {tx}/{ty}")
else:
    taum = np.asarray(z[tau_key])[box]
    print(f"[stress] |tau| from {tau_key}")

rho0 = constants.rho_ocean_ref if hasattr(constants, "rho_ocean_ref") else 1026.0
e_bc = _NEMO_TKE_EBB * taum / rho0
print(f"[bc] rn_ebb={_NEMO_TKE_EBB:g}  rho0={rho0:g}")
print(f"[bc] median |tau| {np.nanmedian(taum):.4e} N/m2 -> "
      f"analytic en(1) {np.nanmedian(e_bc):.4e} m2/s2")
ratio = np.nanmedian(e[:, 0]) / np.nanmedian(e_bc)
print(f"\nVERDICT INPUT: our top interior TKE / analytic surface Dirichlet "
      f"= {ratio:.3f}")
print("  (the top INTERIOR interface sits below the z=0 row the Dirichlet "
      "value pins, so some decay is expected; a ratio near 1 means the "
      "source is arriving, a ratio of 1e-2 or less means it is not.)")
PY2
echo DONE
