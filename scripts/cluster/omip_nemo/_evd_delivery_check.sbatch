#!/bin/bash
#SBATCH --job-name=evddeliv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/evddeliv_%j.out
# DELIVERY CHECK (the day-15 null demands it): the occupancy instrument proves
# the TRIGGER fires, not that the convective coefficient reaches the solve.
# The snapshots carry K_H_diag, the diffusivity the implicit solve consumes.
# In the cold tongue at 00Z the column is stable, so no EVD is EXPECTED there;
# the decisive comparison is the GLOBAL distribution -- if the arm never shows
# cells at the convective value where the control does not, the coefficient is
# not being delivered and the null is plumbing, not physics.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
$PY - <<'PY2'
import numpy as np
B="/burg-archive/glab/users/pg2328/_base2_wt/results/omip_nemo/trp_base2_d30/snapshot_day0015.npz"
E="/burg-archive/glab/users/pg2328/_evd_wt/results/omip_nemo/trp_evd_d30/snapshot_day0015.npz"
for tag, p in (("control", B), ("evd arm", E)):
    z = np.load(p)
    if "K_H_diag" not in z:
        print(f"{tag}: NO K_H_diag in the snapshot"); continue
    K = np.asarray(z["K_H_diag"]); wet = np.asarray(z["land_mask"]) > 0.5
    k0 = K[..., 0][wet]
    k0 = k0[np.isfinite(k0)]
    print(f"{tag}: first interface over {k0.size} wet columns "
          f"median {np.median(k0):.3e}  p99 {np.percentile(k0,99):.3e}  max {k0.max():.3e}")
    for thr in (1.0, 10.0, 50.0):
        print(f"    fraction >= {thr:5.0f} m2/s : {(k0>=thr).mean():.5f}")
    K3 = K[..., :3][wet]
    print(f"    any of the top 3 >= 50 m2/s : {(K3>=50).any(axis=-1).mean():.5f}")
PY2
echo DONE
