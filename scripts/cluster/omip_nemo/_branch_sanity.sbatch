#!/bin/bash
#SBATCH --job-name=brsanity
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/brsanity_%j.out
#
# THE TEST I SHOULD HAVE RUN FIRST.
#
# Six bisection arms all came back STILL NaN -- including one that dropped the
# ENTIRE GM/Redi set. When dropping every candidate still leaves the failure,
# the failure is not in the candidates. The house rule for this exact
# situation is "when a working configuration exists, diff against it FIRST".
#
# So: run the PLAIN configuration -- none of the adopted flags, nothing but
# grid/mesh/levels/timestep/IC/forcing, i.e. what the known-good
# trp_base2_d30 baseline ran -- on THIS BRANCH.
#
# PRE-REGISTERED:
#   NaN   => THE BRANCH IS BROKEN. My own commits today touched shared code
#            (the EOS S=0 adjoint guard, the GM/Redi dry-column AD guards, the
#            applicator signature, the monthly sampler). The adopted card is
#            then INNOCENT and the bisection was chasing the wrong thing.
#   CLEAN => the branch is fine and the blowup really is a card/config
#            interaction, so resume bisecting combinations rather than groups.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/brsanity_plain
echo "HEAD: $(git log --oneline -1)"

$PY -u scripts/run/run_omip_core2.py \
  --grid tripole \
  --mesh /burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc \
  --nlev 75 --dt 150 --years 0.00548 \
  --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --forcing-path /burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_mod \
  --output "$OUT" --snapshot-every-days 1 2>&1 | tail -25
echo "=== exit $? ==="

$PY -u - "$OUT" <<'PY'
import numpy as np, glob, os, sys
snaps = sorted(glob.glob(os.path.join(sys.argv[1], "snapshot_day*.npz")))
if not snaps:
    print("NO SNAPSHOT -- died before writing one"); raise SystemExit(1)
d = np.load(snaps[-1])
bad = {k: int(np.isnan(np.asarray(d[k], float)).sum())
       for k in ("T", "S", "u", "v") if k in d}
print(f"\n{os.path.basename(snaps[-1])} NaN counts: {bad}")
print("VERDICT:", "THE BRANCH IS BROKEN -- plain config NaNs, card is innocent"
      if sum(bad.values()) else "branch is CLEAN -- the blowup is card/config")
PY
echo DONE
