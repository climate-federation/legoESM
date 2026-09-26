#!/bin/bash
#SBATCH --job-name=nanbis
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nanbis_%j.out
#
# THE ADOPTED CARD BLOWS UP: every velocity is NaN by day 5 and every wet T/S
# cell is NaN. Confirmed against a known-good reference snapshot read through
# the SAME probe, which came back clean (0 NaN, -2.13..30.88 C).
#
# Five things moved at once, which is exactly why a multi-change baseline
# cannot attribute anything -- so bisect with ONE-VARIABLE arms.
#
# DROP names which group to REMOVE from the card:
#   ocean  - the whole GM/Redi + operator set
#   msck33 - only --gm-msc-stabilize (and with it the implicit_K33 pairing),
#            the newest change and one that was NEVER RUNNABLE before today
#   ice    - the SI3 exchange set and bulk snow
#   fchl   - linear forcing time interpolation and the SCRIP chlorophyll
#
# PRE-REGISTERED: 2 days is ample -- the full card is already all-NaN at day 5.
#   CLEAN (no NaN in u, v, T, S) => the dropped group CAUSED the blowup.
#   STILL NaN                    => that group alone does not explain it.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
DROP=${DROP:-none}
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nanbis_${DROP}
echo "HEAD: $(git log --oneline -1)"; echo "DROP=$DROP -> $OUT"

OCEAN="--gm-treguier --gm-kappa-min 0 --gm-slope-scheme nemo_iso_lap --gm-slope-positions nemo_native --gm-bolus-advection through_fct --gm-msc-stabilize --redi-coefficient nemo21 --redi-aht0 900"
ICE="--ice-exchange nemo_si3 --ice-snow bulk --ice-snow-flooding on"
FCHL="--forcing-time-interp linear --chl-remap nemo_scrip"
# Added after BOTH first arms came back STILL NaN at DAY 1: the cause is not
# the ocean set and not the msc/K33 pair, so it must be something every arm
# HELD FIXED. The two candidates are the prognostic ice itself and the
# freezing scheme -- and `unesco` was MY judgement call when building the
# card, which makes it exactly the kind of choice to test rather than trust.
FIXEDICE="--prognostic-sea-ice --ice-freeze-scheme unesco"
case "$DROP" in
  freeze) FIXEDICE="--prognostic-sea-ice" ;;
  noice)  FIXEDICE="" ;;
  ocean)  OCEAN="" ;;
  msck33) OCEAN="--gm-treguier --gm-kappa-min 0 --gm-slope-scheme nemo_iso_lap --gm-slope-positions nemo_native --gm-bolus-advection through_fct --redi-coefficient nemo21 --redi-aht0 900" ;;
  ice)    ICE="" ;;
  fchl)   FCHL="" ;;
esac

$PY -u scripts/run/run_omip_core2.py \
  --grid tripole \
  --mesh /burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc \
  --nlev 75 --dt 150 --years 0.00548 \
  --woa-init \
  --forcing-path /burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_mod \
  $FIXEDICE \
  $OCEAN $ICE $FCHL \
  --output "$OUT" --snapshot-every-days 1 2>&1 | tail -25
echo "=== exit $? ==="

$PY -u - "$OUT" <<'PY'
import numpy as np, glob, os, sys
snaps = sorted(glob.glob(os.path.join(sys.argv[1], "snapshot_day*.npz")))
if not snaps:
    print("NO SNAPSHOT -- run died before writing one"); raise SystemExit(1)
d = np.load(snaps[-1])
bad = {k: int(np.isnan(np.asarray(d[k], float)).sum())
       for k in ("T", "S", "u", "v") if k in d}
print(f"\n{os.path.basename(snaps[-1])}  NaN counts: {bad}")
print("VERDICT:", "CLEAN -- the dropped group CAUSED the blowup"
      if sum(bad.values()) == 0 else
      "STILL NaN -- that group alone does not explain it")
PY
echo DONE
