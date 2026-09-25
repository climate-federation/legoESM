#!/bin/bash
#SBATCH --job-name=3wayreg
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/3wayreg_%j.out
#
# IS THE REPORTED SST/MLD REGRESSION REAL? (user, 2026-09-11)
#
# The three snapshots behind the 2026-09-09 figures (trp/mpas/fesom_base2_d30)
# are UNCHANGED on disk since 2026-09-06, and no run since has written them.
# So this re-runs the SAME instrument on the SAME states: if the numbers come
# back identical to the stored 2026-09-09 report.json, the model has not moved
# and the apparent regression is an instrument/plot difference, not physics.
# If they differ, the instrument itself changed under us and THAT is the
# finding. Either way this is a measurement, not an argument.
#
# Pre-registered: tripole-vs-NEMO global SST rmse 0.4664 / bias +0.0641
# (2026-09-09, threegrid_d30_mpas). Identical => no regression.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

echo "=== input provenance (mtimes prove whether the states moved) ==="
ls -l --time-style=+%Y-%m-%d_%H:%M $R/trp_base2_d30/snapshot_day0030.npz \
   $R/mpas_base2_d30/snapshot_day0030.npz $R/fesom_base2_d30/snapshot_day0030.npz

for pair in "mpas MPAS" "fesom FESOM2"; do
  set -- $pair
  echo; echo "########## tripole vs $2 ##########"
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/trp_base2_d30/snapshot_day0030.npz \
    --mpas    $R/$1_base2_d30/snapshot_day0030.npz \
    --label-tripole tripole --label-mpas "$2" \
    --nemo-gridt "$GW" --nemo-time-idx 5 --res-deg 1.0 \
    --out-dir $ROOT/results/omip_nemo/threeway_recheck_$1 2>&1 | tail -30
done

echo; echo "########## DELTA vs the stored 2026-09-09 report ##########"
$PY - <<'PY2'
import json, pathlib
R = pathlib.Path("/burg-archive/glab/users/pg2328/_omip_wt/results/omip_nemo")
for old, new, lbl in (("threegrid_d30_mpas", "threeway_recheck_mpas", "MPAS"),
                      ("threegrid_d30_fesom", "threeway_recheck_fesom", "FESOM2")):
    po, pn = R / old / "report.json", R / new / "report.json"
    if not pn.exists():
        print(f"{lbl}: no new report at {pn}"); continue
    o, n = json.load(open(po)), json.load(open(pn))
    print(f"\n--- {lbl}: 2026-09-09 vs now ---")
    for fld in ("SST", "SSS", "MLD"):
        # codex 9701531 [MEDIUM]: this loop hard-coded "mpas_vs_nemo", but the
        # report keys the second model by its LABEL ("MPAS_vs_nemo",
        # "FESOM2_vs_nemo"), so every companion score was silently skipped and
        # only the tripole row was ever compared. Take the sides from the
        # report itself, and fail loudly if a side present then is missing now.
        sides = [k for k in o["fields"][fld] if k != "unit"]
        assert sides, f"{lbl} {fld}: no comparison sides in the stored report"
        for side in sides:
            go = o["fields"][fld][side].get("global")
            gn = n["fields"][fld].get(side, {}).get("global")
            if go and not gn:
                print(f"  {fld:4s} {side:18s} MISSING from the new report")
                continue
            if not go:
                continue
            for m in ("bias", "rmse"):
                d = gn[m] - go[m]
                flag = "  <== MOVED" if abs(d) > 1e-9 else ""
                print(f"  {fld:4s} {side:18s} {m:5s}  "
                      f"{go[m]:+.6f} -> {gn[m]:+.6f}  delta {d:+.2e}{flag}")
PY2
echo DONE
