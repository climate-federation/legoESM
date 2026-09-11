#!/bin/bash
#SBATCH --job-name=maskdiff
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/maskdiff_%j.out
#
# QUANTIFY CODEX'S LAND-MASK MAJOR.
#
# codex: "our mesh includes wet cells that are dry in the supplied oracle
# domain, preventing same-domain reproduction". That is a claim with a number
# attached, and the number has never been taken. It matters TWICE: it bounds
# how exact the SCRIP remapping can be, AND it affects every score we compute
# against the GATEWAY file, because a cell wet on one side and dry on the
# other is not a comparison.
#
# The oracle's own mesh_mask files in EXP00 are PER-MPI-RANK and would need
# stitching; the GATEWAY grid_T output is already on the full grid and is the
# exact file the scorer uses, so it is the right mask to compare against.
#
# PRE-REGISTERED, no threshold to game -- this is a census, reported either way:
#   * cells OUR mesh calls ocean that the oracle leaves missing;
#   * cells the oracle reports that our mesh calls land;
#   * both as a fraction of our wet points, and by latitude band so it is
#     clear whether the disagreement is coastal, polar, or everywhere.
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -u - <<'PY'
import numpy as np, xarray as xr
GW = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/"
      "RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc")
MESH = "/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc"

g = xr.open_dataset(GW, decode_times=False)
sst_name = [v for v in g.data_vars if v.lower() in ("tos", "sosstsst", "thetao")]
print("grid_T vars:", list(g.data_vars)[:10])
assert sst_name, "no SST variable found"
sst = np.asarray(g[sst_name[0]])
while sst.ndim > 2:
    sst = sst[0]
print(f"oracle SST field {sst.shape}")

m = xr.open_dataset(MESH, decode_times=False)
tmask = np.asarray(m["tmask"]).squeeze()
surf = tmask[0] if tmask.ndim == 3 else tmask
print(f"our tmask surface {surf.shape}")
# ALIGNMENT MATTERS AND A NAIVE TRUNCATION IS WRONG. The oracle's 331x360
# fields are the INTERIOR of our 332x362 grid at python [0:331, 1:361] -- that
# is the ncks slice recorded in the weight files' own history, not a guess.
# My first version of this probe truncated to [:, 0:360] instead, a one-column
# shift that INFLATED the apparent mask disagreement. An off-by-one here does
# not error, it just makes every coastline disagree.
if surf.shape != sst.shape:
    assert surf.shape == (332, 362) and sst.shape == (331, 360), (
        surf.shape, sst.shape)
    surf = surf[0:331, 1:361]

ours_wet = surf > 0.5
oracle_wet = np.isfinite(sst) & (np.abs(sst) > 1e-12) & (np.abs(sst) < 1e20)
print(f"\nour wet cells:    {ours_wet.sum()}")
print(f"oracle wet cells: {oracle_wet.sum()}")
only_ours = ours_wet & ~oracle_wet
only_oracle = oracle_wet & ~ours_wet
print(f"\nWET FOR US, DRY FOR THE ORACLE: {only_ours.sum()} "
      f"({100*only_ours.sum()/max(ours_wet.sum(),1):.3f}% of our wet cells)")
print(f"WET FOR THE ORACLE, DRY FOR US: {only_oracle.sum()} "
      f"({100*only_oracle.sum()/max(oracle_wet.sum(),1):.3f}% of its wet cells)")

lat = np.asarray(m["gphit"]).squeeze()[0:331, 1:361]
print("\nby latitude band (disagreeing cells):")
for lo, hi, label in ((-90, -60, "Antarctic"), (-60, -30, "S midlat"),
                      (-30, 30, "tropics"), (30, 60, "N midlat"),
                      (60, 91, "Arctic")):
    band = (lat >= lo) & (lat < hi)
    d = (only_ours | only_oracle) & band
    w = ours_wet & band
    if w.sum():
        print(f"  {label:10s} {d.sum():6d} disagree of {w.sum():6d} wet "
              f"({100*d.sum()/w.sum():5.2f}%)")
PY
echo DONE
