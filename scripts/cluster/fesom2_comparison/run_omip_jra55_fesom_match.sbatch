#!/bin/bash
# legoESM OMIP-2 ocean+ice forced by the SAME JRA55-do v1.4.0 fields and the
# SAME PHC3.0 winter initial condition that FESOM2's CORE2 hindcast uses
# (arXiv:2606.11356, arXiv:2608.01546).
#
# What is matched to FESOM2 (namelists in /work/ab0995/a270301/fesom2/config):
#   - forcing dataset and files (JRA55-do v1.4.0, 3-hourly, DKRZ pool)
#   - initial condition (phc3.0_winter.nc, in-situ -> potential temperature)
#   - vertical mixing scheme (KPP)
#   - level COUNT and total depth (47 levels, 6250 m)
#   - bottom drag coefficient: already matched WITHOUT a flag.  The bathy
#     lat-lon config sets bottom_drag_r = 2.5e-3 (run_omip.py:1161), equal to
#     FESOM's namelist.oce C_d = 0.0025.  Do NOT pass --bottom-drag-cd0: it is
#     read only when --bottom-drag-scheme != legacy (_apply_drag_iwm_overrides,
#     run_omip.py:241), so on the default lane it is a silent no-op.
#   - timestep (2400 s = FESOM's step_per_day = 36)
#   - bulk formulae family (NCAR / Large & Yeager) at 10 m for wind, T and q
#   - level PLACEMENT: --dz-ref-file feeds FESOM's exact 47-level thicknesses,
#     derived below from the mesh's own core2_zaxis.txt
#   - sea-ice rheology: EVP, 120 subcycles, P*=30000, e=2, C=20,
#     Delta_min=1e-11 (namelist.ice)
#   - river runoff is ROUTED to wet cells (500 km, FESOM's runoff_radius).
#     The run prints its own routed/unrouted budget at setup; see
#     docs/ocean/fidelity/fesom2_gap_analysis.md section 3.5 for the
#     measured loss with and without it.
#
# What is NOT matched, and cannot be without new code (see
# docs/ocean/fidelity/fesom2_gap_analysis.md):
#   - the mesh: 1 deg regular lat-lon here vs a 126858-vertex triangulation
#   - bathymetry (ETOPO on 1 deg vs FESOM's per-vertex depth.out)
#   - the GM closure: legoESM's diagnostic skew flux vs FESOM's Ferrari 2010
#     streamfunction BVP (oce_fer_gm.F90::fer_solve_Gamma).  Not implemented.
#
#   sbatch scripts/cluster/fesom2_comparison/run_omip_jra55_fesom_match.sbatch
#
#SBATCH --job-name=omip_fesom_match
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --account=bb1596
#SBATCH --output=%x_%j.log

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/levante-ocean
MAIN=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
SCRATCH=/scratch/b/b381103/fesom2_comparison
DAYS=${DAYS:-30}

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

cd "$REPO"

# FESOM's exact CORE2 level thicknesses, from the mesh's own z-axis file, so
# the vertical grid matches rather than merely having the same level count.
ZAXIS=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/core2_zaxis.txt
DZ=$SCRATCH/core2_dz.txt
"$REPO/.venv/bin/python" - "$ZAXIS" "$DZ" <<'EOF'
import sys
kv = {}
for line in open(sys.argv[1]):
    if "=" in line:
        k, v = line.split("=", 1)
        kv[k.strip()] = v.split()
lo = [float(x) for x in kv["lbounds"]]
up = [float(x) for x in kv["ubounds"]]
dz = [a - b for a, b in zip(lo, up)]
assert len(dz) == int(kv["size"][0]) and all(d > 0 for d in dz), dz
open(sys.argv[2], "w").write("\n".join(f"{d:.6f}" for d in dz) + "\n")
print(f"[levels] wrote {len(dz)} thicknesses, total {sum(dz):.1f} m")
EOF

"$REPO/.venv/bin/python" scripts/run/run_omip.py \
    --grid latlon \
    --resolution 180x360 \
    --nlev 47 \
    --dz-ref-file "$DZ" \
    --H-max 6250.0 \
    --dt 2400.0 \
    --days "$DAYS" \
    --precision fp64 \
    --bathymetry "$MAIN/data/bathymetry/etopo_1deg.nc" \
    --vertical-mixing-scheme kpp \
    --forcing-mode jra55_do_tropical \
    --jra55-cache "$SCRATCH/jra55_do_fesom_1958_1958_1deg_noleap.zarr" \
    --jra55-sea-ice \
    --ice-dynamics evp \
    --ice-n-evp 120 \
    --ice-p-star 30000 \
    --ice-e-yield 2.0 \
    --ice-c-strength 20.0 \
    --ice-delta-min 1e-11 \
    --runoff-routing nearest \
    --runoff-radius-km 500 \
    --woa-init \
    --woa-t "$SCRATCH/phc3_winter_woa_format.nc" \
    --woa-s "$SCRATCH/phc3_winter_woa_format.nc" \
    --output "$SCRATCH/omip_fesom_match_${DAYS}d" \
    --diag-every 36

echo "=== done ==="
