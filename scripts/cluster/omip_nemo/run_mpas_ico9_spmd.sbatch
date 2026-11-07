#!/bin/bash
#SBATCH --job-name=mpas_ico9
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/logs/mpas_ico9.%j.log
#
# MPAS-ocean at ico9 (2.62 M columns, ~15 km, 47 levels), JRA55-do forcing,
# thermodynamic sea ice, fp64, 16 GPUs with one process per GPU.
#
# TIMESTEP: 120 s, adopted 2026-09-11 from a measured step-length ladder at this
# resolution (60 / 120 / 150 / 180 s, three days each, everything else held
# identical and the diagnostics sampled at the same model time in every arm):
#
#   60 s   stable, the reference
#   120 s  stable; day-3 surface temperature differs from the 60 s run by
#          1.6e-3 K bias and 3.1e-3 K RMS against a field whose own spatial
#          standard deviation is 9.92 K, worst single cell 0.38 K; sea surface
#          height differs by 0.95 mm RMS.  0.43 -> 0.68 simulated years per day.
#   150 s  UNSTABLE: |u| 0.20 m/s at step 19, 4.5 m/s at step 38, dead at step 57
#   180 s  UNSTABLE: |u| 1.06 m/s by step 16, dead at step 32
#
# The failures are a sharp threshold rather than growing error, so the limit is
# a stability bound somewhere in (120, 150) s and 120 s carries roughly a 20%
# margin.  Raising it again needs the marginal feature identified first, not a
# larger number.
#
# The forcing input costs about a quarter of the wall time and is charged per
# model time, not per step, so the speed-up from the longer step saturates
# below the factor the step-count ratio suggests.
#
# Usage: sbatch [--export=ALL,DAYS=3,DT=120] run_mpas_ico9_spmd.sbatch
set -uo pipefail
DT=${DT:-120}
DAYS=${DAYS:-3}
# TKE (FESOM2's scheme) adopted 2026-09-14 for the cross-lane campaign: the KPP run lost
# 1.44 K of global SST in 30 days by draining the tropical top 50 m downward; the
# half-day one-variable ladder at ico6 gave KPP -0.136 K, TKE -0.065, constant-K -0.034.
VMIX=${VMIX:-tke}
# Prognostic TKE with the FESOM2 FORCA20 constants (user decision 2026-09-15):
# TKE carried on the state, Pr = clamp(6.6 Ri, 1, 10), constant Av/Kv
# backgrounds 1e-4/1e-5, no Bryan-Lewis floor, surface flux coefficient 3.75.
# The previous arm ran legoESM's diagnostic TKE (re-seeded each step).
TKE_CARD=${TKE_CARD:-fesom2}
# SSS restoring matched to FESOM2 (user decision 2026-09-15): piston 10 m / 60 d
# = 1.929e-6 m/s (was 5e-7), monthly PHC2 climatology target picked by the
# model calendar day (was the PHC3 winter slice of the initial condition), and
# the area-weighted global mean of the restoring tendency removed each step.
SSS_PISTON=${SSS_PISTON:-1.929e-6}
SSS_TARGET=${SSS_TARGET:-phc2_monthly}
SSS_MEAN=${SSS_MEAN:---sss-restoring-remove-mean}
# Shortwave as FESOM2: 0.54 of the post-albedo flux penetrates on the Sweeney-2005
# chlorophyll two-band profile (monthly climatology), open-water albedo 0.1
# (CORE2 namelist.ice); the lanes' historical Jerlov-II/0.94/0.06 is
# --sw-penetration auto --ocean-albedo 0.06.
CHL_CLIM=${CHL_CLIM:-/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc}
OCEAN_ALBEDO=${OCEAN_ALBEDO:-0.1}
# Bottom drag as FESOM2: quadratic C_d 0.0025 with no background kinetic energy
# (the lanes' legacy law is r=0.001 over u_bg=0.1 -> asymptotic C_d 0.01).  The
# 100 m BBL spreading stays: FESOM applies its drag implicitly at the bottom
# cell, our explicit drag on a 0.3 m partial cell would sign-reverse u.
BOTTOM_CD=${BOTTOM_CD:-0.0025}
# River runoff as FESOM2: the constant CORE2 climatology in place of the JRA55-do
# friver records (both kg m-2 s-1; delivered where the product puts it).
RUNOFF_FILE=${RUNOFF_FILE:-/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/CORE2_runoff.nc}
# Lateral dissipation matched to the FESOM2 reference (user decision 2026-09-15):
# no GM/Redi (FESOM runs none at this resolution), no Laplacian viscosity, no
# Laplacian Smagorinsky; a constant biharmonic B_h = gamma0 * h^3 with FESOM's
# gamma0 = 0.003 m/s at this mesh's spacing h ~ 14 km (ico9 mean cell area
# 1.95e8 m^2) = 1e10 m^4/s, plus a biharmonic Smagorinsky C_smag = sqrt(gamma1)
# = sqrt(0.1) = 0.316, the coefficient FESOM's flow-aware term gamma1*|du|*len*h^2
# reduces to under legoESM's B = C^2 * Delta^4 * |D| convention.  FESOM's
# quadratic gamma2 term has no legoESM analogue and is dropped.  The dx^3-scaled
# biharmonic vorticity damping K_zeta is OFF (user decision 2026-09-15): the
# model-derived dx^3 value is ~2e11 m^4/s at ico9, 20x the FESOM-scaled B_h, so
# leaving it on would make the lane's biharmonic dissipation non-FESOM.
K_ZETA=${K_ZETA:-0}
B_H=${B_H:-1.0e10}
C_SMAG=${C_SMAG:-0.316}
# Diagnostic cadence held at a fixed MODEL time (2880 s) so runs at different
# timesteps stay comparable; it also sets the forcing block length, and a block
# much longer than this exhausts an 80 GB card at this resolution.
DIAG=${DIAG:-$(python3 -c "print(max(1, round(2880.0 / $DT)))")}
WT=${LEGOESM_REPO:?export LEGOESM_REPO=<legoESM checkout> (sbatch runs a spool copy, so $0 cannot locate it)}
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
PP="$WT/src"; for p in atmosphere core coupler ice land ml ocean tools; do PP="$PP:$WT/packages/$p"; done
export PYTHONPATH=$PP
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/mesh_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
F=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/forcing
# Initial state: FESOM2-JAX's own cached FORCA20 field (user decision 2026-09-15),
# so every lane starts from the reference run's state; the PHC3 per-level donor
# fill fabricated water in isolated deep basins on the 1/12-degree mesh.
FESOM_MESH=${FESOM_MESH:-/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20}
OUT=${OUT:-/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/mpas_${RES:-ico9}/j$SLURM_JOB_ID}
mkdir -p "$OUT"; cd "$WT"
NGPU=$(( SLURM_JOB_NUM_NODES * 4 ))
echo "=== MPAS ico9: nodes=$SLURM_JOB_NODELIST gpus=$NGPU dt=$DT days=$DAYS diag_every=$DIAG $(date) ==="
t0=$(date +%s)
srun --ntasks=$NGPU --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 \
  --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/run/run_omip.py --grid mpas --resolution "${RES:-ico9}" --mpas-lloyd 0 \
    --nlev 47 --dt "$DT" --days "$DAYS" --precision fp64 \
    --bathymetry "$F/ETOPO_2022_v1_60s_N90W180_surface.nc" \
    --forcing-mode jra55_do_tropical --jra55-cache "$F/jra55_do_fesom_1958_1958_1deg_noleap.zarr" \
    --jra55-sea-ice --ice-dynamics none --ice-categories 1 --frazil --no-gpu-interp \
    --woa-init --ic-cache-dir /work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/ic_cache --ic-from-fesom-mesh "$FESOM_MESH" --mpas-k-zeta-bih "$K_ZETA" \
    --vertical-mixing-scheme "$VMIX" --tke-card "$TKE_CARD" --diag-every "$DIAG" \
    --no-gm-redi --A-h 0 --C-smag-lap 0 --B-h "$B_H" --C-smag "$C_SMAG" \
    --sss-piston-velocity "$SSS_PISTON" --sss-restoring-target "$SSS_TARGET" $SSS_MEAN \
    --sw-penetration sweeney_2band --chl-clim "$CHL_CLIM" --ocean-albedo "$OCEAN_ALBEDO" \
    --bottom-drag-scheme nemo_quadratic --bottom-drag-cd0 "$BOTTOM_CD" --bottom-drag-ke0 0.0 \
    --runoff-source core2_climatology --runoff-file "$RUNOFF_FILE" \
    --enable-mpas-spmd --multicontroller \
    --output "$OUT" 2>&1 | tee "$OUT/run.log" | grep -vE "^WARNING|^I0000|^E0000" | tail -70
rc=${PIPESTATUS[0]}
echo "run_rc=$rc wall_s=$(( $(date +%s) - t0 ))"
echo "=== DONE $(date) ==="; exit $rc
