#!/bin/bash
#SBATCH --job-name=tkeabl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkeabl_%j.out
#
# ============================================================================
# WHICH TERM MAKES OUR TKE 3-4x NEMO'S IN ONE STEP?  One variable per arm.
# ============================================================================
#
# ESTABLISHED (job 9404770, same instrument, NEMO's own state and `en`):
#   zero steps   K_M / NEMO avm_k = 0.76-0.95   (we start slightly WEAK)
#   one 3600 s step, vs NEMO avm (EVD-free, Prandtl-free: nn_evdm=0):
#       band          e_ratio(all)  e_ratio(active)  f_active  K_M/avm
#       antarctic         3.255         3.205         0.031     1.755
#       SH midlat         2.825         2.758         0.026     1.675
#       tropics           4.035         3.896         0.013     1.659
#       NH midlat         3.982         3.960         0.029     2.239
#       arctic            4.063         4.036         0.053     1.634
#   The two e-ratios agree, so this is NOT the rn_emin floor talking, and the
#   K ratio tracks sqrt(e_ratio) — the length is sweep-capped, not growing.
#
# SO A SOURCE TERM IS TOO LARGE.  Dissipation is ruled OUT as the cause by the
# same job: our lengths are 0.68-0.86x NEMO's on the dissl-covered set, and a
# SHORTER l_eps dissipates MORE, not less.  The remaining sources in the card
# are the NEMO surface Dirichlet condition, the Langmuir (lc) injection, and
# the near-inertial (etau) injection — plus the dissipation length choice,
# kept in as a control that should move the answer the WRONG way.
#
# ARMS, each ONE field off the card, all scored on the identical instrument:
#   0  baseline                     (the card)
#   1  lc=False                     NEMO ln_lc off
#   2  etau_mode=none               NEMO nn_etau=0
#   3  surface_bc=veros_flux        the weaker pre-#1326 surface input
#   4  tke_mxl_choice=4             NEMO nn_mxl=2, l_eps=min -> MORE dissipation
#
# PRE-REGISTERED read on e_ratio(active), Antarctic band (baseline 3.205):
#   THE CULPRIT      : an arm brings it below 1.6 (more than half the excess).
#   CONTRIBUTOR      : 1.6 to 2.6.
#   NOT INVOLVED     : above 2.6, i.e. under ~20% of the excess removed.
# Arm 4 is a CONTROL, not a candidate: it can only ADD dissipation, so if it
# alone drops e_ratio below 1.6 while arms 1-3 do nothing, the reading is that
# our sources are right and NEMO's shorter dissipation length is doing the
# work — a different conclusion, and it must be reported as such.
#
# The probe REFUSES a no-op override (raises if the value equals the card's),
# so an arm cannot silently report "no effect" because the flag never applied.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
FULL=$TRD2/restart_8760_full.npz
TF=$TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc
OUT=$REPO/results/omip_nemo/tke_source_ablation
echo "HEAD: $(git -C $REPO log --oneline -1)"
[ -r "$FULL" ] || { echo "FATAL: no $FULL (run _zero_step_closure.sbatch first)"; exit 1; }
NFAIL=0
arm () {
  local tag="$1"; shift
  echo; echo "########## ARM $tag"
  $PY -u scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
    --restart-npz $FULL --mesh-mask "$MESH" \
    --trd-tfile $TF --trd-rec 0 \
    --out-json $OUT/$tag.json "$@" 2>&1 \
    | grep -aE "^\[cfg\]|^\[card\]|^\[C[123]\]|^band|^antarctic|^SH_mid|^tropics|^NH_mid|^arctic|^ONE STEP|FATAL|Error|error:|raise|ValueError|SystemExit"
  local rc=${PIPESTATUS[0]}
  echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1))
}

arm baseline
arm no_lc            --cfg-override lc=False
arm no_etau          --cfg-override etau_mode=none
arm veros_surface_bc --cfg-override surface_bc=veros_flux
arm nnmxl2_control   --cfg-override tke_mxl_choice=4
arm nemo_z0_placement --cfg-override tke_surface_bc_level=nemo_z0
# ---- ARMS 6-7: the TRANSPORT and SINK of TKE, not its sources -------------
# Arms 0-5 showed the excess is created BY THE STEP and that only the surface
# source matters -- but killing a source is not a fix when the source value is
# already right (en(k=0) matches NEMO at 0.944 median). What the step does
# with that energy is the remaining freedom, and two card defaults govern it:
#   alpha_tke = 30.0. This multiplies the TKE vertical-diffusion coefficient.
#     Its OWN __param_spec__ reference reads "NEMO avm x1 (zdftke); Veros/
#     Gaspar 30" -- so the card is running the Veros value against a NEMO
#     oracle, and NEMO's tridiagonal (zdftke.F90:407-409) uses
#     -0.5*rn_Dt*(avm(k+1)+avm(k)) = -rn_Dt * mean(avm), i.e. exactly x1.
#   dissipation_discretization = "backward_euler". NEMO linearises the
#     Kolmogoroff sink as a NEWTON split: zfact2 = 1.5*rn_Dt*rn_ediss on the
#     diagonal (:241,:414) plus zfact3 = 0.5*rn_ediss added back explicitly
#     (:242,:419) -- the correct Jacobian for eps ~ e^{3/2}. Ours puts 1.0 on
#     the diagonal with no add-back, a WEAKER implicit sink.
# Both would let a correct surface value overshoot, and neither is selected by
# the ORCA1 card. PRE-REGISTERED on antarctic KM_rat (baseline 1.846, and the
# zero-step value on the same interfaces is 0.861, so ~1.0 is the target):
#   CONFIRMED : <= 1.2.   CONTRIBUTOR : 1.2-1.6.   REFUTED : >= 1.6.
arm alpha_tke_nemo   --cfg-override alpha_tke=1.0
arm nemo_diss_split  --cfg-override dissipation_discretization=nemo_1p5_split

echo; echo "=== ALL DONE (failed arms: $NFAIL) ==="
exit $((NFAIL > 0))

# ---- ARM 5, added after arms 0-4 localised the excess to the surface term --
# TKEConfig.tke_surface_bc_level defaults to "interior_pinned", which the
# config's OWN docstring calls "one w-level too deep vs NEMO" and ranks as the
# #1 suspect (Phase-2 #1317 T3), with a synthetic-column measurement of
# en(20m) 8.6x / avt(30m) ~50x NEMO from this placement ALONE.  NEMO holds
# en(1) at the z=0 W-point and solves from jk=2; "nemo_z0" prepends the
# virtual surface row so interior interface 0 becomes a solved row.
# Independent support from NEMO's own restart, measured this session: our
# surface formula is RIGHT (en(k=0)/[rn_ebb*taum/rho0] median 0.944 over 56358
# strong-wind cells), and NEMO's own profile falls 7.11e-3 -> 2.51e-3 between
# k=0 and k=1, a factor 2.83 — so pinning the surface value one level down
# over-energises the top interface by about that much, against a measured
# excess of 3.2.  PRE-REGISTERED on antarctic e_ratio(active), baseline 3.205:
#   CONFIRMED : <= 1.6.   CONTRIBUTOR : 1.6-2.6.   REFUTED : >= 2.6.
