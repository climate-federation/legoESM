#!/bin/bash
#SBATCH --job-name=mpashybd5
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpashybd5_%j.out
#
# THE HYBRID ARM BOTH REVIEWERS ASKED FOR: harmonised LATERAL mixing with the
# PRE-harmonisation VERTICAL surface boundary condition, five days, MPAS.
#
# THE PROBLEM IT SOLVES. The MPAS harmonisation moved lateral and vertical
# settings together and produced a null. A null measured across a package
# licenses no statement about any one member, because the members can cancel.
# GLM raised it, codex independently named this arm as the cheapest thing that
# separates cancellation from genuine insensitivity, and the measured
# Smagorinsky magnitude made cancellation concrete rather than hypothetical:
# at the coefficient the harmonisation selected, the flow-dependent viscosity
# at the equator is about 1.0e5 m2/s, which is the entire background the
# harmonisation removed.
#
# BOTH ENDPOINTS ALREADY EXIST AT DAY 5, so only the hybrid is submitted:
#   mpas_mixharm_d30/snapshot_day0005.npz        new lateral + new vertical
#   mpas_orca1faithtint_d30/snapshot_day0005.npz old lateral + old vertical
#
# HONEST LIMIT, stated because it bounds the conclusion. The harmonisation
# differed in six resolved fields; this arm reverts ONE of the two vertical
# ones, tke_surface_bc_level, because it is the only one with a flag. The
# other, tke.surface_flux_coeff absent -> 1.0, cannot be reverted and does not
# need to be: it multiplies the Veros surface flux as
# surface_flux_coeff * (taum/rho_0)**1.5, so 1.0 is the identity, and the field
# was simply absent from the older manifest because it did not exist yet. It is
# a newly-named field at its neutral value on a branch these runs do not even
# take, since both pass --tke-surface-bc nemo_dirichlet. So the harmonisation
# is five effective knobs, not six, and this arm reverts the one vertical knob
# that carries physics.
#
# THE COMMAND LINE IS THE HARMONISED MPAS RUN'S OWN, read from its
# run_manifest.json rather than retyped, with three edits: --years 30/365 ->
# 5/365, a new --output, and --tke-surface-bc-level interior_pinned appended.
# Restart flags removed so this is a fresh integration from the same initial
# condition both endpoints started from.
#
# PRE-REGISTERED, on nino3 day-5 SST bias. Endpoints: old -0.0309 and new
# -0.0225 global, nino3 old and new both near +0.52. Let H be the hybrid.
#   H close to NEW, and NEW close to OLD    -> the lateral package is inert at
#        this horizon and the null is genuine insensitivity.
#   H far from BOTH endpoints               -> CANCELLATION: lateral and
#        vertical each move nino3 and they offset. The harmonisation null is
#        then an artifact and every statement resting on it is void.
#   H close to OLD                          -> the vertical knob carried the
#        whole response and the lateral package is inert.
# "Close" is judged against the cross-grid spread measured at day 5, which is
# 0.060 C; a move smaller than that is not a move.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_arm_mpas_hybrid.txt)
echo "CMD: $PY scripts/run/$CMD"
$PY scripts/run/$CMD
rc=$?; echo "=== rc=$rc ==="
exit $rc
