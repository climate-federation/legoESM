#!/bin/bash
#SBATCH --job-name=codex_stgc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_stgc_%j.out

set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "DIAGNOSTIC REVIEW (not a style review).  The probe
scripts/validate/ocean_fidelity/compare_advection_nemo.py compares OUR tracer
advection operator against NEMO's ttrd_totad/strd_totad on the hourly trend
file EXP00/RUN_TRD2/ORCA1_1h_20010101_20010101_trd1h_*.nc.  After fixing
three real defects it produces tendencies of PLAUSIBLE MAGNITUDE (within
5-100x of NEMO's) but essentially ZERO CORRELATION (0.02-0.09 by region, both
tracers, both schemes).  A correct operator on a correct state cannot be
uncorrelated: find what is still wrong.

Read, in this order:
 - the probe itself (its docstring lists my four ranked candidates);
 - NEMO /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE/
   stprk3_stg.F90 (which transports and which time level tra_adv is called
   with, and in WHICH stage the trends are emitted), TRA/traadv.F90,
   TRA/traadv_fct.F90, TRD/trdtra.F90;
 - our operator: packages/ocean/legoesm/ocean/dynamics/
   ocean_model_latlon_cgrid.py::_compute_advection_flux_div and
   packages/ocean/legoesm/ocean/advection.py::fct_tracer_advection
   (argument shapes, staggering, units, sign of the returned divergences).

Answer concretely:
 1. Which field does NEMO's advection actually act on for the trend written
    at record r, in terms of what IS in this file?  Is any of it
    unrecoverable from the saved diagnostics (if so, say the probe is
    fundamentally limited and what extra XIOS field would fix it -- e.g. is
    there a zFu/zFv/zFw or uocetr_eff output that should be enabled)?
 2. Sign and orientation: does our (div_hut + vert_flux_div) divided by h,
    NEGATED, equal a tendency in the same sense as ttrd_totad?  Show the
    algebra from both sources.
 3. Is w_half's reconstruction (reverse cumsum of the volume divergence,
    zero at the floor, surface zeroed) consistent with what
    fct_tracer_advection expects at index k (interface ABOVE level k, or
    below)?  An off-by-one here would decorrelate everything.
 4. The lift into the (332,362) halo frame with the cyclic columns filled and
    the fold ghost row left zero: does that corrupt more than the two
    excluded rows, given the tripole grid object's fold handling?
 5. Rank what to test FIRST with a cheap experiment, and name the experiment.
Be concrete; cite file:line."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
