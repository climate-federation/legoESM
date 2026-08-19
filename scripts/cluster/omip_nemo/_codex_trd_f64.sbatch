#!/bin/bash
#SBATCH --job-name=codex_trd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_trd_%j.out

# Adversarial review of the tendency-comparator instrument changes (f64 fix +
# Stage B2) — the instruments decide what we believe.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of the NEMO tendency-matching instrument:
scripts/validate/ocean_fidelity/compare_tendencies_nemo.py (recent commits
touching it) and tendency_stageb_worst_column.py.  Oracle sources:
/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE/TRA/trazdf.F90,
TRD/trdtra.F90, ZDF/zdfevd.F90.

Claims to attack (each currently believed):
1. The f32->f64 fix: _fill casts to float64 and the module sets
   JAX_ENABLE_X64 before jax import.  Is there any remaining f32 path into
   the Stage-B solve (jnp.asarray downcast, the solver's internal dtype,
   dz_half computed from f32 inputs elsewhere)?
2. Stage B2's reconstruction S_pre = S(r) - dt*strd_zdf claims trazdf is the
   LAST tracer operator of the RK3 step, so the post-step saved vosaline
   equals the post-zdf field with no atf/swap adjustment in between.  Verify
   against the RK3 step ordering (stprk3*.F90, traatf* if present).  If an
   Asselin-type or ssh/e3t rescaling intervenes, quantify what it does to the
   verdict corr 0.98-0.999.
3. CIRCULARITY: does solving OUR operator on S_pre and comparing to strd_zdf
   constitute a real test of our operator, or does the reconstruction make
   high correlation inevitable regardless of the operator?  Construct the
   argument both ways: what would the numbers look like if our operator were
   (a) identity, (b) explicit instead of implicit, (c) off-by-one in K?
4. The verdict table reports corr/nrmse on (dT2 vs ttrd_zdf) with
   region/evd splits: any masking or fill (S fill 35.0, T fill 0.0, dz fill
   1.0) that could leak into scored cells rather than staying inert?
5. antarctic/evd T nrmse 0.66 corr 0.75 while every other region is
   0.05-0.33: name the most likely physical/structural cause (under-ice qsr
   penetration inside NEMO's zdf? ice-ocean flux BC? snow/ice mask) and a
   cheap discriminating check.
Severity-tagged findings + verdict table."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
