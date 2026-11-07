#!/bin/bash
#SBATCH --job-name=spsa
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/spsa_%j.out
#
# DOES FEEDING PRACTICAL SALINITY INTO A TEOS-10 POLYNOMIAL MATTER?
#
# Our eos.py:1710-1711 documents that legoESM (like Veros and the oracle
# recipes) feeds model potential temperature / practical salinity straight
# into the TEOS-10 form. NEMO under ln_teos10 does NOT: eosbn2.F90:1924 sets
# l_useCT=.TRUE. and :218 documents "TEOS10: SA ... g/kg", so its prognostic
# tracers ARE Conservative Temperature and Absolute Salinity, and it applies
# no conversion anywhere (the input files are pre-converted offline).
#
# Our PRODUCTION card already runs n2_eos_form='teos10' for the TKE N2
# (run_omip_core2.py:863). So the question is live TODAY, not only for gap 10.
#
# PRE-REGISTERED. The metric that matters is N2, not rho: a nearly uniform
# salinity offset shifts density almost equally at every depth and largely
# cancels in a VERTICAL GRADIENT. So rho alone would overstate this.
#   REFUTES a defect: median |dN2|/N2 < 1% -> the SP/SA mismatch is
#     second-order in the quantity the closure actually consumes, and the
#     current card is fine. Report it and move on.
#   CONFIRMS a defect: median |dN2|/N2 > 5% -> the production card is feeding
#     the wrong salinity scale into its mixing closure and that is a finding
#     independent of gap 10.
#   Between 1% and 5%: report the number, claim no direction.
# rho and alpha/beta are reported alongside so the reader can see WHY N2
# behaves differently from density.
#
# Read-only: touches no worktree file. The conversion used is the standard
# GSW linear form SA = 1.00472*SP (the deep-ocean reference-composition
# value); the spatially varying delta_SA anomaly is NOT applied, which makes
# this a LOWER BOUND on the difference in the North Pacific where delta_SA is
# largest. Stated rather than hidden.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY -u - <<'PY'
import numpy as np, jax.numpy as jnp
from legoesm.ocean.eos import make_eos_fn

SNAP = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
        "trp_base2_d30/snapshot_day0030.npz")
d = np.load(SNAP)
key_T = [k for k in d.files if k.lower() in ("t", "temp", "temperature")]
key_S = [k for k in d.files if k.lower() in ("s", "salt", "salinity")]
print("snapshot keys:", [k for k in d.files][:12])
assert key_T and key_S, f"no T/S in snapshot; keys={d.files}"
T = np.asarray(d[key_T[0]], dtype=float)
S = np.asarray(d[key_S[0]], dtype=float)
print(f"T {T.shape} range {np.nanmin(T):.2f}..{np.nanmax(T):.2f}")
print(f"S {S.shape} range {np.nanmin(S):.2f}..{np.nanmax(S):.2f}")

# Wet points only: the model carries zeros on land and those would dominate
# any percentile. Require a physically plausible salinity.
wet = np.isfinite(T) & np.isfinite(S) & (S > 5.0) & (S < 45.0)
print(f"wet points: {wet.sum()} of {wet.size}")
assert wet.sum() > 1000, "too few wet points to trust a percentile"

eos = make_eos_fn("nemo_teos10")

# Depth axis is last; build a pressure proxy in dbar from level index only to
# keep this self-contained -- the SAME p is used on BOTH sides, so it cannot
# manufacture a difference. This is a CONTROLLED comparison in the salinity
# scale alone: exactly one field differs between the two calls.
nz = T.shape[-1]
p = np.broadcast_to(np.linspace(0.0, 5000.0, nz), T.shape).copy()

SA = 1.00472 * S                       # reference-composition conversion
# Temperature is held IDENTICAL on both sides on purpose: this arm isolates
# the salinity scale alone. The theta-vs-CT difference is a separate effect
# and conflating the two would make neither attributable.

rho_sp = np.asarray(eos(jnp.asarray(T), jnp.asarray(S),  jnp.asarray(p)))
rho_sa = np.asarray(eos(jnp.asarray(T), jnp.asarray(SA), jnp.asarray(p)))

drho = np.abs(rho_sa - rho_sp)[wet]
print(f"\nDENSITY: median |drho| = {np.median(drho):.4f} kg/m3, "
      f"p95 = {np.percentile(drho, 95):.4f}")

# N2 ~ -(g/rho0) d(rho)/dz. Compare the VERTICAL GRADIENT, which is what the
# closure consumes. Uniform vertical spacing cancels, so compare the level
# differences directly.
def grad_z(x):
    return np.diff(x, axis=-1)

g_sp, g_sa = grad_z(rho_sp), grad_z(rho_sa)
wet_g = wet[..., :-1] & wet[..., 1:]
# Only where the column is actually stratified: an unstratified cell has a
# near-zero denominator and its ratio is meaningless, not large.
sig = wet_g & (np.abs(g_sp) > 1e-4)
rel = np.abs(g_sa - g_sp)[sig] / np.abs(g_sp)[sig]
print(f"STRATIFICATION (the N2 quantity): {sig.sum()} stratified wet pairs")
print(f"  median |dN2|/N2 = {100*np.median(rel):.3f} %")
print(f"  p95    |dN2|/N2 = {100*np.percentile(rel, 95):.3f} %")

med = 100 * np.median(rel)
verdict = ("REFUTED (<1%): SP-into-TEOS10 is second-order in N2"
           if med < 1.0 else
           "CONFIRMED (>5%): the card feeds the wrong salinity scale"
           if med > 5.0 else
           "INCONCLUSIVE (1-5%): report the number, claim no direction")
print(f"\nVERDICT: {verdict}")
print("NOTE: delta_SA anomaly NOT applied, so this is a LOWER BOUND.")
PY
echo DONE
