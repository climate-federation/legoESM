#!/bin/bash
#SBATCH --job-name=unescocv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/unescocv_%j.out
#
# IS OUR unesco80_eos AN IN-SITU POLYNOMIAL OR A POTENTIAL-TEMPERATURE REFIT?
#
# BOTH reviewers said in-situ, independently: GLM ("the original Millero-Poisson
# form wants in-situ temperature... same order as the TEOS-10 mismatch you just
# refused") and codex (MAJOR, "unesco80 requires in-situ temperature but
# receives potential temperature without conversion; its docstring is wrong").
#
# Our code says otherwise, in two places that CONTRADICT EACH OTHER:
#   eos.py:1278  "T : Potential temperature [degC]"
#   eos.py:2448  "unesco80 (UNESCO 1980 polynomial -- close approximation to
#                 Veros's eq_of_state_type=3 JM95 form"
# The UNESCO 1980 polynomial is the IN-SITU standard; JM95 (Jackett & McDougall
# 1995) is the POTENTIAL-temperature refit. It cannot be both. Two reviewers
# against a docstring is still a vote, so measure instead.
#
# THE DISCRIMINATOR is the published EOS-80 check value, which is defined for
# IN-SITU temperature and only separates the two forms AT DEPTH (at the surface
# theta == T, so a surface check proves nothing):
#     rho(S=35 psu, T=25 degC, p=10000 dbar) = 1062.538 kg/m3   [UNESCO 1983]
#
# PRE-REGISTERED:
#   |rho - 1062.538| < 0.05 kg/m3  -> CONFIRMED in-situ polynomial. Both
#       reviewers are right, eos.py:1278 is wrong, and unesco80 must be
#       REFUSED by the same argument that refuses TEOS-10.
#   |rho - 1062.538| > 0.5 kg/m3   -> REFUTED. It is a theta-form and the
#       reviewers are wrong; keep unesco80 accepted and say so.
#   in between -> INCONCLUSIVE; report the number, change nothing.
# The surface value is printed too, purely to show the check is wired up (it
# CANNOT discriminate, and is not used for the verdict).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY -u - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.eos import unesco80_eos

# eos.py's contract: p is sea pressure in Pa. 1 dbar = 1e4 Pa.
def rho(T, S, p_dbar):
    return float(unesco80_eos(jnp.asarray(T, float), jnp.asarray(S, float),
                              jnp.asarray(p_dbar * 1.0e4, float)))

surf = rho(25.0, 35.0, 0.0)
deep = rho(25.0, 35.0, 10000.0)
print(f"surface rho(35, 25, 0 dbar)      = {surf:.4f}   "
      f"(EOS-80 one-atm check 1023.343; CANNOT discriminate)")
print(f"deep    rho(35, 25, 10000 dbar)  = {deep:.4f}   "
      f"(UNESCO in-situ check 1062.538)")

d = abs(deep - 1062.538)
print(f"\n|delta| from the in-situ check value = {d:.4f} kg/m3")
if d < 0.05:
    print("VERDICT: CONFIRMED in-situ polynomial. Both reviewers right; "
          "eos.py:1278 is wrong; unesco80 must be refused like TEOS-10.")
elif d > 0.5:
    print("VERDICT: REFUTED. Not the in-situ form; reviewers wrong; "
          "keep unesco80 accepted.")
else:
    print("VERDICT: INCONCLUSIVE. Report the number, change nothing.")
PY
echo DONE
