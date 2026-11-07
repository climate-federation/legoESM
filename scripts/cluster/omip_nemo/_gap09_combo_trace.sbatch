#!/bin/bash
#SBATCH --job-name=gap09combo
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gap09combo_%j.out
#
# GAP 9 (lateral tracer operator) — codex ranked it MEDIUM class (i), i.e.
# "three existing GM operator flags omitted". Verified by reading the runner:
# --gm-slope-scheme, --gm-bolus-advection and --gm-msc-stabilize ALL already
# exist, are CLI-exposed, argument-guarded (tripole-only) and individually
# tested. So gap 9 needs NO new code; selecting them is an ADOPTION ASK.
#
# What is genuinely unverified is COMPOSITION: gaps 7+8 added an opt-in
# nemo21 Redi coefficient whose validator demands the native rotated-Laplacian
# path, and nobody has ever built the full ORCA1-faithful combination in one
# config. A flag that exists but has never been selected alongside its
# siblings is not a closed gap.
#
# PRE-REGISTERED. This either constructs and traces, or it raises. A raise
# names a real composition defect in the gaps 7+8 validator and gap 9 does
# NOT close. Green means the ORCA1 operator stack is selectable as one unit.
# This is a CONSTRUCTION + TRACE check only: it asserts nothing about bias.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY -u - <<'PY'
import traceback
import numpy as np, jax, jax.numpy as jnp
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig, TreguierConfig
from legoesm.ocean.physics.lateral_mixing import gm_redi_latlon_cgrid as gm

ok = True

def check(name, fn):
    global ok
    try:
        fn(); print(f"PASS  {name}")
    except Exception as e:
        ok = False
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

# The full ORCA1-faithful operator stack in ONE config: gap 9's three flags
# (nemo_iso_lap operator, bolus through the FCT limiter, stabilizing
# correction) together with gaps 7+8 (unfloored Treguier GM, nemo21 Redi).
ny, nx = 12, 16
f_f = np.broadcast_to(np.linspace(-1e-4, 1e-4, ny)[:, None], (ny, nx)).copy()

def build(**over):
    base = dict(
        slope_scheme="nemo_iso_lap", slope_positions="nemo_native",
        gm_bolus_advection="through_fct", msc_stabilize=True,
        treguier=TreguierConfig(enabled=True, kappa_min=0.0),
        visbeck=type(GMRediConfig().visbeck)(enabled=False)
        if hasattr(GMRediConfig().visbeck, "enabled") else GMRediConfig().visbeck,
        eke=None,
        redi_coefficient="nemo21", redi_aht0=900.0,
        redi_f_f=jnp.asarray(f_f),
    )
    base.update(over)
    return GMRediConfig()._replace(**base)

check("full ORCA1 stack validates",
      lambda: gm.validate_redi_coefficient(build()))

# NON-VACUITY: each guard the validator claims must actually bind.
def must_raise(msg, cfg_kw):
    try:
        gm.validate_redi_coefficient(build(**cfg_kw))
    except ValueError:
        return
    raise AssertionError(f"guard did NOT bind: {msg}")

check("floored GM is rejected",
      lambda: must_raise("kappa_min", {"treguier": TreguierConfig(enabled=True, kappa_min=200.0)}))
check("missing mesh f_f is rejected",
      lambda: must_raise("redi_f_f", {"redi_f_f": None}))
check("non-native slopes are rejected",
      lambda: must_raise("slope_positions", {"slope_positions": "mode_b"}))
check("unknown selection is rejected",
      lambda: must_raise("unknown", {"redi_coefficient": "nemo22"}))

# The tropical law itself, on the composed config: the additive term must
# reach the equator, where the GM coefficient is identically zero.
def tropical():
    from legoesm import constants
    kt = jnp.zeros((ny, nx))            # GM tapered to zero, as at the equator
    ft = jnp.zeros((ny, nx))            # f = 0
    um = jnp.ones((ny, nx + 1)); vm = jnp.ones((ny + 1, nx))
    ku, kv = gm.nemo21_redi_from_gm(kt, ft, jnp.zeros((ny, nx)),
                                    um[:, 1:], vm[1:, :], 900.0, constants.Omega)
    # NEMO ldftra.F90:434 with ahtu=0 and ff_t=0 gives max(0.2*aht0,0) + zDaht
    # = 0.2*900 + 720 = 900. Anything less means the additive term is not
    # reaching the equator, which is the whole point of the construction.
    got = float(jnp.max(ku))
    assert abs(got - 900.0) < 1e-6, f"equatorial Redi {got}, expected 900.0"
    print(f"      equatorial Redi with ZERO GM = {got:.3f} m2/s (oracle: 900.0)")
check("additive term reaches the equator", tropical)

print("\nVERDICT:", "COMPOSES" if ok else "COMPOSITION DEFECT")
PY
echo "=== now the runner's own CLI, same stack ==="
$PY -m pytest tests/unit/test_run_omip_core2_gm_treguier.py \
  tests/ocean/unit/test_omip_gm_redi_coefficients.py \
  tests/ocean/unit/test_gm_bolus_through_fct.py -q 2>&1 | tail -8
echo DONE
