#!/bin/bash
#SBATCH --job-name=gate_negmass
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/gate_negmass.%j.out
# #1029, user decision "hard error, then pick a fix": acceptance for the guard
# and for the transition_exponent knob the remedy needs in order to exist.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "HEAD: $(git -C "$WT" log --oneline -1)"

echo "########## the guard and the thresholds ##########"
"$PY" -m pytest -q tests/unit/test_hybrid_negative_layer_mass.py 2>&1 | tail -16

echo "########## the existing hybrid suites must stay green ##########"
"$PY" -m pytest -q tests/unit/test_hybrid_level_validity.py \
    tests/unit/test_diagnostics_hybrid_pressure.py \
    tests/unit/test_energy_budget_hybrid.py 2>&1 | tail -10

echo "########## the driver refuses a real-orography run at exponent 3 ##########"
# The guard is only worth having if it fires on the PRODUCTION path, not just
# when called directly. Build the driver's own check against a Tibetan column.
"$PY" - <<'DRVPY'
import jax.numpy as jnp
from legoesm import constants
from legoesm.grids.vertical import (assert_hybrid_valid_for_surface_pressure,
                                    make_hybrid_levels)
# phis of a 4500 m plateau, the driver's own hydrostatic reduction at T_init.
for z, T in ((4500.0, 288.0), (2000.0, 288.0)):
    phis = constants.g * z
    p_s = float(constants.p_ref * jnp.exp(-phis / (constants.R_d * T)))
    for expo in (3, 2):
        coord = make_hybrid_levels(40, stretching=2.5, transition_exponent=expo)
        try:
            assert_hybrid_valid_for_surface_pressure(
                coord, p_s, context=f"{z:.0f} m plateau, exponent={expo}")
            verdict = "accepted"
        except ValueError:
            verdict = "REFUSED"
        print(f"  z={z:6.0f} m  p_s={p_s/100:6.1f} hPa  exponent={expo}  -> {verdict}")
DRVPY

echo "########## the new CLI flag round-trips ##########"
"$PY" - <<'PYEOF'
import sys
sys.argv = ["run_amip.py"]
from importlib.machinery import SourceFileLoader
m = SourceFileLoader("_ra", "scripts/run/run_amip.py").load_module()
p = m.build_arg_parser()
for argv, want in (([], None), (["--transition-exponent", "2"], 2),
                   (["--transition-exponent", "3"], 3)):
    a = p.parse_args(argv)
    got = a.transition_exponent
    assert got == want, (argv, got, want)
    print(f"  parse {argv!r:34s} -> transition_exponent={got}")
import argparse
try:
    p.parse_args(["--transition-exponent", "1"])
except SystemExit:
    print("  parse ['--transition-exponent','1'] -> REFUSED (choices gate)")
else:
    raise AssertionError("exponent 1 should be refused by the CLI choices")
PYEOF

echo "########## validate_strict rejects a bad exponent ##########"
"$PY" - <<'PYEOF'
from legoesm.driver.config import ExperimentConfig
base = ExperimentConfig()
g_ok = base.grid._replace(vertical_coord="hybrid", transition_exponent=2)
assert base._replace(grid=g_ok).validate_strict() is None
print("  hybrid + exponent 2: accepted")
g_bad = base.grid._replace(vertical_coord="hybrid", transition_exponent=4)
try:
    base._replace(grid=g_bad).validate_strict()
except Exception as e:
    print(f"  hybrid + exponent 4: REFUSED -- {str(e)[:110]}")
else:
    raise AssertionError("exponent 4 should be refused by validate_strict")
g_sig = base.grid._replace(vertical_coord="sigma", transition_exponent=2)
try:
    base._replace(grid=g_sig).validate_strict()
except Exception as e:
    print(f"  sigma + exponent 2: REFUSED as inert -- {str(e)[:110]}")
else:
    raise AssertionError("a non-default exponent on the sigma lane should be refused as inert")
PYEOF

echo "########## NON-VACUITY: drop the guard, the tests must go red ##########"
cp packages/core/legoesm/grids/vertical.py /tmp/_vert_negm_$$.py
python3 - <<'SABPY'
import pathlib
p = pathlib.Path("packages/core/legoesm/grids/vertical.py")
s = p.read_text()
old = "    if thr <= 0.0 or p_s_min_Pa > thr:\n        return\n"
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, "    return\n", 1))
print("SABOTAGED: the guard now never raises")
SABPY
"$PY" -m pytest -q tests/unit/test_hybrid_negative_layer_mass.py 2>&1 | tail -8
cp -f /tmp/_vert_negm_$$.py packages/core/legoesm/grids/vertical.py
rm -f /tmp/_vert_negm_$$.py
echo "RESTORED guard"
"$PY" -m pytest -q tests/unit/test_hybrid_negative_layer_mass.py 2>&1 | tail -4
echo DONE
