#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=coupfix_gates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_coupfix/logs/coupfix_gates_%j.out
# Validation gates for the coupled-CMIP coupling fixes:
#   FIX-1  segment-accumulated precip was handed to the coupler as a RATE
#   FIX-3  float32 reverse-mode NaN from floored denominators in the sea-ice code
#
# THREE stages, in order:
#   [VACUITY] the new tests are run against UNPATCHED origin/main.  They MUST FAIL
#             there.  A test that passes on the broken code proves nothing, and
#             the FIX-3 test in particular is easy to write vacuously (it goes
#             green if it silently runs in x64, or if every ice category is
#             populated so the floored branch is never taken).  This stage is the
#             non-vacuousness proof required by the guardrail doctrine.
#   [FIXED]   the same tests on the patched branch.  MUST PASS.
#   [REGRESS] the pre-existing coupler/ice/regrid suites on the patched branch,
#             to show the fix did not move a previously-correct path.
#
# PRECISION LANES are deliberate:
#   x64  for the FIX-1 units gate (the standard scientific lane).
#   fp32 for the FIX-3 AD gate -- the whole defect class (floored denominator ->
#        integer_pow(y,-2) overflows -> 0*Inf -> NaN cotangent) is INVISIBLE
#        under JAX_ENABLE_X64=1.  Running that gate in x64 makes it vacuous,
#        which is exactly why the existing suite never caught the bug.
set -uo pipefail

FIXED=/burg-archive/glab/users/pg2328/legoESM_coupfix
PREFIX=/burg-archive/glab/users/pg2328/legoESM_prefix   # detached @ origin/main, UNPATCHED
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

T_UNITS=tests/unit/test_seg_precip_units.py
T_AD=tests/ice/unit/test_ice_float32_grad_safety.py
T_C2=tests/coupler/unit/test_coupled_lane_carry_aux.py
T_XG=tests/coupler/unit/test_grid_remap_wet_mask.py
T_F4=tests/coupler/unit/test_segment_callback_elapsed_dt.py
T_H5=tests/grids/test_tripole_internals.py::TestRotationAngleSignAndSeam
T_F5=tests/unit/test_coupled_ocean_ice_forcing.py

mkdir -p "$FIXED/logs"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== coupling fix gates $(date -Is) ==="
echo "fixed  : $(git -C "$FIXED" rev-parse --short HEAD) [$(git -C "$FIXED" rev-parse --abbrev-ref HEAD)]"
echo "prefix : $(git -C "$PREFIX" rev-parse --short HEAD) (unpatched baseline)"

RC=0

# ---------------------------------------------------------------- VACUITY ----
# Copy ONLY the new test files into the unpatched tree and run them there.
echo
echo "############ [VACUITY] new tests vs UNPATCHED origin/main -- MUST FAIL ############"
mkdir -p "$PREFIX/tests/unit" "$PREFIX/tests/ice/unit"
cp "$FIXED/$T_UNITS" "$PREFIX/$T_UNITS"
cp "$FIXED/$T_AD"    "$PREFIX/$T_AD"
export LEGOESM_REPO="$PREFIX"
source "$PREFIX/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$PREFIX"

# PER-TEST, not per-file.  A file-level exit code is NOT a vacuity proof: one
# genuinely-red test masks a sibling that passes on the broken code, which is
# exactly how a vacuous step_sea_ice case slipped through earlier.  Collect
# every test id and require EACH to be individually RED here.
vacuity_check () {   # $1 = test file  $2 = X64 value  $3 = label  $4 = optional
                     # space-separated substrings of tests ALLOWED to be green
                     # on main (invariant guards, not regression tests).
  local f="$1" x64="$2" label="$3" allow="${4:-}" ids id rc short base
  # Build node ids from the FIXED source by grep, NOT pytest --collect-only:
  # collection depends on the sourced env / conftest / pytest version and once
  # returned nothing under a mismatched PYTHONPATH, spuriously failing the gate.
  # Grep is deterministic. Handle BOTH layouts: a module-level `def test_` (no
  # class -> id = file::method) and a single unittest `class Test...: def test_`
  # (id = file::Class::method). None of these files use >1 test class.
  local cls methods m
  cls=$(grep -oE '^class (Test[A-Za-z0-9_]+)' "$FIXED/$f" | head -1 | awk '{print $2}')
  methods=$(grep -oE 'def (test_[A-Za-z0-9_]+)' "$FIXED/$f" | awk '{print $2}' | sort -u)
  if [ -z "$methods" ]; then
    echo "!!! $label: no test methods parsed from $FIXED/$f -- gate cannot run;"
    echo "    treating as FAILURE (never a silent pass)."
    RC=$((RC+1)); return 0
  fi
  ids=""
  for m in $methods; do
    if [ -n "$cls" ]; then ids="$ids $f::$cls::$m"; else ids="$ids $f::$m"; fi
  done
  echo "$label: checking $(echo "$ids" | wc -l) test(s) individually"
  for id in $ids; do
    short="${id##*::}"
    JAX_ENABLE_X64="$x64" "$PY" -m pytest -q --tb=no "$id" >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      # Green on main. Allowed only for declared invariant guards.
      for base in $allow; do
        if [ "$short" = "$base" ]; then
          echo "ok(invariant-guard, green both sides): $short"; continue 2
        fi
      done
      echo "!!! VACUOUS: $id PASSES on unpatched main -- it proves nothing"
      RC=$((RC+1))
    else
      echo "ok(red-on-main): $short"
    fi
  done
}

# FIX-1's test imports a symbol that does not exist on main, so the whole
# module fails to collect -- a legitimate (if coarse) red.
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q --tb=line "$T_UNITS"
if [ $? -eq 0 ]; then echo "!!! VACUOUS: FIX-1 units test PASSES on unpatched main"; RC=$((RC+1));
else echo "ok: FIX-1 units test correctly RED on unpatched main"; fi

# FIX-3's tests DO collect on main, so each is checked individually.
vacuity_check "$T_AD" 0 "FIX-3 AD"

# C2's module imports a helper that does not exist on main, so it fails to
# collect there (a coarse but legitimate red); check per-test anyway so that
# once the helper lands, any individually-vacuous case is still caught.
mkdir -p "$PREFIX/$(dirname "$T_C2")"
cp "$FIXED/$T_C2" "$PREFIX/$T_C2"
vacuity_check "$T_C2" 1 "C2 carry_aux"
rm -f "$PREFIX/$T_C2"

# H3/H4 cross-grid: the o2a land-exclusion / open-water-fraction / driver-wiring
# cases are red on main; the a2o conservation + identity cases are INVARIANT
# GUARDS (green both sides by design), so this is not a pure red-on-main set --
# per-test check surfaces exactly which are which.
mkdir -p "$PREFIX/$(dirname "$T_XG")"
cp "$FIXED/$T_XG" "$PREFIX/$T_XG"
# test_a2o_uniform_flux_conserved_over_wet_domain uses only pre-existing symbols
# (the a2o remap is unmasked in this design) -- it is an INVARIANT guard that
# must hold on both trees, so it is expected-green, not vacuous.
vacuity_check "$T_XG" 1 "H3/H4 cross-grid" \
  "test_a2o_uniform_flux_conserved_over_wet_domain"
rm -f "$PREFIX/$T_XG"

# F4: module-level function test that runs a tiny ModelDriver and asserts the
# coupler callback fires every SEGMENT with elapsed dt. On unpatched main the
# callback is diag-gated -> wrong firing/dt -> red.
mkdir -p "$PREFIX/$(dirname "$T_F4")"
cp "$FIXED/$T_F4" "$PREFIX/$T_F4"
vacuity_check "$T_F4" 1 "F4 elapsed-dt"
rm -f "$PREFIX/$T_F4"

# H5/H6: the new TestRotationAngleSignAndSeam class is inserted into an existing
# multi-class file. Copy the FIXED file into prefix and run ONLY that class on
# unpatched main -- sin_alpha_v is sign-flipped and dlon is not unwrapped there,
# so every case is red. (Generic vacuity_check picks head -1 class, wrong for a
# 4-class file, so this is targeted.)
_h5f="tests/grids/test_tripole_internals.py"
mkdir -p "$PREFIX/$(dirname "$_h5f")"
cp "$FIXED/$_h5f" "$PREFIX/$_h5f"
for _m in test_h5_vpoint_sin_matches_uface_convention \
          test_h5_consumer_matrix_is_orthonormal_roundtrip \
          test_h6_seam_cell_matches_interior_neighbor; do
  JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no \
    "${_h5f}::TestRotationAngleSignAndSeam::${_m}" >/dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "!!! VACUOUS: H5/H6 ${_m} PASSES on unpatched main"; RC=$((RC+1))
  else echo "ok(red-on-main): ${_m}"; fi
done
# restore the pristine main copy of this shared file
(cd "$PREFIX" && git checkout -- "$_h5f" 2>/dev/null) || rm -f "$PREFIX/$_h5f"

# F5: the ocean-ice-forcing test file pre-exists on main with different content;
# copy the FIXED version and check the DISCRIMINATING slab case is red on main.
# The v2 case is an invariant guard (v2 keeps f_ocean*P unchanged -> green both
# sides), so it is expected-green and not counted vacuous.
_f5f="tests/unit/test_coupled_ocean_ice_forcing.py"
cp "$FIXED/$_f5f" "$PREFIX/$_f5f"
JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no \
  "${_f5f}::TestOceanIceForcing::test_precip_over_slab_ice_excludes_land_fraction" >/dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "!!! VACUOUS: F5 slab-precip test PASSES on unpatched main"; RC=$((RC+1))
else echo "ok(red-on-main): F5 test_precip_over_slab_ice_excludes_land_fraction"; fi
# test_ice_cell_gets_open_water_flux_scaled_by_f_ocean also changed its precip
# assertion (0.2x -> equal); confirm it too is red on main.
JAX_ENABLE_X64=1 "$PY" -m pytest -q --tb=no \
  "${_f5f}::TestOceanIceForcing::test_ice_cell_gets_open_water_flux_scaled_by_f_ocean" >/dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "!!! VACUOUS: F5 open-water-scaling test PASSES on unpatched main"; RC=$((RC+1))
else echo "ok(red-on-main): F5 test_ice_cell_gets_open_water_flux_scaled_by_f_ocean"; fi
(cd "$PREFIX" && git checkout -- "$_f5f" 2>/dev/null) || rm -f "$PREFIX/$_f5f"

rm -f "$PREFIX/$T_UNITS" "$PREFIX/$T_AD"

# Control for the one known-fragile regression test: it asserts
# `"Traceback" not in stdout`, and on a CPU-only node the JAX CUDA plugin
# prints a CUDA_ERROR_NO_DEVICE traceback into stdout.  Run it UNPATCHED to
# establish it fails there too -- i.e. environment artifact, not a regression
# introduced by this branch.  Do not assert; just record both outcomes.
echo
echo "--- [CONTROL] known-fragile mpas export test on UNPATCHED main ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q \
  tests/coupler/unit/test_mpas_surface_flux_export.py
echo "control rc=$? (expected NONZERO on a CPU node -> pre-existing, not ours)"

# ------------------------------------------------------------------ FIXED ----
echo
echo "############ [FIXED] same tests on the patched branch -- MUST PASS ############"
unset PYTHONPATH
export LEGOESM_REPO="$FIXED"
source "$FIXED/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$FIXED"

echo "--- [x64] FIX-1 precip units ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_UNITS";      RC=$((RC + $?))
echo "--- [fp32] FIX-3 sea-ice reverse-mode AD (MUST be float32) ---"
JAX_ENABLE_X64=0 "$PY" -u -m pytest -q "$T_AD";         RC=$((RC + $?))
echo "--- [x64] C2 coupled-lane _carry_aux guard ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_C2";         RC=$((RC + $?))
echo "--- [x64] H3/H4 cross-grid coastal wet-mask ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_XG";         RC=$((RC + $?))
echo "--- [x64] F4 coupler elapsed-dt (per-segment callback) ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_F4";         RC=$((RC + $?))
echo "--- [x64] H5/H6 tripole rotation sign + branch-cut unwrap ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_H5";         RC=$((RC + $?))
echo "--- [x64] F5 precip-over-ice (no double-count on OMIP) ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q "$T_F5";         RC=$((RC + $?))

# The C2 guard fires on a marker set by the coupled drivers, NOT on the presence
# of a segment_callback -- that hook is a GENERAL per-segment diagnostic with
# several non-coupled users, and gating on it would turn these two SPMD/tiled
# parity tests red.  They go through the exact dispatch branches being guarded,
# so they are the direct regression control for that design choice.
echo "--- [x64] control: non-coupled segment_callback users must STILL pass ---"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q \
  tests/parallel/test_operator_split_spmd_driver_parity.py \
  tests/parallel/test_operator_split_tiled_cube_driver_parity.py
RC=$((RC + $?))

# ---------------------------------------------------------------- REGRESS ----
echo
echo "############ [REGRESS] pre-existing suites on the patched branch ############"
JAX_ENABLE_X64=1 "$PY" -u -m pytest -q \
  tests/coupler/ \
  tests/ice/ \
  tests/unit/test_conservative_regrid.py \
  tests/unit/test_conservative_regrid_curvilinear.py \
  tests/unit/test_conservative_regrid_unstructured.py
RC=$((RC + $?))

echo
echo "COUPLING FIX GATES COMPLETE rc=$RC $(date -Is)"
exit "$RC"
