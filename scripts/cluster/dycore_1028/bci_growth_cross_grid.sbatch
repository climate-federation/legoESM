#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=bci1028
#
# #1028 / #1081 — the cross-grid baroclinic-instability growth-rate
# discriminator that both issues name as the next owed experiment.
#
# QUESTION.  After the top sponge and A_h sinks were removed, the cd-grid cube
# still equilibrates the Held-Suarez jet at ~13.8 m/s vs icosahedral 27 and
# lat-lon 66.  Is the residual a GROWTH deficit (the cube's baroclinic eddies
# never reach the amplitude that feeds the jet) or an EQUILIBRATION deficit
# (eddies grow normally, the equilibrated balance is drained)?
#
# PRE-REGISTRATION (written before submission; do not edit after results).
#   Numbers produced, per grid, by scripts/validate/bci_growth_rate.py from the
#   SAME analytic Jablonowski-Williamson perturbed IC:
#     (i)  d ln(eddy KE)/dt in an EARLY (2-6 d) and a LATE (6-10 d) window,
#     (ii) the same rates from eddy temperature variance at a fixed level,
#     (iii) the eddy-KE ratio to a reference grid AT EACH SNAPSHOT TIME.
#
#   The discriminating statistic is (iii)'s SHAPE, not a single slope ratio.
#     * ratio flat in time and below one  -> the two grids' eddies grow at the
#       same rate from different starting amplitudes: the growth OPERATOR
#       agrees and the difference is upstream (IC projection / regridding).
#     * ratio falling through the growth phase -> the cube's eddies grow more
#       slowly: consistent with the conversion deficit (a).
#     * ratio near one early and falling only after saturation -> eddies grow
#       fine and the equilibrated state is drained: consistent with (b).
#   READING RULE, quantitative and fixed BEFORE the numbers were read (GLM-5.2
#   round 2 correctly objected that "consistent / inconsistent" with no number
#   is as unfalsifiable as the arbitrary cutoff it replaced).  All windows are
#   the EARLY window unless stated:
#     GATE (per arm, else that arm is not quotable at all): R^2 >= 0.95, rate
#       > 0.1 /day, and eke(t_end)/eke(t=0) > 10 — the last one is what keeps a
#       regridding floor or a saturated series from being read as growth.
#     READ AS GROWTH-DEFICIENT only if ALL THREE hold:
#       (i)   sigma_cube(EKE)  <= 0.75 * sigma_ref(EKE);
#       (ii)  the t=0-normalised EKE ratio falls by >= 25 % from day 2 to day 6;
#       (iii) the NATIVE, regrid-free p_s' rate also has
#             sigma_cube <= 0.75 * sigma_ref.
#     READ AS NOT GROWTH-DEFICIENT if sigma_cube >= 0.9 * sigma_ref in the
#       early window on BOTH the EKE and the native p_s' series.
#     Anything else is UNRESOLVED.  A reading that flips between the ico and
#     lat-lon references, or between the EKE and eddy-T-variance fields, is
#     reported as a disagreement and not resolved by picking one.
#
#   No binary CONFIRM/REFUTE threshold on a single slope ratio is declared,
#   deliberately: the rule above requires a regrid-free series to agree.  Both
#   adversarial reviewers (codex, GLM-5.2) showed that a fixed cutoff on a
#   single-run, single-window slope ratio is unfalsifiable here: there is no
#   calibration mapping a J-W growth rate to an equilibrated Held-Suarez jet
#   speed, and the early/late windows straddle nonlinear saturation.  This
#   experiment reports "consistent / inconsistent with" and nothing stronger.
#
#   A rate is only quotable if the EARLY and LATE windows agree; where they do
#   not, the wave has saturated inside the window and no single rate exists.
#   The eke(t=0) column is the regridding-floor control: a grid whose remap
#   injects a time-constant eddy floor (cube panel seams are the candidate)
#   shows an anomalous t=0 value and its fitted slope is biased DOWNWARD.
#
#   Why no cheaper offline test: a growth rate only exists once the wave has
#     been integrated; no saved checkpoint holds a cross-grid comparable eddy
#     state.  These are 10-day 1.25-degree runs (~20 s each on one A100), far
#     under the 1 GPU-hour threshold that would demand more justification.
#
# LIMITS ACCEPTED (both reviewers; recorded rather than silently carried):
#   * EKE is computed on the shared 181x360 canvas, so each arm arrives through
#     a different effective interpolation filter.  That biases the LEVEL per
#     grid and cancels from a RATE only insofar as the filter is constant in
#     time.  Native-grid EKE with per-grid area weights is the stronger
#     instrument and is NOT built here.
#   * One deterministic run per arm: no ensemble, so no error bar on the ratio.
#   * Winds are the lowest model level (~988 hPa at L40 sigma, identical index
#     and identical uniform-sigma column on every arm), not the 850/500 hPa
#     surfaces where baroclinic conversion peaks.  The eddy temperature
#     variance at level 27 is the independent-field cross-check for exactly
#     this reason; v'T' would need 3-D winds, which the snapshots do not carry.
#
# CONTROLLED COMPARISON.  All arms are the SAME matrix case (hydrostatic /
# baroclinic, 10 days, DEFAULT_NLEV, matrix-default dt and precision per grid)
# and differ ONLY in grid type.  The fourth arm is the ONE-VARIABLE diffusion
# control: cube with LEGOESM_AH_SCALE=0.1, the same reduction the #1028 200-d
# Held-Suarez arms used.  Matrix defaults are kept everywhere else so the
# protocol stays byte-fixed to the #1028 baseline (in particular the cube PE
# stays float32 by design; forcing x64 here would change the compared thing).
#
# The worktree is put FIRST on PYTHONPATH so every legoesm.* subpackage
# resolves to this checkout and not to the shared one (which other sessions
# edit concurrently).
set -euo pipefail

WT=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/bci-1028
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
OUT="${OUT:-${WT}/results/bci_1028}"

cd "${WT}"
PYTHONPATH="$(ls -d "${WT}"/packages/*/ | paste -sd:)":"${WT}/src":"${WT}"
export PYTHONPATH
export JAX_PLATFORMS=cuda

echo "sha=$(git rev-parse --short HEAD)  host=$(hostname)  $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PY}" -c "import jax; print('jax', jax.__version__, jax.devices())"

run_arm () {          # $1 = label, $2 = --grid value, $3.. = env assignments
    local label="$1" grid="$2"; shift 2
    echo "=== arm ${label} (grid=${grid}) ${*} $(date -u) ==="
    # A nonzero exit is EXPECTED here and must not abort the sweep: --test
    # selects the sigma AND hybrid entries of this case, and the hybrid
    # coordinate carries negative layer mass for p_s < 663.9 hPa, so the hybrid
    # member NaNs at ~day 10 on every grid.  sigma is the #1028 protocol
    # coordinate and is the one the verdict below rests on; the hybrid member
    # is scored best-effort only.  The per-case PASS/FAIL stays in the arm's
    # summary.json, so nothing is being swallowed silently.
    env "$@" "${PY}" -u scripts/matrix/run_atmosphere_test_matrix.py \
        --only hydro --test =baroclinic --grid "${grid}" \
        --no-cross-grid-plots -o "${OUT}/${label}" \
        || echo "[arm ${label}] matrix exited nonzero — see ${OUT}/${label}/summary.json"
}

run_arm latlon     latlon
run_arm ico        icosahedral
run_arm cube       cubed_sphere
run_arm cube_ah01  cubed_sphere LEGOESM_AH_SCALE=0.1

score () {             # $1 = vertical coordinate
    local vert="$1"
    "${PY}" -u scripts/validate/bci_growth_rate.py --self-test \
        latlon="${OUT}/latlon/hydrostatic/baroclinic/latlon/72x144/${vert}" \
        ico="${OUT}/ico/hydrostatic/baroclinic/icosahedral/ico5/${vert}" \
        cube="${OUT}/cube/hydrostatic/baroclinic/cubed_sphere/C36/${vert}" \
        cube_ah01="${OUT}/cube_ah01/hydrostatic/baroclinic/cubed_sphere/C36/${vert}" \
        --reference "${2:-ico}" \
        --json-out "${OUT}/growth_rates_${vert}_ref${2:-ico}.json"
}

# sigma carries the reading (the #1028 protocol coordinate) and must succeed.
# ico is the primary reference: it is the grid closest to the ~30 m/s
# Held-Suarez target, whereas lat-lon overshoots it at 66, so a ratio against
# lat-lon compares the cube to a core that is itself wrong by ~2x (GLM-5.2
# review S6).  Both references are reported; a reading that flips between them
# is not a reading.
echo "=== growth rates (sigma, reference ico) ==="
score sigma ico
echo "=== growth rates (sigma, reference latlon) ==="
score sigma latlon
# hybrid is best-effort: its members NaN at ~day 10 on the negative-layer-mass
# coordinate, so a failure here is expected and is NOT allowed to mask sigma.
echo "=== growth rates (hybrid, best-effort) ==="
score hybrid ico || echo "[hybrid] not scorable (expected — negative layer mass)"
echo "DONE $(date -u)"
