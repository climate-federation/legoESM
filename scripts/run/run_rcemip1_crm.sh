#!/usr/bin/env bash
# RCEMIP1 (Wing et al. 2018) plane-CRM ocean RCE at SST 300 K — replication driver.
#
# This is the configuration that actually deep-convects, promoted out of the
# untracked scratch scripts it previously lived in (scripts/tmp/run_rce60.sh and
# scripts/tmp/_micro_assess.sh) so a clean checkout can reproduce it.
#
# Two phases, matching how the reference result was produced:
#
#   spinup      128^2 x 100, dx = 4 km, dt = 6 s, RRTMGP radiation + clouds,
#               Kessler microphysics, band-limited theta' seed, run to 60 days.
#               Gets the domain to a convecting, statistically steady RCE
#               state; checkpoints every day.
#   production  restart from that checkpoint with RRTMGP + clouds and the
#               microphysics scheme you actually want to assess, then grade the
#               time mean against the published RCEMIP 300 K ranges via
#               scripts/validate/compare_rce_vs_rcemip_sam.py.
#
# Usage:
#   scripts/run/run_rcemip1_crm.sh spinup                   # 60 d, resumable
#   scripts/run/run_rcemip1_crm.sh production morrison      # 1 d RRTMGP restart
#   DAYS=1 NX=32 PLATFORM=cpu scripts/run/run_rcemip1_crm.sh spinup   # smoke
#
# Env overrides: DAYS NX NY NLEV DX DT OUT SPINUP_OUT PLATFORM PY CKPT CKPT_EVERY
#                SEED_AMP SEED_KMAX HYPERDIFF RAD_INTERVAL VALIDATE
#                SPINUP_RAD SPINUP_MICRO
#
# THE SETTING THAT DECIDES WHETHER IT CONVECTS AT ALL
# ---------------------------------------------------
# run_rcemip_plane.py defaults to --theta-noise-amp 0.0 with --seed-kind
# smooth_k1, i.e. NO initial perturbation. A horizontally uniform rest state
# stays horizontally uniform, so the run completes cleanly, conserves mass and
# exits 0 — while being a pure radiative-equilibrium column replicated across
# every grid point, with max|w| of order 1e-4 m/s and no convective cells.
# Nothing in the log flags it. docs/dev-notes/CRM_faithful_SAM.md records the
# same trap at iter-175/176 ("launch bug: --theta-noise-amp default 0.0 =>
# un-seeded RCE stays laminar") and as CONV-TRIGGER #83. band_noise rather than
# smooth_k1 matters because smooth_k1 puts all the seed energy in one k=1
# cosine, which organises a single domain-filling circulation instead of a
# population of cells.
#
# The seed only stays stable in the resolution/timestep pairing below: nlev=100
# with dt=6 s. The same seed at nlev=30 / dt=20 s goes non-finite within ~25
# outer steps, so do not mix and match the vertical grid and the timestep.
set -euo pipefail

cd "$(dirname "$0")/../.."

PHASE=${1:-spinup}
MICRO_ARG=${2:-}

NX=${NX:-128}
NY=${NY:-$NX}
NLEV=${NLEV:-100}
DX=${DX:-4000}
DT=${DT:-6}
SEED_AMP=${SEED_AMP:-0.05}
SEED_KMAX=${SEED_KMAX:-8}
HYPERDIFF=${HYPERDIFF:-1.0e7}
RAD_INTERVAL=${RAD_INTERVAL:-150}
PLATFORM=${PLATFORM:-cuda}
PY=${PY:-.venv/bin/python}
SPINUP_OUT=${SPINUP_OUT:-results/rcemip300_60day}
VALIDATE=${VALIDATE:-1}
# Extra run_rcemip_plane.py flags, appended verbatim to both phases (e.g.
# DRIVER_FLAGS="--hard-saturation-adjustment --hard-sat-adjust-threshold 1.0").
# The threshold flag is --hard-sat-adjust-threshold: --hard-sat-threshold, the
# name this example used to carry, is not a flag any more and argparse rejects
# it. The IN-SCHEME adjustment above is the preferred mechanism; the post-step
# hook is the separate --poststep-saturation-drain family (#1559 renamed it out
# of the collision), and the driver refuses both at once.
DRIVER_FLAGS=${DRIVER_FLAGS:-}

# RRTMGP everywhere, including the spin-up. Gray radiation is cheaper but it
# UNDER-DRIVES the circulation: docs/dev-notes/CRM_faithful_SAM.md #85 measured
# w_RMS ~0.07 m/s and an inverted w'^2 profile under gray, versus healthy
# mid-tropospheric convection under RRTMGP, and traced the whole
# convective-intensity deficit to gray under-driving. A state spun up under gray
# is therefore not the state RRTMGP would have equilibrated to, and the
# production leg inherits that bias. Set SPINUP_RAD=gray to trade fidelity for
# wall-clock.
SPINUP_RAD=${SPINUP_RAD:-rrtmgp}
SPINUP_MICRO=${SPINUP_MICRO:-kessler}
# Pass --clouds explicitly only for the band model, where it means something.
# run_rcemip_plane.py defaults --clouds ON regardless, so this controls the
# command line, not the cloud setting (use --no-clouds to actually disable).
if [ "$SPINUP_RAD" = "rrtmgp" ]; then SPINUP_RAD_EXTRA="--clouds"; else SPINUP_RAD_EXTRA=""; fi

# Noise suppression: the RRTMGP loader and the CUDA probe are both very chatty.
FILT='hwloc|mailing|topology|ignore this invalid|^\*|kernel mode|UserWarning|jnp\.|return lax|set_T_sfc|jax_enable_x64|See https|truncated|astype|array_dict'

day_steps() { $PY -c "print(int(round(86400.0/$DT)))"; }

common_flags() {
    echo "--nx $NX --ny $NY --nlev $NLEV --dx $DX --dt $DT \
          --T-sfc 300.0 --insolation rcemip \
          --precision float32 --semi-implicit \
          --radiation-interval $RAD_INTERVAL --hyperdiff $HYPERDIFF"
}

case "$PHASE" in

spinup)
    DAYS=${DAYS:-60}
    OUT=${OUT:-$SPINUP_OUT}
    STEPS=$($PY -c "print(int(round($DAYS*86400.0/$DT)))")
    DAY_STEPS=$(day_steps)
    CKPT_EVERY=${CKPT_EVERY:-$DAY_STEPS}
    mkdir -p "$OUT"

    echo "RCEMIP1 spin-up: ${NX}x${NY}x${NLEV}, dx=${DX} m, dt=${DT} s, ${DAYS} d (${STEPS} steps)"
    echo "  ${SPINUP_RAD} radiation + ${SPINUP_MICRO}; seed amp=${SEED_AMP} K band_noise kmax=${SEED_KMAX}"
    echo "  output: ${OUT}"

    # --restart latest makes this idempotent: re-running resumes rather than
    # restarting from the IC, so a 60-day run can be done in sittings.
    LEGOESM_RCEMIP_PLANE_FP32=1 JAX_PLATFORMS=$PLATFORM $PY \
        scripts/run/run_rcemip_plane.py $(common_flags) \
        --radiation "$SPINUP_RAD" $SPINUP_RAD_EXTRA \
        --microphysics "$SPINUP_MICRO" \
        --seed-kind band_noise --seed-kmax "$SEED_KMAX" \
        --theta-noise-amp "$SEED_AMP" \
        --steps "$STEPS" --print-every 4000 \
        --snapshot-days 1.0 --snapshot3d-days "15,30,45,60" \
        --snapshot-heights "1000,5000,9000,12000" \
        --checkpoint-every "$CKPT_EVERY" \
        --restart latest $DRIVER_FLAGS \
        --output "$OUT" 2>&1 | { grep --line-buffered -vE "$FILT" || true; } | tee "$OUT/run.log"

    # Plots are a convenience, not the deliverable: a short run has too few
    # snapshots to render, and that must not fail the wrapper.
    $PY scripts/plot/plot_rcemip_snapshots.py "$OUT" 2>&1 | { grep --line-buffered -vE "$FILT" || true; } | tail -3 || true
    $PY scripts/plot/plot_rcemip_3d.py "$OUT" 2>&1 | { grep --line-buffered -vE "$FILT" || true; } | tail -3 || true
    ;;

production)
    MICRO=${MICRO_ARG:-morrison}
    DAYS=${DAYS:-1}
    OUT=${OUT:-results/rcemip1_crm_$MICRO}
    # `|| true` is load-bearing: under `set -e` + `pipefail` an unmatched glob
    # makes ls exit 1, which fails the assignment and kills the script before
    # the friendly message below can print.
    # rce_checkpoint.save writes ckpt_%09d.npz, so lexicographic order already
    # matches numeric order; sort -V keeps that true if the padding ever changes.
    CKPT=${CKPT:-$(ls -1 "$SPINUP_OUT"/checkpoints/ckpt_*.npz 2>/dev/null | sort -V | tail -1 || true)}
    if [ -z "$CKPT" ]; then
        echo "no spin-up checkpoint in $SPINUP_OUT/checkpoints — run the spinup phase first" >&2
        exit 1
    fi
    SPIN_STEPS=$($PY -c "
import re,sys
print(int(re.search(r'ckpt_0*(\d+)\.npz','$CKPT').group(1)))
")
    RUN_STEPS=$($PY -c "print(int(round($DAYS*86400.0/$DT)))")
    STEPS=$($PY -c "print($SPIN_STEPS + $RUN_STEPS)")
    # ~10 profiles over whatever window was asked for. Scaling this to the RUN
    # length rather than to a fixed day means a short production leg still
    # produces validator input instead of silently producing none.
    SNAP_EVERY=$($PY -c "print(max(1, $RUN_STEPS // 10))")

    # Double-moment schemes cannot inherit single-moment (kessler) condensate:
    # bare mass with zero number concentration is inconsistent and blows up.
    case "$MICRO" in
        kessler|sundqvist) EXTRA="" ;;
        *)                 EXTRA="--restart-reset-condensate" ;;
    esac

    mkdir -p "$OUT"
    echo "RCEMIP1 production: micro=${MICRO}, RRTMGP + clouds, ${DAYS} d from ${CKPT}"
    echo "  output: ${OUT}"

    LEGOESM_RCEMIP_PLANE_FP32=1 JAX_PLATFORMS=$PLATFORM $PY \
        scripts/run/run_rcemip_plane.py $(common_flags) \
        --radiation rrtmgp --clouds --microphysics "$MICRO" \
        --steps "$STEPS" --print-every 2400 --snapshot-every "$SNAP_EVERY" \
        --restart "$CKPT" $EXTRA $DRIVER_FLAGS \
        --output "$OUT" 2>&1 | { grep --line-buffered -vE "$FILT" || true; } | tee "$OUT/run.log"

    if [ "$VALIDATE" = "1" ]; then
        # Average only post-restart snapshots; the window starts at the
        # checkpoint step so none of the spin-up state leaks into the mean.
        $PY scripts/validate/compare_rce_vs_rcemip_sam.py "$OUT" \
            --window-start "$SPIN_STEPS" --no-plot \
            2>&1 | { grep --line-buffered -vE "$FILT" || true; } | tee -a "$OUT/run.log"
    fi
    ;;

*)
    echo "usage: $0 {spinup|production [microphysics-scheme]}" >&2
    exit 2
    ;;
esac

echo "Done: ${OUT}"
