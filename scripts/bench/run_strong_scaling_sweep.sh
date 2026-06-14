#!/usr/bin/env bash
# Local strong-scaling sweep across resolutions for all three grids,
# on both CPU (default backend) and GPU (via gpu_env.sh).  Captures
# steps/s for each (grid, resolution, backend) cell and writes a JSON
# summary to ``results/scaling/strong_sweep.json`` for the plotter.
#
# Per cell the BCW benchmark integrates 2 days of the JW dry baroclinic
# wave; the steps/s reported is the mean over the warm window (after
# the JIT compile step).  Skip cells that are known unstable
# (cubed-sphere C24 — see scaling.md §3.b).

set -e
cd "$(dirname "$0")/../.."

mkdir -p results/scaling
SUMMARY="results/scaling/strong_sweep.json"
TAG="${TAG:-iter201_strong}"

run_one() {
    local backend="$1" grid="$2" res="$3" dt="$4" days="$5" extra="$6"
    local label="${backend}_${grid}_${res}"
    local out_file="output/baroclinic_wave_diagnostics_${TAG}_${label}.npz"
    rm -f "$out_file"

    if [[ "$backend" == "gpu" ]]; then
        env -i HOME=$HOME PATH=$PATH JAX_ENABLE_X64=1 bash -c \
            "source scripts/data/gpu_env.sh >/dev/null && \
             PYTHONPATH=. .venv/bin/python scripts/run/run_baroclinic_wave_benchmark.py \
                 --grid '$grid' --resolution '$res' --dt '$dt' --days '$days' \
                 --tag '${TAG}_${label}' $extra 2>&1" \
            | grep -E "Integration complete|Wall time|drift|BLOWUP" || true
    else
        JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python \
            scripts/run/run_baroclinic_wave_benchmark.py \
            --grid "$grid" --resolution "$res" --dt "$dt" --days "$days" \
            --tag "${TAG}_${label}" $extra 2>&1 \
            | grep -E "Integration complete|Wall time|drift|BLOWUP" || true
    fi
}

echo "==== Strong-scaling sweep (TAG=$TAG) ===="
# Use 1 day per cell so the full sweep finishes inside an iteration
# budget; warm-step throughput is unchanged.  CPU runs at higher
# resolutions are expensive, so the higher-resolution CPU rows are
# deliberately limited (no C96 CPU).
#
# iter-219 scan-steps auto-tuning: the BCW benchmark accepts
# ``--scan-steps auto`` to pick the iter-212 honest per-grid
# recommendation (spectral=24, cubed-sphere=24, icosahedral=1).  Use
# a single uniform flag here so future per-grid retunes only need
# to land in the benchmark, not in this driver.
SS="--scan-steps auto"

for backend in gpu cpu; do
    echo
    echo "--- backend=$backend ---"
    run_one $backend spectral 21 600 1 "$SS"
    run_one $backend spectral 42 300 1 "$SS"
    run_one $backend icosahedral 4 150 1 "$SS"
    run_one $backend icosahedral 5 75  1 "$SS"
    # cubed-sphere C48 / C96 (skip C24 — see §3.b resonance).  C96
    # CPU run is intentionally skipped (>10 min on this host).
    run_one $backend cubed-sphere 48 150 1 "$SS"
    if [[ "$backend" == "gpu" ]]; then
        run_one $backend cubed-sphere 96 75 1 "$SS"
    fi
done

echo
echo "All cells written to output/*$TAG*.npz; summarize with scripts/plot/plot_scaling.py"
