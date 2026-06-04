#!/bin/bash
# Full validation: finish current resolution + double resolution
set -euo pipefail
cd "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM"

PYTHON=".venv/bin/python"
export JAX_ENABLE_X64=1

LOG="results/validation_full.log"
mkdir -p results
echo "=== Full Validation Run Started: $(date) ===" | tee "$LOG"

run_atm() {
    local desc="$1"; shift
    echo "--- ATM: $desc ---" | tee -a "$LOG"
    $PYTHON scripts/matrix/run_atmosphere_test_matrix.py --quick "$@" 2>&1 | tee -a "$LOG"
    echo "--- ATM: $desc DONE: $(date) ---" | tee -a "$LOG"
}

run_ocn() {
    local desc="$1"; shift
    echo "--- OCN: $desc ---" | tee -a "$LOG"
    $PYTHON scripts/matrix/run_ocean_test_matrix.py --quick "$@" 2>&1 | tee -a "$LOG"
    echo "--- OCN: $desc DONE: $(date) ---" | tee -a "$LOG"
}

# =====================================================
# PHASE 1: Finish base resolution (C36 / 72x144 / ico5 / T21)
# =====================================================
echo "=== PHASE 1: Base Resolution Remaining ===" | tee -a "$LOG"

# These were already completed: sw (all grids), hydro cubed_sphere, hydro latlon (partial)
# Run remaining subsets:
run_atm "hydro latlon remaining" --only hydro --grid latlon --output results/atmosphere_latlon2
run_atm "hydro icosahedral"      --only hydro --grid icosahedral --output results/atmosphere_ico
run_atm "hydro spectral"         --only hydro --grid spectral --output results/atmosphere_spectral
run_atm "nonhydrostatic all"     --only nh --output results/atmosphere_nh

# Ocean
run_ocn "base resolution" --output results/ocean

echo "=== PHASE 1 COMPLETE: $(date) ===" | tee -a "$LOG"

# =====================================================
# PHASE 2: Double Resolution
#   Atmosphere: C72 / 144x288 / ico6 / T42
#   Ocean:      C48 / 72x144 / ico4 / T42
# =====================================================
echo "=== PHASE 2: Double Resolution ===" | tee -a "$LOG"

# Atmosphere 2x (per-grid with appropriate resolution)
run_atm "2x cubed_sphere C72"  --grid cubed_sphere --resolution C72    --output results/atmosphere_2x
run_atm "2x latlon 144x288"   --grid latlon        --resolution 144x288 --output results/atmosphere_2x
run_atm "2x icosahedral ico6" --grid icosahedral    --resolution ico6   --output results/atmosphere_2x
run_atm "2x spectral T42"     --grid spectral       --resolution T42    --output results/atmosphere_2x

# Ocean 2x (per-grid with appropriate resolution)
run_ocn "2x cubed_sphere C48" --grid cubed_sphere --resolution C48    --output results/ocean_2x
run_ocn "2x latlon 72x144"   --grid latlon        --resolution 72x144 --output results/ocean_2x
run_ocn "2x mpas ico4"       --grid mpas           --resolution ico4   --output results/ocean_2x
run_ocn "2x spectral T42"    --grid spectral       --resolution T42    --output results/ocean_2x

echo "=== PHASE 2 COMPLETE: $(date) ===" | tee -a "$LOG"
echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
