#!/usr/bin/env bash
# Run ocean MPI conservation tests with valid rank counts.
#
# The cubed-sphere grid has 6 faces.  MPI ranks must divide into the
# 6-face topology, so only certain rank counts are valid:
#
#   Valid: 1, 2, 3, 6, 12, 24, ...  (1-3 divide 6; >6 must be a multiple of 6)
#   Invalid: 4, 5, 7, 8, 9, 10, 11, ...
#
# Usage:
#   ./scripts/run_ocean_mpi_tests.sh          # default: 6 ranks
#   ./scripts/run_ocean_mpi_tests.sh 3        # 3 ranks
#   ./scripts/run_ocean_mpi_tests.sh 12       # 12 ranks
set -euo pipefail

NPROCS="${1:-6}"

# ---- validate rank count ----
is_valid() {
    local n=$1
    case $n in
        1|2|3|6) return 0 ;;
    esac
    if [ "$n" -gt 6 ] && [ $((n % 6)) -eq 0 ]; then
        return 0
    fi
    return 1
}

if ! is_valid "$NPROCS"; then
    echo "ERROR: $NPROCS ranks cannot tile 6 cubed-sphere faces."
    echo ""
    echo "Valid rank counts:"
    echo "  Small: 1, 2, 3, 6"
    echo "  Large: any multiple of 6 (12, 24, 48, ...)"
    echo ""
    echo "Usage: $0 [NPROCS]   (default: 6)"
    exit 1
fi

echo "=== Ocean MPI conservation tests ($NPROCS ranks) ==="
exec mpirun -np "$NPROCS" python -m pytest \
    tests/ocean/distributed/test_ocean_mpi_conservation.py \
    -v "$@"
