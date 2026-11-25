#!/usr/bin/env bash
# Fixture-only mpirun replacement for sandboxes that forbid PMIx sockets.
set -Eeuo pipefail
[[ ${1:-} == -np && ${2:-} == 1 ]] || exit 64
shift 2
if [[ ${1:-} == --oversubscribe ]]; then shift; fi
[[ $# -gt 0 ]] || exit 64
exec "$@"
