#!/bin/bash
# Set up the two Julia environments that run CliMA SurfaceFluxes.jl's
# universal functions (the stability-function oracle for psi_m / psi_h).
# Usage: build.sh <julia binary> <SurfaceFluxes.jl checkout> <env dir>
#   v1.5.1 = fd42723427b5264b3e5f8ee9f622cc7b2e94fdae (Businger, Gryanik, Grachev)
#   v0.12.0 = 167c5144c0171cc425432ad26d6569552a73ee11 (last tag with Beljaars-Holtslag)
# The tier-2 audit ran Julia 1.10.10 with ClimaParams 1.3.3 (v1.5.1) and
# 0.10.35 (v0.12.0).  Needs network access the first time (Pkg).
set -euo pipefail
J=${1:?julia}; SF=${2:?SurfaceFluxes.jl checkout}; E=${3:?env dir}
git -C "$SF" diff --quiet HEAD -- src || { echo "SurfaceFluxes.jl src/ has local changes" >&2; exit 1; }
[ -d "$E/SF_v0120" ] || git -C "$SF" worktree add -q "$E/SF_v0120" 167c5144c0171cc425432ad26d6569552a73ee11
[ -d "$E/SF_v151" ] || git -C "$SF" worktree add -q "$E/SF_v151" fd42723427b5264b3e5f8ee9f622cc7b2e94fdae
[ "$(git -C "$E/SF_v0120" rev-parse HEAD)" = 167c5144c0171cc425432ad26d6569552a73ee11 ] || { echo "SF_v0120 wrong SHA" >&2; exit 1; }
[ "$(git -C "$E/SF_v151" rev-parse HEAD)" = fd42723427b5264b3e5f8ee9f622cc7b2e94fdae ] || { echo "SF_v151 wrong SHA" >&2; exit 1; }
git -C "$E/SF_v0120" diff --quiet HEAD -- src && git -C "$E/SF_v151" diff --quiet HEAD -- src || { echo "SF worktree src/ has local changes" >&2; exit 1; }
mkdir -p "$E/jl151" "$E/jl0120"
$J --project="$E/jl151" -e 'using Pkg; Pkg.develop(path=ARGS[1]); Pkg.add(name="ClimaParams", version="1.3.3"); Pkg.add("DelimitedFiles"); Pkg.instantiate()' "$E/SF_v151"
$J --project="$E/jl0120" -e 'using Pkg; Pkg.develop(path=ARGS[1]); Pkg.add(name="ClimaParams", version="0.10.35"); Pkg.add("DelimitedFiles"); Pkg.instantiate()' "$E/SF_v0120"
