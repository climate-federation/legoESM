#!/bin/bash
# Submit a Ginsburg campaign job, first creating the log directory SLURM opens
# for --output. SLURM does NOT create the --output parent dir; a raw
# `sbatch aimip_ace2loss.sbatch` on a fresh checkout fails at scheduling with
# "Unable to open file". This helper reads the #SBATCH --output directive from
# the target wrapper and mkdir -p's its dir first.
#
# Usage:  ./submit.sh aimip_ace2loss.sbatch [extra sbatch args...]
#
# ponytail: relative --output paths (results/...) resolve against the CWD sbatch
# is run from — run this from the repo root, same assumption the wrappers make.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
job="${1:?usage: submit.sh <wrapper.sbatch> [sbatch args...]}"
shift
target="$here/$job"
[ -f "$target" ] || { echo "no such wrapper: $target" >&2; exit 1; }

out=$(awk -F= '/^#SBATCH[[:space:]]+--output=/{print $2; exit}' "$target")
[ -n "$out" ] && mkdir -p "$(dirname "$out")"

exec sbatch "$target" "$@"
