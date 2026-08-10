#!/usr/bin/env bash
# Per-rank srun wrapper: run the target under `nsys profile` for the ranks in
# $PROFILE_RANKS, plain $PY_BIN for every other rank.
#
# Profiling all 64 ranks is not an option (~400 MB each), and profiling an
# arbitrary subset answers nothing: skew is measured BETWEEN PARTNERS, so the
# selected ranks must actually exchange with each other AND sit on one node
# (see the launcher for why cross-node timestamps are not comparable).
set -uo pipefail

: "${NSYS_BIN:?}"
: "${NSYS_OUTDIR:?}"
: "${PROFILE_RANKS:?}"
: "${PY_BIN:?}"
: "${SLURM_PROCID:?}"

# Rank -> node map, so co-location of the profiled ranks is a RECORD and not
# an assumption about how Slurm laid the job out.
printf 'rank=%s\tnode=%s\tpid=%s\n' "$SLURM_PROCID" "$(hostname -s)" "$$" \
    >> "${NSYS_OUTDIR}/_rank_nodes.tsv"

in_list() {
    local r="$1" x
    local IFS=','
    for x in $PROFILE_RANKS; do [ "$x" = "$r" ] && return 0; done
    return 1
}

if in_list "$SLURM_PROCID"; then
    out="${NSYS_OUTDIR}/rank_${SLURM_PROCID}.nsys-rep"
    # Flags VERIFIED against nsys 2023.2.3 on Levante:
    #   --trace: valid tokens are cuda,nvtx,cublas,cusparse,mpi,oshmem,ucx,
    #            osrt,cudnn,opengl,... There is NO 'nccl' token in this
    #            version, so NCCL kernels are captured as ordinary CUDA
    #            kernels (which is all the analysis needs -- it matches
    #            ncclDevKernel_SendRecv by name).  Asking for nccl here would
    #            abort the run.
    #   --export=sqlite: exists as a profile sub-option in this version.
    #   osrt is deliberately OMITTED: it adds thread-level overhead, and
    #            perturbing the very timing under test is the one thing this
    #            capture cannot afford.
    # NO --delay/--duration: this run is ~7.3 s of compile plus 12 steps of
    # ~10 ms, so any delay long enough to skip compile also skips the entire
    # steady state.  Capture everything; the analysis drops warmup cycles.
    # NO `--` separator before the application: nsys 2023.2.3 parses a bare
    # `--` as an ambiguous long-option abbreviation and aborts with "option
    # is ambiguous and matches ..." (verified — it killed job 26771842's four
    # profiled ranks). The application simply follows the flags.
    # `env -u QUADD_INJECTION_PROXY` is REQUIRED, not hygiene. nsys sets that
    # variable for its injection library, and JAX treats any *_PROXY env var
    # as distributed-coordinator proxy configuration -- it prints "JAX
    # detected proxy variable(s) ... may cause a hang of
    # distributed.initialize" and then does exactly that. Only the PROFILED
    # ranks get the variable, so they diverge from the other 60 and the whole
    # job deadlocks in init (observed: job 26771961 hung with precisely four
    # such warnings, one per profiled rank, and produced nothing).
    # Unsetting it here is safe: nsys's injection is already active via
    # LD_PRELOAD by the time this exec runs, so the variable has done its job.
    exec "$NSYS_BIN" profile \
        --trace=cuda,nvtx \
        --output="$out" \
        --force-overwrite=true \
        --export=sqlite \
        env -u QUADD_INJECTION_PROXY "$PY_BIN" "$@"
else
    exec "$PY_BIN" "$@"
fi
