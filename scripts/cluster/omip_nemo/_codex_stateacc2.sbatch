#!/bin/bash
#SBATCH --job-name=cdx_stacc2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_stacc2_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Round 2. Review commit HEAD (git show HEAD) against your round-1 HOLD on the same files. Round-1 findings and what changed:
1 restart fingerprint -> state_accumulate added to _RESTART_FP_EXCLUDE, test covers all three flags;
2 --mean temporal bias -> finite time_days required; with --use-mean-fields the mean is duration-weighted by flux_mean_window_s and windows must tile (gap/overlap -> SystemExit); otherwise labelled SAMPLE MEAN; NEMO tos online_operation/5-day span enforced;
3 start-of-step sampling -> add() now follows the step at both host-loop sites (T_1..T_n), new source test;
4 per-row support -> one declared support from NEMO+first row, every row must be finite on it, retained area printed per box;
5 scan test -> guard block scoped to this guard (split at next top-level if);
6 oracle averaging -> read from the file: tos op=average interval 3600 s; to = @toce_e3t/@e3t freq_op 5d (file_def_nemo-oce.xml:79); to[k=0]-tos = 0.000 in nino3, max 0.074 globally; we keep a PLAIN mean, bound recorded;
7 dtype -> sums in jax.dtypes.canonicalize_dtype(float64).
Numbered, severity-tagged: (a) any of the seven still open or regressed? (b) is the post-step move correct w.r.t. what the fluxes sf and the MLD now describe (sf applied over the step that produced the sampled state)? (c) does canonicalize_dtype(float64) under JAX_ENABLE_X64=1 give float64 and under x64-off float32 without a warning, and is the sum still on device? (d) any way the tiling check passes on windows that do not actually tile (e.g. first window shorter after a restart)? (e) is the plain-vs-thickness-weighted bound of 0.074 a safe waiver for POLAR SSS (so is @soce_e3t/@e3t too) or should S_mean be thickness-weighted? Concrete file:line." 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="

