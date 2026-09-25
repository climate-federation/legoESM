#!/bin/bash
#SBATCH --job-name=cdx_stacc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_stacc_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Adversarial review of the UNCOMMITTED diff (run: git diff). Three files:
scripts/run/run_omip_core2.py (_SurfaceFluxAccumulator gains state=None: sums state.T.data /
state.S.data per step, drained as T_mean/S_mean into the snapshot extra; new --state-accumulate
flag; scan-lane SystemExit guard; both host-loop call sites pass state),
scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py (fixes from your previous review: --mean
now on a fixed finite common support with no nan_to_num, duplicate-day and empty-box are hard
errors, boxes come from compare_omip_nemo._BOXES, NEMO temporal metadata is READ from the file
attrs and time_counter_bounds, new --use-mean-fields reads T_mean),
tests/ocean/unit/test_surface_flux_accumulator.py (two new tests).
Purpose: NEMO ORCA1 writes 5-day MEANS; we compared 00 UTC snapshots and half the nino3 SST
bias was diurnal phase. T_mean makes the comparison window-matched.
Numbered, severity-tagged findings, file:line, what to change:
1. Is the state mean correct for a z-star column (T lives on cells whose thickness varies with
   SSH in time)? A plain time-mean of T is what NEMO's tos/thetao 5-day mean IS (check: NEMO
   xios averages the field, not thickness-weighted) -- state whether that is true and cite.
2. Is state.T.data the prognostic at the END of the step or the START at each call site (the
   add is called before the step call)? Does that mis-align the window by one step vs the
   flux mean, and does the drain happen before or after the snapshot's state is written?
3. Memory: T/S are 3-D device arrays summed every step on GPU -- any device/host thrash,
   dtype accumulation error in float32 over ~1440 steps, or growth of the snapshot npz?
4. In the probe: does --use-mean-fields silently fall back to instantaneous T when T_mean is
   absent (a silent fallback is banned here)? Is the finite-common-support mean still biased
   when the support shrinks (e.g. sea-ice cells)? Any remaining way --mean can manufacture a
   cooling or warming?
5. Are the two new tests non-vacuous (would they fail if the state sum were dropped, or if the
   scan guard were removed)?" 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="
