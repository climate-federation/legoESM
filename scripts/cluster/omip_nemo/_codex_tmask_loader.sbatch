#!/bin/bash
#SBATCH --job-name=cdx_tmask
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_tmask_%j.out
# Dual review (codex; GLM done) of the masked NEMO-IC regrid fix.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=$WT/omip_nemo/cdx_tmask_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarial review of 'git diff ac990e395~1' (commit ac990e395 plus uncommitted follow-ups) in packages/ocean/legoesm/ocean/forcing/nemo_native_fields.py, scripts/run/run_omip_core2.py and tests/ocean/unit/test_nemo_native_fields.py. Context: NEMO ORCA1's IC files (woce_{temp,salt}_monthly_init_4p2.nc, (12,75,331,360)) and sss_climatology_for_restoring.nc are FLOOD-FILLED over land/below-seafloor (40914 of 106208 finite surface cells are tmask=0 land; 79025 at level 70); NearestWetRegridder took isfinite as wet so MPAS/FESOM coastal columns sampled land fills (FESOM Ligurian column got S~31 at 200 m, eta 1.3 m after one step, NaN by step 12). Fix: read_nemo_tmask_interior (eORCA1.2 mesh_mask tmask, interior [:-1,1:-1], coordinate-proven to 7.6e-6 deg by nemo_src_tmask_for which RAISES on shape or coordinate mismatch), src_tmask per level into load_nemo_monthly_init_ts and load_nemo_sss_restoring_climatology via _source_wet = isfinite & tmask; an all-land source level leaves NaN for the pre-existing below-seafloor fill; all four driver call sites pass nemo_src_tmask_for(args.mesh|_MESH, <file>). Attack: (1) the tripole native-embed path (coords_match short-circuit) — is any tripole IC value changed other than masked cells? (2) the per-level loop: the regridder built from T is reused for S (measured finiteT&~finiteS == 0 at all levels) — any level where the mask differs between the two files? (3) the all-land-level continue: T_out/S_out are np.empty — is every level either written or set to NaN before the fill loop? (4) the fill loop after masking: coastal FESOM/MPAS columns deeper than NEMO's local bathymetry now inherit NEMO's local bottom value — correct? any path to a level-mean fill for a real wet column? (5) call sites: args.mesh default _MESH = data/grids/eORCA1.2_mesh_mask.nc; latlon/eORCA025 runs that pass another --mesh with these ORCA1 files now RAISE — enumerate cards under scripts/cluster that would (grep --mesh and --nemo-monthly-init / --sss-restore-file). (6) the test: does it fail if src_tmask is ignored (non-vacuity) and does the coordinate-mismatch case exercise the real code path? Rank CRITICAL/MAJOR/MINOR, file:line, CONFIRMED vs PLAUSIBLE, under 500 words." > $LOG 2>&1
echo "codex rc=$?"
grep -q "Failed to refresh token\|refresh token was already used" $LOG && echo "CODEX AUTH FAILURE"
sed -n '/^Verdict\|^\*\*\|^- \*\*/,$p' $LOG | head -80
echo DONE
