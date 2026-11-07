#!/bin/bash
#SBATCH --job-name=cdx_merge
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_merge_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=$WT/omip_nemo/cdx_merge_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarial review of merge commit 0a4398266 (origin/main -> fix/omip-gm-treguier-one-variable, 97 commits of NEMO test-case / WS-RK3 / OVERFLOW fidelity work). Git reported NO conflicts, but per this repo's rules a merge is the highest-risk change class, so audit it as if it had them. Exactly ONE file was modified by both sides: packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py. OUR side (the OMIP branch) changed the north-fold pivot handling in tvd_to_v_points (~line 424: pivot_row_stored selects f[-2] vs f[-1] before grid.fold.perm_T) and in een_e3f_h_vtx (~line 2160, pivot-layout F/vertex row). THEIR side added nemo_up3 to VALID_VERTICAL_MOMENTUM_SCHEME and its dispatch in _bc_vertical_momentum_advection, made pgf_scheme='nemo_sco' RAISE when t_depth_ref is absent (previously an arithmetic-midpoint fallback), and threaded momentum_flux_transport_velocity into _bc_horizontal_momentum_advection_flux_form / latlon_cgrid_ocean_baroclinic_tendencies. TASKS: (1) read the merged file and confirm the two edit regions did not interleave semantically — in particular that the fold/pivot code still runs for every caller and that no branch now reads a variable defined only on one side; (2) the nemo_sco t_depth_ref change is a behaviour change from silent-fallback to raise: does any OMIP card or code path on this branch (grep run_omip_core2.py, scripts/cluster/omip_nemo/*.sbatch) select pgf_scheme=nemo_sco with a z_coord lacking t_depth_ref, i.e. would a previously-working tripole run now abort? (3) latlon_cgrid_ocean_baroclinic_tendencies gained a keyword argument — verify every call site on this branch still matches, including the SPMD/sharded lane and any partial application; (4) did the merge bring any change to the vertical-mixing, freshwater, sea-ice or forcing modules our OMIP lane depends on (compare git diff HEAD~1..HEAD for those paths) that would silently change our running cards; (5) any duplicated or now-dead code from the two sides touching the same helpers. Post-merge tests: our surface 184 passed, their NEMO/RK3 surface 90 passed. Rank CRITICAL/MAJOR/MINOR, cite file:line, CONFIRMED vs PLAUSIBLE, under 500 words." > $LOG 2>&1
echo "codex rc=$?"
grep -q "Failed to refresh token\|refresh token was already used" $LOG && echo "CODEX AUTH FAILURE"
grep -vE "^\s*$" $LOG | tail -45
echo DONE
