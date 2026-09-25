#!/bin/bash
#SBATCH --job-name=cdx_tkebud
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_tkebud_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_fluxacc_wt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD), dirty $(git -C $WT status --porcelain | wc -l) ==="
codex exec --sandbox read-only -C $WT \
"Adversarial review of the UNCOMMITTED diff (run: git diff) in this worktree. It adds a
per-term TKE step budget (TKEStepBudget, return_budget=) to the backward-Euler TKE
solver in packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py, a unit test
tests/ocean/unit/test_tke_step_budget.py, and a --mode-a-budget stage in
scripts/validate/ocean_fidelity/compare_tendencies_nemo.py that prints our per-term
rates at NEMO's seed beside NEMO's own terms rebuilt from the ORCA1 restart with
zdftke.F90 (NEMO 5.0.1) formulas. Numbered, severity-tagged findings only, no praise:
1. Is the budget identity production+external+buoyancy+dissipation+transport+pin+floor
   +etau == tke_new-tke_old EXACT for every surface path (Neumann flux row, pinned
   Dirichlet, nemo_z0 extended matrix)? Check the extended-matrix slice _sys =
   (a_ext[...,1:], b_ext[...,1:], c_ext[...,1:], concat(e_sfc, e_new)) and the
   diag_sinks reconstruction b_diff_eff = b_sys - (1 + dt*(diss_w*diss_rate +
   buoy_sink_rate)). Could a term be silently mis-assigned while the residual still
   closes (e.g. the z=0 face flux landing in transport vs surface, the 1.5-split
   dissipation's explicit half, veros_positivity path returning early)?
2. Does return_budget change the primal tke_new anywhere (extra ops, dtype)? Any
   retrace/JIT hazard from the Python-bool flag?
3. In _seed_budget_terms: verify the index mapping (our interface j = NEMO W-level
   array index j+1; e3t[a-1]=dz_c[:, :-1], e3t[a]=dz_c[:, 1:], e3w[a]=dz_half), the
   sign and factor of the NEMO transport rate 0.5*[lw*(e_{a-1}-e_a)+up*(e_{a+1}-e_a)]
   against zdftke.F90:404-414 (zcof = -0.5*rn_Dt, zdiag = 1 - zzd_lw - zzd_up + ...),
   and eps_n = -c_eps*dissl*en against lines 414/419. Is NEMO's transport here the
   explicit estimate on en_old while ours is on e_new, and does that matter at dt=150?
4. Is the printed 'ratio = median of per-column ours/NEMO' well-defined where NEMO's
   term crosses zero (transport)? Suggest the robust statistic if not.
5. Any way the pre-registered branches in scripts/cluster/omip_nemo/_tke_seed_budget.sbatch
   cannot be reached by this instrument (vacuity), or a branch that would be misread?
Be concrete: file:line, what is wrong, what to change." 2>&1 | tail -150
echo "=== codex exit ${PIPESTATUS[0]} ==="
