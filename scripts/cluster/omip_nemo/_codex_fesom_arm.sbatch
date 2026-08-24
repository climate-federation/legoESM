#!/bin/bash
#SBATCH --job-name=codex_fesom
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_fesom_%j.out

# Mandatory adversarial review of the FESOM2 (fesom-jax) CORE2 arm.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of the new FESOM2 cross-grid arm:
scripts/run/run_fesom_core2.py, tests/unit/test_run_fesom_core2_cli.py,
scripts/cluster/omip_nemo/_fesom2_core2_d30.sbatch,
scripts/cluster/omip_nemo/_score_fesom_xgrid.sbatch.  The fesom_jax library
it drives is at /burg-archive/glab/users/pg2328/fesom_jax_repo (read its
integrate.py, surface_forcing.py, mesh.py, phc_ic.py, state.py as needed).

History: run 1 omitted kpp/gm/ice configs (integrate defaults them OFF) and
Arctic SST relaxed to air temperature; run 2 passed KppConfig()/GMConfig()/
IceConfig() and scored SST rmse 1.48 / corr 0.989 vs NEMO Jan.  Attack:
1. Does the day-chunked loop (fresh integrate call per day, state threaded)
   lose anything a single 1440-step integrate would keep -- ice restart
   fields, forcing phase (dates_for_steps called per day with day*steps then
   sliced: off-by-one?), is_first_step semantics?
2. The published hindcast card: does the fesom-jax CORE2 example/README run
   with EXACTLY KppConfig()/GMConfig()/IceConfig() defaults, or does the
   paper configuration set non-default fields (tke instead of kpp? ale_cfg?
   visc)?  Name any silently-defaulted config that changes the physics.
3. write_snapshot conventions: land_mask from node_layer_mask[:,0] -- does
   that mark ice-shelf-cavity nodes (ulevels>1) as land?  CORE2 has none,
   but assert or document.  T orientation guard: state.T is (nl, nod2D);
   could nod2D ever equal nl and silently transpose wrong?
4. The scoring approach: passing the FESOM snapshot through the '--mpas'
   slot relies on _load_legoesm + regrid handling 1-D node clouds
   identically to MPAS cell clouds -- verify regrid_curv_to_latlon's
   assumptions (2-D vs 1-D lat/lon) are actually met.
5. The mean-SST progress print is an UNWEIGHTED node mean on a refined
   mesh -- fine for liveness, but is it labelled so nobody quotes it?
6. Env/protocol: JAX_ENABLE_X64=0 in the sbatch for the foreign model while
   the scorer runs x64 -- any place the f32 snapshot values get compared at
   tolerances that assume f64?
Severity-tagged findings + verdict table."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
