#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_hs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_hs_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/burg-archive/home/pg2328/.codex
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a GLM-authored assembly: commit 0ed4a0c14 (moist-dyn + all review fixes) --
apply_held_suarez_step + the --physics held_suarez wiring in
scripts/validate/fv3_native/full_step_oracle_parity.py. Cite file:line, classify
BLOCKER/MAJOR/MINOR, smallest fix, and say where you found NOTHING.

CONTEXT. This assembles the end-to-end 1-step Held-Suarez parity for the duo lane
(dry, hydrostatic, nwat=0): fv_dynamics (already ported) -> apply_held_suarez_step
(= fv_phys + fv_update_phys) -> compare vs the pinned oracle run_hs_1step_gfs.
apply_held_suarez_step composes certified pieces: d2a2c_vect_duo (D->A winds),
held_suarez_tend (forcing, tendencies on the CELL domain (n,n,km)),
exchange_agrid_scalar_halos (u_dt/v_dt one-cell halos across 6 faces), and
fv_update_phys_dry_duo (apply, full domain (m,m,km), m=n+2*ng). Oracle pinned tree:
/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean.

ATTACK SPECIFICALLY:
1. AXIS/WINDOW GLUE. press[t]['peln'] is (i,k,j) shape (n,km+1,n); press[t]['pe']
   is (i,k,j) shape (n+2,km+1,n+2) (a 1-cell ring). The code does
   peln_c=transpose(peln,(0,2,1)) and pe_c=transpose(pe[1:n+1,:,1:n+1],(0,2,1)) to
   feed held_suarez_tend which wants (n,n,km+1) k-last cell-domain. VERIFY against
   the producer p_var_hydrostatic (packages/core/legoesm/core/fv3_native_dynamics.py
   :198-234): is (0,2,1) the right transpose, and does pe[1:n+1,:,1:n+1] pick the
   compute cells (pe fills pe[1:n+1,k,1:n+1])? A wrong transpose is SHAPE-INVISIBLE
   on square faces.
2. TENDENCY SCATTER + HALO EXCHANGE. Tendencies are computed on (n,n,km), written
   to full (m,m,km) at [ng:ng+n,ng:ng+n], halos zero, then u_dt/v_dt halos exchanged
   with exchange_agrid_scalar_halos(f6, tile 1..6, n, ng). Is scattering the cell
   result into a zero-halo full array then exchanging equivalent to the Fortran,
   which fills the compute domain and then does a whalo=ehalo=shalo=nhalo=1 group
   update (fv_update_phys.F90:645,698)? Does exchange_agrid_scalar_halos need all 6
   faces populated before exchanging tile t (it does)? Is t_dt correctly NOT
   exchanged (fv_update_phys only exchanges u_dt/v_dt)?
3. THE GATE. --physics held_suarez must (a) redirect ONLY the step deck to
   run_hs_1step_gfs, (b) gate the STEP deck for do_held_suarez=.true.,
   adiabatic=.false., nwat=0, hydrostatic, fv_sg_adj<=0, consv_te 0, no OTHER
   physics switch, and (c) check the IC (zerostep) as INERT not as a physics deck.
   Read check_deck_matches_the_arm and the ic/step loop: is the split correct? Can
   an explicit --step-run bypass the gate? Does the dynamics-only path still refuse
   physics decks unchanged?
4. COMPLETENESS. Between fv_dynamics and fv_update_phys, does the solo driver
   (driver/solo/fv_phys.F90:533-591) do anything else for do_Held_Suarez that this
   omits (a state touch, a diagnostic that feeds back, a units change)? Is pdt=dt
   (the atmos step) the right Held-Suarez timestep? Is scoring u/v/pt/delp on the
   compute domain vs run_hs_1step_gfs the right comparison, and is delp really
   unchanged by HS?
5. Is the report-only default (no --max-rel) acceptable for a first run, and is
   there any silent-pass path (a returncode-0 that certifies nothing)?
PROMPT
)" 2>&1
echo "CODEX_HS_REVIEW_DONE"
