#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_m456
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_m456_%j.log
# Adversarial review of the top three modules of the 3-D layer:
#   4  fv3_acoustic_3d.py   (the sub-step driver)
#   5  fv3_tracer2d.py      (the data-derived trip count)
#   6  fv3_dynamics.py      (the full fv_dynamics step)
# Module 3 and the shared instruments were reviewed already; these three
# have not been, and per CLAUDE.md the review is mandatory before done.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL CODE REVIEW of three modules and their gate files:

  packages/core/legoesm/core/fv3_acoustic_3d.py  tests/grids/test_fv3_acoustic_3d.py
  packages/core/legoesm/core/fv3_tracer2d.py     tests/grids/test_fv3_tracer2d.py
  packages/core/legoesm/core/fv3_dynamics.py     tests/grids/test_fv3_dynamics.py

CONTEXT. This branch ports a loop-faithful NumPy fp64 FV3 duo-grid dycore to
JAX. The NumPy twin of each file (fv3_native_acoustic_3d, fv3_native_tracer2d,
fv3_native_dynamics) IS THE SPECIFICATION: any behavioural difference is a port
bug by definition, never a finding about the oracle. Read
docs/atmosphere/fv3_duo_jax_lane_strategy.md and
docs/atmosphere/fv3_duo_jax_lane_STATE.md first -- rules R1/R1a/R4/R7,
conventions C1-C7, and the lessons list. All three modules were authored by
GLM-5.3 in parts and assembled, so SEAMS BETWEEN PARTS are the highest-yield
place to look: one already shipped a defect where the tail phase was handed
un-exchanged uc/vc (fixed, lesson in the STATE doc) and it passed every test in
the file at the time.

ATTACK IN THIS ORDER. Cite file:line and name the concrete failure each defect
enables -- a wrong number that still looks physical is worse than a crash and
should be ranked accordingly.

A. MODULE 4, fv3_acoustic_3d.py -- WHICH PROGRAM RUNS AT WHICH SUB-STEP.
   1. The loop is peel / scan / peel because it == 1, the middle, and
      it == n_split are three different programs. Check each against
      dyn_core.F90: the first refreshes the scalars and seeds gz from zs on
      the NH arm; the last takes the pk snapshot the remap reads and calls
      pe_halo. Is anything that belongs to one peel executing in the scan, or
      vice versa? Does the composition degrade correctly at n_split = 1 and 2,
      where the scan is empty?
   2. THE gz <-> zh CADENCE: the first sub-step exchanges gz and SAVES
      zh = gz; every later one RESTORES gz = zh. Backwards, the run stays
      finite and merely wrong.
   3. THE HAND-OFFS BETWEEN PHASES. The C-grid phase's outputs feed the D-grid
      phase which feeds the tail. Every array crossing a phase boundary: is the
      RIGHT GENERATION handed over -- specifically, is anything passed that the
      spec had already exchanged or already overwritten? This is where the one
      known defect lived; assume there is another.
   4. The scan carry: the spec mutates state, the NH carry and the flux
      capacitors in place. Enumerate what it writes and confirm each comes back
      through the carry. A key the spec writes and this lane drops is silent.

B. MODULE 5, fv3_tracer2d.py -- THE ONLY NON-MECHANICAL NODE IN THE PORT.
   nsplt = int(1 + cmax) per level (fv_tracer2d.F90:247) is DATA-DERIVED; a
   lax.scan needs a static trip count, so this lane runs NSPLT_MAX iterations
   and masks the inactive ones.
   1. IS THE MASK EXACTLY A NO-OP? Not approximately -- exactly. Trace what an
      inactive iteration computes and writes. Anything that accumulates,
      divides, or touches a capacitor on an inactive pass is a defect that no
      norm-based test will catch cleanly.
   2. IS THE SCHEDULE ITSELF RIGHT? cmax is a max-reduce over faces
      (the six faces are the MPI ranks, :258 mp_reduce_max). Reducing over the
      wrong axis, or over the halo as well as the compute window, changes the
      trip count -- check the reduction extents against the spec's loops
      (:217-231), including the dead `k < npz/6` first arm under integer
      division.
   3. THE frac RESCALE (:262-291) is applied to cx/xfx/mfx/cy/yfx/mfy over
      DIFFERENT index ranges. Are those ranges reproduced, and is the rescale
      applied exactly once per level?
   4. THE CAP. Resolved nsplt > NSPLT_MAX must refuse loudly (C5), never clip.
      Is the flag computed from the true resolved value or from the clipped
      one? Can it be lost under jit? Is check_nsplt_schedule reachable from
      every caller, including the jit factory's default path?
   5. stop_gradient on the schedule: is it placed so the derivative through the
      ADVECTION survives, or does it also cut a path that should carry one?

C. MODULE 6, fv3_dynamics.py -- COMPOSITION.
   1. THE pt ROUND TRIP: pt enters as temperature, is converted to theta_v at
      :396-408, and is converted back inside the remap (fv_mapz.F90:209-217).
      The two halves are in different modules. Is the conversion done exactly
      once each way, with the pkz generation the spec uses?
   2. THE k_split LOOP: dp1 is re-snapshotted per outer iteration
      (fv_dynamics.F90:472-478) and the tracers advect against THAT. Is the
      snapshot inside the loop? A hoisted one is correct at k_split = 1.
   3. THE TRACER HAND-OFF: tracer_2d_1L runs between the acoustic loop and the
      remap, and the remap then carries q. Are the capacitors handed over in
      the state the tracer routine expects (un-averaged d_sw1 fluxes), and is
      the frac rescale it returns propagated to whoever reads them next?
   4. THE REFUSALS: zvir != 0 and consv_te != 0 reach unported code. Are the
      guards on the actual paths, and are there OTHER unported arms reachable
      from the public signature with no guard at all? Name them.
   5. p_var_hydrostatic / p_var_nonhydrostatic: check_args defaults to False,
      so the spec's `if (ptop <= 0)` raise is off by default. Is that
      defensible, and does any caller in this repo pass a ptop it has not
      validated?

D. ALL THREE: STATIC/DYNAMIC AND JIT. dt and the state are dynamic; ctx, km and
   the deck constants are static. The make_*_jit factories must bake exactly
   the static set -- no more (a baked dynamic value is a wrong answer that
   caches) and no fewer (a traced static value is a retrace per timestep). Any
   Python branch on a traced value, any jnp.where whose arms are not both
   total, any donate_argnums.

E. THE GATE FILES. For EACH test, ask: could this pass on a broken module?
   - test_fv3_tracer2d.py leans on two claims: that a level's answer cannot
     depend on other levels' schedules, and that the amplitude sweep actually
     crosses a step. Are both true of this code, or does some coupling (a
     shared reduction, a shared capacitor) make the first one false -- in
     which case the test is asserting a property the module does not have?
   - test_fv3_dynamics.py's mass gate compares the two lanes' DRIFT rather than
     asserting conservation. Is the scale it normalises by meaningful, or can
     it pass trivially?
   - the pt range gate assumes theta_v on this deck is outside [150, 400] K.
     Verify that against the actual fixture; if theta_v could land inside the
     range the gate is vacuous.
   - every TOL-PENDING bound: say which are obviously too loose to fail.

F. ANY COMMENT OR DOCSTRING CLAIM THAT IS NOT TRUE OF THE CODE. This repo
   treats prose as a testable claim; the shape justifications and the "the
   spec mutates X" enumerations are the ones most likely to be stale.

Output findings ranked BLOCKER / MAJOR / MINOR with file:line, the concrete
failure, and the smallest fix. Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_M456_REVIEW_DONE
