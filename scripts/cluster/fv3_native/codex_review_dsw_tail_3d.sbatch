#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_tail3d
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_tail3d_%j.log
# Adversarial review of module 3 of the 3-D layer: fv3_dsw_tail_3d.py.
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
ADVERSARIAL CODE REVIEW of packages/core/legoesm/core/fv3_dsw_tail_3d.py
and its gate file tests/grids/test_fv3_dsw_tail_3d.py.

CONTEXT. This branch ports a loop-faithful NumPy fp64 FV3 duo-grid dycore to
JAX. The contract is docs/atmosphere/fv3_duo_jax_lane_strategy.md and the live
state docs/atmosphere/fv3_duo_jax_lane_STATE.md -- read both, especially rules
R1/R1a/R4/R7, conventions C1-C7, and lesson 19.

THE SPECIFICATION is legoesm.core.fv3_native_dsw_tail_3d. IT IS THE AUTHORITY:
any behavioural difference is a port bug by definition. Diff the two routine by
routine. The module was authored by GLM-5.3 in three parts and assembled; the
seams are `dsw_tail_phase_3d` | `dgrid_pressure_phase_3d` |
`nh_exchanged_area6` + `dgrid_nh_pressure_phase_3d`, and a seam is exactly
where an import, a convention or a returned key goes missing.

ATTACK IN THIS ORDER, cite file:line, be concrete about the failure each
enables.

1. THE CADENCE, against the spec and the oracle lines the spec quotes.
   - d_sw3 per (face, level) -> BARRIER 2 per LEVEL with all six faces
     present -> KE -> d_sw4 -> d_sw5 -> d_sw6. Is any stage fed a value from
     the wrong side of the barrier?
   - THE KE ASSEMBLY IS THE TRAP: `ke = 0.5*(ubbtemp*vbbtemp + ubb*vbb)` must
     mix PRE-barrier `ubbtemp`/`vbb` with POST-barrier `vbbtemp`/`ubb`.
     Blending all four, or none, gives a finite physical-looking field. Check
     it against dyn_core.F90:1080-1085 and the spec's own lines.
   - the KE window: the spec writes into `ke[ring, ring]` with
     `ring = slice(ng, ng+npx)` on an `(m_a+1, m_a+1)` zeros array. Is the JAX
     `.at[ring, ring].set(...)` the same window, and is everything outside it
     the same value the spec leaves there?
   - `d_con = 0.0` and the `damp_w` fallback (`cfg.damp_v` on the NH arm, 0.0
     on the hydrostatic one).
2. THE PRESSURE CHAINS.
   - `pk`/`gz` come back SCRATCHED from one_grad_p (a2b_ord4 replace=.true.).
     Is the `remap_step` snapshot taken BEFORE that call, and is `pk_remap`
     ABSENT when remap_step is false?
   - the NH tail's ORDER: update_dz_d -> riem_solver3 -> pe_halo (remap_step
     only) -> pln_halo under use_logp ELSE pk3_halo -> zh and pkc exchanges
     per interface level -> gz = zh*grav over the TWO-CELL halo box ->
     second pkc exchange under square_domain -> nh_p_grad. Any reordering,
     any exchange applied to the wrong array, any halo box off by one.
   - trap #6 from the spec: `csw_press[t]['pkc']` is the SAME storage
     riem_solver3 overwrites and nh_p_grad then scratches. On a functional
     lane that aliasing is gone -- is the RIGHT generation of pkc handed to
     each consumer?
   - `nh_exchanged_area6` caches on the ctx in the spec. What does the JAX
     twin do, and can it return a stale array under jit?
3. FUNCTIONAL FIDELITY (C4). The spec mutates: state u/v, csw_outs uc/vc,
   dsw_outs, tail_outs u/v/w, delz6, and every member of the `nh` carry.
   Enumerate what the spec writes and check EACH one comes back in a returned
   dict. A key the spec writes and this lane neither returns nor documents as
   unreachable is a silent divergence.
4. STATIC/DYNAMIC AND JIT. dt dynamic, ctx/km/deck static; the make_*_jit
   factories must bake exactly the static set. Any Python branch on a traced
   value, any `jnp.where` whose arms are not both total, any donate_argnums,
   any retrace-per-timestep.
5. THE GATE FILE. For each test: could it pass on a broken module?
   - the barrier gates compare pre- and post-barrier stacks -- are they
     comparing arrays that CAN differ, and is the i = ie+1 ring check reading
     the axis it claims?
   - the parity gate skips a key that is absent from either lane. Does it
     assert a MINIMUM set, or could it silently compare nothing?
   - `test_ke_mixes_blended_and_unblended_members` asserts two candidate
     assemblies differ ON THIS FIXTURE. Does that actually discriminate the
     implemented one, or only show the fixture is not symmetric?
   - the gradient gates run the adjoint identity AND
     `assert_fd_truncation_scaling`. Is the ladder's precondition satisfiable
     on these operands, and would a wrong Jacobian really move the ratio here?
6. ANY COMMENT OR DOCSTRING CLAIM THAT IS NOT TRUE OF THE CODE, especially the
   ones asserting which stage owns an extent, what is recoverable from what,
   and the shape justifications.

Output findings ranked BLOCKER / MAJOR / MINOR with file:line, the concrete
failure, and the smallest fix. Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_TAIL3D_REVIEW_DONE
