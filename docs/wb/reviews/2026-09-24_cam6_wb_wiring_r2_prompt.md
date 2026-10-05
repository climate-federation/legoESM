ROUND 2 re-review of the spectral cam6_clubb READ-side wiring, worktree
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6 (uncommitted, on 938855bde).
Round 1: codex HOLD (producer-gate bypass in the split factory CONFIRMED;
cold-start zero carry CONFIRMED; lag/remat/publication/signature clean),
GLM SHIP conditional on an explicit missing-carry raise (B-2).

CHANGES SINCE ROUND 1, each answering one finding:
 1. aimip_params.py make_aimip_classical_spectral_physics: before splitting,
    `if rad_cfg.use_clubb_cloud_fraction and turb_cfg.scheme != "clubb": raise
    ValueError(...)` -- the same producer gate as combined.py, applied to the
    RESOLVED pair. Test: cam6_clubb + louis + split_rad=True now raises
    (test_the_split_factory_refuses_cam6_without_a_producer). Codex #1.
 2. radiation/integration.py spectral core: `if _cam6_cf_active and phys_state
    is None: raise ValueError(...)` naming the caller contract
    (spectral_rollout threads it; spectral_amip_rollout does not and cannot run
    cam6_clubb). Test: fn(state, grid, sigma) with no carry raises. GLM B-2.
 3. aimip_params.py gray branch: the use_clubb_cloud_fraction flag I had added
    there is REMOVED (that branch never receives a cloud scheme; the flag had no
    consumer). Pre-existing behaviour restored. GLM F-4.
 4. Tests: nonhydrostatic refusal now matched on "spectral_pe" so the lane list
    is pinned (GLM G); the convection-bridge publication test uses a distinct
    value per (column, level) and asserts exact equality after the reshape, so
    column ordering is proven, not just shape (codex E).
 Result: 9 tests pass on the wired tree; the 7 pre-existing ones fail 7/7 on
 the pre-wiring commit (control worktree).

NOT CHANGED, deliberately -- OWNER DECISION PENDING, do not treat as a defect
to fix here: the cold-start zero cloud-fraction carry in the first radiation
window (codex #2 / GLM B-1/F-1). MEASURED context I will hand the owner: the
production AMIP coupled driver seeds cloud_fraction from the SAME zero-init
carry at cold start (model_driver.py ~14684: `_seed_carry("cloud_fraction",
_seed_ps.cloud_fraction)`, comment "real zero-init cloud_fraction here"), so
option (a) "accept the transient" reproduces production's cold start exactly,
while option (b) "one diagnostic non-rad pass on the IC to fill the carry
before the first radiation call" would be a departure from production. The
difference in exposure: AMIP's first window is 1 h of a 60-day run; the WB
training rollout's first window is 3 h of a 6 h sample, every sample.

ATTACK the four changes: is the split-factory gate placed before ANY
side-effecting build? Does the missing-carry raise fire at TRACE time inside
jax.checkpoint (it must -- phys_state is a Python None there, not a tracer)?
Is anything in the gray revert now inconsistent? Do the tightened tests still
fail on the pre-wiring commit? Then: SHIP or HOLD, and if you have a view on
(a) vs (b) for the owner, give it as a recommendation with the deciding factor,
not as a change.
