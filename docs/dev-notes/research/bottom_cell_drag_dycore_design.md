# Bottom-cell linear bottom drag — split-explicit dycore design

Companion to `docs/dev-notes/research/why_westward_drake.md` and
`docs/dev-notes/research/bottom_drag_literature.md`. Designs the migration from
the current depth-mean bottom drag (`(1 − dt·r/H)` on `U_bar` inside
the BEBT substep) to a bottom-cell drag that lives in the baroclinic
momentum tendency. Targets `LatLonCGridOceanModel` first; the
identical pattern carries over to `MPASOceanModel`.

## 0. Where the current drag actually lives — auditor notes

There is more drag in the lat-lon C-grid pipeline than the prompt
suggests. Three things are happening simultaneously:

1. **Bottom-cell linear drag in the baroclinic tendency.** Already
   present at
   `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:545–552`:
   `du_dt[…, −1] += −r · u[…, −1] / dz_bot_u`,
   `dv_dt[…, −1] += −r · v[…, −1] / dz_bot_v`. Computed on the **full**
   velocity (not the perturbation), explicit Euler.
2. **Slow-forcing fold-in.** In
   `ocean_model_latlon_cgrid.py:374–378` the entire tendency
   (including the bottom-cell drag) is depth-averaged into
   `F_slow_u/v` and the perturbation `du_dt − F_slow_u` is applied to
   the 3D state. `F_slow_u/v` is then passed to
   `barotropic_substeps_latlon_cgrid` as constant forcing each
   substep. So the depth-mean of the bottom-cell drag *already*
   reaches the barotropic momentum equation through the slow-forcing
   channel.
3. **Independent depth-mean drag on `U_bar` inside BEBT.**
   `barotropic_latlon_cgrid.py:340–346` multiplies `U_bar_new`,
   `V_bar_new` by `implicit_bottom_drag_factor(dt_s, r, H_u/H_v)` once
   per substep. This is independent of (1) and (2).

The MPAS path (`ocean_pe_mpas.py:246–249`,
`barotropic_mpas.py:253–255`) has the same triple structure.

So the "non-standard" depth-mean drag is currently **double-counted on
top of** an already-present bottom-cell drag. The diagnostic finding
that `u_bot` is unconstrained at −7 cm/s with Drake transport at
−405 Sv is consistent with: the explicit bottom-cell drag is present
but weak, and the depth-mean drag is silently fighting it instead of
removing the integration constant cleanly.

The job is therefore not "add bottom-cell drag" — it is "remove the
double counting and make the bottom-cell drag the sole, well-posed
sink, with implicit treatment for stiffness safety". The minimum
correctness fix is to delete one of the two pathways; the proper
dycore-clean version is the design below.

## 1. Where the new drag should land

**Punchline. Bottom-cell drag is a baroclinic-tendency-level forcing.
It must live in the per-RK / per-baroclinic-step momentum tendency,
applied at the deepest wet level only, with implicit treatment, and
its depth-mean *must* be allowed to enter the barotropic solver via
the existing `F_slow_u/v` channel — but no drag may be applied a
second time on `U_bar` inside the BEBT substep.**

The barotropic substep operates on a 2D depth-integrated transport,
on a `dt_s = dt / n_substeps ≈ 20 s` clock with a `c = √(gH) ≈ 200
m/s` gravity-wave CFL. The actual bottom-momentum sink is *not* a
2D phenomenon; the bottom cell is one layer of a 3D column with
strong stratification. Treating it as 2D is what allows the thermal-
wind integration constant to drift (see `why_westward_drake.md`
section 3). The right home is `latlon_cgrid_ocean_baroclinic_tendencies`
(file `ocean_pe_latlon_cgrid.py`), which is called once per
baroclinic step on the slow `dt = 300–600 s` clock.

## 2. Implicit vs explicit treatment

**Punchline. Use per-cell implicit, multiplier
`α = 1 / (1 + dt · r / dz_bot)`, applied to `u_new[…,−1]` and
`v_new[…,−1]` after the explicit tendency has been integrated. This
costs one divide per face, is unconditionally stable, smoothly
extends to quadratic drag, and stays simple under autodiff.**

Numerics. With `r = 1.1e−3 m/s`, `dz_bot ≈ 100–200 m`, and
`dt = 600 s`, the explicit factor `dt·r/dz_bot ≈ 3.3e−3 to 6.6e−3` is
fine for stability. But we want to keep the door open to
(a) thinner ridges where `dz_bot` shrinks under z-star compression,
and (b) a quadratic option `r_eff = C_d |u_bot|` which spikes during
storms. Implicit is essentially free (a scalar reciprocal per face)
and removes a stiffness failure mode forever. It also removes the
need for an `eps` floor on `dz_bot`, since `(1 + dt·r/dz_bot)^{−1}`
is bounded in [0,1] for any positive `dz_bot`.

The implementation pattern, applied **after** the explicit Euler
update in `ocean_model_latlon_cgrid.py:381–381` and **before** the
forward-backward Coriolis call:

```
alpha_u = 1.0 / (1.0 + dt * r / jnp.maximum(dz_bot_u, eps))
alpha_v = 1.0 / (1.0 + dt * r / jnp.maximum(dz_bot_v, eps))
u_star = u_star.at[..., -1].multiply(alpha_u)
v_star = v_star.at[..., -1].multiply(alpha_v)
```

This is mathematically equivalent to a backward-Euler step on the
linear ODE `du/dt = −(r/dz_bot) u` applied for one baroclinic step.

## 3. Identifying the bottom cell

**Punchline. legoESM uses z-star with full (non-partial) cells, and
every column carries `nlev = z_coord.n_levels` layers. The "bottom
cell" of a wet column is index `−1` along the level axis, period.
For dry columns `u_mask`/`v_mask = 0` already zeros the drag.**

Confirmed at `src/legoesm/ocean/vertical.py:46–123`: the z-star
coordinate stores `dz_ref` of shape `(nlev,)` and constructs
physical layer thickness as `h_k = dz_ref[k] · (eta + H_bathy)/H_max`
for every (i, j). Bathymetry shows up only as the column-wide
Jacobian `J = (η + H)/H_max`; there is no per-column truncation of
the level axis. This is why the existing
`ocean_pe_latlon_cgrid.py:549–550` indexing `u[..., −1]` with
`dz_bot = dz_ref[-1] · interp_cell_to_uface(J)` is correct.

The face-centred bottom thickness must be the **maximum** of two
neighbouring J's, not the minimum, when one neighbour is land
(J=0): use `interp_cell_to_uface(J)` exactly as today, then floor
with `eps`. The `u_mask` at the face is zero whenever either
neighbour is land, which kills the drag tendency at coastlines and
matches the existing pattern.

For full-cell physical bottom drag at coastlines that ride higher
than `−H_max`, this gives `dz_bot ≈ dz_ref[-1] · J ≈ dz_ref[-1] · H/H_max`,
i.e., `dz_ref[-1] · (true H)/H_max`. Over a 4000 m column with
`H_max = 5500 m` and `dz_ref[-1] ≈ 300 m`, the effective bottom-cell
thickness is ≈ 218 m — physically the right scale.

## 4. Interaction with the barotropic substeps — the load-bearing decision

**Punchline. Option (a): remove the depth-mean drag from the
barotropic substep entirely. The bottom-cell drag's depth-mean is
already carried into the barotropic equation through the existing
slow-forcing decomposition. Adding `(1 − dt·r/H)` per substep on top
of that is exactly the double-count that makes `u_bot` unconstrained.**

The split-explicit pipeline as it stands:

  1. `tendencies()` returns `du_dt` (which now includes the
     bottom-cell drag at the deepest level only).
  2. `step()` computes
     `F_slow_u = sum_k(du_dt · h_u) / H_u`, the depth-mean tendency
     including the bottom-cell drag.
  3. `du_dt_pert = du_dt − F_slow_u[..., None]` is applied to 3D.
  4. `barotropic_substeps_latlon_cgrid` advances `U_bar` with
     `F_slow_u` injected each substep.

So the depth-mean of the bottom-cell drag *is already* a slow forcing
on `U_bar`. The depth-mean drag at `barotropic_latlon_cgrid.py:340–346`
is therefore redundant and structurally different in time-discretisation
(per-substep multiplicative vs per-baroclinic-step additive forcing).
That mismatch lets the column average drift relative to what the
deep-cell sink is asking for.

Option (b) — keep a small residual depth-mean drag for barotropic
stability — is rejected. Barotropic stability is set by the BEBT
implicit blend (`bebt = 0.2`), divergence damping
(`barotropic_div_damp`), and `barotropic_diffusion_alpha`, none of
which require bottom drag for stability. MOM6, MITgcm, NEMO, POP
all run with the bottom drag *only* at the bottom cell of the
baroclinic equation; their barotropic substeps see drag only via
the slow-forcing channel.

Option (c) — explicit projection of the bottom-cell drag onto the
barotropic mode — is exactly what the `F_slow_u` accumulator
already does. We do not need a separate term.

Concrete edits:

  - Delete `if config.bottom_drag_r > 0:` block at
    `barotropic_latlon_cgrid.py:340–346` when the new mode is
    selected. Keep the helper itself in `ocean_tendency_common.py`
    for backward-compatibility and the `depth_mean` mode (Section 6).
  - Identical change in `barotropic_mpas.py:253–255`.

## 5. Conservation, masking, and AD-compatibility

**Punchline. The implicit factor is a JAX-pure scalar multiply with
no Python branching on traced values; it slots cleanly into the
existing tendency. Conservation is unchanged because drag is a
momentum sink, not a mass sink — the only conservation diagnostic
to update is the kinetic-energy budget term.**

Required guards:

  - `α = 1 / (1 + dt · r / max(dz_bot, eps))` with
    `eps = 1e−10` to avoid division-by-zero when `J → 0` on dry
    columns. The `u_mask` at the face zeroes the drag in those
    cells before the multiplier applies.
  - The drag at `[..., −1]` only — every level above is untouched.
    This is `at[..., -1].multiply(α)` (XLA in-place scatter), which
    differentiates cleanly because Equinox/JAX VJP for `.at[].set/multiply`
    is well-defined and dense.
  - Land masking: multiply the *tendency* by `u_mask_3d` after the
    drag is applied (already done at line 606 of
    `ocean_pe_latlon_cgrid.py`). Or, if implicit drag is applied in
    the step function (Section 2 pattern), apply `u_mask_3d` after
    the multiply. Either is correct; do not double-mask.

KE diagnostic: existing band-momentum / KE budget code in
`docs/dev-notes/research/why_westward_drake.md` section 6 already plans for an
`F_botdrag` term. The formulation needs the *true* bottom stress
`τ_b = ρ₀ · r · u[..., −1]` not the depth-mean form
`ρ₀ · r · U_bar`. This is mentioned as a follow-up in the test plan.

AD note: `jnp.maximum(dz_bot, eps)` is differentiable everywhere
except at the kink, which is irrelevant for any wet column where
`dz_bot ≫ eps`. No `jnp.where` on traced values is needed because
the choice between modes is a static config branch.

## 6. Backward-compatibility and config switch

**Punchline. Add `bottom_drag_location: Literal["depth_mean",
"bottom_cell"] = "depth_mean"` to `LatLonCGridOceanConfig` and
`MPASOceanConfig`. Default to current behaviour. Switch the
controller in `step()` and the bottom-cell tendency in
`latlon_cgrid_ocean_baroclinic_tendencies` on this static literal —
it is a Python `if` on a static field of a NamedTuple captured in
the JIT closure, so no `jnp.where` two-branch trace, per the JAX
rule for on/off feature gating.**

Concrete schema additions:

  - `bottom_drag_location: str = "depth_mean"` — `"depth_mean"`
    keeps today's `(1 − dt·r/H)` on `U_bar` and disables the
    explicit bottom-cell tendency (sets the lines 545–552 block to
    pass-through). `"bottom_cell"` enables implicit bottom-cell
    drag in `step()` and removes the BEBT depth-mean drag.
  - `bottom_drag_implicit: bool = True` — controls the per-cell
    implicit multiplier vs explicit `du_dt` form. Default to
    implicit because it is unconditionally stable; explicit kept
    only for unit-testing the linearisation.

Do **not** add a deprecation wrapper around `implicit_bottom_drag_factor`.
It still has a legitimate use under `bottom_drag_location =
"depth_mean"`. Per CLAUDE.md, private helpers are not removed
without removing all callers; this helper has callers in both
barotropic schemes.

The structural enforcement test
`tests/ocean/unit/test_no_scheme_duplication.py:174–179` requires
`implicit_bottom_drag_factor` to appear in every `barotropic_*.py`
that has bottom drag. With the new mode that test must be widened
to "this helper appears, OR the file references
`bottom_drag_location` and gates the helper on it". Pencil this in
as part of the test diff.

## 7. Test coverage

**Punchline. Two unit tests plus a refresh of the no-scheme-duplication
test. The unit tests must verify the *physical* effect, not just
the field shapes.**

  1. **Flat-bottom rest-state spinup with uniform wind, dual-mode
     comparison.** 5° × 5°, flat 4000 m bottom, uniform `τ_x = 0.05 Pa`,
     no Coriolis variation in latitude, `bottom_drag_r = 1.1e−3 m/s`,
     no GM, `n_lev = 10`. Run 30 days for both `bottom_drag_location`
     values. Equilibrium `U_bar` should match
     `τ_x / (ρ₀ · r) ≈ 4.4 cm/s` to within 5% in both modes (this is
     the unstratified case where both formulations are equivalent).
     This is the necessary smoke test that the bottom-cell mode
     reduces correctly to the depth-mean answer when there is no
     stratification.
  2. **Drake-channel-like setup, `u_bot` decay diagnostic.**
     Reuse `legoesm.ocean.experiments.acc_channel`. Initialise with
     a barotropic `u₀(z) = +5 cm/s` everywhere, no wind, drag on,
     stratified IC. Verify that with `bottom_drag_location =
     "bottom_cell"` the bottom-layer velocity decays with timescale
     `dz_bot / r ≈ 218 m / 1.1e−3 m/s ≈ 2.3 days`, while
     `U_bar` decays only as the column rebalances. Under
     `"depth_mean"`, `u_bot` should *not* decay — this is the bug
     we are fixing, and the test asserts the formulations are
     genuinely different.
  3. **Update `tests/ocean/unit/test_no_scheme_duplication.py`** so
     the bottom-drag-helper enforcement is parameterised on the
     mode, or gated behind a `# allow when bottom_drag_location` comment.

Optional but cheap: a third test verifying gradient compatibility
through `eqx.filter_value_and_grad` to confirm no shape change or
dtype promotion when toggling `bottom_drag_location`.

## 8. Risk register

  - **Thin bottom cells (z-star compression in shallow ridges).**
    With `dz_ref[-1] ≈ 300 m` and `H_max = 5500 m`, a 500 m column
    has `dz_bot ≈ 27 m`, so `dt · r / dz_bot ≈ 0.024`. Implicit
    multiplier is `α ≈ 0.976`, fine. With explicit drag, an
    aggressive `r = 5e−3 m/s` would push this to `α ≈ 0.89` and
    explicit *might* go unstable at the timestep we use; another
    reason to default to implicit.
  - **Land-mask corner cases.** The drag tendency must inherit
    `u_mask` at the *face* (averaged from cell-centred land mask).
    Using the existing `interp_cell_to_uface(J)` plus post-multiply
    by `u_mask_3d` is sufficient. The runtime check
    `_assert_runtime_invariants` already verifies face-mask
    consistency under
    `enable_runtime_checks`.
  - **Cell-thickness mismatch between baroclinic and barotropic
    sweeps.** The bottom-cell tendency uses `dz_bot` from the
    pre-barotropic Jacobian. Across the barotropic substep,
    `eta` and therefore `dz_bot` evolve. For a 600 s baroclinic
    step the change in `dz_bot` is `O(dη)` ~ cm scale, completely
    negligible. We should *not* re-evaluate the drag inside the
    barotropic substep just to follow η — that is exactly the
    depth-mean drag mistake.
  - **Pre-existing callers of `implicit_bottom_drag_factor`.**
    Two callers (lat-lon C-grid and MPAS barotropic). With
    `bottom_drag_location = "bottom_cell"` neither caller fires the
    helper, but the helper still exists. The duplication test
    needs widening (Section 7). The deprecated `physics`-pipeline
    `BottomDragConfig` continues to work since it operates on the
    physics_fn route, *not* through the dycore drag — but turning
    it on simultaneously with `bottom_drag_location = "bottom_cell"`
    would re-introduce double-counting. Document this as
    "exclusive: pick one route". Add an assertion in
    `_validate_config`.
  - **MPAS parallel path.** Identical change applies, but the MPAS
    bottom-cell drag at `ocean_pe_mpas.py:246–249` is *already*
    explicit on full velocity. Migration is the same: switch to
    implicit multiplier in the step function, delete the BEBT
    depth-mean drag.
  - **Conservation diagnostics.** `ocean_conservation_fixer` works
    on volume/heat/salt, none of which the drag touches; no fixer
    change. KE budget code (if/when added) must use the bottom-cell
    stress.
  - **GM/Redi interaction.** Per `why_westward_drake.md`, the real
    cause of Drake-band sign flip is GM-on-tracers without
    GM-on-momentum, not the drag formulation. Fixing the drag
    will improve the diagnosis (cleaner integration-constant
    behaviour) but is not by itself expected to give correct ACC
    transport in flat-bottom mode. Land this fix expecting a
    cleaner residual but residual still nonzero; budget term
    F_botdrag will be smaller and have unambiguous sign.

---

## Bottom line: code-change recipe in 5 bullets

  1. **Add config flag.** `bottom_drag_location: str = "depth_mean"`
     to `LatLonCGridOceanConfig` (`state.py:386–438`) and the MPAS
     equivalent. Add `bottom_drag_implicit: bool = True`.
     Validation in `_validate_config` rejects the combination of
     `bottom_drag_location = "bottom_cell"` with the legacy physics
     pipeline `BottomDragConfig` to prevent double-counting.

  2. **Disable the BEBT depth-mean drag in `bottom_cell` mode.**
     `barotropic_latlon_cgrid.py:340–346`: gate behind
     `if config.bottom_drag_r > 0 and config.bottom_drag_location == "depth_mean":`.
     Same change at `barotropic_mpas.py:253–255`.

  3. **Replace explicit bottom-cell tendency with implicit multiplier
     in `step()`.** In `ocean_model_latlon_cgrid.py` after the
     `u_star, v_star = state.u.data + dt * du_dt_pert` update at
     lines 380–381 and *before* `_forward_backward_coriolis_3d`, add
     ```
     if config.bottom_drag_location == "bottom_cell" and config.bottom_drag_r > 0:
         dz_bot_u = z_coord.dz_ref[-1] * jnp.maximum(interp_cell_to_uface(J_pre), 1e-10)
         dz_bot_v = z_coord.dz_ref[-1] * jnp.maximum(_interp_to_v_points(J_pre), 1e-10)
         alpha_u = 1.0 / (1.0 + dt * config.bottom_drag_r / dz_bot_u)
         alpha_v = 1.0 / (1.0 + dt * config.bottom_drag_r / dz_bot_v)
         u_star = u_star.at[..., -1].multiply(alpha_u)
         v_star = v_star.at[..., -1].multiply(alpha_v)
     ```
     where `J_pre` is the pre-step Jacobian already needed for
     `h_u_pre`. Remove the explicit
     `du_dt.at[..., -1].add(...)` block at
     `ocean_pe_latlon_cgrid.py:545–552` so it does not contribute
     in `bottom_cell` mode (gate it on the same flag).

  4. **Mirror in MPAS.** Apply the implicit multiplier after the
     `du_dt_full` integration in the MPAS step function, gate the
     existing explicit add at `ocean_pe_mpas.py:246–249` on the
     `bottom_drag_location` flag, gate the BEBT depth-mean drag
     similarly. The slow-forcing decomposition that already exists
     at `ocean_pe_mpas.py:257` carries the depth-mean of the
     bottom-cell drag into the barotropic equation correctly.

  5. **Tests.** Two unit tests (flat-bottom equivalence, Drake
     `u_bot` decay) under
     `tests/ocean/unit/test_bottom_cell_drag.py`, plus
     widen `test_no_scheme_duplication.py:174–179` so the
     helper-presence check is allowed to be gated on
     `bottom_drag_location`. Run
     `JAX_ENABLE_X64=1 .venv/bin/python -m pytest
     tests/ocean/unit/test_bottom_cell_drag.py
     tests/ocean/unit/test_bottom_drag_sponge.py
     tests/ocean/unit/test_no_scheme_duplication.py` plus a quick
     `acc_channel` smoke run.

After these five bullets the dycore has a single, well-posed
bottom-momentum sink that lives where every other GCM puts it; the
thermal-wind integration constant is no longer free; and the
diagnosis in `why_westward_drake.md` can finally separate the GM-on-
momentum gap from the drag-formulation gap.
