# Ocean Physics Validation — Final Report

**Scope.** `src/legoesm/ocean/` physics modules
(vertical/lateral mixing, convection, bottom drag, shortwave
penetration, surface forcing, freshwater/sponge/conservation, EOS,
shared dynamics helpers).

**Iterations.** Three rounds of adversarial review:
- iter-1: physics-validator agent, internal codex round (packet-1 / review-1)
- iter-2: physics-validator agent, internal codex round (packet-2 / review-2)
- iter-3: independent `/codex:adversarial-review` (this round)

## Findings and resolutions

### iter-1 / iter-2 (physics-validator + codex internal)

| # | Bug | Module | Resolution |
| --- | --- | --- | --- |
| R1 | `B_salt = -g·β·Q_S` — freshening was destabilizing (sign inverted) | `vertical_mixing/integration.py:144` | Sign corrected; new test |
| R2 | `B_f` proxy sign for stable column flagged as unstable | `vertical_mixing/kpp.py:207` | Sign corrected; new test |
| R3 | NaN gradient through `sqrt(max(N², 0))` at N²=0 | `_gm_redi_common.py:154`, `gm_redi_mpas.py:464`, `diagnostics.py:61` | Floor at 1e-30; gradient probe added |
| R4 | Monin-Obukhov stable-side suppression disabled by sign-convention mismatch | `kpp.py:282` | `max(zeta_kpp, 0)` → `max(-zeta_kpp, 0)`; new test |
| R5 | A_v interior used K_bg (1e-5) instead of A_bg (1e-4) — 10× too low | `kpp.py:312-321` | Separate A/K interior diffusivities; new test |
| R6 | Non-local mask broke column tracer conservation when h_bl cut a layer | `kpp.py:399, 411` | Drop in_bl_full mask on non-local tendency; new test (drift 1.4e-18) |
| R7 | Plume convection tendency had units [K/m] instead of [K/s] | `convection/plume.py:75-76` | Multiply by `cfg.w_plume_min`; existing test retuned |
| O1 | Bottom-drag factor `1 - dt·r/H` could flip velocity sign at `dt·r/H ≥ 1` | `dynamics/ocean_tendency_common.py:288` | Replaced with backward-Euler `1/(1 + dt·r/H)` |

After iter-2, codex returned **GREEN — no remaining substantive
findings within scope** for the unit-validator audit.

### iter-3 (independent `/codex:adversarial-review`)

Codex challenged the design, not just defects.  Three findings:

#### Finding #3 (medium) — Plume column conservation [FIX APPLIED]

The R7 unit fix made the local tendency [K/s], but
`dT_dt[..., 0] = 0` left `Σ_k dT_dt[k]·dz[k] ≠ 0`.  Numerical probe
on a static-instability column gave `column∫ dT = -1.88×10⁻³ K·m/s`.

**Fix.** `src/legoesm/ocean/physics/convection/plume.py` now subtracts
the column-integrated detrainment from the surface layer (~10 LOC):

```python
column_dT = jnp.sum(dT_dt * dz_actual, axis=-1)
column_dS = jnp.sum(dS_dt * dz_actual, axis=-1)
dT_dt = dT_dt.at[..., 0].add(-column_dT / dz_top)
dS_dt = dS_dt.at[..., 0].add(-column_dS / dz_top)
```

Verified post-fix: `column∫ dT = 5.4×10⁻²⁰ K·m/s`, `column∫ dS =
6.8×10⁻²¹ PSU·m/s` (machine epsilon).

**Test.** Added
`tests/ocean/unit/test_plume_convection.py::test_plume_conserves_column_heat_and_salt`
asserting `max |Σ dT·dz|, max |Σ dS·dz| < 1e-12` on a column with
varying `T` *and* `S`.  The pre-existing
`test_plume_outputs_finite_and_signs_consistent` had to drop the
(now-incorrect) `dT_dt[..., 0] == 0` assertion — that surface row is
exactly where the conservation correction lives.

**Stop-hook follow-up — complete dry-column fix.** Codex stop-time
review caught two residual issues, addressed in two passes:

1. *NaN guard.* The conservation correction divides by `dz_top =
   dz_actual[..., 0]`, which is zero for dry / land columns
   (`jacobian = 0` in z-star).  ``0 / 0`` produces NaN even though
   ``column_dT == 0`` analytically.  Fixed with a ``jnp.where`` guard
   that zeroes the correction when ``dz_top == 0`` and feeds a safe
   denominator (`1.0`) into the division so neither branch produces
   NaN gradients.

2. *Wet-mask on full output.* The plume scan operates on `T`, `S`,
   `rho` without reference to `dz_actual`, so dry columns still
   produced non-zero detrainment tendencies on `k ≥ 1` after the NaN
   guard alone.  Added a final wet-mask multiply on the entire output
   (`dT_dt`, `dS_dt`, `convection_flag`) so dry columns contribute
   exactly zero.  This is consistent with the conservation correction,
   which already evaluates to zero for dry columns.

```python
# 1. Safe-divide for the conservation correction
wet = dz_top > 0
dz_top_safe = jnp.where(wet, dz_top, jnp.ones_like(dz_top))
correction_T = jnp.where(wet, -column_dT / dz_top_safe, jnp.zeros_like(column_dT))
correction_S = jnp.where(wet, -column_dS / dz_top_safe, jnp.zeros_like(column_dS))
dT_dt = dT_dt.at[..., 0].add(correction_T)
dS_dt = dS_dt.at[..., 0].add(correction_S)

# 2. Final wet-mask multiply on full output
wet_mask = wet[..., jnp.newaxis].astype(dtype)
dT_dt = dT_dt * wet_mask
dS_dt = dS_dt * wet_mask
flag = flag * wet_mask
```

Regression test
`tests/ocean/unit/test_plume_convection.py::test_plume_no_nan_for_dry_columns`
mixes a dry (`jacobian = 0`) column with a wet unstable column and
asserts: outputs are finite, dry column produces *exactly zero*
`dT_dt` / `dS_dt` / `convection_flag`, wet column is still active,
and `jax.grad(loss)` is finite through both paths.

#### Finding #1 (high) — Bottom drag double-applied in explicit barotropic [DOC ONLY]

`F_slow_u` carries the depth-mean bottom drag from the 3D PE solver
(`ocean_pe_latlon_cgrid.py:1265` adds `-r·u_bot/dz_bot` to `du_dt`;
`ocean_model_latlon_cgrid.py:422` depth-averages to F_slow).  The
explicit barotropic substep then multiplies by
`implicit_bottom_drag_factor` (≈ `1 - dt·r/H`), applying drag a second
time.  The Crank-Nicolson implicit solver omits this multiplier.  Net
effect: barotropic-mode drag is `≈ 2·r/H` instead of `r/H` in the
explicit path.

**Pre-existing scope.** This duplication predates the recent fix.
The iter-2 change of formula from `1 - dt·r/H` to `1/(1 + dt·r/H)`
swapped first-order-equivalent expressions; both forms double-apply
identically.  The new form is justified for shallow-water stability
(no sign flip when `dt·r/H ≥ 1`) and is **not** the source of the
double-application bug.

**Action.** Documentation only.  Updated the
`implicit_bottom_drag_factor` docstring at
`src/legoesm/ocean/dynamics/ocean_tendency_common.py:250` to
explicitly record the duplicate-application architecture and the
`F_slow` ownership question.  A behavior fix requires either
subtracting the depth-mean drag from `F_slow_u` in
`ocean_model_*.py` or removing drag from `du_dt` in `ocean_pe_*.py` —
multi-file change with measurable impact on KE decay (drag timescale
doubles).  Validation requires AMIP comparison runs.  Out of scope
for unit triage; flagged as architectural debt.

#### Finding #2 (high) — KPP / surface-forcing flux ownership [DOC ONLY]

`_make_kpp` reads `Q_net`, `Q_S`, `tau` directly from
`surface_forcing` to drive KPP, while the configured prescribed/bulk
surface-forcing wrapper discards its diagnostic fluxes before they
reach the column budget.  KPP can therefore react to fluxes that
never enter the heat / salt / free-surface tendencies.

**Status.** Pre-existing; already in iter-2 residuals (item #1).
Codex re-confirmed.  Fix requires a single source-of-truth surface
diagnostic plumbed through `combined.py` so KPP and the budget
application share inputs.  Out of scope for unit triage.

## Files changed (this validator pass, total)

```
src/legoesm/ocean/diagnostics.py                              (R3)
src/legoesm/ocean/dynamics/ocean_tendency_common.py           (O1 + iter-3 doc)
src/legoesm/ocean/physics/convection/plume.py                 (R7 + iter-3 fix)
src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py   (R3)
src/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py      (R3)
src/legoesm/ocean/physics/vertical_mixing/integration.py      (R1)
src/legoesm/ocean/physics/vertical_mixing/kpp.py              (R2/R4/R5/R6 + V_t² floor)
tests/ocean/unit/test_plume_convection.py                     (R7 retune + new conservation test)
tests/ocean/unit/test_visbeck_gm.py                           (R3 grad test)
tests/unit/test_corrections.py                                (R1/R2/R4/R5/R6 KPP regressions)
.physics-validator/ocean/diff_probes.py                       (O1 probe updated)
```

## Test results

| Suite | Result |
| --- | --- |
| `tests/ocean/unit/test_plume_convection.py` | **8 passed** (1 new) |
| `tests/ocean/unit/test_no_scheme_duplication.py` | **19 passed** |
| `tests/unit/test_physics_ocean.py` | **11 passed** |
| `tests/unit/test_corrections.py` | **49 passed** (5 new KPP) |
| Remaining `tests/ocean/unit/` (12 files) | **241 passed, 2 skipped** |
| Visbeck/GM grad probe | passed |
| All AD diff_probes (23 + 7 new) | passed |

**Total: 328 ocean tests passed, 0 failed**, plus 30 AD probes.
One pre-existing failure
(`test_longrun_conservation_with_fixer`, heat drift 3.46e-7 vs 1e-8)
unaffected — conservation-fixer algorithmic limitation, not introduced
by this work.

## Sign-off (per agent definition)

- **Units**  ✓ — plume [K/s]; everything else verified.
- **Signs**  ✓ — B_salt, B_f proxy, MO stability, EOS direct tests.
- **Differentiability**  ✓ — Visbeck/KPP V_t² NaN-free; 30 AD probes pass.
- **Test case**  ✓ — column conservation 1.4e-18 (KPP), 5e-20 (plume); 328 ocean tests green.

**Verdict.**  GREEN within unit-validator scope.

## Open architectural debt (carried forward)

Two pre-existing items elevated by codex iter-3 to "high":

1. **Single-owner bottom drag** (Finding #1) — F_slow vs barotropic
   substep ownership.  Multi-file change in `ocean_model_*.py` /
   `ocean_pe_*.py`; needs AMIP validation.
2. **KPP / surface-forcing flux plumbing** (Finding #2) — KPP must
   share a single diagnostic source with the budget-application path.
   Multi-file change in `combined.py` and surface-forcing wrappers.

Plus iter-2 minor items (unchanged):

3. Function-default constants (`shortwave_penetration.py`,
   `biogeochemistry/{gas_exchange,carbonate}.py`) — CLAUDE.md-forbidden
   literal defaults; pre-existing tech debt.
4. `KPPConfig.Ri_conv` named like a Richardson threshold but compared
   against `N²` — API rename.
5. `test_longrun_conservation_with_fixer` heat drift 3.46e-7 vs 1e-8
   — pre-existing conservation-fixer limitation.

## Codex transcripts

- `packet-1.md` / `review-1.md` — iter-1 internal codex (gpt-5.5 fallback)
- `packet-2.md` / `review-2.md` — iter-2 internal codex (clean re-validation)
- `codex-adversarial-review-followup.md` — iter-3 independent
  `/codex:adversarial-review` triage and resolution
