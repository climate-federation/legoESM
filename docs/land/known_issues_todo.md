# Land model — known issues / development TODOs

Developer-facing backlog of **deferred** land numerical-robustness issues found during
the LMIP single-scheme spin-up validation (2026-07). All are currently **left as-is**
(the runs complete and are globally stable/equilibrating); each is a degenerate-regime
singularity that the safety nets (per-column NaN-revert guard, dry-ψ floor, elastic
storage) *catch but do not cure*. Fixing any of these is a physics/numerics change →
requires codex adversarial review + a targeted regression test that reproduces the
failure before/after.

Priority is by scientific impact on the LMIP biophysics validation, not incidence.

---

## TODO-1 — Two-leaf canopy Newton singularity in the LAI→0 (leafless) limit
**Status:** open · **Severity:** low incidence, self-correcting, but pollutes GPP in
the boreal spring green-up we validate · **Owner:** unassigned

**Symptom.** "Spurious NaN cells in GPP", NH spring (Feb–May, **peak April**), 49–69°N.
Not a GPP bug: GPP is NaN only when the *whole cell* is NaN — it's the per-column
revert guard masking a transient non-finite state (100% correlate with the `reverted`
tape flag; 0.1–0.7% of steps; cells recover). The failing cells are LAI≈0 (96% <0.1),
snow-covered (83%), frozen soil (median 259.6 K) — the snowmelt/freeze→thaw/green-up
transition.

**Root cause (trace-confirmed).** The two-leaf canopy Newton solve goes singular as
LAI→0:
1. **PRIMARY** — `canopy/stability.py:355-361`: `Rb_Sun = rb / max(LAI·fSun, 1e-6)`.
   As LAI→0, `fSun → CI (~0.75)` (NOT small; `radiative_transfer.py:237-241`), so
   `LAI·fSun → 0`, `Rb` blows up ~1/LAI (10⁴–10⁷ s/m), the Jacobian rows scale with
   `Rb~1/LAI`, `cond(J)` explodes, and `jnp.linalg.solve` (`solver.py:527`) returns a
   non-finite step. **The degeneracy anchor (`solver.py:275`) is MIS-KEYED on `fSun`**
   (fires when fSun<0.08) so it never engages in the low-LAI limit (fSun≈0.75). The
   `delta` sanitizer (`solver.py:536`) catches the non-finite step but the solve has
   already given up → state poisoned → revert.
2. **COMPOUNDING** — `stability.py:140-159, 173-189`: MOST stability regime branches
   take `log`/`cbrt` of NEGATIVE args in the UNSELECTED `jnp.where` branches. The
   primal is finite but `jax.jacfwd(F)` differentiates THROUGH the NaN branch →
   non-finite Jacobian. Classic "`jnp.where` protects the value, not the gradient".
3. **MINOR** — `canopy/energy_balance.py:145,174` (+ `stability.py:70`):
   `q_sat = eps·e_s/(p−(1−eps)·e_s)` denominator UNFLOORED while sibling `RH_c`
   (`energy_balance.py:179`) floors its denom.

Photosynthesis/Ci/stomatal path is CONFIRMED ROBUST — GPP NaN is inherited from the
Newton state `Ci`, not generated.

**Fix direction.** (a) Re-key the degeneracy anchor on `LAI·fSun` (or LAI) so sunlit
AND shaded leaf eqns collapse toward the bare-soil limit as the canopy vanishes — the
single change most aligned with the signature. (b) Make MOST branches `where`-safe
(clamp `log`/`cbrt` args INSIDE each branch). (c) Floor the `q_sat` denominator to
match `RH_c`. Ship with a low-LAI/cold regression test that reproduces the singular
Jacobian. Memory: `gpp-nan-lowlai-canopy-singularity`.

---

## TODO-2 — Richards spurious mass in arid cells (dry-soil ill-conditioning)
**Status:** open · **Severity:** low incidence (<0.3% land) but violates mass
conservation + corrupts arid-region runoff/storage · **Owner:** unassigned

**Symptom.** A handful of **arid subtropical** cells (lats ~−25° to 33°: Sahara,
Kalahari, Australian interior) carry impossible soil moisture in the prognostic
restart — `theta` DOUBLES with depth to **6.66** (6.7× porosity), `psi` to 1.6×10⁵ m
of head — and emit runoff **~10 orders of magnitude larger than any water entering**
(e.g. col 9711 = 17°N/18°W: precip ~1e-6 kg/m²/s, runoff spikes to 5×10⁴). The cell is
already draining at near domain-max runoff AND still filling → water is **created out
of nothing**. Baked into the restart → propagates into warm-start ICs.

**Root cause.** Bone-dry soil is where the mixed-form Richards solve is ill-conditioned:
specific moisture capacity `C = dθ/dψ → 0` as θ→θ_r, so the tridiagonal diagonal
(`C/dt`) goes near-singular. On a rare wetting transient the `dψ` overshoots to
positive head; the **elastic-storage relief valve** (`θ = θ_sat + S_s·θ_sat·ψ`,
`richards.py:428-432`; intentionally NO θ≤θ_sat clip — the compressible term is
load-bearing for well-posedness at saturation) absorbs it as huge-finite θ instead of
NaN. Bounded (no positive feedback, input-limited) but spurious.

**Fix direction.** Regularize the DRY-SOIL conditioning, NOT the drainage:
(a) a specific-moisture-capacity floor (min `C = dθ/dψ`); (b) a per-step wetting-`dψ`
overshoot limiter; (c) check the surfdata pedotransfer for degenerate θ_sat/θ_r/k_sat
in these arid texture classes (needs Derecho surfdata to confirm singular-capacity vs
bad-params). **REJECTED alternatives:** a θ_sat cap (breaks the numerics — elastic
storage is needed); a bottom-`k_sat` floor (the water isn't real and the cell already
drains at max runoff — a floor just routes spurious mass into subsurface runoff,
corrupting the budget, and re-plumbs every poorly-drained/frozen cell globally).
Memory: `arid-richards-spurious-mass`.

---

## TODO-3 — Multi-layer snow column unstable at global scale
**Status:** open · **Severity:** blocks `snow_scheme="multilayer"` for production ·
**Owner:** unassigned · **Detail doc:** `phase2b_snow_thermal_plan.md`

**Symptom.** `snow_scheme="multilayer"` (the CLM-faithful prognostic snow column, Phase
2b) blows up in snow/cold regions: 10-yr global run had 46% of land cells revert,
final `T_soil` to ~10⁹ K, `snow_depth` to ~10¹⁴ kg/m² (large-finite, so the NaN-guard's
"PASS" was misleading). 100% confined to snow latitudes; tropics/mid-lat clean. The
production LMIP config runs the stable `single` scheme instead.

**Root cause.** The canopy→column→soil coupling is an **explicit operator split** with
a **hard `thermal_active_swe` on/off threshold**. The soil side is semi-implicit
(Robin) but the snow-column base takes `G_bottom` as a fully explicit prescribed flux;
with a thin base layer + large interface conductance + dt=3600 s it overshoots. Under
time-varying forcing a cell whose SWE oscillates across the threshold also flips the
soil BC discontinuously → growing oscillation. Unit tests passed (constant forcing);
they lacked time-varying forcing + threshold oscillation.

**Fix direction.** Solve snow layers + soil layers as ONE coupled implicit tridiagonal
column each step (CLM combined snow+soil), with the interface as an internal implicit
interface — no lagged explicit seam. Prerequisite: a deterministic time-varying-forcing
multi-step stability test that reproduces the blowup. Also: cold-bias + snow-tower
realism limits of `single` remain (documented). Memory: `mlsnow-unstable-global`.

---

## Cross-cutting note
TODO-1 and TODO-2 are the same class of bug: a **degenerate-regime singularity**
(vanishing canopy; bone-dry soil) that the run-level safety nets bound but don't fix.
The common remedy pattern is to add the missing regime-limit regularization (anchor
re-key / capacity floor) rather than lean on the revert guard, which silently masks
fluxes (and pollutes the exact fields — GPP, arid runoff — used for validation).
