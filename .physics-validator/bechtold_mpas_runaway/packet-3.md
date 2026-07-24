# Adversarial review packet — round 3 (converging on the static-review findings)

Same reviewer. Rounds 1-2 in `review-1.md`, `review-2.md`. I concede your
round-2 methodological corrections and report verification of your two new
code items (one with a correction back to you). This is a FINDINGS-ONLY static
review — the definitive root-cause attribution requires an instrumented
GPU experiment that is OUT OF SCOPE here; I want your agreement on which CODE
findings are settled vs which are experiment-gated.

## I concede (round-2 corrections accepted)
- **Factor labels were wrong.** A = Bechtold `dx_m=0` vs physical `dx_m` (both
  Bechtold). B = malformed LW-OD ON vs zero-OD. SBM-vs-Bechtold is a separate
  scheme contrast; pinned-vs-transient is a BUNDLED deck factor (ozone / SW
  aerosol / GHG / volcanic / radiation cadence), not B. Adopted below.
- **"Reduced OLR" from B is an inference, not shown.** A pure absorber also
  emits; net TOA sign is state-dependent. Needs paired radiation calls (current
  LW-OD vs zero LW-OD on the identical state) → ΔOLR, Δsurface-LW, vertical ΔLW
  heating. `_aerosol_lw_od×heating` was a sloppy shorthand; withdrawn.
- **One-checkpoint factorial is invalid across schemes.** The MPAS restart guard
  rejects a convection-scheme mismatch when the physics carry is present
  (`model_driver.py:6224`, which I had read as F5). Need identical dynamical
  fields with deliberately scheme-compatible carries; stripping physics state
  resets radiation/turbulence memory (transient). Accepted.
- **Leaf water closure is narrower than "MPAS step conserves".** The bridge
  routes convective condensate/rain into TRACERS for later microphysics
  (`integration.py:664`); the full-step budget must include all condensate
  species + microphysics + surface precip, not the convective leaf alone.
  Accepted (my R=0 result is explicitly leaf-scope).

## Your two round-2 items — verified (B confirmed; floor CORRECTED back to you)
- **B activation gate — CONFIRMED.** `model_driver.py:6157-6160`:
  `_ext_forcing = radiation∈{rrtmg,rrtmgp} AND (_ozone_ext_active OR
  _aerosol_active OR _ghg_active OR bool(_experiment))` — `_aerosol_lw_active`
  is NOT in the OR-list, so volcanic-LW ALONE never calls
  `_precompute_external_forcing`. BUT probe_bech40's deck is FULL transient
  (ozone+aerosol+volcanic) ⇒ `_ozone_ext_active`/`_aerosol_active` true ⇒ the
  gate opens ⇒ `_forcing_daily["aerosol_lw_od"]` is set iff `_aerosol_lw_active`
  (`model_driver.py:6483-6485`). So B is plausibly LIVE in probe_bech40; still
  record the 3 diagnostics (resolved `volcanic_aerosol_lw`, `_aerosol_lw_active`,
  day-1 area-mean/max `aerosol_lw_od`).
- **MPAS temperature floor — CORRECTION: it CANNOT cause this runaway.**
  `primitive_eq_mpas.py:887-892` is `T = maximum(T, T_min)` — a LOWER bound
  (clamps COLD air up), behind the #930 nu_vert4_T cure and a #915 daily
  `T<100 K` ABORT guard. A WARM runaway (T→377 K) never triggers a lower floor,
  so the T-floor is exonerated as the warm source. The **tracer non-negativity
  floor** (`:900-905`, `max(q,0)` — creates water) and the **pressure-mass
  fixer** are legitimate ledger rows (documented "negligible", but instrument,
  don't assume). Agreed those belong in the ledger; the T-floor does not.
- **Wind spike ≠ Bechtold CMT — accepted.** MPAS zeros Bechtold cell winds /
  edge-wind tendency (`integration.py:400,720`); the |u| spike is a
  GWD/dycore pressure-gradient (thermal-wind) response to the heating — a
  SYMPTOM, consistent with my reading.

## Settled CODE findings (my proposed sign-off list)
1. **A — CONFIRMED bug:** MPAS never wires `bechtold_dx_m` from cell area →
   `_ifs_ztaures→1.0` → deep CAPE closure ~3× too vigorous on ~2.2° cells; the
   config comment "the driver sets it from the grid" is false. Amplifier, not a
   net source.
2. **B — CONFIRMED bug (activation-conditional):** volcanic `ext_earth`
   extinction fed to the LW **absorption** slot with no (1−ω) scaling
   (`rrtmgp.py:599-605` API contract violated) AND column AOD spread by
   pressure-mass (`surface_utils.distribute_column_aod_to_layers`) →
   stratospheric aerosol placed in the troposphere. TOA sign needs the paired
   ΔOLR diagnosis.
3. **F1 — CONFIRMED dead guard:** `test_bechtold_column_conservation.py:47`
   references removed `cape_sink_heating_ratio` → 3 tests fail at collection →
   the leaf enthalpy/water guard gives no protection.
4. **Convection is conservative — CONFIRMED (leaf):** R=∫(c_p Ṫ+L_v q̇_v)dp/g ≈ 0
   and ∫(q̇_v+q̇_c+q̇_r)dp/g ≈ 0 across 6 soundings × {L20,L30} with probe_bech40
   flags. The original "heating without moisture sink" mechanism is refuted at
   leaf scope (NOT a full-MPAS-step proof).
5. **F2/F3/F4 — CONFIRMED:** pe inert under `use_ifs_inplume_precip=True`; SBM
   ignores pe; the resolved 0.0 (legacy-schema default) is harmless to SBM.
6. **F7 — CONFIRMED false positive** (`bechtold.py:2491` post-conversion residual).
7. **Experiment-gated (NOT settled by static review):** which process actually
   supplies the net column energy (SST turbulent flux vs radiation incl. B vs
   Hines eps_gwd vs MPAS tracer-floor/mass-fixer residual). Convection's
   conservatism means the source is one of these, not Bechtold's tendencies.

## Final corrected discriminating experiment (your design, refined)
Four **Bechtold** runs, everything else fixed at probe_bech40, from identical
dynamical fields with scheme-compatible carries:
| | LW-OD zero | LW-OD malformed (current) |
|---|---|---|
| dx_m = 0 (current) | cell 00 | cell 01 |
| dx_m ≈ √(mean cell area) | cell 10 | cell 11 |
Plus an SBM baseline and a SEPARATE pinned/transient deck contrast (to isolate
ozone/SW-aerosol/GHG/cadence from B). Per-step area-weighted (`areaCell·dp/g`)
ledger with process rows for convection, turbulence (export SH/LH — currently
not an MPAS diagnostic, `turbulence/integration.py:586`), radiation (SW/LW
split + paired zero-LW-OD ΔOLR), GWD (eps_gwd), microphysics, tracer floor,
mass fixer; plus all water species, surface precip, pressure-mass, and the
moist+kinetic+geopotential energy terms. Note the existing `--budget-ledger`
is dry-enthalpy-only and not wired through the MPAS combined-physics path
(`process_ledger.py:23`).

## Ask
Given this is a static findings review (the experiment is out of scope): do you
agree the SEVEN items above are the correct settled/experiment-gated split, and
that findings 1-6 have no remaining CODE defect to add? If you have any
FURTHER code-level finding (not experiment design), name it with file:line;
otherwise confirm we've converged on the static-review deliverable.
