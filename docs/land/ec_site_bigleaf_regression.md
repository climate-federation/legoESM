# EC-site regression check — big-leaf vs multi-layer canopy (Phase-3 FvCB adoption)

## Question
Did PR #897 (route big-leaf SimpleSEB photosynthesis through canonical FvCB C3+C4;
retire the C3-only `farquhar_photosynthesis`) regress land fluxes at the 4 runbook
EC sites (US-MMS DBF, DE-Obe ENF, US-Ton oak-savanna, DE-Hai DBF)?

## Constraint (honest)
The runbook's Section-B model run needs the DifferBESS `<SITE>_driver_v2.nc` half-hourly
FLUXNET driver files (100–220 MB/site, SW/LW/T/q/wind/precip/CO2 + per-step LAI/Vcmax
+ observed fluxes). Those are NOT checked in and NOT on disk; DE-Obe/DE-Hai are not even
in the local FLUXNET2015 mirror. The checked-in `ec_site_example_data/*.nc` hold model
OUTPUTS + LAI + obs only — no forcing — so the model is not re-drivable from them.
There is also no big-leaf EC-site runner (the harness is two-leaf-only).

=> The runbook's skill-vs-obs evaluation is NOT reproducible in this environment.
   What IS runnable is a controlled BEFORE/AFTER comparison of the changed code on
   identical, site-representative forcing at the per-site LAI recorded in the committed fixture.

## Method (controlled, one variable = code revision)
- Baseline = git 658433222 (pre-Phase-3, C3-only Farquhar) in an isolated worktree.
- HEAD     = PR #897 (canonical FvCB).
- One probe script (`scripts/validate/ec_site_bigleaf_regression_probe.py`) run under each revision's packages via
  PYTHONPATH swap; imports ONLY the stable public API whose signature is identical at
  both revisions (`compute_effective_beta`, `c3_photosynthesis`, `c4_photosynthesis`).
- Forcing: 4 sites × {median, p90 real LAI} × SW∈{100,300,600,900} × β_soil∈{0.3,0.6,1.0},
  CO2=400, site-typical T_air/q. Per-site LAI + T_air/q come from the committed fixture
  `scripts/validate/ec_site_regression_lai.csv` (its header records provenance: LAI =
  median/p90 of the `lai` variable in the ec-site branch's `ec_site_example_data/<site>.nc`,
  transcribed because those NetCDFs are not on this branch; T_air/q are representative).
- Big leaf driven with `land_params=None` => pure C3 at BOTH revisions. The measured
  delta is therefore the WHOLE SimpleSEB coupled-photosynthesis path change, not just an
  equation swap: canonical FvCB kernel + changed effective Jmax/Rd defaults + `beta_soil`
  now folded into `Vcmax25_eff` (so soil stress also scales Jmax/Rd). All intended physics.
- Two-leaf canopy checked via its photosynthesis KERNELS (the only surface Phase-3 touched;
  the full canopy flux is a pure consumer of them).
- HEAD-only production-path coverage: the probe also drives the big leaf through the SAME
  `land_params` override SimpleSEB uses (per-site Vc_max25/g1/LCMA/fC4) for fC4 in {0, 1},
  so the new C3/C4 blend + override are actually exercised (not just the fC4=None default);
  the test asserts fC4 changes GPP at every site (C4 > C3 at warm light, as expected).
- The HEAD probe output is pinned as a committed golden (`ec_site_regression_golden.json`);
  `test_ec_site_bigleaf_regression_probe.py::test_matches_committed_golden` fails (rtol 1e-3)
  if a future change drifts the documented GPP shifts / per-site magnitudes / C3-C4 blend /
  canopy kernels — so this artifact is a regression pin, not a loose sanity check.
  Run it with:

      JAX_ENABLE_X64=1 python -m pytest \
          tests/land/integration/test_ec_site_bigleaf_regression_probe.py

  NOTE: the default CI gate (`.github/workflows/ci.yml`) runs `tests/unit/`, top-level
  `tests/test_*.py`, and a narrow `tests/atmosphere/` smoke — the whole `tests/land/` tree
  (this test included) is collected but NOT executed. Wiring an `integration-smoke` step
  that runs this file needs a maintainer push with the GitHub `workflow` scope; until then
  the golden is enforced on any manual/local run, not on PRs.

## Results

### 1. Two-leaf / multi-layer canopy — no regression
- `c3_photosynthesis`/`c4_photosynthesis` outputs are **bit-identical** baseline vs HEAD
  (max|Δ| = 0.00e+00, byte-for-byte).
- `git diff 658433222..HEAD` shows the canopy FLUX modules — `canopy/solver.py`,
  `surface_scheme/two_leaf_canopy.py`, `canopy/clm_ml_interface.py` — are **unchanged**;
  Phase 3 only factored the shared photosynthesis wrappers around identical public outputs.
- Unchanged flux code + bit-identical kernels ⇒ the full two-leaf canopy flux is unchanged.
  (The end-to-end two-leaf EC run itself was not executed — no drivers — but there is no
  code path by which it could differ.)

### 2. Big-leaf SimpleSEB — intended change, no breakage
- All 96 points finite; GPP ≥ 0; **0 sign flips**; monotone light-response preserved at
  both revisions.
- GPP magnitudes physically sane: 2.6–24.7 µmol CO₂ m⁻² s⁻¹ (loose sanity vs tower-obs p90
  US-MMS 20.7 / DE-Obe 16.3 / US-Ton 8.3 / DE-Hai 19.1 — magnitude in range only; NOT a
  matched skill comparison, see caveats).
- GPP change distribution (96 pts): median **−17.5%**, mean −15.7%, full range
  **−32.2% … +18.8%**. **86/96 decrease, 10/96 increase.** The increases are ALL at LOW
  light (SW=100) + wet soil — coherent: FvCB's electron-transport-limited rate slightly
  exceeds the old Farquhar light response at low light, and is lower at high light. So the
  shift is DOWN in the median / high-light regime, UP in the low-light corner — not
  universally down.

Productive regime (SW=900, β_soil=1.0, p90 LAI), GPP in µmol CO₂ m⁻² s⁻¹:

| site | pft | LAI | base GPP | HEAD GPP | Δ% |
|------|-----|-----|----------|----------|----|
| US-MMS | DBF | 5.9 | 24.68 | 21.97 | −11% |
| DE-Obe | ENF | 4.2 | 18.40 | 13.44 | −27% |
| US-Ton | SAV | 2.1 | 11.49 | 10.59 | −8%  |
| DE-Hai | DBF | 5.8 | 22.65 | 18.66 | −18% |

- `beta_eff` (stomatal moisture factor driving land ET) tracks GPP (e.g. DE-Obe
  0.791→0.634) — internally consistent (GPP and transpiration co-vary through gs); LE would
  move with it. Not an independent defect.
- Largest change at cold ENF DE-Obe: the big leaf uses a FIXED acclimation reference
  `_TGC_REF_BIGLEAF_C = 25 °C`; a boreal site acclimated to a colder Tg is slightly
  mis-acclimated. Documented simplification (the two-leaf canopy carries per-column TgC).
  Main big-leaf sensitivity.

## Verdict
- **Multi-layer / two-leaf canopy: no regression** — photosynthesis kernels bit-identical
  and the canopy flux code is unchanged in the diff.
- **Big-leaf SimpleSEB: no breakage; an intended physics change.** Instantaneous
  photosynthesis is finite, ≥0, monotone, sane magnitude; GPP moves median −17.5%
  (range −32%…+19%) as the full coupled path switches to canonical FvCB (Jmax-limited +
  acclimation + soil-stress on Jmax/Rd). Consistent with the land-carbon equilibrium realism
  gate (0 hard-check failures, C3+C4). No claim about skill vs observations.

## Caveats (controlled-comparison doctrine)
Synthetic representative forcing, fixed T_leaf=T_air, single-point instantaneous (no
diurnal/seasonal integration or leaf energy-balance solve). Therefore this is a
BEFORE/AFTER magnitude+sanity check, NOT the runbook's NSE skill-vs-obs evaluation; the
"vs obs p90" column is indicative only. A definitive skill comparison requires the DifferBESS
drivers + a big-leaf EC runner, neither available here.
