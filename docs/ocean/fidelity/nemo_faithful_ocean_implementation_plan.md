# NEMO-faithful DINO ocean — implementation plan

**Purpose.** Turn the term + join audits (2026-08-06) into an ordered, effort-and-leverage-ranked
plan to get legoESM's DINO ocean to NEMO fidelity. Grounded in two authoritative audits, both
cited file:line on both sides; not an estimate.

## Where we are (audit-confirmed, not estimated)

NEMO's DINO step = 106 CALLs. **0 uncovered.** Breakdown:
- **69 N/A** — I/O, dead namelist branches, diagnostics, debug dumps. Nothing to do.
- **21 DONE** — 16 FAITHFUL (AT BAR) + 5 CEILING (matched to NEMO's own arithmetic-noise floor).
- **27 DEBT** — 25 at 1e-4..1e-6 with mostly-identified causes + 2 NEAR-CLASS.
- **4 GAP** — and the audit split these correctly: only **1 is architectural**, 3 are measurement gaps.

**⇒ ~80% done by the gate's own accounting; the campaign has BEEN the port.** Differentiability,
calibration (params as leaves), and the modular framework all survive — established this session,
not at risk.

## The two axes, and which one holds the remaining physics

- **TERMS** (the gate's 53 rows): mostly done. The residual DEBT is 1e-4..1e-6 and, per the noise
  floor (0.091 Sv), climate-INERT individually — six same-state exonerations confirm no single
  term's formula owns a climate metric.
- **JOINS** (composition/time-level, the gate is blind to these): where BOTH confirmed defects
  lived (surface placement — fixed; form stress — open). This is the frontier.

## THE PLAN — four workstreams, ordered by (leverage x readiness)

### W1 — Close the form-stress defect (THE frontier; highest leverage, unknown effort)
The join audit localized two concrete suspects in the bottom/barotropic-accumulation path where
the deficit lives (mid-depth density heave at the sill flank):
- **W1a — `mlf_baro_corr` ordering (CONFIRMED DIFF).** lego reconciles barotropic/baroclinic
  BEFORE the implicit vertical solve (`_leapfrog_step:7055-7098`); NEMO does it AFTER
  (`stpmlf.F90:467`, post-`dyn_zdf:305`). A bottom-drag x depth-mean coupling — exactly the
  form-stress mechanism. **Cheapest check needs NO new instrumentation:** NEMO's existing
  `stp_dump_state_and_bt` dumps write `uu/vv(Naa)` both pre- and post-`mlf_baro_corr`; diff them
  at the bottom level near the sill flank. If the AFTER-solve correction moves the bottom velocity
  ~0, this demotes. If not, it is a candidate fix (reorder lego to reconcile after the solve).
- **W1a — RESOLVED 2026-08-06: IMMATERIAL for this recipe, suspect ELIMINATED (CONFIRMED, test-backed).**
  NEMO's `mlf_baro_corr` corrects depth-mean shifts that `dyn_zdf` injects (wind stress
  `dynzdf.F90:354-360`, implicit drag `:305-313`). legoESM's solve has NONE of those inside it
  under the kamm card: `surface_stress_implicit=False` (`dino.py:211`, never overridden), drag
  folded onto the depth-mean directly (`zdf_drag_in_matrix`), and `test_baroclinic_only_round_
  trip_conservative_at_zero_Av` PROVES the solve is depth-mean-exact. Nothing for a post-solve
  reconciliation to correct ⇒ the ordering DIFF is a tested no-op here. GAP row closed:
  `mlf_baro_corr_probe.py` committed; correction is a uniform-with-depth shift (per-column std
  6e-19), 30-39% of RMS velocity globally. (Config-scoped: revisit if a card ever sets
  `surface_stress_implicit=True` outside `explicit_substep` — already guarded by a raise.)
- **W1b — `fix_eta_drift` compensator (CONFIRMED, quantified).** An always-on volume-drift
  projector with NO NEMO analogue, masking the barotropic solver's own ~2.5e-7 volume
  non-conservation (`dino.py:1397-1403`). It sits in the ssh/e3-commit path. Test: with/without,
  integrated over years, how much of the form-stress residual it explains. A real fix means making
  the barotropic solve conserve volume faithfully rather than patching it.
- **W1c — the flank density-tendency budget** (the accumulation instrument the session converged
  on): budget the flank column's density tendency term-by-term over years, both models, find which
  term's TIME-INTEGRAL diverges while its instantaneous value matches. This is the general
  instrument; W1a/W1b are the two specific leads to check first.
**Effort: 2-4 lanes to identify, then the fix (1 config-reorder to a scheme change). This is the
one genuinely unknown-cost item — but it is NOT a fundamental barrier (same equations, same grid;
if NEMO does it we can transcribe it); it is enumeration effort on the join layer.**

### W2 RESULT (2026-08-06) — NOT a port; ONE surgical defect found (3rd form-stress suspect)
Term-by-term matrix diff of NEMO `trazdf.F90:118-293` vs lego `_apply_implicit_vertical_mixing`
(`ocean_model_latlon_cgrid.py:5424`) + `thickness_weighted_tracer_combine` (`:854-902`), both read.
- **`ah_wslp2` / GM-Redi vertical term concern RETRACTED**: not omitted — config-gated on
  `gm_redi.implicit_K33` (mirrors NEMO's `l_ldfslp`), threaded end-to-end. RHS content-identity,
  diagonal thickness (both Kaa), zero-flux BCs, timestep (rDt=2dt): all CONFIRMED IDENTICAL.
- **REAL DEFECT (previously unnamed): the implicit flux-coefficient DIVISOR uses the wrong
  time-level thickness.** NEMO uses `e3w(Kmm)` (NOW/pre-solve interface, `trazdf.F90:219-220`);
  lego's default `dz_half = 0.5*(e3t(Kaa,k)+e3t(Kaa,k+1))` is built from the AFTER state
  (`naa_expl.eta`, `:5523-5533`). Under moving z* these differ O(dη)/step, in the DIFFUSIVITY
  NORMALISATION, at mid-depth interfaces — same class and order as the surface-placement defect
  and the `thickness_weighted_tracer_combine` fix, acting exactly where the density heave lives.
- **Fix scope: SMALL, not a port.** The `implicit_vmix_dzw_slot` machinery (#428) already exists —
  it's wired to Veros's reference thickness, not NEMO's `e3w(Kmm)`. Add a `nemo_kmm` slot value
  (NOW-time e3w from `state.eta`) + a conservation test. → moves to W1 (a form-stress suspect),
  NOT W2. So W2's "1 real port" item is REDUCED to a surgical option-add.
- Flagged not-checked: `legoesm_nemo_like_v1`'s `adaptive_implicit_vertadv` (ln_zad_Aimp fold; recipe renamed from `nemo_v1` 2026-09-02) — separate lane.

### W2 — The architectural port (1 real item)
- **`tra_zdf` (z*-volume-form + GM/Redi vertical fold).** The one genuine implementation gap:
  lego's plain backward-Euler solver does combine-then-correct where NEMO folds RHS into the
  implicit solve with `e3t(Kaa)T(Kaa)=e3t(Kbb)T(Kbb)+2dt e3t(Kmm) RHS` inside the matrix. Target
  algebra matches (CONFIRMED both sides); the two solve PATHS' equivalence is UNSURE (needs the
  two tridiagonal matrices diffed term-by-term — not done in the reading audit). Also folds the
  `avt+ah_wslp2` GM/Redi vertical term lego omits. **Effort: 1 substantial lane to diff the
  matrices + decide port-or-prove-equivalent; if a real port, ~1 week.** NB `dyn_zdf` is the SAME
  solver's momentum path — resolve together.

### W3 — Close the 3 measurement GAPs (cheap, high documentation value)
`dyn_zdf`, `traldf_iso_lap tendency`, `mlf_baro_corr` — code exists and executes, NEMO dumps
already on disk, just no probe. **Effort: ~1 lane each (an afternoon of probe-writing), not dycore
work.** Closes 3 of the 4 GAP rows to a real measured status. Do these first — they're free and
they may themselves surface a W1-relevant number (`mlf_baro_corr`'s probe directly serves W1a).

### W4 — Grind DEBT to the class bars (mechanical, parallelizable, background)
25 DEBT + 2 NEAR-CLASS rows to POINTWISE 1e-15 / ACCUMULATING 1e-12 / CONDITIONED->CEILING.
Mostly conditioning candidates and identified-cause tails. Climate-inert individually (noise
floor), so this is fidelity-completeness, NOT the path to closing the ACC excess — keep it
background, do not let it block W1. Known sub-items: zdftke composite (largest, 7% ratio),
dyn_ldf e3-weighting (OPEN ESCALATION — faithful transcription WORSENS it, needs Dhruv),
traadv_fct family, dyn_vor, dom_qco_r3c band.

## Sequencing (one lane at a time, cheap agents, Fable orchestrates)

1. **W3 first** — free, and `mlf_baro_corr`'s probe feeds W1a. (~3 lanes)
2. **W1a + W1b** — the two localized join suspects, existing dumps. (~2-3 lanes) — this is the
   decisive fork: it tells us if the form-stress defect is a bounded join fix or needs W1c.
3. **W1c** if W1a/W1b don't own it — the trajectory density-tendency budget. (1-2 lanes)
4. **W2** — the tra_zdf/dyn_zdf port, in parallel with W1 (independent). (~1 week)
5. **W4** — background grind, never blocking.

## Cost, honestly

- **Known-bounded (W2+W3+W4): ~3-5 weeks**, mostly mechanical, all scoped and cited.
- **W1 (form stress): the one unknown.** If W1a or W1b owns it -> folds into days. If it needs the
  full accumulation instrument and a scheme-level fix -> weeks. NOT open-ended research: it is a
  join, joins are enumerable, and the two leads are already localized to the bottom/barotropic path.
- **Success = the 90-day acceptance gate (built, demonstrated) green at 1x** (ACC 0.09 Sv), all
  budgets closing, DEBT->0. The gate is the finish line and it already runs.

## What this does NOT require (settled this session)
- NOT starting from scratch (~80% done). NOT abandoning differentiability (translation is
  differentiable — FESOM2-JAX proves it). NOT freezing params (structure is faithful, params stay
  tunable leaves). NOT a bespoke non-shared dycore (the faithful pieces are selectable blocks in
  shared modules — the recipe strategy). The one intrinsic limit: the momentum operators
  interoperate only within the lat-lon C-grid family, which is a property of the grid, not the code.
