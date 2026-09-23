# DINO / NEMO fidelity debt register (#1226)

Written 2026-07-27 in response to: *"how many more 'good enough's have you left
in there?"* — a full audit of every term where I declared PASS / MATCHED /
CLOSED / WAIVED while the measurement was not actually exact.

**The bar** (standing user directive): NEMO-faithful, corr 1.0 **and** ratio 1.0.
Not "close", not "within tolerance", not "attributed to X" on the strength of a
single black-box analysis.

**Method required to clear an item** (learned the hard way): dump the routine's
**internals** from NEMO, compare **stage by stage**, and name the first
deviating stage. Output-only comparisons hide compensating errors — that is how
`dyn_adv` passed for a day with a perfect KEG masking a broken ZAD, and how
"threshold chatter"/"irreducible amplification" got recorded as verdicts when
they were only ever hypotheses.

---

## A. Revoked acceptances (already reopened)

| # | term | measured | label I gave it | why it is debt |
|---|---|---|---|---|
| 8 | `traadv_fct` limiter | tendency 0.9923 | "FAITHFUL" | `nonosc` is deterministic; a true transcription must hit roundoff. Prime suspect for spurious diapycnal mixing. **IN PROGRESS** |
| 4 | `zdf_mxl` | 99.88% levels (12 cols) | "ACCEPTED" | "below input precision" never proven by comparing internals |
| 10 | `dyn_vor` EEN | 0.9999; bottom levels 1.04–1.07 | "ACCEPTED" | two fix attempts inert ⇒ mechanism still unknown |
| 7 | eiv transport | u 0.9985 / v 0.9954 | "near, accepted-for-now" | "diffuse, no lead" = not investigated to internals |
| 11 | `zdftke` composite | 0.9976 (257 cells) | "CLOSED" | "EVD threshold chatter" is a hypothesis, not a verdict |
| 5 | `ldf_slp` bottom row | \|x\| 1.0255 | "irreducible amplification" | asserted from a substitution test, not from internals |

## B. Called PASS/MATCHED with a ratio that is NOT 1 (never revoked — new debt)

| term | corr | ratio / error | what I said |
|---|---|---|---|
| `eos_rab` α | 1.000000 | median 4.7e-6 (β bit-exact) | "PASS" |
| `bn2` | 1.0000000000 | median \|rel\| 6.96e-6 | "PASS" |
| `ldf_eiv` κ | 1.000000 | \|x\| **1.000608** | "CLOSED" |
| `dyn_hpg` | 1.000000 | \|x\| **1.000045** | "MACHINE-EXACT" |
| `dyn_spg_ts` outputs | 0.9996–0.99999 | `puu_b` \|x\| **0.9871**, `un_adv` **1.0067** | "VERIFIED" — a 1.3% ratio gap, never explained |
| ATF filters | u 0.999969 | u \|x\| **0.9955** | "MATCHED" — 0.45% |
| `dyn_ldf` | 0.9979 / 0.9994 | \|x\| **1.0039** | "MATCHED" — 0.4% |
| `traadv_fct` fluxes | 0.99994 | \|x\| 1.0001 | "faithful" |
| slopes (interior) | ≥0.9988 | \|x\| 1.0011 | "CLOSED" |

## C. Never verified at all (waived, deferred, or structurally skipped)

| item | status | note |
|---|---|---|
| `mlf_baro_corr` | algebra-verified only | inlined at -O3; needs a `_step_impl` diagnostics hook. NEMO dumps 8883-8886 already exist |
| `dom_qco_r3c` **r3u/r3v** | never compared | `nemo_io` reader lacks `hu_0`/`hv_0`; only the T-point `r3t` was checked |
| `lbc_lnk` sign convention | deferred | needs a different harness |
| `zdf_mxl_turb` | UNVERIFIED | found by the runtime trace; turbocline depth, consumers unchecked |
| `zdf_drg_nonlin` + `dyn_drg_init` | "interface-covered" | bottom drag, never term-isolated |
| `dyn_cor_2d` (69×/step) | "interface-covered" | per-substep barotropic Coriolis, never term-isolated |
| tracer advection **salinity** | never compared | only `jp_tem` was dumped/compared; S assumed to follow |
| `traadv_fct` clean tendency | worked around | `trd` dumps are Krhs-contaminated; comparison used flux reconstruction, not a byte-level tendency |

## D. Unexplained anomalies (no owner, no hypothesis under test)

1. **legoESM recovers only 68% of its own ACC from its own density field**, while NEMO is thermal-wind self-consistent (1.46 by the same measure, i.e. consistent within the method's bias). A genuine momentum/reference-velocity residual, never chased.
2. **The 90-day twin's `u_surf` got WORSE** (0.9797 → 0.9691) after the momentum RHS was made machine-exact. Never explained; implies error downstream of the explicit RHS.
3. `dyn_spg_ts` `puu_b` \|x\| 0.987 — the entry-seed hypothesis was falsified (fix inert); no replacement hypothesis.
4. NEMO's `grid_T` history output is **numerically wrong** (votemper ×9.6 surface → ×290 deep; restarts are fine). Cause not diagnosed — an XIOS thickness-weighting/normalisation issue is suspected. Any past analysis reading NEMO T/S from `grid_T` is invalid.

## D2. Named campaign tasks opened by the PR #1640/#1634/#1645 dual review (2026-08-23)

Each of these is a REAL work item with an owner slot, not a caveat. None is started.

| # | task | why it exists | cost estimate |
|---|---|---|---|
| D2.1 | **Re-tune bottom drag against the corrected drag.** | Two independent fixes weakened the seafloor drag: the face-control-volume fix (`4734c2d5f`) and the factor-2 removal (#1645). Combined vs the original code: **~2× weaker in open water, ~4× weaker over stepped bathymetry**. Any `r_bot`/`Cd` value tuned against the OLD drag is now off by that factor. No tuned value was changed in the merge, so every calibration inherited from before those two commits is stale. | one calibration sweep over the drag coefficient on the shipped card + a gate re-run; no new instrument needed |
| D2.2 | **Phase sweep + reference-antiphase control for the seasonal-clock attribution.** | The clock A/B sampled TWO phases (0 and 180 days). That cannot yield a share of the error, and every percentage derived from it is retracted (see `PREREG_gate90_corrected_clock.md`). Sound attribution needs (a) a dose-response curve at 0/45/90/135/180 days showing a smooth collapse, and (b) a control in which the **reference** model runs with artificially antiphased forcing — that arm isolates the season with ZERO model difference. | 5 twin arms + 1 NEMO arm at 30 days each; the 30-day window is where the effect was originally measured |
| D2.3 | **Semiannual blind spot.** | An exact 180/360-day antiphase flips only the ODD harmonics, so the semiannual component was IN PHASE in both arms. Neither arm can see semiannual error, including whatever part of the residual is semiannual. Unquantified. Falls out of D2.2 for free if the sweep includes a 90-day offset. | subsumed by D2.2 |
| D2.4 | **Close the Euler-start reconciliation gap for real.** | The forward-Euler first step returns before NEMO's second reconciliation site, so step 1 commits a depth-mean deposit NEMO removes. Now warns loudly (`_warn_euler_start_skips_after_reconcile`) instead of being silent, but the gap is still there. Closing it needs `_step_impl` to surface the barotropic depth mean on its `_apply_implicit_vmix=True` path — a return-contract change at ~8 call sites. A raise was rejected: the shipped twin defaults to `bridge_before=False` and would be refused. | ~8 call sites + their tests; or sidestep entirely by defaulting the twin to `--bridge-before` (a physics change to shipped runs — needs its own A/B) |
| D2.5 | **Select the reference face-thickness rule from the DOMAIN BUILDER, not by hardcoding — and make the whole chain agree.** | NEMO builds `e3u_0` **two different ways**, and the discriminator is the builder, *not* `ln_zco` vs `ln_zps`: DINO/usrdef grids AVERAGE (`cfgs/DINO/MY_SRC/zgr_lib.F90:231`, reached via `zgr_sco_mi96` on **both** branches of `usrdef_zgr.F90:108-139`), DOMAINcfg partial-step grids take the MIN (`tools/DOMAINcfg/src/domzgr.F90:1166`, inside `SUBROUTINE zgr_zps` `:987`). So `min_cell_to_uface` is **also** NEMO's own rule for that grid class, not merely the MOM6 `hFacW` convention. The reconciliation now hardcodes the mean — right for the only card that ships the option (a DINO-built grid), inert everywhere today, and wrong on a DOMAINcfg partial-step grid. **Related, same row:** the reconciliation's TARGET `btu_exp` comes from the barotropic solve, whose column depth uses the MIN rule, so on a grid where the two rules differ the weighting and the target disagree end-to-end. Not built: no such card exists, and the two rules are **indistinguishable on every card that does**, so a selector would ship untested. | small, but must arrive WITH the first DOMAINcfg partial-step card that enables the option, not before |
| D2.6 | **Move the gate's certification check off the caller and onto the shared path.** | `acceptance_gate_90d.main()` now withholds the verdict off-claim, but `gate90_time_series.py` and `reconcile_confirm.py` call `print_gate` directly and still issue full PASS/FAIL on any candidate. Root-cause placement is inside `print_gate` or `load_candidate`, which the siblings already use. | small; blocked only on deciding whether the siblings *want* a verdict |
| D2.7 | **Widen the ladder content hash to the topology, or rename the stamp.** | It hashes the four 1-D reference arrays only, so two runs on different bathymetry/wet-dry staircases hash identically. That is enough for the ladder-mode question it was built for, and it is **not** a full grid identity — the stamp name oversells it. | one-line change to the hash input; needs a decision on comparability with already-recorded stamps |
| D2.8 | **Same two-arm share defect, different experiment, untriaged.** | `kamm_twin_90d.py:1228` still reads "the recorded 10-yr A/B **closes 98.7 %** of the ACC gap" — the same ownership-from-two-arms construction retracted for the seasonal clock, on the `surface_tendency_placement` experiment. Not touched here because it is a different experiment with evidence this triage did not review. | one measurement review; the wording fix is one line once someone checks the arms |

Checked and closed by the same review, recorded so they are not re-opened:
* **Diurnal misphasing — N/A.** The reference sets `ln_diu_cyc = .false.` (`RUN_90D_TWIN/namelist_cfg:34`), so `usrdef_sbc.F90:572-576` assigns `qsr = pqsr_dayMean` with no time-of-day term; legoESM's port has no diurnal code at all. Neither side carries a diurnal cycle, so a 12-hour phase error is unobservable here.
* **Calendar-indexed ancillary fields — none found.** T\* and Qsr are the only calendar-time consumers in the DINO forcing path and both already route through the corrected clock. Salinity restoring is allocated once and never recomputed; EMP/runoff is off (`ln_emp_field=.false.`); shortwave penetration uses a fixed Jerlov type with no chlorophyll climatology; wind stress is latitude-only.
* **The "0.044 Sv margin vs an unvalidated floor" worry — superseded.** The 90-day floor has since been measured directly (`4e652b101`, `1dba0b733`, `5f173cfff`): single-run ACC spread is **1.15e-05 Sv**, so the margin is ~2700× the measured floor, not "less than one floor". The 0.091 Sv constant is a cross-model **tolerance**, not a 90-day noise floor, and is validated as the right scale only at the **1-year** horizon (`dino_verdict360_result.md`, P1 CONFIRMED).

## D3. NEXT-ACTIONS REGISTER (2026-08-26) — the ranked queue after consolidation

Authoritative ranking for the campaign as it stands after the wall-flicker
stack. Full context and every citation:
`docs/ocean/fidelity/dino_campaign_synthesis.md`. Nothing below is started.

### #1 — The Coriolis PAIR. Fix both halves together, never one.

**Ranked first.** legoESM builds the Coriolis parameter at the vertex as the
average of the two adjacent tracer-row values; NEMO evaluates it at its own
f-point latitude. Measured against NEMO's dumped `ff_f`
(`dino_wall_fixed_bias.md`, "The sibling error this arm uncovered"):

| | median | RMS | at the walls | at the equator |
|---|---|---|---|---|
| v-face zonal metric gap (the arm just run) | +1.86e-05 | 2.08e-05 | +3.35e-05 | +2.9e-09 |
| **Coriolis at the vertex** | **−5.48e-05** | **6.15e-05** | −2.50e-05 | −9.19e-05 |

The two sit in the **same** EEN rotation coefficient with **opposite sign** and
**complementary latitude shape** — signed profiles correlate at +1.000, and in
`e1v · f` they **partially cancel**. The override arm removed the **smaller**
half and scored PARTIAL (16.2% basin / 24.9% wall rows). **Rule 8 applies: fix
the pair together.** The registered design is three arms — f alone, `e1v` alone
(done), and both together — with the verdict pre-registered on the **joint** arm.
This campaign has four recorded faithful-but-worse instances from raising
fidelity on one half of a cancelling pair; do not add a fifth.

**Half of it is a constant, not a scheme.** The gap decomposes into a uniform
−1.578e-05 and a latitude-varying −3.90e-05. legoESM's `constants.Omega` is
**7.292e-05**; NEMO's is **7.2921150830e-05** (confirmed from `phycst.F90` and
independently from the dumped `ff_f`). Tracer latitudes match NEMO's `gphit` to
1.4e-14 degrees, so this is a constant, not a grid difference.

**Free premise check first, before any compute** (code read, 2026-08-26, not a
measurement): the DINO card already pins NEMO's rotation rate as a top-level
`omega` field (`experiments/dino.py:1299`, from
`NEMO_CONSTANTS_CONFIG.Omega = 7.292116e-05`, `ocean/constants_config.py:70`),
and one consumer of the 4-significant-figure literal was already fixed earlier
in the campaign (the eddy-coefficient path, where Ω is squared into κ_GM —
status board fixed bug #14). Meanwhile `packages/core/legoesm/constants.py:12`
still reads `Omega = 7.292e-5` and the lat-lon C-grid Coriolis builder reads
`getattr(grid, "omega", constants.Omega)`
(`ocean/dynamics/latlon_cgrid_operators.py:3278,3296`). **The run plausibly
carries two rotation rates at once.** Establish by grep which f-consuming sites
see the card's pin and which fall back, then decide between a `ConstantsConfig`
pin, a card wiring change, or both.

**Cost:** the same offline freeze-and-vary substitution as the completed metric
arm, three arms. No new instrument.

### #2–#8 — the rest of the queue

| # | action | register item | cost |
|---|---|---|---|
| 2 | **Split the gate's ACC row into channel + basin, scored separately.** The full-section reducer sums a walled sub-polar gyre into a number labelled "circumpolar" and is **opposite in sign to the channel it nominally measures**. Reached independently by the substep lane and by the verdict run. **Proposal to the owner** — gate constants and reductions are not changed by a lane. | new | one gate-definition change + a re-score of recorded runs; no new compute |
| 3 | **Phase sweep + reference-antiphase control** for the seasonal-clock attribution. | D2.2 (+ D2.3 subsumed) | 5 twin arms + 1 NEMO arm at 30 days each |
| 4 | **Re-calibrate bottom drag against the corrected drag** (~2× weaker in open water, ~4× over stepped bathymetry after two independent fixes). | D2.1 | one coefficient sweep + a gate re-run |
| 5 | **Close the step-1 reconciliation gap for real** — a return-contract change so `_step_impl` surfaces the barotropic depth mean on its `_apply_implicit_vmix=True` path. | D2.4 | ~8 call sites + their tests |
| 6 | **fp64 snapshots on two ensemble members (days 30/90).** fp32 storage is a live confound twice over: it bounds the measured ensemble spread from above, and it manufactured the "NEMO convects more often" finding (casting NEMO to fp32 reproduces 98.7–99.9% of the apparent gap). Days 10–70 are currently unmeasurable at fp32. | new | rerun two existing members; no model change |
| 7 | **The multi-year horizon.** Everything certified is one year from a common state; the deep ocean has not adjusted, and two verdict metrics are recorded as **upper bounds only** because their spread is still growing. **REGISTERED, NOT RUN (2026-08-30):** `PREREG_multi_year_climate_equivalence.md` freezes a six-member-per-side, horizon-matched 20-year climate-statistics ensemble; `dino_multi_year_climate_equivalence_handoff.md` carries the exact held arms. | registered / held | the largest compute item on the board |
| 8 | **The from-rest raise decision.** Needs an owner's call, not a lane's. | carried | decision |

### On the board, no work assigned

- ~~**The wall-row RESCORE**~~ — **DONE. It landed 2026-08-23 and this entry was
  stale on the day it was written**, quoting the source file's to-do list rather
  than the section beneath it that had already executed it. Re-run at HEAD
  2026-08-27: **nothing reopens.** Friction's mask-dimensionality leg is exactly
  0.0 under both weightings and its magnitude leg moves 6.7541e-05 → 5.6500e-05
  (16%, same order). Its ENRICHMENT leg does cross the 3× bar (2.35× → 5.16×) —
  reported, not buried, and far too small to matter. **Lateral friction is
  EXONERATED, not provisional.**
- ~~**The VERTEX area is not NEMO's `e1f·e2f`**~~ — **CLOSED 2026-08-27, the
  same day it was opened.** The construction diff is named and the fix shipped
  inside `metric_convention="nemo_isotropic"`. legoESM built the vertex area as
  the exact spherical cap between adjacent TRACER latitudes; NEMO forms it as
  the product `e1f·e2f` of two scale factors taken at the F-point's own
  Mercator latitude (`usrdef_hgr.F90` :97/:109/:114/:118). On a Mercator
  coordinate `sin φ = tanh(Δλ·j)` gives `d(sin φ)/dj = Δλ·cos²φ`, so the cap is
  the EXACT interval integral of `cos²φ` where NEMO's product is its MIDPOINT
  value — **exact quadrature versus the midpoint rule**, gap
  `(Δλ²/12)(3sin²φ − 1)`, reproduced to a measured/predicted ratio of
  **0.999984** including the sign change at ±35.26°. The corrected area
  reproduces NEMO's own closed form to **3.1e-15** relative, from 4.10e-05.
  Because `pphif == pphiv` (:97 and :96 carry the same +0.5 offset), NEMO's
  F-cell area IS the square of its v-face width, so this reuses the width the
  previous fix corrected rather than deriving a second latitude.
  **PAIR ANALYSIS: UNPAIRED-SAFE, measured not argued** (`vertex_area_pair_
  analysis.py`). The area and the vertex Coriolis are the same defect at the
  same point and meet in the F-point absolute vorticity `ζ + f`; on a
  pre-registered bar of 1/10, the area half is **1.51e-04** of the Coriolis
  half at the median (p99 6.9e-02), because `|ζ|` median 2.05e-08 sits four
  orders below `|f|` median 1.02e-04. Removing it moves the total
  absolute-vorticity error by +0.050%. In the other active channel — the
  NEMO-faithful lateral viscosity, which divides by `e1f·e2f` — `f` does not
  appear at all, so it is structurally unpaired there. The one genuinely
  self-cancelling use of the vertex area (the Smagorinsky/om4p25 raw-stress
  path) is INACTIVE on the DINO card.
  **THE COST, named:** the (cap, cell-average) pair made the discrete curl of
  solid-body rotation equal `f` EXACTLY; the fix lands that at +1.17e-05
  median. NEMO's own pair is worse (−2.73e-05 median, 1.02e-04 max), so the
  twin's internal consistency now sits between legoESM's and the oracle's.
  **One-state response: INERT** — against a pre-registered 1% bar, the wall-row
  fixed-bias residual moved −0.011% and the meridional deposit −0.040%, both
  toward NEMO. No multi-state arm run.
- **A pre-existing NaN in the adjoint of the sea-surface-height-average face
  depth**, found by review while checking the v-face fix and NOT caused by it.
  The barotropic face-depth helper divides by the v-face cell area without a
  positivity guard, which is `inf` on the two zero-width end-wall rows; the
  forward run is rescued by a later replace, but the reverse pass multiplies
  that `inf` by a zero cotangent and yields **96 NaN gradient entries** (2 rows
  × 48 columns), identical under both metric conventions. The repo already
  guards the identical pattern correctly one file away with a
  `where(width > 0, 1/width, 0)`. One-line fix; matters to anyone
  differentiating through the barotropic solver. NOT actioned here — out of
  scope for the register items, recorded so it is not lost.
- **PLAUSIBLE, not confirmed — the overturning / heat-transport diagnostic's
  v-face width.** One reviewer reported that this diagnostic reconstructs the
  face metric rather than reading the model's own, at ~3e-03 relative on a
  stretched grid (90x the gap just fixed). Reading the code, the primary branch
  DOES read the stored width, and the fallback that does not is reached only
  when the grid carries no face-latitude axis — which is not the DINO case, and
  I did not reproduce the number. Recorded as a flagged path to check, NOT as a
  measured defect, and pre-existing either way.
- The **grown-noise census** at days 5/10 in both models — states already on
  disk. The convective switch's ~300× rectification is established at ≳1e-10 and
  **silent at 1e-14**, so on current evidence the edge does not explain the
  models' amplification asymmetry at the ensemble kick amplitude.
- The **wind-placement term**, now unlocked as measurable offline (its dump slot
  is allocated and never written; the placement difference is real).
- The **closed sub-basin mass budget** — the control volume nobody drew.
- The **78–435× kick-amplification asymmetry** (computational-mode probe named).
- **Eighteen of nineteen** DINO probes still call `tendencies_with_diagnostics`
  without a before-level state (see §D3.1 below). Per-probe judgement, not a
  mechanical sweep.

### D3.1 — Instrument debt opened by the wall-flicker stack

`LatLonCGridOceanModel.tendencies_with_diagnostics` silently dropped the
before-level argument its sibling accepts, so every lateral-friction term in
every oracle budget on this card compared now-level against before-level state.
Fixed by adding an `ldf_state` parameter (`513339cba`); the term's difference
fell **262×** (1.5947e-9 → 6.0875e-12 m/s²) against a registered bar of 50×.
**Nineteen probes call the wrapper; eighteen still do not pass a before-level
state.** Any budget result from those eighteen predating the fix is suspect for
its lateral-friction term.

### D3.2 — What is closed, so it is not re-opened

Every row here was tested and eliminated with a decisive number. The full table
with citations is `dino_campaign_synthesis.md` §2.3. Do not re-test:

**Now ON this list — lateral friction (2026-08-27).** Free-slip is identical at
0/372,528 corners and the viscosity ablation makes it a lever rather than an
owner. The earlier caveat here — that this rested on the retracted layer-averaged
instrument — is **RETRACTED twice over**: the rescore had already landed, and the
ablation is scored on 90-day Sv transports, which carry no vertical weighting at
all and so could never have been touched by that instrument. **Do not re-test.**

surface forcing / wind (1e-10 per row, torque constant across all 36 windows) ·
water masses (4e-5 kg/m³, with the density-*gradient* caveat) · bottom drag
(coefficient bit-exact 0/9758; assembled increment 3.9e-18) · EEN vorticity scheme (fixed, transport-inert) · e3f at dry-neighbour
vertices (95× too small, 0% closure) · face-depth divisor (fixed, 0.0000 Sv) ·
every post-tendency stage at the matched step (wrong sign **and** wrong shape) ·
salinity advection (improves at day 180 on every part) · isoneutral slopes
(99.3–99.9% collapse on NEMO's own density) · the implicit vertical momentum
solve (fixed; divisor-independent under the barotropic split, an exact reason) ·
per-step barotropic injections (retention-corrected, 8–11× short and wrong sign)
· the meridional sidewall flicker (transient, depleted in member differences) ·
the half-step impulse lag (owned by the twin's Euler start; nothing
re-baselines) · the convective switch as a *differential* rectifier (shared with
NEMO; legoESM's fires **less**) · barotropic time filter (0.02%) · face-thickness
convention (~1%, anti-aligned) · rigid meridional position error (0.8%).

**And the conclusion drawn from the whole set:** the deficit is **not producible
by any operator difference at a matched state**. It must be rectification —
sub-floor per-step differences amplified through the basin's own feedback over
the year. The next experiments are feedback-class, a different cost and design
class from anything above.

---

## E. `ldftra` isoneutral diffusivity (Redi, `nn_aht_ijk_t=20`) — instrumented 2026-07-27

Investigated as the leading suspect for the 13%-too-weak upper-ocean meridional
T gradient (costs ACC via thermal wind). **Verdict: NOT the cause** — the
coefficient magnitude is bit-exact; a small (~2.6e-5) v-face discretization
residual was found and is now tracked, but it is far too small to explain 13%.

- **Formula correction (source read + live dump both confirm)**: NEMO's
  `nn_aht_ijk_t=20` (`ldftra.F90:290-296,322-326`, `ldfc1d_c2d.F90:106-155`)
  computes `ahtu/ahtv = (½·rn_Ud)·max(e1u,e2u)^1` — i.e. **`rn_Ld` is DEAD for
  this branch** (`aht0 = ½·rn_Ud·rn_Ld` is computed but never used by
  `CASE(20)`; only the printed `aht0` diagnostic references it). The task
  brief's "aht0 = 1350 m²/s (=½·rn_Ud·rn_Ld)" is a **documentation red
  herring** — NEMO's own `ocean.output` prints the value actually used:
  `"maximum reachable coefficient (at the Equator) = 1501.1854665553683 m2/s"`
  (`ldftra.F90:325`, `zah_max = zUfac*(ra*rad)^inn`). Confirmed independently
  by a live `kt==nit000` dump of `ahtu`/`ahtv` (MY_SRC/ldftra.F90, units
  8960-8963, `cfgs/DINO/RUN_1226_AHTU`).
- **legoESM's `K_h_base = 0.5*cfg.U_T*grid.radius*grid.dlon`** (dino.py:2520)
  evaluates to **1501.1854665553683** for both the default and
  `nemo_faithful_dino_config()` grids — bit-identical to NEMO's `zah_max`,
  because `grid.radius=constants.R_earth=6371229.0` and `grid.dlon≈1°` in
  radians reproduce `ra*rad*rn_e1_deg` exactly. The task's other stated
  number ("kappa_Redi = 1443.4475639955465") does not reproduce from current
  source with either grid path checked — likely stale/from a different
  config snapshot; the live value is 1501.1854665553683.
- **Field comparison (u-point, all 36 levels, both hemispheres)**: corr
  1.0000000000, ratio 1.0000000000, rel-err median 0, p90 1.7e-16 — roundoff.
  **AT BAR.** legoESM's `static_kappa_redi_override` (T-point `cos(lat)`
  field) → `interp_cell_to_uface` (same-row average) is a no-op for a
  field that is constant along a row, so it reproduces NEMO's direct
  `cos(gphiu)` evaluation exactly.
- **v-point — FIXED 2026-07-27 (tier-2 item 1)**. The original note here
  ("`interp_cell_to_vface` averages `cos(φ_j)` and `cos(φ_j+1)`... `avg(cos) ≠
  cos(avg)`") was **WRONG** — `static_kappa_redi_override` never called
  `interp_cell_to_vface`; grep + `git log -p --follow` confirm it has never
  existed in that function. That description was a probe-reimplementation
  artifact (Rule 0: "trace the real dispatch chain, a probe that re-implements
  a path is not evidence about that path"), not a description of the code that
  actually ran. The REAL cause: legoESM built ONE T-point field
  (`K_h_base*cos(lat_T)`) and reused it unshifted for BOTH the u-face and
  v-face Redi fluxes (`zfu`/`zfv` in
  `nemo_iso_lap_tracer_tendency_latlon_cgrid`, plus the shared
  `nemo_iso_w_kappa_sums`/`nemo_iso_a33` w-point kappa averages that feed the
  `ln_traldf_msc` implicit K33). NEMO's `nn_aht_ijk_t=20` does NOT do this:
  `ldf_c2d('TRA', ...)` (`ldfc1d_c2d.F90:141-145`) evaluates `ahtu` and `ahtv`
  as two INDEPENDENT arrays, `ahtu(ji,jj)=zUfac·MAX(e1u,e2u)^inn` and
  `ahtv(ji,jj)=zUfac·MAX(e1v,e2v)^inn` — i.e. `∝cos(lat_u)` and `∝cos(lat_v)`
  respectively, each at its own point. `cos(lat_u)==cos(lat_T)` exactly on
  this grid (u shares its T-row's latitude) so the u-face was already exact;
  `cos(lat_v)` is a genuinely different (row-shifted) value that legoESM was
  never computing at all.
  **Fix**: `static_kappa_redi_override` now returns `(kappa_T, kappa_v)`,
  with `kappa_v = K_h_base·grid.cos_lat_v[1:]` (the true v-face latitude,
  north-face-of-cell-j convention matching `grid.dx_v[1:,:]`). A new
  keyword-only `kappa_Redi_v`/`kappa_redi_v_override` (default `None` ⇒ reuse
  the existing field, bit-identical for every other closure — Visbeck/EKE/
  GEOMETRIC/Treguier are genuinely T-point quantities that legitimately DO
  face-average the same way for u and v) threads this through
  `nemo_iso_lap_tracer_tendency_latlon_cgrid`, `nemo_iso_w_kappa_sums`,
  `nemo_iso_a33`, `gm_redi_tracer_tendency_latlon`, and
  `compute_isoneutral_K33_latlon`.
  **Verified against the actual NEMO dump** (`ldftra_dump_{ahtu,ahtv,gphiu,
  gphiv}.bin`, `RUN_1226_AHTU`, reproducible via
  `scripts/validate/ocean_fidelity/dino_1226/ldftra_ahtv_compare.py`):
  pre-fix v-face corr 0.9998618 / ratio 1.0000380 (NOT the previously
  recorded 1.0000260 — the earlier number came from whatever ad-hoc
  reimplementation produced the false "interp_cell_to_vface" story, not from
  this code); post-fix corr 1.0000000 / ratio 0.9999999704 — matches the
  u-face's own bit-exact quality. Ground-truth unit test:
  `tests/ocean/unit/test_dino_experiment.py::TestDinoLatLonModelConfig::test_static_kappa_override_row_scaling`.
- **Residual open question**: the ahtu/ahtv coefficient is now the LEAST
  likely explanation for the 13% front deficit found so far. The search for
  that deficit should move to a different term (candidates from this same
  ledger: `traadv_fct` vertical upstream flux over-clip 1.7-1.9x — row
  "traadv_fct VERTICAL upstream flux" above, added by a parallel
  investigation — or the slope/taper chain `ldf_slp`).

---

## Honest count

- **6** acceptances revoked
- **9** terms labelled PASS/MATCHED with a non-unit ratio
- **8** never verified at all
- **4** unexplained anomalies

(Counts above are the 2026-07-27 audit's and are unchanged; §D3 carries the
current queue, and `dino_campaign_synthesis.md` carries the campaign-level
ledger of fixes, retractions, PRs and review rounds.)

The step-by-step methodology was created precisely to prevent this, and then I
applied it with a tolerance it does not have. Nothing in A–C is known to be
wrong; the point is that **none of it is known to be right** at the stated bar,
and the campaign has now produced enough retractions to make that distinction
load-bearing.


## F. RETRACTED findings — read this before re-investigating

Retractions must live where the next session looks, or they get re-discovered
at full cost. This section is that place.

| retracted claim | why it died | date |
|---|---|---|
| "Tracer operator COMPOSITION differs from NEMO (GM/Redi mutates the field advection reads)" | TRUE for `nemo_dino_kamm` (forward_euler) but **NOT for `nemo_dino_kamm_mlf`**, the recipe under test: `_leapfrog_step` passes hardcoded `_ab2_scope_override="advective"` (ocean_model_latlon_cgrid.py:6836,6868), routing GM/Redi to the ADDITIVE `_diss_dT_incr` bucket; the mutation branch (:3873) is dead code there. Final update :6925 is additive. Only the single first from-rest step mutates (1 of 58,400). The probe hand-reconstructed the mutating path and mislabelled it "actual code order" without tracing dispatch. NOTE: an earlier session had already retracted this in `dino_session_2026_07_25.md` — it was re-discovered because that retraction was not carried forward. | 2026-07-27 |
| "N² is 54% too large" | interface index off-by-one in the comparison; correctly aligned = corr 1.000000 | 2026-07-26 |
| "hmlp is 14% too deep" | unvalidated depth lookup; control test now reproduces NEMO's hmlp to 0.0 m | 2026-07-27 |
| "slopes are 9% too large" | signed-sum ratio on a SIGN-CHANGING field inflated a 1.2% magnitude bias ~9x | 2026-07-27 |
| "wslpi corr 0.9585" | alignment artifact; a proper ±2 offset scan peaks at 0.999110 | 2026-07-27 |
| "col_stretch recovers (1+r3t)" | `h_partial` is the STATIC at-rest thickness; measured identically 1.0 | 2026-07-27 |
| "dry-vertex e3f_0 fallback explains the EEN bottom bias" | controlled A/B was byte-identical | 2026-07-27 |
| "the vertical upstream flux is corr 0.91" | OFFSET ARTIFACT — the probe headline scored `offset=+1`; at the correct `offset=0` the advecting transport, upstream flux and upstream tendency all match at corr >= 0.9977. (A REAL but climate-inert dry-cell masking bug was found while chasing it: w-face clips 1126-1256 -> 603 vs NEMO 662, commit 01c1f226a.) | 2026-07-27 |
| "the harness biases EVERY measurement ~0.44% (e3t default) + 30-60 ppm (dy_v vs e2v)" | MOSTLY REFUTED by measurement. The e3t default affected ONLY `eiv transport`; dyn_ldf/ATF/dyn_spg_ts/zdftke/ldf_slp/ZAD/salinity are identical under off vs both. The dy_v/e2v part is refuted outright: on non-tripolar grids `gradient_*_cgrid`/`divergence_cgrid` recompute face metrics inline and NEVER read grid.dx_u/dy_u/dx_v/dy_v, so substituting NEMO's metrics is a structural no-op. | 2026-07-27 |
| "dyn_hpg is AT BAR (1.000000008)" | Measured on a DEGENERATE from-rest state (ssh=0, zonally-uniform IC) where `du` is identically zero on both sides and the zuap/stretch terms vanish. On the actual Y5 twin state it stays 1.000045. The number certified the trapezoid p'/EOS/g/gradient path only. | 2026-07-27 |
| "ldf_eiv kappa has an INDEPENDENT operator defect (corr 0.975)" | MEASUREMENT ARTIFACT. The probe compared lego's raw T-POINT kappa against NEMO's `paeiu`, which is the U-FACE AVERAGE (ldftra.F90:741, 0.5*(zaeiw(i)+zaeiw(i+1))*ssumask) -- a different quantity; the probe's own comment wrongly asserted they were the same. Routed through the existing `nemo_kappa_gm_to_faces`: corr 0.975163 -> 0.999995, ratio 1.032510 -> 0.999958. Every named intermediate (zn/zah/zhw/zRo/zaeiw) was ALREADY at 0.99999-class -- there was no deviating stage. NOTE the 3x3 alignment scan still picked offset (0,0) because both fields are smooth: **a passing alignment scan does NOT prove you are comparing the right QUANTITY.** | 2026-07-28 |
| "traadv_fct SALINITY is broken (corr 0.203 / ratio 3.17), the most alarming row" | MEASUREMENT ARTIFACT. All five traadv_fct rows were comparison-convention errors, not live defects. The `fluxes` row picked its k-offset by maximising correlation against the Krhs-CONTAMINATED trd dumps (which carry tra_sbc/tra_qsr from earlier stpmlf calls and peak at the WRONG offset +1); the contamination-free reconstruction peaks sharply at 0. Reproducible at HEAD: salinity 0.999952/0.999900, tendency(T) 0.999991, fluxes 0.999986-0.999999, vertical upstream flux 0.999986. The CANCELLATION MECHANISM is real (corr(horiz,vert) = -1.0000 for S vs -0.998 for T, so S genuinely is the sensitive detector) -- but with components at five nines the amplified residual does not exist. | 2026-07-28 |
| "the FCT limiter is the climate lever" | centered (unlimited) advection reproduces fct2 to 4 decimals over 5 years; ACC identical | 2026-07-27 |

**Five probe/relay retractions in one day** (composition reconstruction,
vertical-flux offset). EVERY headline number must carry its own offset/alignment
scan BEFORE it is reported as a finding — an agent reporting a bare correlation
without one is reporting an unverified quantity.

**Rule 0 applies to OUR code too.** A probe that re-implements a code path is
not evidence about that code path — trace the dispatch chain.
