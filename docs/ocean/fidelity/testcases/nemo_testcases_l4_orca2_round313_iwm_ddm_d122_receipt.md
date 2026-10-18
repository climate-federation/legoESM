# ORCA2 round 313 — V4's internal-wave mixing and V3's double diffusion under Decision 122

Date: 2026-10-18 (Claude round; codex review: paused (quota)).

Status: **LANDED-pending** (operator review + push gate).

## Headline

- Rung 10 (the only ORCA2 card running internal-wave mixing) now runs V4's
  NEMO-literal internal-wave mixing with `ln_tsdiff = .true.` and V3's
  literal double diffusion, as ORCA2's deck sets them.  Both pass D122 on
  both labels: 0 exact rows lost, first over-bar row unchanged.
- Both lean toward NEMO.  Rung 10 kt-10 stage-3 RMS (independent),
  control -> shipped: T 4.230229e-3 -> 4.027470e-3 K, S 2.388732e-3 ->
  2.369889e-3, u 3.568795e-3 -> 3.558958e-3 m/s, v 3.234058e-3 ->
  3.229765e-3 m/s; census 174 toward / 9 away of 183 moved rows.
- One code fix was needed (ours, not imported): NEMO gives salt its own
  matrix only with double diffusion on, and then adds the isoneutral term to
  salt as to heat.  Our model refused that combination.
- Seven-pair decision `PASS_R278_D122_DECISION`; rung 0 and OMT-1..4 move
  0 rows; GYRE, DINO and VORTEX x4 step jaxprs identical.
- The three ORCA2-lane tests red since round 308: the expectations were
  stale, not the code (re-pinned).

## Provenance

Base `bc12548d7` (round 312).  Preregistration `b4dbe8853` (frozen before
code and runs).  Import `718300c8a`; stale-test re-pins `c31e1a19f` (arms
ran here first and refused); salt-matrix fix `224fa9688` (arms); selection
`a2daf2f0f`; tests `543545299` (seven pairs ran here); statement test
`881be2d14`; receipt and map after.  Gate:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round313_iwm_ddm_d122_gate.py`
(round-285 ladder, round-276 classifier, round-278 decision).  CPU, fp64,
libm, production JIT.  Evidence `orca2_rounds/round313/` (`arms/`,
`arms2/`, `final/`, `jaxpr*/`, `config_*.json`, `subcensus.txt`).

## Survey

- Internal-wave mixing: only rung 10 runs it, on the generic form with
  `tsdiff` off.  The deck sets `ln_zdfiwm = .true.`
  (`ORCA2_OMIP_L4/EXP00/namelist_cfg:396`) and `ln_tsdiff = .true.`
  (`ORCA2_OMIP_L4/EXP00/namelist_cfg:424`).  Rung 0 and OMT-1..4 run
  constant mixing with it off.
- Double diffusion: off on all six cards.  Rung 10 refused it while
  internal-wave mixing was on: "zdf_iwm's avs update is missing".  Checked
  in code: TRUE — the generic internal-wave path adds only to the heat and
  momentum diffusivities, never to salt's.  The `DDM` arm below reproduces
  the refusal.
- Settings, rung 10 vs the deck (differing rows only):

| setting | deck | rung 10 before | rung 10 after |
|---|---|---|---|
| `ln_zdfddm` (`ORCA2_OMIP_L4/EXP00/namelist_cfg:393`) | T | F | T |
| `ln_tsdiff` (`ORCA2_OMIP_L4/EXP00/namelist_cfg:424`) | T | F | T |

  Agreeing (no row): `ln_zdfiwm` T; `ln_mevar` F
  (`ORCA2_OMIP_L4/EXP00/namelist_cfg:423`); `rn_avts` 1e-4, `rn_hsbfr` 1.6
  (`ORCA2_OMIP_L4/EXP00/namelist_cfg:394-395`, already the card's values,
  now stated on the card); the six maps from `zdfiwm_forcing_orca2.nc`
  (`cn_dir = './'`, the `sn_*` lines); V4's forced TKE floors (rn_emin
  1e-10, rmxl_min 1e-3, avmb 1.4e-6, avtb 1e-10), which rung 10 already
  carried.  The statement form (generic -> literal) is not a namelist row.
- V3's two fixes: the divisor guard lives in the double-diffusion module,
  already byte-identical to the vmix tip here (round 309); rung 10 already
  reads EVD's step-entry trigger (round 309).

## Code imported, not re-implemented

gridladder-vmix `1fc44b9b98` (V4 landed): `internal_wave_mixing.py`,
`k_profiles.py`, `tke.py` copied byte-identical (here they equalled the vmix
base `0b9443cef2`); its four model-file hunks applied cleanly; its literal
statement test imported (5 passed).  The V4 ladder card is not imported.

## Our fix (commit `224fa9688`)

- NEMO builds salt's own matrix only when double diffusion is on
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:175`, in `tra_zdf_imp` at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:134`).  Without it, the
  internal-wave avs is computed but never read.  Our model fed it to salt
  anyway; it now drops it unless double diffusion is on.
- With an isoneutral operator NEMO adds the same term to salt as to heat:
  ORCA2 runs the stabilizing-correction form
  (`ORCA2_OMIP_L4/EXP00/namelist_cfg:302`), heat `avt + akz`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:185`), salt `avs + akz`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:203`).  Our model refused; it
  now adds the term heat already gets.
- Not ladder-certified: no V or G rung runs isoneutral mixing.  Measured on
  rung 10 only (below).  Inert on V3/V4 (no isoneutral term, double
  diffusion on) and on every card without the literal forms.
- Evidence it acts: at the import tree the `IWM_LIT`, `IWM` and `BOTH` arms
  refused on the isoneutral combination (`arms/`); after the fix they run,
  and `IWM_LIT` = `IWM` row for row on both labels (tsdiff does nothing
  without double diffusion, as line 175 says).

## D122 rows on rung 10 (tree `224fa9688`, vs the round-312 card)

| arm | indep moved | indep census | given moved | given census |
|---|---:|---|---:|---|
| `IWM_LIT` (literal form, tsdiff off) | 183 | 173/10/0 | 183 | 172/11/0 |
| `IWM` (literal + tsdiff) | 183 | 173/10/0 | 183 | 172/11/0 |
| `DDM` (on the generic-IWM card) | refused | — | refused | — |
| `BOTH` (deck) | 183 | 174/9/0 | 183 | 173/10/0 |
| `IWM` -> `BOTH` (double diffusion's own increment) | 183 | 163/20/0 | 183 | 161/22/0 |

Every running arm: 0 exact rows lost; first over-bar row equal (kt-1 entry
ssh independent, kt-1 stage-1 T given entry); kt-10 ssh max error
0.277218 -> 0.277208 m (indep), 0.261699 -> 0.261690 m (given).

## Seven pairs (tree `543545299`, control = this selection stripped)

| pair | rows moved | toward/away/equal | exact rows (control = shipped) | first over-bar | max kt-10 ssh err (m) |
|---|---|---|---:|---|---|
| rung 0 | 0/200 | 0/0/0 | 5 | kt-1 stage-1 T, equal | 0.394172999 unchanged |
| OMT-1 | 0/160 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.746990074 unchanged |
| OMT-2 | 0/200 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.348811723 unchanged |
| OMT-3 | 0/200 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.363340965 unchanged |
| OMT-4 | 0/200 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.394004281 unchanged |
| rung 10 indep | 183/200 | 174/9/0 | 4 | kt-1 entry ssh, equal | 0.277217979 -> 0.277208045 |
| rung 10 given | 183/200 | 173/10/0 | 5 | kt-1 stage-1 T, equal | 0.261699458 -> 0.261689511 |

Decision: `PASS_R278_D122_DECISION` (`decide7_D122.json`).

Instrument checks (whole ladders): import-tree controls (`c31e1a19f`) =
round 312's shipped rows 7/7, and rung 10's at `224fa9688` too; same-tree
controls = round 312's shipped rows 7/7; shipped rung 10 = the `BOTH` arm,
both labels.

## Resolved configs (tip vs round 312, 572 fields; rung 10 635)

Rung 10: double diffusion off -> on, its form hybrid -> literal; internal-wave
form generic -> literal, tsdiff off -> on; the two salt/heat labels leave the
unmeasured list.  Rung 0, OMT-0..4: only the new form field appears, at
`generic` (rung 0 states the previous forms back).

## GYRE, DINO, VORTEX

Step jaxprs at `224fa9688` and at `543545299` equal round 312's: GYRE
(127,783 lines), DINO (123,235), VORTEX zco/zps flux and vector (55,510 /
47,484 each).

## Predictions

| ID | Verdict | Result |
|---|---|---|
| R313-P1 | **CONFIRMED** | import-tree controls = round 312's shipped rows, 7/7. |
| R313-P2 | **CONFIRMED** | `DDM` refuses at trace, the predicted reason. |
| R313-P3 | **REFUTED, then CONFIRMED after the fix** | at the import tree every literal arm refused (isoneutral salt term not built); after `224fa9688` all run and move 183 rows. |
| R313-P4 | **CONFIRMED** | `IWM` and `BOTH` pass (c)+(d) on both labels; both lean toward. |
| R313-P5 | **CONFIRMED** | constant-mixing cards 0 rows moved, configs unchanged but the new field; GYRE/DINO/VORTEX identical. |

## The three red tests (since round 308)

All three expected the uncarried after-sea-level form.  Round 308 moved
the ORCA2 card to the carried slot under Decision 132 + addendum 104 (G6
certified), measured under D122.  So the expectation was wrong, not the
code; re-pinned, no card setting changed:
- `test_orca2_card_states_the_after_ssh_form_and_the_cfl_cap` (two pins);
- `test_orca2_card_selects_resolved_rk3_sh2` — the same pin, plus a stale
  Langmuir plant that set the value the card already carries since round
  312 (a no-op plant); it now plants the old vectorized form;
- `test_decision_78_census_switches_only_gyre_and_vortex_vector` — ORCA2
  now executes the carried route.

The round-312 gate's docstring no longer says the terminal-step arm is
refused without the raw-length arm.

## Gates and tests

- New `tests/ocean/fidelity/test_orca2_round313_selection.py` (8): with the
  card selection reverted, 2/8 fail; tree restored and clean.
- Updated: the round-1 ladder gate's expected unmeasured list, the
  round-309 selection test (rung 10 now runs double diffusion), the rung-0
  card test, the IWM-maps test.
- Neighbouring battery (selection rounds 305-313, recipe, rung-0 card,
  defaults, constructibility, IWM maps, decision-43 gate, imported
  statement test), clean tree: 197 passed.
- Default gate (land.sh orca2 list, JAX_ENABLE_X64=1): citation gate +
  private-import ratchet + TKE + recipe + freshwater + barotropic state +
  ORCA2 parallel receipt 143 passed; face-mask rank 2 passed.
- This receipt's citation gate: PASS, 10 citations, 0 failures, 0 map
  entries failing (10 new entries; first run flagged an unpinned range
  extent, fixed).  76 pins of the model file and builder shifted by this
  round's edits were re-anchored (round-312 tool, mechanical offsets).
- No direct unit test of our salt-matrix fix; it is covered by the rung-10
  arms only (refused before, `IWM_LIT` = `IWM` after).
- The frozen phase-2 and round-66 gates still list the two labels; they
  read historical artifacts and no test runs them against the new card.
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Configuration choices

- ASKED (addendum 108): literal internal-wave mixing with `ln_tsdiff`, and
  literal double diffusion, on the cards that run them (rung 10).
- UNASKED, offered for revert: rung 0 states the internal-wave and
  double-diffusion forms back to their previous values (round 307's
  pattern); the card now states `rn_avts`/`rn_hsbfr` explicitly (same
  values).
- UNASKED, offered for revert: our salt-matrix fix (needed to run either
  feature on rung 10; transcribes the cited NEMO lines; not ladder-certified).
- Still off: `nn_mxlice`, G8's fold rows (D137), the G6 fold statements, G9.

## OPEN

1. The salt-side isoneutral term has no ladder rung; a rung with isoneutral
   mixing would certify it.
2. Unchanged: `nn_mxlice = 2` waits for the card's own ice thickness; D137
   fold rows off; G6 fold statements wait on G9.
