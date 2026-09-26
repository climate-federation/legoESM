# NEMO testcase L2 GYRE round 150 — card-owned live LDF geometry

Date: 2026-09-22

Status: **LANDED**.  The round-149 live-geometry landing no longer requires
bridge-carried F-point operands to construct the momentum-LDF geometry.  The
only recipe that resolves the complete executed condition (RK3-WS plus
`nemo_div_curl` plus `nemo_e3`) is GYRE-zco; it now builds T/U/V/F geometry
from its own SSH, reference thickness, masks, and grid.  The generic
`build_nemo_gyre_recipe()` card remains its declared vector-Laplacian,
e3-weighting-off configuration and no longer evaluates a dead exact-route
operand builder.  This corrects the four red generic-card tests without a
selector or configuration change.

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round150.md` at
`701069ee2`.  Candidate physics was measured clean at `5aff6eda5`; the generic
certification and recipe-derived census were measured at descendants containing
only gate, test, citation, and documentation changes.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round150/`.

## Compiled statement and landed repair

The record's compiled `dynldf_lev` multiplies curl by live F thickness,
divergence by live Kbb T/U/V thicknesses, and divides the final U/V increments
by live Kmm thicknesses at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:157-176`.
The F ratio is the explicitly parenthesised, area-weighted four-T-cell SSH
average at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:273-286`.
The configured `nn_e3f_typ=0` reference F thickness is the masked four-T-cell
quarter sum, with dry F points replaced by `e3f_3d`, at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynvor.f90:911-936`.

The shared transcription now accepts the card's own grid, reference T
thickness, and T mask and rebuilds the compiled F reference/live fields at
`vertical.py:378-473`.  It deliberately materialises the stored F-cell area
before the final division, preserving the compiled association under
production JIT.  The production route is selected only by the complete source
condition at `ocean_model_latlon_cgrid.py:5278-5280`; GYRE supplies its
step-entry tuple at `ocean_model_latlon_cgrid.py:5327-5362` and its stage tuple
at `ocean_model_latlon_cgrid.py:5862-5921`.  Recorded bridge fields remain an
oracle check in the developed-state walk, not a production dependency.  No
state field, stabiliser, carried state, or selector was added.

The operator note's premise that the generic recipe executes the same exact
statement was **REFUTED by the resolved card**: it is RK3-WS, but its lateral
operator is `vector_laplacian` and its e3 weighting is `off`.  Changing those
selectors would be a configuration decision; this round does not make one.

## Frozen predictions and falsifiers

| prediction | frozen falsifier | result |
|---|---|---|
| generic recipe completes and all 15 certified endpoints equal base `903625dd1` | any exception or endpoint move | CONFIRMED: PASS, 0 moved rows |
| GYRE ladder and month equal the round-149 after arm | any of 70 ladder rows or 30 daily snapshots moves | CONFIRMED after the JIT repair |
| day-180 F geometry and isolated LDF U/V remain BIT | any consumed cell differs | CONFIRMED |
| only GYRE-zco executes the complete route | any resolved in-scope recipe also resolves all three selectors | CONFIRMED |
| guard and entry plants fail | either remains green | CONFIRMED |

The first implementation at `ddb70da1a` **REFUTED** the stronger initial
bit-identity prediction: the ladder remained exact, but day 30 was
`6.888193516133112e-5 K`, rather than
`6.888193513796918e-5 K`, and the day-29/day-30 snapshots moved.  The card-owned
`area_q` multiplication had been fused into the division under production JIT.
The one-variable materialisation boundary at `5aff6eda5` restored all 30 daily
snapshots byte-for-byte.  The failed arm and score remain in
`candidate_month/` and `candidate_month_gap.json`; they were not promoted.

## Local developed-state proof

The admitted day-180 state and the recorded operands are unchanged from round
149.  Through the production JIT closure:

| row | unequal / cells | max abs |
|---|---:|---:|
| live F thickness | 0 / 16,530 | 0 |
| live U Kbb thickness | 0 / 17,400 | 0 |
| live V Kbb thickness | 0 / 17,100 | 0 |
| isolated LDF U | 0 / 17,400 | 0 |
| isolated LDF V | 0 / 17,100 | 0 |

The full post-LDF accumulators retain the already-registered upstream RHS
debt; the LDF terms themselves are BIT.  The production entry-SSH plant moves
49 consumed LDF-U cells (max `1.5881867761018131e-22`), prints
`PLANT-FIRED`, and exits 1.

## GYRE trajectory and year disposition

| headline | round-149 before | repaired route | movement |
|---|---:|---:|---:|
| kt2 T max abs | `1.4210854715202004e-14` | `1.4210854715202004e-14` | 0 |
| kt2 S max abs | `2.1316282072803006e-14` | `2.1316282072803006e-14` | 0 |
| kt2 U max abs | `2.7377110452773967e-12` | `2.7377110452773967e-12` | 0 |
| kt2 V max abs | `3.2849219221489645e-12` | `3.2849219221489645e-12` | 0 |
| kt3 T max abs | `8.659371033559182e-7` | `8.659371033559182e-7` | 0 |
| kt3 S max abs | `7.027288972949464e-8` | `7.027288972949464e-8` | 0 |
| day-30 T3D RMS | `6.888193513796918e-5 K` | `6.888193513796918e-5 K` | 0 |

The oracle-relative comparison reports PASS over all 70 rows, zero moved
rows, first-over-bar kt2 U/V on both arms, and no kt1 bar loss.  All 30 final
daily NPZ files are byte-identical to round 149.  Per the preregistered repair
protocol and the operator's explicit round-150 proof list, no new year run was
required after this zero-move month.  The inherited round-149 year values
remain day 240 `1.644671864406711e-2 K` and day 360
`1.1223450861560211e-2 K`; these are inherited, not newly measured this round.
This corrective landing preserves the already-admitted Decision-43/45 arm; it
is not a new magnitude candidate claiming an additional month improvement.

## Recipe-derived blast radius

| card | momentum integrator | operator | e3 weighting | executes exact route |
|---|---|---|---|---|
| GYRE-zco | rk3_ws | nemo_div_curl | nemo_e3 | yes |
| generic NEMO-GYRE | rk3_ws | vector_laplacian | off | no |
| LOCK_EXCHANGE-zco | rk3_ws | vector_laplacian | off | no |
| OVERFLOW-zps | rk3_ws | vector_laplacian | off | no |
| DINO nemo_dino_kamm | euler | nemo_div_curl | off | no |
| DINO nemo_dino_kamm_mlf | euler | nemo_div_curl | off | no |

The generic card's own three-step certified gate was run before at exactly
`903625dd1` and after at `ad7321f93`: both reports PASS, all certifications are
unchanged, and all 15 endpoint arrays have zero unequal cells.  Its four
starting red tests are included in the now-green full recipe test file.  DINO
does not execute the changed source condition, so no DINO trajectory row is
claimed.  ORCA2 is **UNMEASURED-WITH-SPEC** here: its separate ocean-only card
is not part of this Decision-43 census; when it selects the complete
RK3-WS/`nemo_div_curl`/`nemo_e3` route, its own-state geometry must pass the
ORCA2 10-step gate before certification.

## Controls, review, and tests

The generic production guard was removed in a planted working copy: the real
generic step then called the forbidden exact-route builder, raised the planted
exception, and pytest reported `1 failed in 3.54s`.  With the guard present,
the same test passes.  The own-mesh mask plant and bridge/own-input equivalence
are covered by the five-test QCO file, which reports exactly `5 passed in
55.26s`.  The final candidate-sensitive four-file gate reports exactly `61
passed in 28.17s` after the receipt and citation map were committed.

The required combined `tests/ocean/fidelity tests/ocean/unit -n 12` run did
not produce a trustworthy regression verdict: it exhausted memory, workers
crashed, and xdist ended with an internal assertion.  Its exact terminal
summary is `186 failed, 7614 passed, 153 skipped, 2 xfailed, 62 warnings, 38
errors in 1514.42s`.  A mechanical comparison extracts 186 failed IDs, of
which 174 are absent from the frozen 87-ID list; the retained traceback shows
`MemoryError` and a crashed worker, so those 174 are recorded in
`full_failed_ids_not_in_frozen_base.txt` but are not misreported as 174 code
regressions.  Every changed-path focused test is green, including the four
tests that were red at the starting tip.

The required read-only Codex review was attempted after the implementation.
Its verdict is unavailable; the log ends verbatim: `Error: failed to initialize
in-process app-server client: Read-only file system (os error 30)`.  This is
**independent review unavailable in-sandbox**, not approval and not a
DO-NOT-SHIP verdict.

The final citation gate reports PASS over 7 citations, zero unmapped, zero
failures, and an empty map audit.  Shifting the compiled LDF citation by two
lines reports FAIL and exits 1.  No NEMO build or run was requested, and no
NEMO source was modified.

## OPEN — round 151

Run the operator-ordered amplification test deferred by the mandatory repair:
three independent 360-day from-rest legoESM members whose only difference from
member 0 is the harness temperature-perturbation pattern at amplitudes
`1e-8`, `1e-6`, and `1e-4 K`.  Score each against member 0 at days
30/60/90/120/180/240/300/360.  Preregister the threshold verdict from note AP:
if either larger perturbation reaches at least 0.3 times the `1.64e-2 K`
day-240 gap while `1e-8 K` stays near the deterministic floor, report threshold
amplification; otherwise return to the developed-state tracer operator walk.
No physics, configuration change, or NEMO acquisition is needed.
