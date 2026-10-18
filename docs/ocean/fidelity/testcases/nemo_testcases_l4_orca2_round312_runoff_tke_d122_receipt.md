# ORCA2 round 312 — surface-only runoff (D126) and V1's TKE forms (D135) under Decision 122

Date: 2026-10-18 (Claude round; codex review: paused (quota)).

Status: **LANDED-pending** (operator review + push gate).

## Headline

- D126: the shipped card (rung 10) now puts runoff at the surface only, as
  NEMO's ORCA2 deck.  It was the only ORCA2 card spreading runoff (150 m,
  inherited from the generic recipe default); rung 0 and OMT-0..4 already
  resolved 0.
- D135: rung 10 now runs V1's penetration below the mixed layer (with its
  mask, native htau and libm EXP), the literal raw mixing length with the
  dummy-bottom step, and the literal Langmuir source.  Every group passed
  D122 (c)+(d) on both labels.  The under-ice mixing length (`nn_mxlice = 2`)
  stays off: it needs an ice thickness the card does not compute itself.
- Size: the penetration is the large term.  kt-10 stage-3 RMS error on
  rung 10 (independent), control -> shipped: T 9.218320e-3 -> 4.230229e-3 K,
  S 2.686689e-3 -> 2.388732e-3, u 4.729218e-3 -> 3.568795e-3 m/s,
  v 4.461012e-3 -> 3.234058e-3 m/s, ssh 2.905153e-2 -> 2.903973e-2 m.
  Census 183 toward / 0 away of 183 moved rows, both labels.
- Seven-pair decision: `PASS_R278_D122_DECISION`.  GYRE, DINO, VORTEX x4
  step jaxprs identical.

## Provenance

- Base `f075a5cd5` (round 311).  Preregistration `459f20e54` (frozen before
  code and runs).  Import tree `ea782817a` (builder inputs + gate; arms ran
  here).  Selection `c04c693ca`.  Tests + same-tree control `1f0eb1661`
  (seven pairs ran here).
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round312_runoff_tke_d122_gate.py`
  (round-285 ladder, round-276 classifier, round-278 decision).  CPU, fp64,
  libm, production JIT.  Evidence `orca2_rounds/round312/` (`arms/`, `sub/`,
  `final/`, `final2/`, `jaxpr/`, `config_*.json`, `subcensus.txt`).
- Code imported, not re-implemented: the TKE forms are gridladder-vmix
  `0b9443cef2`'s (`tke.py` byte-identical to its tip); the values are V1's
  card (`_orca2v1_tke_vertical`).  The two inputs they read and rung 10's
  builder lacked — native gphit and the domcfg's level-jpk thickness — are
  loaded the same way V1's builder does.

## NEMO statements

Runoff.  `ORCA2_OMIP_L4/EXP00/namelist_ref:474` and `:478` set
`ln_rnf_depth = ln_rnf_depth_ini = .false.`; `:480`'s `rn_dep_max = 150` is
read only when one is true.  So sbc_rnf_init
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:289`) sets `nk_rnf = 1` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:486-489`, and sbc_rnf_div
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:238`) takes the surface-only branch
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:279-282`.

TKE.  The deck sets `ORCA2_OMIP_L4/EXP00/namelist_cfg:406` (`nn_mxl = 3`),
`:410` (`nn_etau = 1`) and `:415` (`nn_mxlice = 2`); the rest from
`ORCA2_OMIP_L4/EXP00/namelist_ref:1236` (`ln_mxl0`), `:1243` (`rn_mxl0`),
`:1245-1246` (`ln_lc`, `rn_lc`), `:1252` (`nn_htau = 1`).  The card
already carried `nn_mxl = 3`, `ln_mxl0`, `rn_mxl0 = 0.04`, `ln_lc` and
`rn_lc = 0.15`; what was missing were the forms below.  In tke_tke
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:199`) and tke_avn (`:531`):

| group | statement | line |
|---|---|---|
| ETAU | `nn_etau = 1` increment `* wmask * tmask(1)`, its EXP | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:498-500` |
| ETAU | `nn_htau = 1` htau from native gphit | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:864` |
| MXL | literal raw length `MAX(rmxl_min, SQRT(2*en/zrn2))` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:644-645` |
| MXL | dummy-bottom `e3t(jpk)` step in the bottom-up scan | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:690-692` |
| LC | Langmuir depth `imlc = mbkt+1` when no level crosses | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:354` |

(a) V1 certification: gridladder-vmix receipts rounds 3 (Langmuir), 5
(mask, raw length, terminal step), 7-8 (htau, EXP), addendum 101.  (b) V1's
round-7 receipt found every cited routine byte-identical between the V1 build
and `ORCA2_OMIP_L4`; the lines above were read in `ORCA2_OMIP_L4` this round.

## D122 rows on rung 10 (import tree, vs the round-311 card)

Census = row RMS toward / away / equal over the moved rows.  0 exact rows
lost and first over-bar row unchanged in every arm, both labels (first
over-bar: kt-1 entry ssh independent, kt-1 stage-1 T given entry).

| arm | indep moved | indep census | given moved | given census |
|---|---:|---|---:|---|
| RNF (D126) | 183 | 61/122/0 | 183 | 62/121/0 |
| ETAU_CORE (penetration + nn_htau=1) | 183 | 183/0/0 | 183 | 183/0/0 |
| ETAU (all five) | 183 | 183/0/0 | 183 | 183/0/0 |
| MXLRAW | 183 | 41/142/0 | 183 | 37/146/0 |
| MXLTERM | 179 | 89/90/0 | 179 | 88/91/0 |
| MXL (both) | 183 | 61/122/0 | 183 | 48/134/1 |
| LC | 183 | 170/13/0 | 183 | 170/13/0 |
| TKE (ETAU+MXL+LC) | 183 | 183/0/0 | 183 | 183/0/0 |
| ALL (TKE+RNF) | 183 | 183/0/0 | 183 | 183/0/0 |

Apart, each statement on top of its partner (`sub/`, `subcensus.txt`;
same rule, 0 exact rows lost each):

| step | indep moved, census | given moved, census |
|---|---|---|
| core -> + mask | 183, 108/75/0 | 183, 108/75/0 |
| core -> + native htau | 176, 101/64/11 | 178, 82/83/13 |
| core -> + libm EXP | 174, 83/83/8 | 179, 63/103/13 |
| core -> all five | 183, 108/75/0 | 183, 108/75/0 |
| raw length -> + terminal step | 183, 114/69/0 | 183, 116/67/0 |
| ETAU -> TKE | 183, 131/52/0 | 183, 131/52/0 |
| TKE -> ALL | 183, 135/48/0 | 183, 132/51/0 |

Registered, not a veto (D122): the two mixing-length statements and surface
runoff lean away on their own (5th-7th digit at kt 10); together with the
penetration everything leans toward.  kt-10 stage-3 RMS per arm is in
`subcensus.txt`.

## Seven pairs (tree `1f0eb1661`, control = this selection stripped)

| pair | rows moved | toward/away/equal | exact rows (control = shipped) | first over-bar | max kt-10 ssh err (m) |
|---|---|---|---:|---|---|
| rung 0 | 0/200 | 0/0/0 | 5 | kt-1 stage-1 T, equal | 0.394172999 unchanged |
| OMT-1 | 0/160 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.746990074 unchanged |
| OMT-2 | 0/200 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.348811723 unchanged |
| OMT-3 | 0/200 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.363340965 unchanged |
| OMT-4 | 0/200 | 0/0/0 | 4 | kt-1 stage-1 T, equal | 0.394004281 unchanged |
| rung 10 indep | 183/200 | 183/0/0 | 4 | kt-1 entry ssh, equal | 0.277237788 -> 0.277217979 |
| rung 10 given | 183/200 | 183/0/0 | 5 | kt-1 stage-1 T, equal | 0.261719286 -> 0.261699458 |

Decision: `PASS_R278_D122_DECISION` (`decide7_D122.json`).

Instrument checks (whole ladders compared): import-tree controls = round
311's shipped rows, 7/7; same-tree controls = import-tree controls, 7/7;
shipped rung 10 = the ALL arm, both labels; shipped rung 0 / OMT-1..4 =
their controls; the selection-tree D122 runs at `c04c693ca` and
`1f0eb1661` agree row for row.  My sub-census script reproduces the
round-276 classifier's numbers on every control-relative arm.

## Resolved configs (selection vs base, 572 fields; rung 10 635)

Rung 10, nine rows: runoff depth 150 -> 0; penetration none -> below the
mixed layer; htau constant 10 m -> latitude; mask off -> on; htau and EXP
evaluation vector -> scalar libm; raw length factored -> literal; terminal
step off -> on; Langmuir vectorised -> literal.  Rung 0, OMT-0..4: none.
All seven cards also gain the level-jpk thickness array (consumed only
under the terminal-step flag) and rung 10's grid the native gphit (only
under the literal htau).

## GYRE, DINO, VORTEX

Step jaxprs on the selection tree equal round 311's: DINO (123,235 lines),
GYRE (127,783), VORTEX zco/zps flux and vector (55,510 / 47,484 each).  The
diff touches only the ORCA2 builders and the ORCA2/GYRE Langmuir check
(GYRE's value unchanged).  GYRE ten-step/day-30 not rerun: jaxpr identical.

## Predictions

| ID | Verdict | Result |
|---|---|---|
| R312-P1 | **CONFIRMED** | import-tree controls = round 311's shipped rows, 7/7. |
| R312-P2 | **CONFIRMED** | RNF moves 183 rows, both labels; 0 lost; first over-bar equal (census leans away 61/122). |
| R312-P3 | **CONFIRMED** | ETAU 183/0/0 both labels, (c)+(d) pass. |
| R312-P4 | **CONFIRMED** | native htau and libm EXP differ from the core in 176/178 and 174/179 rows. |
| R312-P5 | **REFUTED (part)** | MXLTERM does NOT refuse: the guard asks for the raw e3t(jpk) operand, not the literal raw length — my misreading.  So the two are independent statements, each measured alone and together; MXLRAW and MXL pass. |
| R312-P6 | **CONFIRMED** | LC 170/13/0, passes. |
| R312-P7 | **CONFIRMED** | TKE and ALL pass; seven-pair decision passes. |
| R312-P8 | **CONFIRMED** | only rung 10's nine rows change; GYRE/DINO/VORTEX jaxprs identical. |

## Gates and tests

- New `tests/ocean/fidelity/test_orca2_round312_selection.py` (10): with the
  selection reverted 3/10 fail; with only rung 0's stated-off fields
  reverted 6/10 fail; tree restored, `git status` clean.
- Updated: round 307's selection test (the card now runs the penetration),
  one recipe unit test (its synthetic ORCA2 card now carries the literal
  Langmuir form the validator requires).
- Neighbouring ORCA2 selection (rounds 305-312), recipe, rung-0 card and
  card-default tests: see the battery line below.  One failure there is
  pre-existing: the after-SSH form test expects the pre-D132 value (round
  308 moved it; the field is not in this round's config diff).
- The frozen round-279 runoff audit pins the then-shipped 150 m and would
  now refuse; it is a historical gate, no test runs that audit.
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Configuration choices

- ASKED (D126): runoff depth 150 -> 0 on rung 10.
- ASKED (D135): the ETAU, MXL and LC groups on rung 10; `nn_mxlice` stays 0.
- UNASKED, offered for revert: rung 0 (and OMT-0..4 through it) states the
  eight TKE fields at their previous values, so constant-mixing cards'
  configs do not change (round 307's pattern).
- Required by the ask, stated: the ORCA2 builder loads native gphit and the
  level-jpk thickness for every card derived from it (inert unless selected).
- Not touched: G8 fold rows (D137 off), double diffusion (V4), G6 fold
  statements (G9).

## OPEN

1. `nn_mxlice = 2` waits until the card computes its own ice thickness.
2. Unchanged: D137 fold rows off; double diffusion (V4); G6 fold statements
   (G9's stage-1 V transport mask); internal-wave tsdiff gap (V4 import).
