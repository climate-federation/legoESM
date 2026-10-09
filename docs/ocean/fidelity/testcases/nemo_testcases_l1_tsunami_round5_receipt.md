# RECEIPT — TSUNAMI lane, round 5 (build B4j: the j-periodic step)

Date 2026-10-08. Lane tip at start `4b991198b242`. Preregistered in
`PREREG_nemo_testcases_l1_tsunami_round5.md` (commit `e8009718c`), before any
round-5 code or measurement. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round5/`.

Status: **LANDED. TSUNAMI AT THE BAR for the key_RK3 deck** (DECISION 100):
the card's own 100-step chain stays within the trajectory gate's bar
(normalised max <= 1e-15) of NEMO's record on ssh, uu_b, vv_b, u and v at
every kt = 1..100. Worst row: ssh 7.7e-16 normalised at kt = 90. Not
bit-identical: every row has unequal cells at the 1e-16 relative floor.

## 1. What NEMO does at a periodic j-seam (read from the compiled source)

The reference run is one process (`ocean.output` lines 61-62,
`jpni = jpnj = 1`; `nn_comm = 1` selects the point-to-point exchange).
Then `l_SelfPerio(3:4) = l_Jperio .AND. jpnj == 1` (cited below) and every
self-periodic side is filled `jpfillperio`. The southern halo takes the
northern interior rows and the northern halo the southern ones, by a plain
assignment with no sign. The sign argument is used only by the north fold,
which runs only when `l_IdoNFold`, and that needs `l_NFold` (false on
TSUNAMI).

Per point type on a doubly periodic domain: T, U, V and F halo rows are all
copies of the opposite interior rows, same point type, sign +1, offset
exactly one period. In legoESM's (ny+1) v-face layout: south face row 0 =
north face row ny, the j copy of the u-face column 0 = column nx closure
the i-seam already uses.

Correction to the preregistration's line ranges (the statements are the
same): the halo source rows are lines 2026-2033 (prereg said 2026-2034) and
the copy loop is 2041-2046 (prereg said 2040-2045).

## 2. What was built (one shared implementation)

Searched first: the round-2 y-wrap (`halo_latlon.meridional_periodicity`),
every reader of it, and every j-neighbour statement the TSUNAMI step
executes (a line trace of one step, `trace_lines.pkl`). The existing wrap
already reached the lat-axis pad, the pole-row zeroing and the face-mask
builders. It did not reach the NEMO-literal statements: they build
north/south neighbours with hard zero (or copy) rows.

- **Three helpers in the y-wrap module**: north neighbour, south neighbour,
  and native-north-faces to (ny+1) faces. Under the flag each calls the
  existing lat-axis pad (wrap mode). Without it each returns the old
  expression verbatim. Under MPI/SPMD with the flag on they raise rather than
  wall the seam.
- **Routed through them** (every j-wall statement on the executed path, plus
  the two unexecuted siblings in the same functions): the barotropic surface
  pressure gradient, the EEN Coriolis neighbours and output faces, the
  southern F-point copy, both barotropic seeds, the qco F-mask and F
  thickness north neighbours, the live V thickness/ratio/reciprocal faces, the
  hpg V faces, and the F-point vertex map.
- **jit cache**: the hpg kernel is a module-level jit, so the y-wrap is now a
  static argument (a cached walled trace cannot be served to a periodic card).
- **Card**: `build_tsunami_zco_card` builds its face masks inside
  `meridional_periodicity(ln_Jperio)` (the seam V faces are wet, as NEMO's
  are), and `TSUNAMI_UNMEASURED` is now empty: B4 (both seams, measured on
  the record) and B4j are closed, so the execution gate admits the card.
- **No default changed.** The y-wrap stays off by default. Caller grep
  (`caller_grep.txt`): the only `j_periodic=` selection is the TSUNAMI card's;
  the only `meridional_periodicity(` callers are that card, the TSUNAMI
  ladder, and the hpg kernel's own static.

## 3. The A/B on the 100-step record (INDEPENDENT label)

OFF = round-4 commit `4b991198b242` (walled j-seam), run in a throwaway
worktree with the same harness. ON = `445e00145620` (B4j). Same harness, the
`record100` arm, kt = 1..100. Normalised max = max abs / reference max abs.

| field | first kt over bar, OFF | worst normalised, OFF | first kt over bar, ON | worst normalised, ON | kts over bar, ON |
|---|---|---|---|---|---|
| ssh | 15 | 9.8e-03 (kt 75) | none | 7.7e-16 (kt 90) | 0 |
| uu_b = u | 15 | 2.5e-03 (kt 42) | none | 3.5e-17 (kt 98) | 0 |
| vv_b = v | 15 | 3.2e-03 (kt 81) | none | 3.8e-17 (kt 98) | 0 |

Max abs / rms (cell of max as (j, i)):

| kt | arm | ssh | uu_b | vv_b |
|---|---|---|---|---|
| 15 | OFF | 2.7e-13 / 6.4e-15 (0, 39) | 6.7e-15 / 1.6e-16 | 2.1e-13 / 3.6e-15 |
| 15 | ON | 3.6e-16 / 2.5e-17 (90, 45) | 1.8e-17 / 2.8e-18 | 1.7e-17 / 2.8e-18 |
| 20 | OFF | 6.1e-05 / 2.1e-06 (200, 39) | 2.2e-06 / 7.1e-08 | 2.7e-05 / 7.7e-07 |
| 20 | ON | 3.4e-16 / 3.2e-17 | 2.3e-17 / 3.8e-18 | 2.2e-17 / 3.8e-18 |
| 50 | ON | 4.3e-16 / 4.2e-17 | 2.3e-17 / 4.9e-18 | 2.0e-17 / 4.9e-18 |
| 100 | OFF | 5.1e-03 / 1.7e-03 | 1.6e-03 / 4.6e-04 | 2.0e-03 / 5.4e-04 |
| 100 | ON | 5.5e-16 / 4.8e-17 (80, 38) | 2.9e-17 / 5.4e-18 | 2.7e-17 / 5.4e-18 |

**CONFIRMED**: the walled j-seam owned the round-4 record departure (OFF
reproduces round 4's kt = 15 row exactly; ON removes it). The ON maxima sit in
the interior, not on a seam row.

## 4. Certification set (stamped `424e59d1bfe6`)

| gate | result |
|---|---|
| record100, kt = 1..100 | no field over the bar; re-run after the last commit (`c0fdb9b55`, which only touches statements TSUNAMI does not execute or where f is constant) row-identical, 518 of 518 rows |
| kt = 1..10 ladder, INDEPENDENT | every row AT-BAR; ssh 2.5e-16 at kt = 10 |
| kt = 1..10 ladder, GIVEN-NEMO-ENTRY: whole step, stp_2D rhs, dyn_spg_ts substeps, handoff, stage-local | every row AT-BAR; rhs at the 1.09e-19 floor |
| geometry identity | `GEOMETRY IDENTICAL`, 35 rows EXACT (vmask included, with the seam faces now wet); plant `--plant "grid dx_u"`: `GEOMETRY DIFFERS`, exit 1 |
| j-seam translation equivariance (unit) | 0 unequal cells on eta, uu_b, vv_b (round 2's strict xfail now passes); plant (j-wrap off): fires |
| i<->j transposition, symmetric bump on the seam corner, f = 0 (unit) | eta = eta.T and u(j, i) = v(i, j), bit for bit; the walled step and the f-on step both break it (non-vacuity) |
| closed cards bit-identical | state sha256 after 3 steps equal at `4b991198b242` and `c0fdb9b55`: LOCK_EXCHANGE `b3aa27fa80dbdefd`, OVERFLOW `72ad6c2d6a9be145`, GYRE `b8cb5ecd4dc4b467`, VORTEX `b0ef8aaf1dc75f91`, VORTEX_VEC `ce76bd63f0441d3f`, VORTEX_SMT4_VEC `1cc2f09098310c7f` |

**Every kt = 1..10 row against round 4** (`r4_vs_r5_rows.txt`; independent,
given entry, rhs, substeps, stages; round 4 has no handoff file): 546 rows
compared, 61 moved, none changed status; max abs fell on 12, rose on 3, all by
about one ulp at interior cells: independent kt = 10 vv_b, v 1.65e-17 ->
1.68e-17 at (84, 36); stage 2 kt = 4 v 8.7e-19 -> 1.3e-18 at (76, 47). The
moves start at kt = 5 (given entry) and kt = 8 (independent). P5 predicted no
move at all: **REFUTED as worded**. The bump's tail is not zero at the seam
from kt = 1 (substep error rows on the j = 1 seam row at 1e-38 .. 1e-47 in
round 4 became 1e-306 .. 1e-313 here), so the wrap reaches the record before
the wave does. How a one-ulp change reaches an interior cell by kt = 4..10 is
**not attributed** (one PLAUSIBLE route: XLA fuses the wrapped pads
differently from the zero rows; not measured).

## 5. Predictions scored

| id | prediction | result |
|---|---|---|
| P1 | j-seam equivariance bit for bit; plant fires | **CONFIRMED** |
| P2 | transposition symmetric bit for bit | **CONFIRMED** with f = 0 (the prereg allowed that); f on breaks it, as predicted |
| P3 | ON at the bar through kt = 100; OFF reproduces round 4 | **CONFIRMED**: worst 7.7e-16; OFF kt = 15 ssh 2.7e-13 at (0, 39), identical to round 4 |
| P4 | closed cards unchanged | **CONFIRMED**: 6 of 6 state digests equal |
| P5 | kt = 1..10 rows unchanged to the bit | **REFUTED**: 61 of 546 rows moved by about one ulp, none changed status (section 4) |

## 6. ORCA2 pointer

What ORCA2 rung 0 shares with this program, now certified at the bar on
TSUNAMI over 100 steps (every shared statement drives the external mode
the whole run):

- **Split-explicit substeps**: the dyn_spg_ts loop with `nn_bt_flt = 1`
  boxcar weights, forward start, AB3 extrapolation, the surface pressure
  gradient, the face-depth update and the transport accumulation.
- **EEN in the barotropic Coriolis**: the literal coefficient build and the
  source-ordered quotient, in planetary-only form (f-plane; no relative
  vorticity because advection is off). ORCA2 adds relative vorticity and a
  non-uniform f; those are not certified here.
- **The cyclic i-exchange**: ORCA2's E-W cyclic U/V/T/F copy. The wave
  crosses the i-seam from kt = 5 and stays at the bar through kt = 100.
- **The j-exchange ORCA2 does NOT share.** ORCA2 is closed in the south and
  folds in the north (`l_NFold`, `lbc_nfd` with the sign argument). This
  round's j-wrap is the self-periodic copy, which ORCA2 never runs, and every
  rewritten statement is verbatim-unchanged with the flag off, so ORCA2 is
  unaffected by construction. Its southern `ff_f` copy (round 112's
  landing) is untouched off-flag.

ORCA2's held items: none is **refuted**. This **narrows** the rung-0
external-stage ssh debt. The loop's statements, run uncoupled from the 3-D
stages' baroclinic forcing and the coast, carry NEMO's state at the floor
for 100 steps. So ORCA2's per-column external-stage difference is PLAUSIBLY
in what ORCA2 feeds the loop or in what TSUNAMI does not exercise: masks and
coast (`rn_shlat`), the north fold, relative-vorticity EEN, the slow forcing.
That matches round 112's open list: southern `e3f_0vor` and mask cells,
northern-fold V. It does not certify any of those.

## 7. Review

[pending: single review (codex), run after this receipt is committed]

## 8. Choices made this round

| choice | ASKED or UNASKED |
|---|---|
| build B4j: the card's step honours `j_periodic` | ASKED (round brief, item 1) |
| the TSUNAMI card builds its masks inside the y-wrap (seam V faces wet) | ASKED (part of item 1; NEMO's vmask; geometry gate EXACT) |
| `TSUNAMI_UNMEASURED` emptied, so the execution gate admits the card | ASKED (item 3, conditional on the record at the floor, which holds) |
| the hpg kernel takes the y-wrap as a jit static | UNASKED (cache correctness; no behaviour change off-flag; offered for revert) |
| helpers raise under MPI/SPMD with the flag on, rather than wall | UNASKED (a refused combination TSUNAMI never runs; offered for revert) |
| two unexecuted siblings (reference-mesh seed, F-point vertex map) also routed | UNASKED (same statement, same function; verbatim off-flag; offered for revert) |
| closed-card digest probe kept as evidence, not committed | UNASKED (`r5_closed_card_digest.py`, sha256 `823cb1303855da0d`) |

## 9. Other gates

| gate | result |
|---|---|
| card tests | `20 passed`; with the prognostic-barotropic suite `28 passed` |
| ratchets (dispatch hardening, constants, inline coefficients, private imports) + barotropic and lat-lon halo suites | `3 failed, 5841 passed, 9 skipped`. All three reds fail identically at the round-4 commit: `test_jra55_do.py` literal 273.15, `land/restart.py` inline coefficient, `test_polar_filter_accepted` (checked in the worktree) |
| citation gate | see section 10 |
| NOT gated (honour system) | dual review: codex only (the headless brief names one review). Controlled comparison: the A/B differs only in the B4j commits (same harness file at both). Non-vacuity: every new test shown to fail with its feature removed (sections 4, 5) |

Evidence sha256 prefixes: record100 OFF `a68deba65a5d383d`, ON
`37102dd7ee242505`; ladder independent `48abc05ae9d527df`, given_entry
`3ce131d0ddcb5d69`, rhs `8ce9c8f2181d36f3`, spgts `ec377dcfa863bb5a`,
handoff `fe29176729b98d3a`, stages `d65948b4c4d3f7a7`, record100 (certified)
`7750cd2d49c64479`; geometry `ef0ce940518b08a2`; caller grep
`13c01f63af1ee99f`.

## 10. Citation gate

[filled after the gate run]

## 11. OPEN, for the next round

1. TSUNAMI is at the bar, not bit-identical: the 1e-16 relative floor (round
   4's unattributed 1.09e-19 stp_2D rhs floor, `stp2d.f90:137-138`, is the
   first unequal statement) is the only remaining gap.
2. The interior one-ulp moves at kt <= 10 under B4j (section 4) are
   unattributed.
3. The y-wrap has no MPI/SPMD band exchange (the helpers refuse it).
4. Shipped-deck (leapfrog, no key_RK3) TSUNAMI stays unscored (DECISION 100).

## Citations

The exchange: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/mppini.f90:432`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:1868`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2026-2033`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2041-2046`.
The fold that does not run: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2111-2112`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/mppini.f90:582`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:561`, and its per-point
sign cases `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:584`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:639`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:684`.
The remaining floor: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:137-138`.
