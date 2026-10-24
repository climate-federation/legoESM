# ORCA2 round 330 — full-depth TKE bottom boundary stays outside the solve

Date: 2026-10-24

Base: `355880d5b796`; preregistration `3c27c2f6f599`; implementation
`ddd2821b7194`; cross-tree D122 instrument `f1836e669e7d`; bottom-index census
instrument `e3fbb42b3e57`.

Disposition: **HELD-pending** (operator GPU acquisition, final gates and
review).

## Answer

**CONFIRMED owner.**  NEMO writes the bottom TKE boundary at `mbkt+1`, but a
full-depth column puts that value at `jpk`, outside the `2:jpkm1` tridiagonal
solve.  Rung 10's comparison card instead clipped that out-of-range index onto
its last carried solve row and replaced the row by an identity equation.
Keeping the scatter only when the raw index is inside the carried array removes every
deepest-interface returned-energy mismatch (4,757 -> 0) and every
deepest-interface viscosity mismatch (4,596 -> 0).

The review hypothesis that `en(jpk)` feeds the last solve row is therefore
**REFUTED**.  It is written, but the terminal solve assignment contains no
`en(jpk)` term.

## Source-order walk

**CONFIRMED, compiled record build.**  The TKE operands and all four local
matrix arrays extend through `jpk`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:222-240`).  The
surface boundary is an explicit held row
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:273-276`).  With
drag enabled, the bottom value is written to `en(mbkt+1)`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:287-295`).

The matrix and RHS are built only for `2:jpkm1`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-428`).  The
forward recurrence has the same bounds, the terminal row is assigned directly
as `zd_lw(jpkm1)/zdiag(jpkm1)`, and back substitution starts at `jpk-2`; the
post-solve mask is also limited to `2:jpkm1`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:459-477`).

Consequently:

- **Full depth:** `mbkt=jpkm1`, so the BC write is `en(jpk)`.  It is present in
  the array but is not read by the terminal row or back substitution.
- **Shallower:** `mbkt+1<=jpkm1`.  The dry boundary row has zero `tmask`, hence
  zero off-diagonals and unit diagonal; its held `en` value participates in
  back substitution before the final `wmask` zeroes that dry row.
- **Returned viscosity:** NEMO computes it only through `jpkm1`, from the
  solved `en` (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:709-711`).

**CONFIRMED, card.**  Rung 10's ORCA2 recipe selects the bottom boundary
(`nemo_testcase_recipe.py:2042`) and converts NEMO's one-based bottom T-cell
index to `bottom_level=bottom-1` (`nemo_testcase_recipe.py:1820-1827`).  The
card carries 29 zero-based solve rows.  The old scatter clipped every index
into `[0,N-1]`; the correction retains the clipped gather index but ANDs the
row selector with `0 <= raw_idx < N` (`tke.py:1803-1834`).  Thus a shallower
`bottom_level=28` remains an identity row while full-depth `bottom_level=29`
does not overwrite solved row 28.

**CONFIRMED, V1 untouched.**  Lane B's V1 deck turns drag off
(`nemo_testcase_v1_orca2tke/namelist_cfg:193`); its recipe consequently sets
`bottom_tke_bc=False` and includes that value in the V1 selector
(`gridladder_vmix/nemo_testcase_recipe.py:5083,5099`).  V1 never executes the
changed scatter, so its certified values are unchanged by construction.

## Record census and literal replay

**CONFIRMED.**  The admitted round-328 two-rank record reports `jpk=31`, 29
carried solve rows, 4,757 columns at `mbkt=30` (`mbkt+1=jpk`), and 2,760 at
`mbkt=29`.  Full-depth `bottom_level=29` is outside the card array; its deepest
returned-energy family is zero-based `k=28`.  Evidence:
`round330/replay/candidate_with_bottom_census.json`, SHA-256
`295cd3219d78a6a62d042fa9670a2778ae5b8d18816f7da1a669c581bc6be6f9`.

The frozen replay before -> after is:

| boundary | all unequal | `k=28` unequal |
|---|---:|---:|
| returned `en` | 5,661 -> 901 | 4,757 -> 0 |
| post-TKE `avm` | 5,530 -> 931 | 4,596 -> 0 |
| consumed final `avm` | 5,524 -> 925 | 4,596 -> 0 |

The consumed-viscosity maximum absolute error falls from
`1.2502789085852083e-2` to `4.805254826944355e-7`.  No level below 28 gains an
inequality; the last three earlier returned-energy counts are 7 -> 6, 11 -> 9,
and 6 -> 6.  Both replay admissions and all planted corruptions pass.  Baseline
SHA-256 `71298c7b9ab7e5fcbaa826c79523e96bf73c2c08d1668b1ff39e6b08c75a6293`;
candidate SHA-256
`54417d4329f908e65125789bb010524d22652e029e22fffa1f48ef729cf95d31`.

The focused old-clamp plant fails at the full-depth row because it restores
the clipped Dirichlet value `0.0007`; the restored candidate test class is
11/11 green.  This confirms the control is non-vacuous.

## Decision 122 / 139 — all seven pairs

**CONFIRMED.**  `decide7.json` is `PASS_R278_D122_DECISION`, SHA-256
`e92b47ef5482e56d0cf515844e5d688d0da25a33a96043254bea33528a6dc000`.

| card / label | exact rows lost | first over-bar | toward / away / equal moved rows |
|---|---:|---|---:|
| rung 0 independent | 0 | equal | 0 / 0 / 0 |
| OMT1--4 independent (each) | 0 | equal | 0 / 0 / 0 |
| rung 10 independent | 0 | equal | 183 / 0 / 0 |
| rung 10 given NEMO entry | 0 | equal | 183 / 0 / 0 |

The cross-tree wrapper admits exactly baseline `a48dd2956c616` and candidate
`ddd2821b7194`, normalizes only arm/commit admission metadata in deep copies,
and delegates all rows and Decision 139 later-first logic to the reviewed
classifiers.  Its exact-loss plant fires.

## Scope and gradients

**CONFIRMED.**  Normalized one-step JAXPRs for rung 0 and OMT0--4 are byte-for-
byte identical to the round-329 admitted programs; status-file SHA-256
`8a5a6381b4443f28ad44b65434a9ce966bba2a811bd7a4b235fd18bd522f20cb`.
The shared implementation change is active only on the comparison card's
selected bottom-TKE path among these cards.

**CONFIRMED.**  Rung-10 N=1 from the comparison card's own state has zero
non-finite state cotangents, zero non-finite cotangent leaves, and zero
forward-parity non-bit leaves.  Reverse/forward relative errors are
`2.22e-14` (T direction), `1.39e-15` (SSH direction), and `1.65e-15`
(`rn_ud` T; reverse `-4.511704487033998e-5`).  Peak RSS is 74.5 GiB.  Evidence
SHA-256 `88044d158531b30f3257c8dd48601c79cec166caa3bf4dc17f574543b2618e18`.

## DINO shared-branch effect

**CONFIRMED, selected.**  DINO's `nemo_dino_kamm_mlf` recipe selects the
bottom boundary (`dino.py:1287`).  The committed fp64 probe instantiates that
recipe's literal TKE solve on both DINO grids with inexact, nonuniform operands
(`nemo_testcase_l4_orca2_round330_dino_probe.py:88-170`).

- Default 35-level grid: 7,306 columns have `bottom_level == N == 34`.
  The normalized TKE program changes (SHA-256
  `d4827052b5761dac7317e429e461d20097b90a695935daf351703d7764b42724`
  -> `4d4712197d8fc0c0e422f5d6655f4ae1dde2477ce50664f5cc64c48e6ac7ce62`).
  Three chained solve values move on every step: array SHA-256
  `1ac5ab30...`, `8865506a...`, `a2d3f282...` -> `07286693...`,
  `d626e022...`, `bd2c1c16...`; the old pin keeps the maximum at
  `7.003482e-4`, while the solved maxima are `5.1976535e-5`,
  `2.5940056e-5`, and `1.6933280e-5`.
- NEMO-faithful 36-level grid: no wet column has an out-of-solve positive
  index; 428 dry columns carry `bottom_level=-1`.  The normalized TKE program
  changes, but all three chained arrays are bit-identical before/after (SHA-256
  `f0a999a2...`, `6eba3f56...`, `cd64a369...`).  The complete faithful-grid
  model program also changes (SHA-256 `bb0c501f...` -> `f2be22cb...`).  Its
  full three-step GPU values and the certified day-30 NEMO comparison are
  pending the operator acquisition below because this sandbox exposes no GPU.

Evidence is under `orca2_rounds/round330/dino/`.  The synchronous acquisition,
which refuses a busy GPU and stops its own exact PID below 150 GiB available
memory, is `orca2_rounds/round330/dino/run.sh`.

## Gates

- Preregistration frozen before payload inspection; P1--P4 all confirmed.
- Offline replay: `PASS_R328_DYNZDF_OFFLINE_REPLAY`; all admission plants fired.
- Decision 122/139: seven censuses and aggregate decision pass; exact-loss
  plant fired.
- Focused TKE bottom-boundary class: 11 passed; old-clamp plant failed.
- Required receipt citation gate: PASS, 8 citations, 0 unmapped, 0 failures,
  0 map-audit failures; its full pytest gate is 17 passed.
- Default ORCA2 landing battery: 147 passed in 952.64 s, including all 6
  private-import ratchet tests.
- Strengthened literal full-depth test: candidate 2 passed; old clamp 2 failed,
  including expected binary64 word `4535269685926082061` -> pinned
  `4559595583697369799`.
- DINO faithful-grid three-step and month gate: **PENDING ACQUISITION**.

independent review: operator (Claude)

## Choices and open debt

- Keep `en(jpk)` outside the carried solve instead of adding a synthetic row:
  **UNASKED**, selected because it is the literal compiled solve topology and
  preserves the existing 29-row certified V1 representation.
- No configuration value changes: **ASKED**.
- Do not extend or run the acquisition: **ASKED conditional**, because the
  admitted record and source-order replay settle the full-depth owner.

**PLAUSIBLE remaining debt.**  The 901 returned-energy inequalities, including
the `k=0` family and first unequal index `[1,119,19]`, are not owned by this
statement.  The top BC and the still-unrecorded `zdf_sh2`, `zdf_drg`, and
`zdf_mxl` inputs remain candidates for that residual only; this round neither
confirms nor excludes them.

**Named debt — legacy no-index path.**  When a caller supplies a bottom value
without per-column bottom indices, the old path still pins the last carried row
unconditionally (`tke.py:1812-1817`).  No current card takes that path; this
round records it and does not change it.

**Named debt — faithful-grid terminal coupling (PLAUSIBLE, pre-existing).**
The faithful-grid solve still carries the `jpkm1` row's upper coefficient into
the assembled recurrence (`tke.py:1664-1670`), coupling it to the carried
`jpk` row even though NEMO's terminal assignment never reads `en(jpk)`.  The
round-330 range fix neither introduces nor repairs that separate topology.
