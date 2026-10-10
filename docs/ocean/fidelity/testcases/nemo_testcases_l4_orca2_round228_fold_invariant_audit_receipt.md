# ORCA2 round 228 — fold-invariant audit

Date: 2026-10-10. Frozen base: `b6daf6603`. Preregistration commit:
`c3e779ea0`. Raw-scenario implementation commit: `d7a4942cc`. Final
classifier commit: `36692fdc9`. Status: **HELD**; no model statement, card
selector, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round228/`.

Every trajectory number below is labelled **independent** or **given NEMO's
entry**. The two labels were executed separately; equality between their
reported values is measured, not assumed.

## Source boundary and method

The active compiled NEMO external-mode statement associates `ua`, `va`,
`hu`, `hv`, `hur`, `hvr`, and `ssha` in one call, including each point type
and sign, at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756`. The
T-pivot V-row source and sign program is the active `CASE ('V')` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718`. After each
RK3 stage, the OMT build associates the live `u`, `v`, `T`, and `S` together
at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:784-796`.

The current compact-grid helper instead cyclically closes `depth_u` and
`inverse_u`, folds the three V arrays, and returns `eta` unchanged at
`barotropic_latlon_cgrid.py:689-743`. That inspection motivated the frozen
hypothesis; it is not itself trajectory attribution.

The committed audit executes rung 0 and OMT-4, each under the independent and
given-NEMO-entry labels, with the complete round-217 unit OFF and ON. It reads
only completed production stage states. For kt=1..7 the live trace first
proves every ordinary state slot bit-identical to the untraced run; kt=8
stages 1-2 are separately compiled completed-state results. The audit scores
only the northern fold band. `zFv` and `ww` are source-ordered reconstructions
from those admitted completed states, not recorded NEMO operands.

All eight scenario gates report `PASS_R228_SCENARIO`. The 56 traced kt=1..7
state comparisons (seven per scenario) are array-identical. No in-executable
observer, NEMO acquisition, or inferred cross-label state was used.

## Guard and frozen predictions

R228-P1 is **CONFIRMED**. Each card contributes 40 admitted NEMO stage frames
(kt=1..10, four frames per step). All SSH-derived live W thicknesses are
finite and positive. The minimum is 9.190740075483527 m on rung 0 and
9.190763602599942 m on OMT-4.

R228-P2 is **REFUTED** by its explicit falsifier. At kt=1 stage 1, NEMO has
zero T/U-point fold residual for `eta`, `r3t`, `e3t`, `e3w`, `T`, `S`, `u`,
`v`, `H_u`, and `r1_H_u`, but the unit-OFF candidate already violates each
relevant identity. The unit therefore does not create the preregistered
ON-only first-boundary violation. For `eta`, both OFF and ON have 35 unequal
pivot/right-half cells: the maximum residual falls from 0.1225196674 to
8.5084593e-05 m on rung 0 and from 0.1224643710 to 8.4328243e-05 m on OMT-4.

R228-P3 is **REFUTED**. There are zero instances of its required sequence
(`u`/`v`/`T`/`S` exact OFF, first non-exact ON, then the same-location
consumer). ON is already mixed at kt=1 stage 1 and every ON run becomes
non-finite in the audited band at kt=8 stage 1; NEMO remains finite through
kt=10. This is the registered blow-up, not proof that either missing
association statement owns it.

## First-boundary result

The following values are maximum absolute fold-band differences from NEMO at
kt=1 stage 1. `D` is ON minus OFF. Independent and given-entry runs produced
the same values but remain separate rows as Decision 52 requires.

| card | label | field | OFF | ON | D |
|---|---|---:|---:|---:|---:|
| rung 0 | independent | SSH (m) | 0.1211666133 | 0.0007343149 | -0.1204322983 |
| rung 0 | independent | T (K) | 0.0013726779 | 0.1774094359 | +0.1760367580 |
| rung 0 | independent | S (PSU) | 0.0011734531 | 3.2847473544 | +3.2835739013 |
| rung 0 | independent | u (m/s) | 0.0617152024 | 0.0012954967 | -0.0604197057 |
| rung 0 | given NEMO's entry | SSH (m) | 0.1211666133 | 0.0007343149 | -0.1204322983 |
| rung 0 | given NEMO's entry | T (K) | 0.0013726779 | 0.1774094359 | +0.1760367580 |
| rung 0 | given NEMO's entry | S (PSU) | 0.0011734531 | 3.2847473544 | +3.2835739013 |
| rung 0 | given NEMO's entry | u (m/s) | 0.0617152024 | 0.0012954967 | -0.0604197057 |
| OMT-4 | independent | SSH (m) | 0.1210007400 | 0.0005926746 | -0.1204080654 |
| OMT-4 | independent | T (K) | 0.0013606316 | 0.1773343083 | +0.1759736767 |
| OMT-4 | independent | S (PSU) | 0.0011712471 | 3.2833563447 | +3.2821850976 |
| OMT-4 | independent | u (m/s) | 0.0607454093 | 0.0002252537 | -0.0605201556 |
| OMT-4 | given NEMO's entry | SSH (m) | 0.1210007400 | 0.0005926746 | -0.1204080654 |
| OMT-4 | given NEMO's entry | T (K) | 0.0013606316 | 0.1773343083 | +0.1759736767 |
| OMT-4 | given NEMO's entry | S (PSU) | 0.0011712471 | 3.2833563447 | +3.2821850976 |
| OMT-4 | given NEMO's entry | u (m/s) | 0.0607454093 | 0.0002252537 | -0.0605201556 |

Across each label's 322 field-stage rows, rung 0 has 188 negative-D, 105
positive-D, nine zero-D, and 20 non-comparable non-finite rows. OMT-4 has
192 negative-D, 102 positive-D, eight zero-D, and 20 non-comparable rows.
Those counts describe cancellation, not a landing vote: the same unit sharply
improves external geometry/U while introducing the 3.28 PSU first-stage
salinity error.

The audit also falsifies its assumed exact identity for reconstructed V-point
diagnostics: NEMO itself has nonzero residuals in all 92 stage rows of each of
`H_v`, `r1_H_v`, `zFv`, and `ww`. Those rows are retained in the artifact but
are diagnostic-only. They cannot support an owner claim until the compact
V-point fold support is derived directly from the compiled two-row program.

R228-P4 is therefore **HELD / UNRESOLVED**. There are zero owner-signature
hits and no monotonic-owner series to classify. Decision 96 is NOT REACHED:
no model statement is proposed and no toward/away census is promoted as a
landing predicate.

R228-P5 is **CONFIRMED**. The guard, fold-sign/permutation, boundary-order,
monotonic-verdict, and scenario-coverage plants all fire. The preserved plant
log ends `STATUS PASS_R228_ALL_PLANTS_FIRED`.

## Artifacts

- `fold_audit.json`: `cc8f6096f623c10af1fc71e04fe9e77e5e264842fd27774b96d55d9c09c0ea38`
- `fold_audit.log`: `8c715da3a7c3a0d5c649e896a365189af5a35c3f69dad692b3d8c7b7c9fc051b`
- `plants.log`: `17e71214c95c3d4a2a942a513daf3beb6bc64e5868a586e53e19a56df096db9c`
- `SHA256SUMS`: `491f680bd3d2405dd82e16a0c7b55a3b5a533d253a670f50d00d78ea378ad0cc`
  (28 JSON/log entries, including all eight scenario JSON artifacts and
  logs). Each scenario JSON stamps measurement commit
  `d7a4942ccfc8613b1c828cef738ea414145d87f5` and a clean worktree.

## Validation and review

The final validation results are:

- focused audit, receipt-citation, and citation-gate tests pass 26/26
  (`focused_tests.log`, SHA256
  `477094a18eeb7bb2b62d98b5e1e15cf9ebc77e87abf296d2556efaa8a55e46ee`);
- the round receipt citation gate passes with four citations, zero failures,
  zero unmapped citations, and zero map-audit failures; the cumulative
  default receipt passes with 274 citations and the same zero counts;
- the rigid-shift plant exits nonzero and fails exactly the shifted stage
  association citation (`citation_plant.json`, SHA256
  `c53ee0a46216c7e064e335b2ab1447503783de5de9df757b2895f6745ba6e5e1`);
- the single prescribed `tests/ocean/fidelity -n 12` battery completes with
  3,106 passes, seven skips, and four registered pre-existing reds in
  2,439.84 s: the certified-year harness stamp, round-35 allow-dirty scope
  ratchet, report-emitter worktree-stamp ratchet, and SI3 scalar-math
  provenance gate (`battery.log`, SHA256
  `fd0a8ea4f01ac00647945af96088e4072a101ff16e639011c9ba8cfdbf3bf8dc`);
- the required `codex exec --sandbox read-only` review could not initialize
  its app-server client because the sandbox denied its PATH-alias write. The
  verdict is **independent review unavailable in-sandbox**, not PASS
  (`independent_review.log`, SHA256
  `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).

No `packages/` file changed, so GYRE, DINO, tanks, rung 0, rung 7/rung 10,
and OMT-4 cannot execute a changed model statement. Their certified
trajectories remain unchanged by construction.

## OPEN

Do not land the round-229 association proposed by the advisory: its frozen
owner signature is refuted. Before any package change, derive and gate the
compact V-point support for the two-row T-pivot program, then bisect the four
round-217 unit parts at kt=1 to separate the SSH/U improvement from the
first-stage T/S regression. The unit remains private, OMT-5 remains blocked,
and the live-W refusal remains registered debt.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.
