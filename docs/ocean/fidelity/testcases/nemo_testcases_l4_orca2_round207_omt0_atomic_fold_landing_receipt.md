# ORCA2 round 207 — OMT-0 atomic fold-unit landing

Date: 2026-10-09. Base: `29bd2d077`. Production commit: `4465ba94c`.
Post-landing replay commit: `47d475f85`. Status: **LANDED**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round207/`.
All OMT-0 numbers are separately labelled **independent OMT-0** or **given
NEMO's entry**; their registered ladder movements are identical. The shipped
rung-10 sea-ice selectors and `unmeasured_features` tuple did not change.

## Result

The exact four-statement unit qualified in round 206 is now the ordinary
production path for cards selecting NEMO's literal flux-form external mode:

- raw reference `hu_0/hv_0` face depths from
  `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:512-519`;
- the unmasked, separately materialised V transport product at
  `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:532,535`; and
- the seven-array post-update association, including the NEMO V sign, at
  `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:712-741` and
  `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:721-730`.

NEMO exposes no selector between these statements. legoESM consequently
selects them from the existing literal external-mode identity predicate; no
deck field, card selector, stabiliser, carried-state choice, or sea-ice field
was added. Cartesian tank cards lacking raw face arrays retain the already
certified compact min-neighbour construction; a half-present raw pair raises.

Ordinary production is row-for-row equal to round 206's frozen private arm
on all 200 **independent OMT-0** and all 200 **given NEMO's entry** rows. For
each label, 195 RMS-moved rows move toward NEMO, zero away, zero score-equal,
five unchanged, and no exact row is lost. The first debt moves toward. The
kt=1 stage-1 SSH RMS/maximum move
`0.006243153742634906 / 0.13136125371061705` to
`0.00023211739258069189 / 0.006278809746666877 m`; kt=10 stage-3 moves
`0.0288100436468264 / 0.5086605459790218` to
`0.0036726314904968077 / 0.07685333444272535 m`.

## DINO correction

Round 206's claim that its DINO processes ended before scoring was a process
observation error: Codex's foreground tool call ended while both integrations
continued. Both retained logs end in the same complete fixed-gate result,
`2.056821682e-03 K` against the `2.244317642e-03 K` bar — **PASS**. The
ordinary promoted-tree log has SHA-256
`4026dcc50417660a2251902e76f045ff74bdaa34f5fd62566bc2bd6e52c3b63e`;
the post-retraction retry has
`ba51562599d41738c5b4727a833f9f4948ee73c42506400456430e61f7293c7d`
and the same snapshot digest and score. The round-206 receipt now carries a
loud correction; its original false conclusion remains visible as history.

## Shared gates

- GYRE ten-step: fresh 70-row oracle-relative comparison PASS, zero maximum
  worsening ULP, first over-bar unchanged at kt=3.
- GYRE year: a fresh 360-day member completed. All twelve 30-day snapshots
  are byte-identical to round 206's certified member. Day-30/day-240/day-360
  SHA-256 are `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`,
  `2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe`,
  and `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646`.
  The certified day-30/day-240/day-360 T RMS therefore remains
  `2.3432419318363155e-06`, `6.5816987106668941e-05`, and
  `5.4077212586815052e-05 K`.
- Independent hierarchy rung 0: fresh 200-row ladder, 0 moved, 0 exact lost;
  first non-bit remains kt=1 stage-1 T.
- Shipped rung 10, **given NEMO's entry**: fresh 200-row ladder, 0 moved,
  0 exact lost; first non-bit remains kt=1 stage-1 T.
- LOCK_EXCHANGE and OVERFLOW: `git diff 68919282db -- packages` is empty, so
  the admitted round-206 results apply to the identical executable package
  tree: LOCK has zero worsening ULP; OVERFLOW has maximum 1.5 ULP, within the
  fixed two-ULP ratchet. A fresh invocation refused before physics because
  these legacy roots' kt=2 entry hashes predate the wrapper's scalar-math-v2
  certification. That refusal is retained in `lock.log`; no control was
  waived and no record was modified.

## Post-landing OMT-0 walk

The passive completed-state replay reproduces all terminal values and keeps
the first over-floor value at substep 2's V metric transport: 1,714 wet cells,
RMS `3.089909796130036e-11`, maximum `9.313225746154785e-10` in `zhV`.
Its source statement is the product cited above at `dynspg_ts.f90:532,535`.

The source-ordered operand split gives:

| operand | wet unequal | wet max | disposition |
|---|---:|---:|---|
| `e1v` | 0 | 0 | dry/halo differences only; substituting it is product-null |
| `va_e` | 2,134 | `8.673617379884035e-19 m s-1` | first effective non-bit operand; adding it closes `zhV` exactly |
| `zhvp2_e` | 0 | 0 | bit-exact |

Thus the next source statement is the forward/AB3 midpoint-V producer at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:461-484`. At OMT-0
substep 2 the compiled branch has `za1=1`, `za2=za3=0`, then evaluates
`va_e = za1*vn_e + za2*vb_e + za3*vbb_e`. This round does not land a second
physics change. The all-recorded-product plant flips the closing predicate
and refuses.

## Preregistered predictions

| ID | disposition |
|---|---|
| R207-P1 | **CONFIRMED**: production and frozen private candidate agree on 400/400 rows. |
| R207-P2 | **CONFIRMED**: 195 toward, zero away/equal, no exact loss, frozen SSH headlines reproduced for both labels. |
| R207-P3 | **CONFIRMED** by fresh GYRE/ORCA2 gates and identical-package transfer of the admitted tank results; the fresh legacy-root wrapper refusal is explicitly retained. |
| R207-P4 | **CONFIRMED**: both DINO logs contain the completed score and frozen hashes. |
| R207-P5 | **CONFIRMED**: substep-2 V transport remains first over-floor; the effective operand is `va_e`. |
| R207-P6 | **CONFIRMED**: census, production/private, citation-shift and V-product replay plants refuse. |

## Validation and review

Focused validation and the full ocean-fidelity battery are recorded below in
the final validation commit. Independent read-only review is also recorded
there; an unavailable sandbox review is not represented as a PASS.

## OPEN

1. Preregister and split the substep-2 midpoint-V statement's three history
   operands and written association offline against the admitted ordered
   record. The production product closes only after `va_e`; do not revisit
   `e1v` or `zhvp2_e` without contrary measurement.
2. When OMT-0's split-explicit solve reaches the bar or a complete cancelling
   unit, start OMT-1 (+ linear implicit bottom drag). OMT-0 has no month by
   Decision 103 because NEMO itself reaches its registered stability boundary
   at kt=11.

No NEMO acquisition is needed. No configuration or sea-ice decision is
pending.
