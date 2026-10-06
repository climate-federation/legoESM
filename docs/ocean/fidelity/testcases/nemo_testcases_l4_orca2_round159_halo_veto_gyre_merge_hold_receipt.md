# ORCA2 round 159 — halo veto and GYRE merge hold

Date: 2026-10-05. Base: `b7b1e747b4`. Preregistration:
`PREREG_nemo_testcases_l4_orca2_round159.md` at `f1d230ef2d`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round159/`.

Status: **HELD**. The exact external-mode halo association is source-exact but
fails its preregistered rung-0 salinity veto. Production halo physics is
unchanged. The subsequently required GYRE-lane merge is committed as
`5e8246929f9bdc97ca6dda6a5c3c165e367f5b1c`, but is not landing-qualified:
the merged EVD composition refuses the rung-0 constant-mixing card before its
first step. This receipt does not call that merge scientifically admitted.

## Claim labels and source statement

All rung-0 numbers below are **independent**. No given-NEMO-entry and
independent number is mixed in the table. The rung-7 ladder is unmeasured
after the rung-0 terminal veto.

The isolated candidate is only the complete seven-field external-mode
association. NEMO associates those fields together at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`, performs the
two-rank U west/east send at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979`, receives and
writes it at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2060-2068`, and
dispatches the north fold at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2105-2113`. The T-pivot
V overwrite and sign are the live no-gather statements at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1712-1738` and
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1747-1766`.

The raw-reference-depth, unmasked-V-transport, and V-materialisation arms
remain OFF. No configuration, initial state, carried state, stabiliser,
sea-ice selector, or `unmeasured_features` entry changed.

## Exact halo gate

`halo_exact.json` is PASS and has SHA-256
`7da3b3ee55fa3bc25f2f33a2659f85b1c2b9a422b67fe573dbee71eeeb26c2df`.

- 520/520 U two-rank comparisons are bit-exact; maximum absolute difference
  is 0.
- The completed U cyclic-then-fold and V cyclic-then-fold results are exact.
- All seven ordinary fields (`T`, `S`, `u`, `v`, `ssh`, `uu_b`, `vv_b`) are
  passive and bit-exact.
- All seven independent plants fire: U source, U pivot sign, V source, V
  permutation, V sign, composition, and selector.

This confirms R159-P1. It does not make the association trajectory-faithful.

## Independent rung-0 candidate

Both the production control and isolated candidate complete 200 rows. The
candidate moves 195/200 rows, loses no bit-identical row, and leaves the first
non-bit boundary at kt=1 stage 1 T. The decisive kt=10 stage-3 table is:

| field | control rms | candidate rms | control max | candidate max |
|---|---:|---:|---:|---:|
| T | 5.963034121233243e-3 | 5.962454410432421e-3 | 8.633902735293519e-1 | 8.633902735163224e-1 |
| S | 1.608894028553539e-3 | 1.651366362185173e-3 | 4.156673855360964e-1 | 4.156724015708448e-1 |
| u | 4.343087152949231e-3 | 4.681891943839918e-3 | 3.640084123187020e-1 | 3.640080332952717e-1 |
| v | 4.356804437803066e-3 | 4.548857024617737e-3 | 5.144146002383860e-1 | 5.144146002832977e-1 |
| ssh | 2.802652392934881e-2 | 3.113461855567756e-2 | 4.283251766668493e-1 | 1.066570750321891e0 |

The S maximum increases by `5.0160347484e-6`, so R159-P3 is **REFUTED** by
its frozen falsifier. R159-P2 is also REFUTED because both ladders were
required to pass and rung 7 was correctly not run past the terminal rung-0
veto. R159-P4/P5 are UNMEASURED. The two ladder artifacts have SHA-256
`f7a471adaf0445ac7703321e6e2f6d019adfeb4ace1771c3925fc2a4fde5eea6`
and `b1ebc1ab8011364ef10f104fdfdffc6fb9393940d3de52202d6c39e6ddb39cc4`.

## Required GYRE merge and conflicts

After the second halo round held, the official local GYRE lane tip
`33c754c716ccf3fc6c50692ca9c30041ff7a7a64` (round 236) was merged with
merge-base `a3be519e0`. There were three conflicted files:

1. The cumulative GYRE receipt had ten stale line-anchor conflicts. The ORCA
   conflict side was retained at each marker, all non-conflicting GYRE prose
   was retained, and the complete receipt was mechanically re-anchored.
2. The citation gate had the corresponding generated-map conflicts. The map
   is the union; all non-conflicting entries from both parents remain, then
   every changed package citation was mechanically re-anchored.
3. The ocean model had four textual conflicts in three combined hunks. The
   `tracer_zdf_trace` and `tracer_ldf_diagnostics` validations, private
   `_step_impl` arguments, and separate dispatch returns were all retained.
   A new mutual exclusion refuses asking for both diagnostic traces in one
   call. With neither hook requested, the production dispatch is unchanged.

The focused EVD and diagnostic-hook tests pass 10/10. The default citation
gate passes 274 citations with zero failures, zero unmapped citations, and
zero map-audit failures; all built-in plants fire. Its artifact SHA-256 is
`21d5a807a4e61e4d6b530d5b5609474ef202c456990b162bd41ffb02838abcdc`.

## Merge gate refusal

The merged independent rung-0 gate refuses before trajectory integration:

> `EnhancedDiffusionConfig.evd_composition="nemo_replace" ... is only defined
> when ... vertical_mixing.scheme='none'; got 'constant'.`

This is an integration gap, not a configuration decision. NEMO copies the
background closure coefficients before calling EVD at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:349-350` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359`; a fired tracer interface is
then assigned `rn_evd` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109`, while the momentum
replacement is separately gated by `nn_evdm` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121-135`. The rung-0 card's constant background plus
`nemo_replace` therefore has a source-defined composition, but the merged
fast path cannot yet represent it without losing which surfaced coefficient
came from which module. Weakening the guard or treating the summed coefficient
as pure EVD would be an unpreregistered physics change, so neither was done.

Because this gate is red, the rung-7, GYRE trajectory/year, DINO, VORTEX,
tank, generic-card, and full ocean-fidelity battery were not claimed for the
merge. The required separate review was unavailable in-sandbox with the exact
failure `failed to initialize in-process app-server client: Read-only file
system (os error 30)`.

ASKED choices: none. UNASKED choices: empty. ACQUISITION_NEEDED: none.

## OPEN — round 160

1. Preregister the one source-defined EVD composition repair for a constant
   vertical-mixing background. Preserve the fail-closed distinction between a
   surfaced closure coefficient and a surfaced EVD coefficient; do not add a
   selector or infer a configuration.
2. Re-run rung 0 first. If it passes, measure rung 7, GYRE's certified ladder
   and year, DINO, VORTEX, tanks, generic cards, the full ocean-fidelity
   battery, and every merge push gate. Register every moved row.
3. Only when the merged tree passes those gates may the merge be called
   landed. The halo association remains HELD and must not be folded into the
   EVD repair.
