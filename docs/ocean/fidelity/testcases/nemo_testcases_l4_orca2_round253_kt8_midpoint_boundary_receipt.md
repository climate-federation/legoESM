# ORCA2 round 253: kt=8 midpoint-SSH failure boundary

Date: 2026-10-12. This is an ocean-only OMT-4 measurement round. The shipped
ORCA2 card, SI3, and its `unmeasured_features` tuple are unchanged. No file
under `packages/` changes in this round.

## Frozen question and claim labels

The committed preregistration is
`PREREG_nemo_testcases_l4_orca2_round253.md`. It asks for the first writer
that turns the complete private fold/transport unit's finite kt=8 entry into
the registered live-W-thickness refusal. Every result below was measured
separately as **independent** (OMT-4's own corrected initial state) and
**given NEMO's entry**. The two reports are identical in every number quoted
below; they were never pooled.

The complete private unit is the round-217 fold/transport unit, the raw V
surface mask and T-pivot neighbour association, the resolved three-dimensional
V mask, and both held external-mode associations. The probe uses the current
production D114 mesh route. It introduces no executable observer: at kt=7 the
existing completed-stage observer reproduced the ordinary state bit-for-bit
for every state leaf under both labels.

## NEMO source order

In the compiled OMT-4 source, NEMO first forms the half-step SSH as
`zsshp2_e = za1*sshn_e + za2*sshb_e + za3*sshbb_e` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:489-493`.
`ts_bck_interp` supplies the startup and AB3-AM4 coefficients in the same
compiled file at `:1496-1531`. The downstream continuity write is
`:546-558`, and the external velocity/depth update follows at `:655-718`.
The external result is installed into the RK3 stage
at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178`.

legoESM evaluates the corresponding literal midpoint interpolation at
`barotropic_latlon_cgrid.py:2006-2025`. This round therefore identifies the
first invalid writer, not yet the first non-bit operand: the OMT-4 oracle
record does not contain the kt=8 substep histories needed for that split.

## Guard control and completed-stage growth

The exact production live-W formula was replayed without a floor: raw
`e3w_0[...,1:] * (1 + ssh/ht0)`. All 40 admitted NEMO states (kt=1..10,
entry plus three stages) pass it; the smallest NEMO value is
`9.190763245379527 m`. The candidate kt=8 entry also passes under both labels:
zero non-finite/non-positive values and minimum `9.37621660239688 m`.
Therefore the runtime guard is exonerated and the failure is created after
the finite entry.

The passive boundary table around kt=7 is:

| boundary | ssh max (m) | u max (m/s) | v max (m/s) | T max (K) | S max (PSU) |
|---|---:|---:|---:|---:|---:|
| kt=7 entry | 0.5830026628364016 | 6.049826683839001 | 6.492610591046075 | 17.018511853803368 | 314.5399949556666 |
| kt=7 stage 1 | 2.5098693564814485 | 25.66494858322057 | 27.548055335486712 | 519.4546880303545 | 9605.002260374864 |
| kt=7 stage 2 | 3.6552289428918647 | 1694043.145535235 | 1803235.5039632872 | 23448.04894381267 | 437529.38234285376 |
| kt=7 stage 3 | 7.091307702123113 | 1.4419824139093897e14 | 2.108577093668666e14 | 8.57148432307801e10 | 2.1524191895507012e12 |
| kt=8 entry | 7.091307702123102 | 1.4419824139093897e14 | 2.108577093668666e14 | 8.57148432307801e10 | 2.1524191895507012e12 |

The first registered greater-than-10 growth at these completed boundaries is
kt=7 stage 1 temperature: `17.018511853803368 -> 519.4546880303545 K`, a
factor `30.522920716729097`. This is upstream explosive state, not a
rounding-level disagreement.

## First invalid writer

The source-ordered external-substep table is decisive and identical under
both labels:

| substep / boundary | invalid live-W cells | minimum live W (m) |
|---|---:|---:|
| 1 entry | 0 | 9.37621660239688 |
| 1 midpoint | 0 | 9.371257223417912 |
| 1 continuity exit | 0 | 9.366375824714254 |
| 2 entry | 0 | 9.366375824714254 |
| 2 midpoint | 0 | 9.361503342315057 |
| 2 continuity exit | 0 | 0.0009091263330853932 |
| 3 entry | 0 | 0.0009091263330853932 |
| **3 midpoint** | **476,499** | **-392.56280049082903** |

The first invalid cell in frozen row-major order is `[j=1,i=49,k=0]`.
Immediately before the writer it is valid. At the midpoint writer, SSH is
`-1296.6280391709458 m` over `ht0=729 m`, so
`1 + ssh/ht0 = -0.7786392855568529` and live W thickness is
`-7.786665855170239 m`. The failure is on the southern closed-boundary row,
not the northern fold. SSH, U, and V preserve their configured fold identities
through substep 3, so no fold-identity violation precedes this boundary.

R253-P1 is **CONFIRMED**: both labels complete kt=1..7, enter kt=8 finite,
and reproduce the same refusal. R253-P2 is **REFUTED and retained**: the first
invalid writer is the substep-3 half-step SSH interpolation, not the
continuity write. R253-P3 and P4 are **CONFIRMED**. R253-P5 is not activated:
the interpolation itself is already the cited NEMO statement, and the absent
oracle histories prevent attributing which of `sshn_e`, `sshb_e`, `sshbb_e`,
or the coefficients first differs. No partial physics landing is made.

## Acquisition request

No admitted OMT-4 stream contains the kt=8 rank-complete external-substep
histories. Rung-0's round-166 stream cannot be substituted because OMT-4 has
already executed its additional modules for seven steps. The committed
launcher
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round253_spg8_acquisition/run.sh`
reuses the admitted round-166 self-describing writer binary with the admitted
round-222 OMT-4 deck. It runs a two-step smoke test first, then the ten-step
record, requires both rank files, parses the record headers and payload sizes,
requires 65 substeps on both ranks, and compares both kt=10 restarts byte for
byte with the admitted OMT-4 run. Seven record/restart plants and a producer-
content plant must fire. `--preflight-only` reports
`ORCA2_ROUND253_OMT4_SPG8_PREFLIGHT_READY`; the content plant fires.

## Gates, tests, and review

The round classifier is `HELD_R253_FIRST_INVALID_WRITER_NAMED`. All nine
measurement plants fire. Focused round-252/253 tests pass. The once-only
`tests/ocean/fidelity -n 12` battery reached 99% with 3,266 explicit passes,
the same four registered pre-existing reds as round 252, and no round-253
failure before an xdist worker stopped returning output. The two printed
in-flight tests were rerun serially; their disposition is recorded in the
evidence log. The full parallel battery was not rerun.

The required `codex exec --sandbox read-only` review was attempted. Its
in-process app-server client could not initialize on the read-only filesystem,
so **independent review unavailable in-sandbox**.

## OPEN

The operator runs the committed acquisition. The next round admits it and
splits substep 3's midpoint statement in compiled order: `sshn_e`, `sshb_e`,
`sshbb_e`, then `za1/za2/za3`. The first non-bit operand owns the upstream
walk. The complete fold/transport unit and both associations remain private;
no stabilizer, floor, clip, or configuration change is authorized.
