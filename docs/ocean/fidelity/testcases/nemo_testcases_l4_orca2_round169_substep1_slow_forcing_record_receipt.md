# ORCA2 round 169 — kt=8 substep-1 update exoneration and slow-forcing record request

Date: 2026-10-07. Base `e421a39e6`; preregistration `1d2344b9b`;
measurement instrument `1eec7846a` with face-registry correction
`b14255e3e`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round169/`.
Verdict: **STOPPED_FOR_RECORD**. The substep-1 vector update and its boundary
exchange are bit-exact given NEMO's recorded operands. The upstream slow
forcing is already finite but explosive, reaching
`1.5216884364408584e51 m s^-2`; the admitted record contains only its final
two-dimensional value, so the owning producer statement cannot yet be split.
No physics or configuration changed.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into this result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Compiled source order

The admitted `ORCA2_OMIP_L4_R166SPG8` executable adds explicit bottom drag at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:680-702`, evaluates the
live vector-form update at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:715-728`, constructs exit
depths at `ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:761-767`, and
associates U/V, both depths, both reciprocals and SSH at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:770-779`. It then carries
the associated velocity into the next external substep at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:836-844`.

Upstream, the same compiled program makes the slow forcing by vertically
reducing the three-dimensional momentum RHS at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:207-218`, then applies drag
and wind in order at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:231-247`; atmospheric
pressure, embedded-ice load and wave-load branches follow at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:254-285`. The new acquisition
records the operands and output at each live boundary without changing their
evaluation order.

## Source-ordered measurement

The round-166 record re-admits with both rank slabs, exactly-once coverage and
20 terminal restarts byte-identical to its additions-only baseline. Under the
unchanged complete private halo arm, independent kt=1..7 completes and kt=8
substep 1 gives:

| source-order row | active unequal cells | candidate non-finite | maximum absolute difference |
|---|---:|---:|---:|
| entry U | 15,789 | 0 | 2.983022220969999 m s^-1 |
| pressure U | 15,789 | 0 | 1.3368383028225998e-3 m s^-2 |
| Coriolis U | 15,787 | 0 | 2.2773689171841686e-4 m s^-2 |
| drag U | 15,789 | 0 | 8.92619341428281e-6 m s^-2 |
| combined trend U | 15,789 | 0 | 2.2773689975288586e-4 m s^-2 |
| slow forcing U | 15,789 | 0 | 1.5216884364408584e51 m s^-2 |
| complete increment U | 15,789 | 0 | 2.528343863624811e53 m s^-1 |
| pre-exchange U | 15,721 | 0 | 7.828465166305654e41 m s^-1 |
| post-exchange U | 15,789 | 0 | 2.528343863624811e53 m s^-1 |

R169-P1 is **CONFIRMED**: entry U is already non-bit. Replacing every update
operand by its recorded NEMO value reproduces the pre-exchange U on all
active cells (0 unequal); the remaining 118 full-domain differences are
outside the scored interior. R169-P2 is therefore **CONFIRMED**: NEMO's
literal ordered update arithmetic is exonerated.

Replaying the recorded association reproduces post-exchange U on every active
cell (0 unequal); 64 unscored full-domain halo values differ. R169-P3 is
**CONFIRMED**: the exchange itself is exonerated for the live active/boundary
domain. The first upstream recorded boundary with explosive magnitude is now
the slow forcing, not the update or exchange. Because the current record does
not contain its three-dimensional reduction, drag and wind operands at kt=8,
this round does not name an owning arithmetic statement.

## Face-average residual

R169-P4 is **REFUTED and retained**. The round-168 1/41 residual is native
U face `[j=76,i=179]`. The pre- and post-association after-SSH operands are
array-identical for this replay, and both leave that face unequal by
0.14741707684196548 m. Thus the residual is not a pre/post-association timing
error. It remains a separate finite face-average debt; it is not the source
of the kt=8 slow-forcing explosion.

R169-P5 is **CONFIRMED**. This is measurement-only: no `packages/` file,
recipe field, card selector, carried-state policy, stabiliser or sea-ice
feature changed. The complete raw-depth/no-extra-V-mask/seven-array/
materialised-`zhV` arm remains private and HELD.

## Acquisition

The committed launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round169_slow8_acquisition/run.sh`.
It creates the new `ORCA2_OMIP_L4_R169SLOW8` target and writes one
self-describing kt=8 record per rank. Its 23 fields cover the exact compiled
depth-reduction, drag and wind boundaries for both components, including the
three-dimensional RHS, compiled face thickness, masks, inverse depths,
surface stress and final slow forcing. The checker parses each field header
and payload length, requires exactly-once global rank coverage and compares
all 20 terminal restarts byte-for-byte with round 166.

The first preflight correctly refused because a whole-array reference to
NEMO's function-like `e3u_0/e3v_0` aliases does not compile. Commit
`421aca17f` corrected the recorder to the actual compiled operands
`e3u_3d/e3v_3d`. The corrected clean-tree preflight reports
`SYNTAX_PROOF_PASS` and `ORCA2_ROUND169_SLOW8_PREFLIGHT_READY`. The
additions-only layout plant exits 69 with `STATUS PLANT-FIRED layout`. The
launcher pins the deck, inputs, source build and all committed acquisition
artifacts by content hash; it never pins the moving producer commit.

## Controls, review and validation

The source-order, all-recorded-update, entry-replay, exchange-replay and
face-registry plants each refuse. The first entry-replay plant revision only
decremented a 15,721-cell census and was therefore vacuous; commit
`15e35e763` replaces it with an empty-coverage violation, which refuses with
`entry-U one-variable replay coverage is empty`. Failed predictions remain
in the report.

Focused coverage passes 25/25. The round receipt's eight citations pass with
zero failures or unmapped spans; shifting the vertical-reduction span by two
lines makes the citation plant fail. The cumulative receipt gate also passes
274 citations with zero failures and zero unmapped spans.

The one required `tests/ocean/fidelity -n 12` invocation collected 2,726
tests and reached 97%, with four failure markers and seven skips, but emitted
no terminal summary and no pytest process remained to poll. It was not run a
second time. The four-marker count equals the registered pre-existing count
from round 168, while the unavailable terminal node IDs mean this round does
not claim an exact ID match. The focused round-169 and citation tests had
already passed independently.

The separate `codex exec --sandbox read-only` review attempt returned
**independent review unavailable in-sandbox** before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.

ASKED choices: continue the compiled-source independent rung-0 walk. UNASKED
choices: empty. No configuration decision is requested.

## OPEN

1. The operator runs the round-169 slow-forcing launcher. Admit the two
   self-describing rank records and the 20 byte-identical terminal restarts.
2. At kt=8, replay the vertical reduction, drag and wind boundaries in the
   compiled `stp2d` order, one recorded operand at a time. The first
   finite-to-explosive statement owns the walk.
3. If that owner is a cited NEMO statement, re-test the complete private halo
   unit atomically under Decision 96 and every shared gate. Otherwise retain
   the unit and the face-average residual as separate HELD debt.
