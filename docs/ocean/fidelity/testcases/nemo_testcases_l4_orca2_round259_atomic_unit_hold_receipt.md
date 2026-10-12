# ORCA2 round 259 — complete atomic fold unit growth hold

Date: 2026-10-12. Base: `588a24344`. Preregistration commit:
`f8b035109`. Measurement commit: `9d38cb48a`. Held-classifier commit:
`148f1699b`. Candidate-removal commit: `34500c897`. Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried-state
contract, stabiliser, sea-ice declaration, or `unmeasured_features` entry at
its tip. It measures the complete private fold/transport unit on OMT-4 under
CPU fp64/libm production JIT, separately labelled **independent** and **given
NEMO's entry**. No NEMO acquisition was needed.

## Compiled statements and candidate

NEMO forms `zub` and `zvb` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279`, then
writes the U/V stage transports from the live face thickness, velocity,
external correction and 3-D masks at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`. The
split-explicit solver associates velocity, depth, inverse depth and SSH at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739`; the
completed RK3 stage associates U, V, T and S at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`.

The measurement commit routes the stage transport's private V-mask operand
through the already-landed D114 resolved-mesh builder only while the complete
atomic test hook is enabled. It combines the round-217 parts, the raw
surface-V-mask/T-pivot pair, the resolved 3-D stage mask, and both held
external-mode association sites. It adds no observer. Traced and untraced
completed states are array-identical for every field at every completed step
under both labels. Because the unit fails the frozen terminal predicate, the
candidate is removed at the round tip; the final model file is byte-identical
to the base.

## Frozen predictions

R259-P1 is confirmed for the finite prefix: the committed tree, CPU fp64/libm
policy, production JIT, D114 route and passivity predicates all hold. R259-P2
is confirmed: at kt=1 stage 1 fold-band T and S are at the campaign floor
under both labels (maxima exactly 0.0; **given NEMO's entry** has 10,831
zero-magnitude T bit differences, while **independent** has none).

R259-P3 is **REFUTED**. The first post-floor greater-than-10x boundary is
kt=1 stage 2 in the fold band: the preceding maximum is
0.027227136128767482 and the next maximum is 6.817836451535307, a factor of
250.40593396570117. R259-P4 is **REFUTED**: both labels complete through kt=4,
then the independent kt=10 attempt refuses while entering kt=5 with
`raw-mesh e3w_int must contain only finite values > 0`. R259-P5 and P6 are not
reached. R259-P7 is confirmed by the eight planted refusals below.

## Passive finite-prefix table

The two labels give identical magnitudes at every listed boundary. `fold max`
and `interior max` are the maximum absolute candidate-minus-NEMO difference
over SSH, U, V, T and S in the named region. All entries are finite.

| kt | stage | fold max | interior max |
|---:|---:|---:|---:|
| 1 | entry | 0 | 0 |
| 1 | 1 | 2.7227136128767482e-2 | 7.105427357601002e-15 |
| 1 | 2 | 6.817836451535307 | 6.764903567064995e-5 |
| 1 | 3 | 13.350001820553338 | 5.6545982659971156e-2 |
| 2 | entry | 13.350001820553338 | 5.654598265997114e-2 |
| 2 | 1 | 18.747061566206472 | 1.1548679089011403e-1 |
| 2 | 2 | 13.375513334793201 | 1.7323018633517107e-1 |
| 2 | 3 | 24.713523840139267 | 3.464603726703421e-1 |
| 3 | entry | 24.713523840139267 | 3.464603726703421e-1 |
| 3 | 1 | 42.6128906108551 | 2.806345648253401e-1 |
| 3 | 2 | 113.95688023834467 | 2.884111085995371e-1 |
| 3 | 3 | 161.56246531047657 | 7.193514055138852e-1 |
| 4 | entry | 161.5624653104766 | 7.193514055138852e-1 |
| 4 | 1 | 399.50447583892776 | 7.145234911517435e-1 |
| 4 | 2 | 3342.291814954385 | 7.34752970768368e-1 |
| 4 | 3 | 78500.21113166603 | 1.0200702166710232 |

At the first surviving boundary, kt=1 stage 2, the fold-band field maxima are
SSH 0.0 m, U 0.0033804433633266525 m/s, V 0.029019283591256526 m/s,
T 0.36823246744026883 K, and S 6.817836451535307 PSU. The T and S argmax is
local `[j=2,i=49,k=0]`. By kt=4 stage 3, fold-band U/V reach
72020.75027297057/78500.21113166603 m/s and S reaches
36006.88870762031 PSU. The kt=5 thickness guard is therefore downstream
damage, not an owner statement.

This round names the first passive boundary, not a source statement. The next
walk must replay kt=1 stage 2 from the exact kt=1 stage-1 state in the compiled
stage order. No attribution is made to the guard or to a later damaged field.

## Controls, tests, and gate disposition

The held classifier status is `HELD_R259_KT1_STAGE2_GROWTH`. All eight
non-vacuity plants refuse: missing label, wrong route, observer movement,
kt=1 floor violation, moved growth boundary, missing finite-prefix boundary,
wrong refusal text, and one-bit blindness. Evidence SHA-256 values are:

- independent failed terminal attempt: `38088c5a1ee2e706a06c69b676238a812dde72bb06060cba16c7a9bede23992e`;
- independent kt=1..4 table: `f55f3bb5a839e7364dde68aeccaa8ecca0ca51c6025bff15296e0a8865f58705`;
- given-NEMO-entry kt=1..4 table: `af27c67ed16826799b988d3b14e72d27925004b9814d35e5c37e1e7c1c23f34a`;
- held classification: `20024fbdc8316bcd81170edb78b14fbe7ce768f26674fc4b4cdd0c60c2c487d6`;
- eight plants: `e12709b21fb5dd1e8824268358202a49feb9b5272485ff13a81836005fd55b20`.

The focused round-259 classifier set passes 11/11. The package candidate was
removed before final validation, so the round tip has no model diff and cannot
move OMT, rung-0, rung-10, GYRE, DINO or tank trajectories. Decision-96 and
the full landing gates are consequently not activated.

## OPEN

The complete atomic unit remains HELD and private. The next round starts at
the kt=1 stage-2 fold-band tracer magnitude boundary, from the exact stage-1
state, and replays the executing stage statements offline in compiled order.
It must name the first writer of the 6.817836451535307 PSU salinity error
before revisiting kt=5 or any later guard. D116 remains moot only if a future
complete unit qualifies under Decision 96; this round makes no decision or
configuration request.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
