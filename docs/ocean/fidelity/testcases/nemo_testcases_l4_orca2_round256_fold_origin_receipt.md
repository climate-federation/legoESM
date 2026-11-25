# ORCA2 round 256: restart the fold-growth walk at kt=1

Date: 2026-10-12. This is an ocean-only, offline OMT-4 measurement round.
The shipped ORCA2 card, SI3 selectors, and its `unmeasured_features` tuple are
unchanged. No file under `packages/` changes.

## Frozen question and evidence

The committed preregistration is
`PREREG_nemo_testcases_l4_orca2_round256.md`. Both **independent** and **given
NEMO's entry** runs use CPU, fp64/libm, production JIT, the production D114
resolved-mesh route, and the complete private fold/transport unit with both
held external-mode associations. Their completed stage states are
array-identical with and without the passive trace through kt=7. The two labels
agree in every value quoted below and remain separately stored.

The first foreground runner disappeared after kt=1 without a model refusal;
it produced no admissible result. The first complete probe then exposed a
reader defect: the admitted frame assembler returns a named mapping, not an
attribute-bearing state. That run completed all 28 boundaries before the
reader refused, and no number from it was used. The committed reader fix was
rerun for both labels. A second classifier fix correctly treats NEMO's
signed-zero-only entry U/V residuals as zero-valued fold identities. Both
failed attempts are retained in the evidence.

## Growth table before attribution

At kt=1 stage 1, before any source attribution, both labels give:

| field | fold-band max vs NEMO | fold-band RMS | candidate fold-identity unequal | NEMO fold-identity unequal |
|---|---:|---:|---:|---:|
| SSH | 0 m | 0 m | 0 | 0 |
| U | 3.469446951953614e-18 m/s | 2.91516208301639e-19 m/s | 1,680 signed-zero-only | 0 valued |
| V | 0.027227136128767482 m/s | 0.002202566269216334 m/s | 4,771 (3,452 signed-zero-only) | 0 valued |
| T | 0.17735225873138316 K | 0.005406472611654442 K | 683 | 0 |
| S | 3.283688696376231 PSU | 0.09830558057556885 PSU | 683 | 0 |

The required complete boundary table follows. Each maximum names the field
carrying it; RMS is that same field's RMS on the named region.

| kt | stage | fold max (field) | fold RMS | interior max (field) | interior RMS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 (S) | 0 | 0 (S) | 0 |
| 1 | 1 | 3.283688696e+00 (S) | 9.830558058e-02 | 7.105427358e-15 (S) | 5.577736344e-17 |
| 1 | 2 | 5.037097792e+00 (S) | 1.478561550e-01 | 6.764903567e-05 (V) | 2.559721162e-07 |
| 1 | 3 | 9.581947880e+00 (S) | 2.898412020e-01 | 5.654598266e-02 (U) | 1.968102270e-04 |
| 2 | 0 | 9.581947880e+00 (S) | 2.898412020e-01 | 5.654598266e-02 (U) | 1.968102270e-04 |
| 2 | 1 | 1.080395856e+01 (S) | 3.178084875e-01 | 8.074628296e-02 (SSH) | 2.763138209e-03 |
| 2 | 2 | 1.145513012e+01 (S) | 3.345091824e-01 | 1.211194244e-01 (SSH) | 4.144707313e-03 |
| 2 | 3 | 1.218304335e+01 (S) | 3.267097175e-01 | 2.839439133e-01 (V) | 7.062395513e-04 |
| 3 | 0 | 1.218304335e+01 (S) | 3.267097175e-01 | 2.839439133e-01 (V) | 7.062395513e-04 |
| 3 | 1 | 1.194707740e+01 (S) | 3.414358166e-01 | 2.806345648e-01 (V) | 9.543691458e-04 |
| 3 | 2 | 1.328758129e+01 (S) | 3.554489828e-01 | 2.884111086e-01 (V) | 9.661563648e-04 |
| 3 | 3 | 1.650110883e+01 (S) | 3.882817266e-01 | 7.193514055e-01 (V) | 1.406000568e-03 |
| 4 | 0 | 1.650110883e+01 (S) | 3.882817266e-01 | 7.193514055e-01 (V) | 1.406000568e-03 |
| 4 | 1 | 1.662122991e+01 (S) | 3.588272750e-01 | 7.145234912e-01 (V) | 1.494519766e-03 |
| 4 | 2 | 1.667485355e+01 (S) | 3.897898813e-01 | 7.347529708e-01 (V) | 1.512315063e-03 |
| 4 | 3 | 2.796230082e+01 (S) | 5.219800225e-01 | 1.020070217e+00 (V) | 1.863865399e-03 |
| 5 | 0 | 2.796230082e+01 (S) | 5.219800225e-01 | 1.020070217e+00 (V) | 1.863865399e-03 |
| 5 | 1 | 2.818819631e+01 (S) | 4.375310971e-01 | 1.011315246e+00 (V) | 1.795948668e-03 |
| 5 | 2 | 2.378549353e+01 (S) | 5.450097510e-01 | 1.037149088e+00 (V) | 1.835971045e-03 |
| 5 | 3 | 4.934013777e+01 (S) | 9.300257428e-01 | 9.621604301e-01 (V) | 1.824611171e-03 |
| 6 | 0 | 4.934013777e+01 (S) | 9.300257428e-01 | 9.621604301e-01 (V) | 1.824611171e-03 |
| 6 | 1 | 6.024034996e+01 (S) | 1.076280750e+00 | 9.451202438e-01 (V) | 2.016940635e-03 |
| 6 | 2 | 1.737321071e+02 (S) | 2.228228554e+00 | 9.624430974e-01 (V) | 2.050355263e-03 |
| 6 | 3 | 3.145399950e+02 (S) | 3.932255272e+00 | 1.111480837e+00 (V) | 2.155330550e-03 |
| 7 | 0 | 3.145399950e+02 (S) | 3.932255272e+00 | 1.111480837e+00 (V) | 2.155330550e-03 |
| 7 | 1 | 9.605002260e+03 (S) | 1.044886669e+02 | 1.083972944e+00 (V) | 6.895368502e-03 |
| 7 | 2 | 1.803235504e+06 (V) | 1.521326278e+04 | 1.096672661e+00 (V) | 6.870613870e-03 |
| 7 | 3 | 2.108577094e+14 (V) | 2.381810820e+12 | 3.542672751e+00 (V) | 8.375029761e-03 |

The first fold-band growth above 10x is therefore kt=1 stage 1: `S` rises
from zero to `3.283688696376231 PSU` (`1.6418443481881155e10` against the
`2e-10` floor). The first interior growth above 10x is kt=1 stage 2:
`V = 6.764903567064995e-05 m/s`. This supersedes the round-249 claim that V
alone was the earliest current-tree fold magnitude; T/S are already larger at
the same first stage.

## Source-ordered discrimination

NEMO applies the completed-stage U/V/T/S boundary association after the stage
updates at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`.
The compiled compact north-fold rules update the stored T pivot's right half
at `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:584-638`, the U
pivot at `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:639-683`, and
the reflected V row at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`. The replay uses those compact rules rather than
replacing the complete stored north row from its southern neighbour.

| field | raw fold-identity unequal | after compiled association | associated fold-band max vs NEMO |
|---|---:|---:|---:|
| T | 683 | 0 | 0.17735225873138316 K |
| S | 683 | 0 | 3.283688696376231 PSU |
| V | 4,771 | 0 | 3.469446951953614e-18 m/s |

Thus the missing post-stage association is source-exactly confirmed as the
first statement required to restore the T/S/V fold invariant. It fully closes
V to rounding in the fold band. It does **not** close the T/S magnitude: the
physical source half already differs, so this round does not claim that the
association owns the whole tracer error. The stage writer upstream of that
physical-half value remains open. U's only identity movement is 1,680 signed
zero bits and carries no finite-valued owner claim.

R256-P1 and P2 are confirmed. R256-P3 is confirmed for V and extends to real
T/S fold-identity violations. R256-P4 names the compiled post-stage
association as the missing invariant but explicitly withholds sole ownership
of the T/S magnitude. R256-P5 is honored: no statement lands. All eight
implemented plants fire: label coverage, production route, passivity,
boundary order, NEMO fold value, growth owner, association owner, and one-bit
detection.

## Gates, tests, and review

This round changes no model, card, recipe, deck, or configuration file. It
cannot move an OMT, rung-0, rung-10, GYRE, DINO, or tank trajectory, so their
landing gates are not activated. Focused round-249/252/253/254/255/256 tests
pass 70/70.

The once-only `tests/ocean/fidelity -n 12` battery collected 3,329 tests and
printed 3,297 passes plus the same four registered pre-existing failures as
round 255, then stopped returning worker output at 99% and was terminated. The
four failures reproduce serially with xdist disabled: SI3 scalar-math `MY_SRC`
provenance, the GYRE round-129 certified-year harness stamp, round-35
process-global allow-dirty scoping, and the report-emitter worktree-stamp
ratchet. The full battery was not rerun.

The receipt citation gate passes all four extracted citations with no failure
or unmapped entry. Shifting the compiled `stprk3_stg` citation by two lines
makes it fail exactly one source anchor. The cumulative default receipt also
passes 274 citations with no unmapped entry; the citation-gate unit suite
passes 17/17. The round, plant, cumulative, and citation-test SHA-256 values
are `ef52241cbd0f69b48b68028e69a36ca2229c0350fba87014dd95aae38a172aef`,
`d3c67f2b43f20ad7ce05a04feddf198965e70e17bdcf7af88e1abc82fab34ada`,
`73dacc574b2ef86f9c7de9f678737281473a26d6b57c396e84d229093cff2351`,
and `82d31e4cb045ddda725712add136a5df606b8bad2112c8a16e06a9ad71ec728a`.

The required `codex exec --sandbox read-only` review was attempted. Its
in-process app-server client could not initialize on the read-only filesystem,
so **independent review unavailable in-sandbox**. The retained attempt has
SHA-256
`8e3330eeb312ddb37b9da73c3ece4a8aa7e2ade4b7bb255573aa487bc89f0f58`.

The independent/given growth JSON SHA-256 values are
`41af05e21b5d5236e5c6875ccab3dc3e141042d145732e221a995f9894143004` /
`8cb66140f17adc9032b40e6ffa078290f4c31b531767cfac7a07b56a08ac49b5`;
the corresponding association replays are
`405c30adc14794d7f79364453e793caaba5965044f2da9630b4a7e2926270c07` /
`117afb624207e7700f93d51d5e4008a4a5eeee2343e8e1877e3fc24408f3d605`.
The classification and plant log are
`9bc51ed6a2e91a02f073310bf37d0e3cf55afa5434baabf29979f3adec23c777` /
`5fda737a01066348f84d53195367219bddb5b6249fc26e850dc8aef49bb010bf`.
The focused, full-battery, and isolated-red logs are
`a2fdb82feb3e596e912c19b5272370bf0d2b778097ad846b8772736f1116282b`,
`b7a8ddab028a82c7acb0e35b0b4c91d5e846a3e338321aa72b1808394147d322`,
and `8c44c01e15203d422a7486b8043ea972f62c48e894cbcdce78495a45eef78e33`.

## OPEN

Decision 116 remains pending and is not re-emitted. Once the operator unlocks
it, score the source-exact post-stage association atomically with the complete
private unit, but do not treat it as the tracer magnitude owner. Independently,
split the kt=1 stage-1 T/S physical-half writer in compiled stage order from
the exact kt=1 entry; the first statement that creates the `0.177352 K` /
`3.283689 PSU` source-half values owns that walk. Do not return to kt=8 until
the kt=1 origin is closed.

Round disposition: **HELD**.
