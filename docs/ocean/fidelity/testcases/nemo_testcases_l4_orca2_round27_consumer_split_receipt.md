# NEMO testcase Lane 4 — ORCA2 card round 27 consumer-split receipt

Date: 2026-09-26

Parent: `aaca19714d`

Status: **HELD — THE ROUND-26 F ARM WAS A TWO-CONSUMER EXPERIMENT.**
The direct discriminator refutes the proposed Kbb/Kmm tendency difference:
their effective wet-face operands and every measured lateral-diffusion
component are bit-identical.  At kt=1 lateral diffusion is exactly zero, but
the raw-F helper arm changes the independent stage-2 EEN vorticity component.
Round 26's “F-curl thickness alone” attribution is therefore withdrawn.  No
model statement lands; every experimental `packages/` change is reverted and
the final `packages/` tree is identical to the parent.

The six sea-ice selectors and the ORCA2 card's `unmeasured_features` tuple are
unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round27/`.

## Compiled statements and executed branch

The admitted build forms lateral diffusion's F curl at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125`,
its Kbb divergence at `:127-129`, and its Kmm divisors at `:132-140`.

The same compiled configuration selects the EEN vorticity consumer.  Its
reference F thickness is built by the selected `nn_e3f_typ` branch at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937`.
The EEN operator takes the reciprocal at `:734-738`, then forms transports and
adds the tendency at `:784-802`.  The admitted `ocean.output` has
`ln_dynvor_een = T` and `ln_dynvor_msk = F`; these are live statements, not a
dead arm.

legoESM calls `nemo_qco_live_vorticity_e3f_cgrid` for both the EEN operand and
the lateral-diffusion operand.  The round-26 raw-F edit changed this shared
helper, so source coverage alone could not assign its trajectory movement to
one consumer.

## Given NEMO's entry — direct lateral-diffusion result

This table uses NEMO's recorded kt=1 or kt=2 entry velocity and SSH.  It is not
an independent legoESM trajectory result.

| discriminator | cells | unequal | maximum absolute difference | verdict |
|---|---:|---:|---:|---|
| kt=2 raw U face thickness vs parent, all storage | 803,640 | 388,856 | 499.9785217898161 m | different |
| kt=2 raw V face thickness vs parent, all storage | 804,600 | 385,310 | 499.9785217898161 m | different |
| kt=2 raw U face thickness vs parent, executed wet faces | 803,640 | **0** | **0** | bit-identical |
| kt=2 raw V face thickness vs parent, executed wet faces | 804,600 | **0** | **0** | bit-identical |
| Kbb-only vs Kmm-only U tendency | 803,640 | **0** | **0** | bit-identical |
| Kbb-only vs Kmm-only V tendency | 804,600 | **0** | **0** | bit-identical |

Both experimental outputs also equal the parent bit-for-bit.  Their grad-div
and curl intermediates each equal the matching parent intermediate in U and V.
Thus the round-26 Kbb and Kmm trajectory summaries did not collide through
cancellation: the substituted values differ only outside executed wet faces,
so both direct physics changes are numerically vacuous on this entry.

The kt=1 recorded face velocity has maximum magnitude exactly zero.  Parent
and raw-F lateral-diffusion U/V arrays both have maximum magnitude exactly
zero, share digest
`1becd346e68494fb565e2a6f1f9381c8c789f1eb819f9bd7a3f998a7f40af3ba`,
and differ in **0 / 803,640 U** and **0 / 804,600 V** cells.

## Independent with Decision-52 SSH — production EEN result

This result starts from legoESM's independent initial T/S/u/v, first requires
those four arrays to equal the admitted kt=1 record bit-for-bit, and loads only
NEMO's recorded initial SSH as Decision 52 directs.  It is not mixed with the
direct-entry table above.

The clean parent's stage-2 vorticity digest is
`032cb7d192afb4a60ec5816ab78504ffb96faa5d19d4e83b61c17d1d462247b2`;
the raw-F helper arm's is
`0e355bcc1a0f3134e27b6a9cff428c93e04384740a164abacce8ab879ba56f4a`.
The arm changes **413,554 / 803,640 U cells**, maximum
`2.1873555668998308e-05 m s-2`, and **412,558 / 804,600 V cells**, maximum
`3.213274876559519e-05 m s-2`.

**RETRACTION:** round 26's claim that “F-curl thickness alone owns the kt=4
refusal” is withdrawn.  Its arm changed both the LDF F-curl operand and the EEN
potential-vorticity operand.  At the first movement, LDF is proven zero and
EEN is proven changed.  This round does not claim that EEN alone owns the
later kt=4 refusal; that requires a consumer-local trajectory arm.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R27-P1 | **REFUTED** | Kbb/Kmm U and V tendencies, their effective wet-face operands, and all measured direct components are bit-identical. |
| R27-P2 | **CONFIRMED** | kt=1 recorded velocity and both parent/raw-F LDF arrays are exactly zero. |
| R27-P3 | **CONFIRMED** | The production stage-2 EEN component moves while the direct LDF zero control remains exact. |
| R27-P4 | **CONFIRMED** | The one-value plant exits 1, and the unit plant refuses either missing source consumer. |

No failed prediction was rewritten.

## Gate, review, and tests

The round-27 outcome gate exits 2 with `HELD`.  Parent and raw-F NPZ digests
are respectively
`3de5dcb29bd4edc45252b2112280d3cd85c5705385e08fd29f2069c77aab50df`
and
`4fbef2e05adb0e2cfb3866466bb8a870385fc424ba1a78768dc48938394cbaac`.
Its one-representable-value kt=1 LDF plant exits 1 with
`planted kt=1 lateral-diffusion violation passed`; the source-inventory plant
is covered by the focused unit test.

REVIEW_RESULT_PLACEHOLDER

CITATION_RESULT_PLACEHOLDER

TEST_RESULT_PLACEHOLDER

No GYRE/DINO/lock-exchange/overflow landing gate is claimed: no model
statement lands, and `git diff aaca19714d -- packages` is empty at the final
tip.

## Choices

ASKED: Decision 54 and round 26's OPEN items authorize this consumer
discriminator and direct tendency comparison.

UNASKED: none.  No configuration value, default, carried state, stabilizer,
scoring rule, sea-ice selector, or NEMO source changed.

## OPEN

1. Decision 54 remains held.  Add a consumer-local raw-F operand seam, first
   for EEN with LDF held at the parent and then for LDF with EEN held at the
   parent.  Re-run the kt=1..10 ladder to assign the later kt=4 refusal without
   another two-consumer arm.
2. Treat round 26's Kbb/Kmm trajectory movements as instrumentation/graph
   perturbations, not NEMO-physics attribution; their direct executed operands
   and tendencies are exact on the admitted entry.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The inherited duplicate citation-map literal keys remain open.
