# ORCA2 round 255: substep-2 SSH writer split

Date: 2026-10-12. This is an ocean-only, offline OMT-4 measurement round.
The shipped ORCA2 card, SI3, and its `unmeasured_features` tuple are unchanged.
No file under `packages/` changes in this round.

## Frozen question and admitted record

The committed preregistration is
`PREREG_nemo_testcases_l4_orca2_round255.md`. Every number below was measured
separately as **independent** (OMT-4's corrected initial state) and **given
NEMO's entry**. The two final reports agree in every quoted value; they were
not pooled.

The operator-run round-253 record remains admitted as
`PASS_R253_OMT4_SPG8_ADMISSION`. This round reads its rank-complete 65-substep
stream and round-254's passive candidate trace without changing either. The
final probe requires CPU, fp64/libm, production JIT, the exact clean tree, the
production D114 resolved-mesh route, both labels, and round 254's admitted
classification.

## Source order

The compiled OMT-4 oracle forms the transport divergence and writes

`ssha_e = (sshn_e - rDt_e * (ssh_frc + zhdiv)) * ssmask`

at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:550-558`.
It then associates all seven external fields at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739`
and rotates the SSH histories at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:781-793`.
legoESM's corresponding boundaries are
`barotropic_latlon_cgrid.py:2050-2096`,
`barotropic_latlon_cgrid.py:2336-2374`, and
`barotropic_latlon_cgrid.py:2403-2408`.

## Instrument calibration retained

The first probe attempted an in-executable observation and continued to the
already-known live-thickness refusal. It produced no admissible scientific
number. The first offline version then reconstructed face transports from
canonical state and missed the candidate divergence by 28 boundary cells;
those values are retracted. The final probe instead consumes the already-
passive named `continuity_du` and `continuity_dv` differences and proves that
they reproduce the candidate divergence and literal continuity output
bit-for-bit. It also exposes, rather than silently folding in, legoESM's
existing post-continuity clamp. These failed calibrations are retained in the
round evidence and are not used below.

## Boundary split

Both labels give the same source-ordered census:

| boundary against NEMO `j002_ssha_e` | unequal / 26,640 | maximum absolute difference | RMS difference | argmax `[j,i]` |
|---|---:|---:|---:|---:|
| literal continuity output | 16,433 | 3.3276385577734643e53 m | 2.1969002391044925e51 m | `[147,48]` |
| candidate post-clamp output | 16,433 | 1.0102432777429826e53 m | 6.189546913905138e50 m | `[147,48]` |

The candidate's existing clamp changes 16,433 cells between those two rows,
with maximum change `2.3173952800304815e53 m`. The seven-array association
then changes 0/26,640 cells, and history rotation changes 0/26,640 cells.
Thus R255-P2 is **CONFIRMED**: continuity is already non-bit; association and
rotation are exonerated. The clamp is a downstream candidate-only boundary,
not the first owner, and this measurement-only round neither changes nor
authorizes it.

## Continuity operand split

The literal candidate divergence replay, candidate output replay, and NEMO
output replay each reproduce their recorded target bit-for-bit. The statement
uses 65 external substeps and `rDt_e = 166.15384615384616 s` in both models.

| operand, in NEMO statement order | unequal / support | maximum absolute difference | RMS difference | argmax `[j,i]` |
|---|---:|---:|---:|---:|
| incoming `sshn_e` | 16,433 / 26,640 | 7.060633126433089 m | 0.09934199315608172 m | `[147,48]` |
| `zhU` | 15,853 / 26,640 | 4.052712074220707e60 | 3.111807255046282e58 | `[147,47]` |
| `zhV` | 15,943 / 26,640 | 4.71247947797403e60 | 3.6692726864399364e58 | `[147,48]` |
| `r1_e1e2t` | 0 / 26,640 | 0 | 0 | none |
| reconstructed `zhdiv` | 16,433 / 26,640 | 2.002745428289585e51 | 1.3222084772388146e49 | `[147,48]` |
| `ssh_frc` | 26,640 / 26,640 | 0 | 0 | signed-zero-only |
| `rDt_e` | 0 / 1 | 0 | 0 | none |
| `ssmask` | 0 / 26,640 | 0 | 0 | none |

The all-cell `ssh_frc` bit census is solely `+0.0` versus `-0.0`; it carries
no finite-magnitude difference and is registered rather than promoted over
the earlier `sshn_e` statement. The first non-bit input in written order is
therefore the incoming current SSH. R255-P3 is **CONFIRMED**, while the
literal continuity arithmetic is exonerated given its inputs.

The final classification is
`HELD_R255_INCOMING_SSH_OWNS_CONTINUITY` under both labels. All nine unique
plants fire: label coverage, round-254 dependency, admission, source order,
NEMO replay, candidate replay, boundary identity, first-nonbit, and one-bit
detection. R255-P1 through P5 are satisfied, including the frozen HELD
disposition.

## Gates, tests, and review

This round changes no model, card, recipe, or configuration file, so no
trajectory can move and the OMT/rung/GYRE/DINO/tank landing gates are not
activated. The round-253 certified states remain the baseline; this receipt
makes no new trajectory claim. Focused round-252/253/254/255 tests pass 51/51.
The citation-gate unit suite passes 17/17; its log SHA-256 is
`c6f27a3fbb310251447c72722453445c6a9f054ace9bbf4863f561ce9ac1428b`.

The once-only `tests/ocean/fidelity -n 12` battery collected 3,320 tests,
printed 3,288 passes and four failures, then stopped returning worker output
at 99%; it was terminated and was not rerun. The four printed IDs were rerun
serially with xdist disabled, and all reproduce their registered pre-existing
conditions: SI3 scalar-math `MY_SRC` provenance, the GYRE round-129 certified-
year harness stamp, the round-35 process-global allow-dirty scoping ratchet,
and the report-emitter worktree-stamp ratchet. The full, focused, and isolated
log SHA-256 values are
`5ea02e63bf0d6a154bdab26e5d035b9951186ff5294922125edcb03b44216120`,
`68b1311a8728129970dd650c838b7baf93deeda1dec6a361217efefd52d55b76`,
and `b3272f20332468d0e11674cde378e18b9a76236a80adbf86dcf2648e18875fa1`.

The receipt citation gate passes all six extracted citations with no failure
or unmapped entry. Shifting the compiled continuity citation by two lines
makes it fail `SYMBOL-NOT-AT-LINE`. The cumulative default receipt also passes
with 274 citations and no unmapped entry. Their JSON SHA-256 values are
`cc9665ae2ef5aafd7d9f8cb1118448c5eeccdbba6bf764d5008417485321e995`,
`bf5a6b0110e6daae8f8f9dce35b710f513011d1275409e382d0dc6defe69507f`,
and `207fc8715420d842111bd3c6087d01410d9b20a0634ff3696f7afe7e0b73e076`.

The required `codex exec --sandbox read-only` review was attempted. Its
in-process app-server client could not initialize on the read-only filesystem,
so **independent review unavailable in-sandbox**. The retained attempt has
SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The independent/given calibrated measurement JSON SHA-256 values are
`123ba5c6c31a028e2c2fa0e855473e33a2ab0a11278862e1a7b4c0756735bdf7` /
`cf841a9bd1ddc1ba852ff7ab288e48822a25bcc9195ae3dd924ed4ffe5862096`.
The classification and plant-log SHA-256 values are
`ea809b648a14fbf96dd6d3e1bc0b2d0bd449b81db270bf7a2486b10e1edac2c6` /
`5cdbb8e790fbc30d96a1940acdb665e0f5936b98edaef29b2d62bd01d9ef6d97`.

## OPEN

Continue one substep upstream without a new acquisition. At kt=8 stage 1,
replay substep 1's literal continuity statement against the admitted
`j001_ssha_e`, in the same compiled order: incoming `i000_sshn_e`, `zhU`,
`zhV`, `r1_e1e2t`, reconstructed `zhdiv`, `ssh_frc`, `rDt_e`, and `ssmask`.
The first non-bit operand owns the next walk. The complete fold/transport unit
and both held external-mode associations remain private. No stabilizer, floor,
clip, configuration change, or acquisition is authorized.

Round disposition: **HELD**.
