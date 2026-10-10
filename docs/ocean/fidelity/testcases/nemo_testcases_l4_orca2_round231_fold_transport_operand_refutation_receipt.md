# ORCA2 round 231 — fold record admission and tracer-operand refutation

Date: 2026-10-10. Frozen base: `ea8427f2d`. Status: **HELD**. Evidence
root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round231/`.

No model statement, card selector, deck physics, carried state, stabiliser,
sea-ice selector, or `unmeasured_features` entry changed. All reported ocean
numbers are labelled either **independent** or **given NEMO's entry**. The two
labels are never pooled.

## Record admission

R231-P1 is **CONFIRMED**. The round-230 rank-1 record has exactly one
non-finite value: local Fortran `[94,106,6]`, global `[184,106]`, record
`tmask=0`. It lies outside both the owned mesh payload and the compiled
stage-1 consumer support. NEMO constructs `zFv` through `jpkm1` and only
through `ntej+nn_hls-1` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.
The live stage-1 centred consumer reads its V faces at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv_cen.f90:150-160`.
The value is therefore unused work-array storage, not an ocean operand.

R231-P2 is **CONFIRMED**. The checker now derives that support from the
compiled loop, retains the self-describing header and named payload walk,
requires every consumed `zFv` value and every other field finite, and reports
the excluded value instead of replacing it. Both rank records admit; both
rank-0/1 step-10 restarts remain byte-identical to the admitted OMT-4 source
run. Rank, field-name, truncation, and consumed-nonfinite plants all fire.
Admission JSON SHA-256:
`02eb8b9b222017e5032d4ac289ff9f33f77afc000432dfcd7f04717f92240b30`.

## Frozen operand discrimination

R231-P3 selects its **non-bit `zFv`** branch under both labels. The passive
side output is array-identical to the ordinary completed state in every state
slot before any operand is read. The candidate complete unit versus NEMO's
post-`tra_adv_trp` V transport has 415,175/799,200 unequal executable-level
slots, maximum absolute difference 397,263.8882070326 at `[147,131,24]`.
On the three-row fold band it has 3,943/16,200 unequal slots with the same
maximum at band-local `[2,131,24]`.

The tracer-halo alternative is **REFUTED**. Given NEMO's entry, the north-face
T and S sums are already exact (0/5,400 unequal each). Independently, S is
exact and T differs only by 3,524 signed zeros, with zero numerical
difference. NEMO's recorded T, S, live `e3t`, and mask halo rows each satisfy
their compiled T-fold identity exactly (0/5,400 unequal). Applying the frozen
offline T-fold correction therefore changes neither endpoint: T remains
0.17733430832081432 K and S remains 3.283356343139289 PSU. R231-P4 is
**REFUTED**, as required when P3's premise is false. Evidence SHA-256 is
`7074a86c003f3c7897309e8f69cfed4dfd4a8535fcaeab1d88ee67a7290f58d3`
for independent and
`b0c28a00cbd03ad5bae08563a95eb9827657582be3e04bc422dcf45debae3bb5`
for given NEMO's entry.

The first non-bit **recorded boundary** is therefore the stage-1 V transport,
not a tracer-fold operand. NEMO first builds the V correction `zvb` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279`, then
builds `zFv = e1v*e3v(Kmm)*(vv(Kmm)+zvb*vmask)` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`, before
the recorded post-`tra_adv_trp` call at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:552-559`.

That product statement itself is **EXONERATED** on the available rank-zero
record: offline replay from NEMO's recorded metric, live thickness, velocity,
correction, and mask is bit-exact for U (0/226,236 unequal, 0 ULP) and V
(0/226,637 unequal, 0 ULP). Its planted one-bit violation fires. Replay JSON
SHA-256:
`0cf924baf713ef3f5b33aeb38eccb7082f6eb59faa8e1240543b527e7e4148c7`.
Thus the surviving owner is upstream in the candidate's stage-1 external-mode
V operands. It is not yet attributable to one statement because the current
rank-complete record contains final `zFv`, T, S, `e3t`, and `tmask`, but not
rank-complete `e1v`, live `e3v`, `vv(Kmm)`, `zvb`, and `vmask`.

## Controls, validation, and review

R231-P5 is **CONFIRMED**. In addition to the four admission plants, the
transport-bit scorer and halo-bit fold-identity controls each fire exactly
once. The correction-sign control compares the changed correction directly
with its unplanted value and fires; this replaced a first attempt whose
endpoint-maximum predicate was correctly caught as vacuous and retained in
`operand_plants.log`. The binding correction control is
`correction_sign_plant.log` (SHA-256
`de3e0ed41c25b78c2542da0a30c61ddc12fa259258e16fc57aabef42919d9371`).

The focused round-229/231 tests pass 5/5. The one required
`tests/ocean/fidelity -n 12` battery was launched only after the process-count
gate returned zero. It reported 3,109 passes, seven skips, and four unrelated
branch-wide reds before its last legacy prediction-plant test failed to
complete; that test also exceeded a bounded 600-second isolated run. The four
reds reproduce in isolation: the registered GYRE year-harness-stamp move, the
round-35 allow-dirty scope ratchet, the registered worktree-stamp ratchet, and
the registered SI3 verbatim-source gate. Logs have SHA-256
`d24a2f4536c8672b244f956c4edab2cf9241e70fe68501bbbaf942a2c4ae828f`,
`57d5595cc2e38fdb4d7b1b6466a81244a3d02826b5f07c0bce70c4e8401ee0aa`,
and `a8fd15ff28de307393503ef55796f2ce7321b4ab88c9983b2b35e27232e7b683`.
No `packages/` file changed, so no GYRE, DINO, tank, rung-0, rung-10, or OMT
trajectory can execute changed model code. The required separate
`codex exec --sandbox read-only` review
could not initialize because its app-server PATH-alias write was denied by
the read-only sandbox. Verdict: **independent review unavailable in-sandbox**
(`independent_review.log`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).

## OPEN

Round 232 resumes at `zvb` before the product. First obtain or locate a
rank-complete, self-describing stage-1 V-transport operand stream containing
`e1v`, live `e3v(Kmm)`, `vv(Kmm)`, `zvb`, `vmask`, `vn_adv`, `r1_hv_0 /
(1+r3v(Kmm))`, and `vv_b(Kmm)`. Offline replay then splits, in source order,
the correction and product at the two compiled `stprk3_stg` spans cited
above, on the fold rank. The first operand that leaves the floor is the next
citable owner.
No tracer-fold exchange or atomic unit lands from this refuted hypothesis.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.
