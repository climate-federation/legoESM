# Vertical-mixing lane — V1 round 4 (vmix_r4)

Status: **HELD. V1 is not certified.** The below-bottom-thickness owner is
confirmed for the 28 full-depth cells it predicts and landed on V1 only. The
preregistered claim that it would close all kt-1 `avm`/`avt` debt is refuted:
the counts improve from 148/147 to 120/119, not zero. Both labels remain DEBT
through kt 10.

Decision 129 is still pending. This round uses the admitted as-built deck:
NEMO's own TKE floors and internal-wave mixing off.

Preregistration:
`PREREG_nemo_testcases_gridladder_vmix_v1_round4.md` at `ea0e58b4e`.

Landing commits:

- `58f654ee2`: V1-only terminal `e3t(jpk)` operand, source-ordered sweep,
  instrument, controls and tests;
- `84af544fc`: stamp the admitted record's intentional absence of a kt-1
  restart;
- `3376af47d`: stamp the kt-1 ZDF frame's missing `dissl`/native-length
  observable;
- `dbcc25bfc`: discriminate NEMO's recorded `taum` from the card's consumed
  value (they are bit-identical);
- `46ce790f6`: mechanically re-anchor every guarded citation shifted by the
  landing.

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1`, admitted in round 3.
Evidence: `gridladder_vmix_rounds/round4/`, all clean artifacts stamped
`dbcc25bfc9b5fa02bfeb6f0e157b3a6dc8a34851` and packages tree
`5f6049bb64c1b85d4204d48bfc5a77f06344141d`.

| file | sha256 |
|---|---|
| `v1_mixing_length.json` | `a1087318ff035b8c6ba2100d7d6fcec60b71f112b57983906d0ea4e01b07e4a4` |
| `v1_mixing_length_plant.json` | `d70e482020b866f1a762f02a36fa5adbd9cc31b26e7d54ad86522120e2860e8d` |
| `v1_independent_kt1_5.json` | `401906029d1f5e582a2d9ac1ad76aad9f4f4d156da50380978b424d5f0a4ef72` |
| `v1_independent_kt6_10.json` | `172cda1565398b890a512b4d0d8f9cbb1ed9507efdb5731a332cf46553592101` |
| `v1_given_entries.json` | `d4a49662aaad8abfc8598f5fd42d0467a771de00d94b789ad033b1c8a25b8166` |

## 1. Fixed source reading and Decision 122

The compiled `tke_avn` opens at
`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdftke.f90:531`. It initialises
both arrays at :594-595, evaluates the raw length at :643-645, executes the
top-down and bottom-up sweeps at :686-692, composes `zmxlm`/`zmxld` at
:694-698, and publishes `avm`, `avt`, and `dissl` at :706-711. In particular,
the first bottom-up iteration is `jk=jpkm1` and reads
`zmxlm(jk+1) + e3t_3d(jk+1)*(1+r3t*tmask)` at :690-692. Since row `jpk` is
dry, that read is its raw, unstretched mesh thickness.

`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90` is byte-identical to the V1 file
(both SHA-256 `0f290190a33e4713bba218047451b35e25327bd589a8a74536a2db506efe81c2`).
For Decision 122, the enclosing production subroutine therefore also opens at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:531`, and the same statement is at
:690-692. This round selects the transcription only on the V1 card; full
ORCA2 remains unchanged pending its own ladder-certified selection.

## 2. What landed

The domain reader now retains the raw dummy-bottom thickness separately while
leaving all live `jpkm1` arrays unchanged
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3680`). Only
V1 installs that row on the coordinate (:3763) and selects the new flag
(:4389). The default is false
(`packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:657`). The
step-entry bundle refuses a missing or wrong-shaped selected operand
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:11770`).
The closure appends it only for the selected scan
(`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:3085`), and the
literal recurrence performs NEMO's `jpkm1` update at :963-981. G1-G5 retain
the old shape and default path.

## 3. Preregistered verdicts and kt-1 cellwise measurement

The committed arm is at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g2_orca2bathy_ladder.py:738`.
Its write-only trace is passive: every returned state leaf is bit-identical to
the ordinary step. The trace's consumed `rn2` and a separate eager bundle have
332,247 unequal cells, maximum `8.239936510889834e-18`; all replay numbers use
the consumed trace operand, not the eager reconstruction. NEMO's recorded
`taum` and the consumed card `taum` are bit-identical over all 16,433 wet
columns, as are the two surface mixing-length anchors.

P1 **CONFIRMED**. Cellwise source-ordered results over 414,119 wet W cells:

| comparison | unequal | over bar | max abs | verdict |
|---|---:|---:|---:|---|
| old 30-row `zmxlm` vs true-row replay | 28 | 28 | `9.225210721922394e-3` | DEBT |
| old 30-row `zmxld` vs true-row replay | 28 | 28 | `4.667682512376814e-3` | DEBT |
| landed `zmxlm` vs true-row replay | 0 | 0 | 0 | exact |
| landed `zmxld` vs true-row replay | 0 | 0 | 0 | exact |
| card-en `zmxlm` vs NEMO-en replay | 21,879 | 0 | `1.4210854715202004e-14` | AT-BAR |
| card-en `zmxld` vs NEMO-en replay | 22,001 | 0 | `1.4210854715202004e-14` | AT-BAR |

P2 **REFUTED**. The record does not contain native mixing-length work arrays.
Where NEMO's background `MAX` is inactive, inversion of recorded `avm`
defines `zmxlm`; that inversion has 127,617 unequal and 45 over-bar cells
against the replay, maximum `1.588915665706736e-10`, so it is a secondary
observable rather than an exact oracle. The direct coefficient rows are:

| comparison | unequal | over bar | max abs |
|---|---:|---:|---:|
| NEMO-input replay `avm` vs record | 109,920 | 2 | `1.5889199678209565e-14` |
| landed card `avm` vs record | 114,690 | 120 | `3.4121551539823997e-1` |
| landed card `avt` vs record | 109,902 | 119 | `1.2375029133760408e-1` |
| card-state final replay vs published card `avm` | 46,593 | 118 | `3.4121551539823997e-1` |

Thus the terminal row removes exactly the 28 attributable over-bar cells from
round 3's 148/147 counts; it does not own the remaining 120/119. `en` itself
remains AT-BAR at kt 1: 35,577 unequal cells, maximum
`6.938893903907228e-18`.

P3 **CONFIRMED**. The config delta test proves only V1 selects the flag and
raw row; a planted false selector makes the card validator fail. The full
132-test TKE/V1 focused suite passes.

## 4. Thickness actually read

There are 16,433 wet columns: 4,757 full-depth and 11,676 partial. Thickness
at each wet bottom and immediately below spans
`1.307202170947221 .. 999.9570435796322 m`. Every one of the 11,676 partial
columns already carries the exact below-bottom row. Full-depth `e3t(jpk)`
spans `20.978521789021215 .. 999.9570435796322 m`; this was the only missing
operand.

The admitted acquisition intentionally writes no kt-1 restart, so the
preregistered kt-1 frame/restart identity control is unavailable and stamped
as such. Its kt-1 ZDF frame also records no `dissl` or native length arrays;
therefore `zmxld` is tested by the source-ordered replay, not claimed as a
directly observed array.

Non-vacuity: one ULP was added to the terminal row. It moved 28 `zmxlm` cells
(maximum `5.684341886080802e-14`) and 14 `zmxld` cells (same maximum); the
artifact gate returned 1. Wrong shapes and non-finite rows are also refused.

## 5. First non-bit statement and next owner

With the surface anchor, consumed `rn2`, geometry and terminal thickness held
fixed, substituting the card's AT-BAR post-step `en` first produces non-bit
lengths at NEMO's raw-length assignment
`zmxlm = MAX(rmxl_min,SQRT(2*en/zrn2))`
(`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdftke.f90:643-645`, enclosing
`tke_avn` at :531): 21,879 `zmxlm` and 22,001 `zmxld` cells differ, all still
AT-BAR.

That is not the remaining over-bar owner. Replaying the final length and
coefficient from the card's published `en` still disagrees with the card's
published `avm` in 118 over-bar cells. The next boundary is therefore the
trace association/lifetime between the final `tke_avn` operands and the
published coefficient, whose NEMO statements are `zsqen`, `zav`, then
`p_avm` at :707-709. Owner label: **UNMEASURED**. The discriminating next
measurement is a passive write-only capture of `tke_curr`, `l_k_final`, and
`K_M` at the return of the closure, compared with both carried state fields.

## 6. Both labels, kt 1..10

Counts are wet unequal cells over all carried levels; `max` is maximum absolute
difference. Every listed row is DEBT. `dissl` additionally has 346,979 unequal
dry cells at every restart. Raw JSON contains every level, depth class, region,
surface field and first unequal cell.

INDEPENDENT:

| kt | `en` count / max | `avm_k` count / max | `avt_k` count / max | `dissl` wet count / max | T count / max |
|---:|---:|---:|---:|---:|---:|
| 2 | 124,320 / `2.24557e-3` | 155,297 / `4.58693e-1` | 153,239 / `4.58693e-1` | 324,740 / `6.50797e-3` | 185,764 / `1.68964e-1` |
| 3 | 137,423 / `1.62133e-3` | 159,857 / `4.25033e-1` | 158,460 / `4.25033e-1` | 358,100 / `6.31972e-3` | 403,896 / `1.42565e-1` |
| 4 | 177,553 / `3.96170e-3` | 172,568 / `4.54101e-1` | 172,864 / `4.15987e-1` | 404,740 / `6.92010e-3` | 428,339 / `1.24019e-1` |
| 5 | 194,178 / `9.49939e-3` | 175,477 / `5.21981e-1` | 176,200 / `5.40607e-1` | 413,359 / `6.73009e-3` | 430,349 / `1.15546e-1` |
| 6 | 205,277 / `1.16961e-2` | 175,383 / `5.24376e-1` | 176,228 / `5.24376e-1` | 414,055 / `7.48712e-3` | 430,522 / `1.31483e-1` |
| 7 | 218,133 / `1.32544e-2` | 177,363 / `5.89395e-1` | 178,379 / `5.89395e-1` | 414,104 / `8.31515e-3` | 430,547 / `9.64013e-2` |
| 8 | 225,017 / `1.32831e-2` | 177,222 / `6.06534e-1` | 178,399 / `6.06534e-1` | 414,111 / `8.93016e-3` | 430,550 / `1.06179e-1` |
| 9 | 232,147 / `1.04587e-2` | 178,370 / `5.32887e-1` | 179,670 / `5.32887e-1` | 414,118 / `8.78880e-3` | 430,552 / `1.42894e-1` |
| 10 | 238,121 / `1.03488e-2` | 178,235 / `5.64327e-1` | 179,607 / `5.64327e-1` | 414,119 / `8.60928e-3` | 430,551 / `2.21605e-1` |

At kt 1, the post-`zdf_phy` frame has: `en` 35,577 unequal / maximum
`6.93889e-18` (AT-BAR), `avm` 114,690 / `3.41216e-1`, and `avt` 109,902 /
`1.23750e-1`. The first-step surface `ssh`, `uu_b`, and `vv_b` remain exact;
u, v, T, S remain DEBT.

GIVEN-NEMO-ENTRY, scored after the registered two-step projection:

| entry kt -> score kt | `en` count / max | `avm_k` count / max | `avt_k` count / max | `dissl` wet count / max | T count / max |
|---:|---:|---:|---:|---:|---:|
| 1 -> 2 | 124,320 / `2.24557e-3` | 155,297 / `4.58693e-1` | 153,239 / `4.58693e-1` | 324,740 / `6.50797e-3` | 185,764 / `1.68964e-1` |
| 3 -> 4 | 127,480 / `1.68899e-3` | 157,898 / `4.39166e-1` | 156,375 / `4.39166e-1` | 350,155 / `6.07779e-3` | 425,512 / `1.02551e-1` |
| 5 -> 6 | 128,816 / `9.31657e-3` | 158,752 / `4.80200e-1` | 157,581 / `4.80200e-1` | 351,125 / `5.83130e-3` | 428,529 / `8.21684e-2` |
| 7 -> 8 | 130,539 / `1.24322e-2` | 160,721 / `6.33501e-1` | 159,893 / `6.33501e-1` | 352,000 / `9.00573e-3` | 428,118 / `1.28594e-1` |
| 9 -> 10 | 131,916 / `9.43272e-3` | 161,865 / `5.80683e-1` | 161,217 / `5.80683e-1` | 352,596 / `8.63115e-3` | 427,581 / `2.35723e-1` |

At each given entry, `ssh`, `uu_b`, and `vv_b` are exact; u, v, T, and S are
DEBT. The two-step projection is DEBT in every table row. V1 is not certified.

## 7. Tests and review

- 132 focused V1/TKE tests passed in 119.31 s:
  `test_nemo_testcase_v1_orca2tke.py`, `test_tke_carried_coefficients.py`,
  `test_tke_nemo_identity.py`.
- Terminal-row selector/helper focus: 3 passed in 12.90 s; post-instrument
  helper checks: 1 passed in 0.46 s.
- Required final gates, synchronously after citation re-anchoring: 23 passed
  in 9.77 s (`test_nemo_testcase_receipt_citation_gate.py` and
  `test_no_private_cross_imports.py`).

independent review: operator (Claude)

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r4-3151526350
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: Decision 129 remains pending — keep NEMO's own V1 floors until IWM is added, or apply ORCA2 effective floors now?
ROUND_STATUS: HELD — terminal thickness landed; V1 remains DEBT and is not certified.
