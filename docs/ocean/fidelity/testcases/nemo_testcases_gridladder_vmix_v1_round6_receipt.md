# Vertical-mixing lane — V1 round 6 (vmix_r6)

Status: **HELD. V1 is not certified.** Two cited owners are named, measured
and landed (both V1-selected). kt 1 is now **AT-BAR on every scored field,
including the TKE energy, viscosity and diffusivity** (the kt-1-only ladder
exits 0). Both labels are still DEBT from kt 2, at much smaller amplitude.

Decisions in force: D129 = 1 (NEMO's own TKE floors, internal-wave mixing
off), D130 = 1 (new vertical-mixing statements stay off on other cards).

Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v1_round6.md` at
`3dd0320df`, committed before measuring.

Commits:
- `3dd0320df` preregistration;
- `da743c240` landing 1: N2 with NEMO's per-operation rounding;
- `9c8763e98` landing 2: TKE shear at the whole-step entry sea level;
- `9de26dab5` citation re-anchor (mechanical);
- `65570d50b` review fixes; `ae651f0da` helper moved to module end;
  `b4a6aba48` citation re-anchor for the review fix (mechanical).

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3).
Evidence: `gridladder_vmix_rounds/round6/` (`manifest.sha256`). Ladders
stamped clean at `9c8763e98ff0572e7b4d1ad1026eef87b9ff2af9`, packages tree
`4e1a8108cf604f029a45be9fd4d53525a16c70bd`. The later package commits change
only the diagnostic path, a refusal and code placement; the kt-2 probe re-run
at `65570d50b` gives the identical kt-2 row (`kt2_probe_review.json`).

| file | sha256 prefix |
|---|---|
| `probe_n2.py` | 5844ca0ffecb1cc0 |
| `probe_kt2.py` | 4ea882423ffbc3a9 |
| `probe_sh2.py` | 0d4588bfc147ae95 |
| `n2_probe_before.json` / `n2_probe_after.json` | 3f8afdd531e122e9 / 6448e6128f40d7b1 |
| `kt2_probe.json` / `kt2_probe_fix.json` | 1a0b41e77b23a5fb / 522fedbbfc19ea33 |
| `sh2_probe.json` (entry ssh) | fdb7f981b7c0bd4d |
| `sh2_probe_control.json` (card's own inputs) | ba2806da618bbabf |
| `sh2_probe_ssh_s1/s2/s3.json` | 745cb31a08440b7b / 602f70d4515c8ddc / d454b580ba64da70 |
| `sh2_probe_fix.json` | 00821c6183176f22 |
| `v1_independent_kt1.json` | a71081bcee818fd2 |
| `v1_independent_kt1_5.json` / `_kt6_10.json` | 42b05d69de9779c6 / f77333df9ed3507f |
| `v1_given_entries.json` | 0ba3b70fac63651a |
| `before_sh2/v1_independent_kt1_5.json` (after landing 1 only) | 2f13e3fb86039f02 |
| `codex_review.log` | f34d613de1f68eb4 |

## 1. The kt-1 N2 owner (P1, P2 CONFIRMED; LANDED)

Instrument: `probe_n2.py`, an offline numpy replay of NEMO's literal
`rab_3d_t` (EOS-80) and `bn2_t`, one binary64 operation per Fortran
operation in source order, fed NEMO's own kt-1 step-entry T/S/ssh. The card's
entry T/S equal NEMO's in every wet cell; the card's entry ssh equals NEMO's
bit-for-bit (asserted). At kt 1 `rn2 = rn2b` (`stprk3.f90:176-180`).

| row (414,119 wet W cells) | before | after landing 1 |
|---|---|---|
| card eager alpha / beta vs NEMO-literal | 0 / 0 unequal | 0 / 0 |
| card **compiled** rn2 (what TKE reads) vs NEMO-literal | 332,247 / max 8.24e-18 | **0** |
| NEMO-input `avm` replay with the compiled rn2 vs record | 4 over 1e-15 abs | 0 unequal |
| NEMO-input `avm` replay with the literal rn2 vs record | **0 unequal** | 0 |
| card `avm` / `avt` over the bar (ladder) | 2 / 1 | **0 / 0** |

- At the 3 over-bar cells the compiled rn2 was e.g. `0x1.9da2cb4edf31dp-31`
  against NEMO's `0x1.9da2cb4ed43e8p-31` at (10,120) w20.
- **First non-bit statement:** none of the card's statements is mistranscribed.
  Evaluated op-by-op they equal NEMO. Under jit, XLA contracts multiply/add
  pairs inside the EOS-80 Horner chains (alpha 215,783, beta 409,554 cells
  unequal), the bn2 assembly and the zrw depth products. NEMO's gfortran -O3
  build (no FMA) rounds each source operation.
- **What landed:** `TKEConfig.nemo_bn2_source_rounded` (default False,
  V1-selected) routes every source operation of `rab_3d_t`/`bn2_t` and the
  zrw products through the repo's existing `nemo_source_round` guard. With the
  flag off the expression trees are unchanged. It raises with the simplified
  EOS and with a closure that never builds the step-entry N2 bundle.

## 2. The first kt-2 over-bar row and its owner (CONFIRMED; LANDED)

**Row:** step 2, the TKE closure inside `zdf_phy` at step entry (before
stage 1), field `en`. It was DEBT in 123,838 full-depth + 418 partial cells,
max 6.40e-6 (16,170 of 16,433 cells at the first interior w level). kt 2
ssh (max 4.48e-15) is produced later in the step.

Ruled out first (measured):
- **The kt-2 entry state:** card vs NEMO's kt-2 step-entry frame on all levels:
  T 1,765 unequal / 1.4e-14, S 1,593 / 2.8e-14, u 9,450 / 3.6e-16,
  v 8,126 / 4.2e-16 (kt 1 has no NEMO restart, so this was unscored before).
- **The surface value en(1):** 0 unequal.
- **The sea-level stretch r3u:** corr(gap, r3u) = -0.06.
- **Dissipation memory:** no coherent signature (mixed signs).

Instrument: `probe_kt2.py` backs NEMO's pre-sweep RHS out of NEMO's recorded
kt-2 `en` through the card's at-bar matrix. Its control (card's own rows vs
card RHS) is 8e-15 at level 2. NEMO-backed vs card RHS was median 4.3e-8
relative. Then `probe_sh2.py`, an offline NEMO-literal `zdf_sh2`
(`zdfsh2.f90:98-113`, with `Kbb = Kmm = Nbb`), was fed NEMO's kt-2 entry u/v/ssh
and NEMO's kt-1 avm. Interior cells only (fold row and seam excluded,
407,907 cells):

| replay sea level | unequal | median rel | max rel |
|---|---|---|---|
| NEMO's kt-2 entry ssh (what NEMO uses) | 407,907 | 5.6e-5 | 5.7e-2 |
| card's own kt-1 state (control) | 407,907 | 5.6e-5 | 5.7e-2 |
| NEMO stage-1 after-ssh | 407,907 | 1.9e-5 | 1.9e-2 |
| **NEMO stage-2 after-ssh (= stage-3 Kmm)** | 39,654 | **0** | **2.2e-13** |
| stage-3 after-ssh | 407,907 | 5.6e-5 | 5.5e-2 |

The card's metric operands (`hu_0`, `e1e2t`, `e1e2u`, `e3w`) equal NEMO's
mesh in every cell. So the card's shear divides by `e3w*(1+r3u)` at the
stage-3 sea level.

**NEMO statement:** `stprk3.f90:185` calls `zdf_phy(kstp, Nbb, Nbb, Nrhs)`
once at step entry (SUBROUTINE `stp_RK3`). `zdfphy.f90:318` passes
`(Kbb, Kmm)` to `zdf_sh2`, whose divisor is
`(e3w_1d(jk)*(1+r3u(Kmm))) * (e3w_1d(jk)*(1+r3u(Kbb)))`
(`zdfsh2.f90:98-108`, SUBROUTINE `zdf_sh2` :56). Both are the entry level.
The card handed the shear the stage-3 Kmm sea level that `dyn_zdf`/`tra_zdf`
correctly read. That is a transcription defect: the card's own code comment
already states the entry-level rule.

**What landed:** `TKEConfig.nemo_sh2_step_entry_ssh` (default False,
V1-selected). It sends the whole-step entry eta to the shear (both RK3 call
sites and the diagnostic path). It raises when a stage eta is given without
the entry eta, and on non-prognostic TKE.

After landing 2, at kt 2: `en` max miss 6.40e-6 -> 5.57e-7; NEMO-backed RHS
vs card median 4.3e-8 -> 3.1e-16. The shear replay with the entry ssh matches
to max 2.2e-13 (39,662 cells unequal, none over 1e-12 relative).

## 3. Both labels, kt 1..10

kt 1 (both labels share it): every surface row is AT-BAR (ladder exit 0):
- ssh, `uu_b`, `vv_b`: 0 unequal;
- u: 560 unequal / 3.6e-16 (round 5: 3,494 / 9.99e-16);
- v: 525 / 3.1e-16;
- T: 169 / 7.1e-15 (round 5: 1,318 / 1.07e-14);
- S: 147 / 2.1e-14.

kt-1 ZDF frame (all AT-BAR):
- `en`: 31,802 / 6.9e-18;
- `avm`: 16,622 / 2.2e-16 (round 5: 99,800 / 1.59e-14 DEBT);
- `avt`: 15,927 / 5.6e-17 (round 5: 95,369 / 1.59e-15 DEBT).

INDEPENDENT, all-level wet cells. Each entry is unequal / full-depth max abs,
with the partial-cell max in brackets. Same metric as round 5's table. Every
row is DEBT.

| kt | `en` | `avm_k` | `avt_k` | `dissl` | T |
|---:|---|---|---|---|---|
| 2 | 73,757 / 5.57e-07 (6.7e-17) | 45,662 / 1.17e-05 (3.2e-15) | 43,690 / 9.55e-06 (3.2e-15) | 38,709 / 1.32e-07 (3.4e-16) | 18,626 / 2.95e-04 (5.6e-15) |
| 3 | 90,392 / 1.04e-06 (4.8e-16) | 65,612 / 3.89e-05 (1.4e-14) | 65,630 / 3.89e-05 (1.4e-14) | 59,918 / 4.24e-06 (1.0e-15) | 57,985 / 2.33e-04 (2.1e-09) |
| 4 | 101,795 / 4.68e-06 (1.1e-12) | 82,168 / 5.97e-05 (1.9e-10) | 82,291 / 2.66e-05 (1.5e-06) | 93,984 / 7.94e-06 (8.6e-12) | 76,800 / 1.86e-04 (1.7e-05) |
| 5 | 107,910 / 3.77e-06 (5.0e-07) | 89,578 / 8.38e-05 (4.5e-10) | 89,846 / 8.38e-06 (2.8e-05) | 107,533 / 6.73e-06 (1.3e-07) | 91,529 / 1.85e-04 (3.3e-04) |
| 6 | 113,368 / 2.29e-06 (1.0e-06) | 95,798 / 1.09e-04 (3.6e-06) | 96,232 / 1.09e-04 (8.1e-05) | 119,001 / 1.04e-05 (2.5e-06) | 106,494 / 4.86e-04 (1.2e-03) |
| 7 | 117,340 / 1.85e-06 (1.4e-06) | 101,435 / 5.72e-05 (5.1e-05) | 102,132 / 5.72e-05 (9.0e-05) | 130,414 / 1.24e-05 (9.4e-06) | 123,355 / 8.70e-04 (2.2e-03) |
| 8 | 122,813 / 1.41e-06 (2.6e-06) | 106,845 / 4.23e-05 (1.4e-04) | 107,787 / 4.23e-05 (1.4e-04) | 144,675 / 1.35e-05 (1.7e-05) | 143,480 / 1.43e-03 (3.7e-03) |
| 9 | 126,203 / 2.23e-06 (5.1e-06) | 111,847 / 1.06e-04 (2.6e-04) | 112,935 / 1.06e-04 (2.6e-04) | 162,921 / 1.96e-05 (2.8e-05) | 160,111 / 2.48e-03 (6.3e-03) |
| 10 | 131,163 / 5.11e-06 (9.4e-06) | 116,179 / 2.45e-04 (5.0e-04) | 117,380 / 2.45e-04 (5.0e-04) | 177,549 / 3.18e-05 (4.9e-05) | 175,440 / 4.32e-03 (1.1e-02) |

Round 5, same record and windows: kt-10 `en` 8.10e-4, `avm_k` 1.30e-2,
`dissl` 8.87e-4, T 1.59e-1. Surface rows at kt 2..10 are all DEBT; e.g. the
kt-2 surface T is 3,183 cells / 2.95e-4 (round 5: 15,370 / 3.32e-4).

From kt 5 the **partial-cell** maxima exceed the full-depth ones
(kt-10 T: 1.1e-2 vs 4.3e-3). That is a separate, unattributed owner.

GIVEN-NEMO-ENTRY (two-step projection; full-depth max abs, unequal cells):

| entry -> score | `en` | `avm_k` | `avt_k` | T |
|---|---|---|---|---|
| 1 -> 2 | 5.57e-07 (73,576) | 1.17e-05 (45,557) | 9.55e-06 (43,582) | 2.95e-04 (18,591) |
| 3 -> 4 | 4.92e-06 (90,576) | 6.36e-05 (65,074) | 2.74e-05 (65,031) | 1.88e-04 (55,942) |
| 5 -> 6 | 1.77e-06 (93,040) | 8.68e-05 (67,508) | 8.68e-05 (67,700) | 4.80e-04 (59,731) |
| 7 -> 8 | 1.06e-06 (94,609) | 4.22e-05 (69,448) | 4.22e-05 (69,828) | 8.93e-04 (62,121) |
| 9 -> 10 | 2.04e-06 (96,861) | 6.90e-05 (71,369) | 6.90e-05 (71,834) | 1.79e-03 (64,303) |

Round 5's single-step surface T from NEMO's entry at kt 7 (2.57e-2) and kt 9
(2.37e-1) is now 4.90e-6 and 5.88e-6. Round 5 called that "a different
owner": it was this shear. Retracted.

**V1 is NOT certified.**

## 4. Decision 122 (ORCA2_OMIP_L4)

`eosbn2.f90`, `zdfsh2.f90` and `zdfphy.f90` are byte-identical between the
V1 build and `ORCA2_OMIP_L4` (SHA-256 `f08e8e47…`, `e8bb2398…`,
`903e0d12…`). `stprk3.f90` differs (V1 carries record instruments), so its
lines are given per build.

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| EOS-80 alpha/beta polynomial | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1281-1333` | `rab_3d_t` (:1256) |
| zrw and N2 assembly | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1631-1640` | `bn2_t` (:1604) |
| rn2b from ts(Nbb) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171` | `stp_RK3` (:62) |
| shear divisor `e3w_1d*(1+r3u)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:98-108` | `zdf_sh2` (:56) |
| `zdf_phy(kstp, Nbb, Nbb, Nrhs)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179` | `stp_RK3` (:62) |

The full ORCA2 card also selects the RK3 shear form with the stage-Kmm sea
level, so it carries the same shear defect. This is a pointer only, not
measured there.

## 5. Selection and caller grep (no-silent-default rule)

- Both flags are set only in the V1 card's vertical block, and the V1
  selector pins both. G1-G5, the full ORCA2 card, DINO, GYRE and every library
  default are unchanged. The V1 config-delta test names both flags, and the
  selector check fires when either is planted back.
- The shear flag is read in one place, the new sea-level helper. Its callers
  are the two RK3 calls of the vertical-mixing routine and the
  vertical-diffusivity diagnostic, and all three now pass the entry sea level.
  The other step programs (AB2, leapfrog, MLF, unsplit) pass their own
  `state.eta` as `eta_now` and no entry eta, so with the flag set they raise
  rather than guess. V1 runs only the RK3 program.
- **RULE 3 question:** both fixes default to False, which keeps the old
  behaviour on other cards. D130 already answers this: stay off until each
  card is measured. No new question.

## Tests and review

All runs synchronous, CPU, fp64 (`JAX_ENABLE_X64=1`):

| suite | result |
|---|---|
| new `test_nemo_bn2_source_rounded.py` (literal-order bitwise under jit; contraction observed on this CPU; refusals) | 4 passed; planting `nemo_source_round` = identity turns the main test red (17,770 / 20,000 unequal) |
| new `test_nemo_sh2_step_entry_ssh.py` | 3 passed; planting the helper back to the stage eta turns it red |
| V1 file + bn2/sh2 units + TKE carried + TKE identity + bn2 + Roquet EOS, at HEAD | 175 passed |
| G2-ladder + G4 + G5, not slow, at HEAD | 51 passed, 11 deselected |
| citation gate + private imports, clean tree at `b4a6aba48` | 23 passed |

Pre-existing failures, not caused by this round: both also fail on a clean
clone of `02b1de6d9`.
- `test_teos10_rab_bn2.py::TestProductionWiring::test_mpas_accepts_nemo_bn2_and_still_refuses_adiabatic`
- `test_nemo_zdfmxl_transcription.py::test_treguier_kappa_uses_the_same_n2_variant_as_the_slopes`

Citations: the round shifted guarded model-file pins. They were re-anchored
mechanically (`reanchor.py`, `reanchor2.py`, `reanchor_band.py` in the
evidence directory), with one range extent widened 122 -> 124 because it now
contains the inserted argument. No anchor text was changed.

Review: **single review (codex)**, `codex_review.log`, verdict
**"VERDICT: DO NOT SHIP"**. Its three findings:
1. **Diagnostic path would raise on V1:** CONFIRMED by reading. Fixed in
   `65570d50b` (passes the entry eta).
2. **No behavioural test for the shear flag; it was silently unused on
   non-prognostic TKE:** CONFIRMED. Fixed with a tested helper that refuses
   non-prognostic TKE. The behavioural evidence is the before/after probe pair
   (`kt2_probe.json` vs `kt2_probe_fix.json`).
3. **`e3w*stretch` in bn2 is not source-rounded:** measured, no change needed.
   The compiled rn2 equals NEMO-literal in all 414,119 wet W cells
   (`n2_probe_after.json`), because a division cannot contract with the
   product feeding it.

No second codex pass was run (weekly quota 86% at round start).
GLM review: unavailable in this session. **Dual review not met.**

## 6. Next owner (registered, UNMEASURED)

At kt 2 the first over-bar row is still `en` (73,757 cells, max 5.6e-7).
It was 6.4e-6 before this round. Candidates, none measured:
- the shear's `r3u` association: the card divides by `e1e2u` where the ORCA2
  build multiplies by `r1_e1e2u`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266-267`, same lines in V1,
  SUBROUTINE `dom_qco_r3c_RK3` :237), plus possible
  contraction in `0.5*(a*b + c*d)`. This is the 2.2e-13 shear residual.
- threshold flips amplifying the residual (Langmuir depth index, the
  Prandtl `MAX`, `MAX(en, rn_emin)`).
- the north-fold row: an equal-and-opposite mirror pair at (147,52)/(147,128)
  was seen before landing 2; it was not re-checked after.

Also open: the partial-cell maxima overtaking the full-depth ones from kt 5.

## Choices made

ASKED (this round's brief): name and land the kt-1 N2 owner; name the first
kt-2 row and its statement; re-score both labels.

UNASKED, offered for revert:
- landing the kt-2 shear fix this round, not only naming it. It is a cited
  transcription defect and V1-selected under D130.
- the two new refusals (simplified EOS with source rounding; shear flag on
  non-prognostic TKE).

UNVERIFIED:
- the kt-2 remaining owner (section 6);
- the full ORCA2 card's response to either flag (pointer only);
- the partial-cell growth from kt 5;
- ladder numbers are stamped at `9c8763e98`. Later package commits were checked
  only by the kt-2 probe re-run (identical), not by a full kt 1..10 re-run.
