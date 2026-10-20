# Vertical-mixing lane — V1 round 7 (vmix_r7)

Status: **HELD. V1 is not certified.** kt 1 is now **bit-identical on every
scored field** under both labels (surface ssh/u/v/T/S/barotropic velocity and
the TKE frame en/avm/avt: 0 unequal cells). Both labels are DEBT from kt 2. The
kt-2 over-the-bar TKE energy lives entirely on the north-fold pivot row.

Decisions in force: D129 = 1, D130 = 1. Codex review: paused (quota).

Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v1_round7.md` at
`fd5ffc797`. It was committed after the offline kt-1 walk and before the
landing and the kt 1..10 ladders; the file says so.

Commits:
- `e76926a87` task 1: wiring tests for the round-6 flags;
- `fd5ffc797` preregistration;
- `f2dc0dd4d` landing: kt-1 TKE energy bit-exact;
- `516ce4cb8` citation re-anchor (mechanical, +3 on four recipe pins).

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3).
Evidence: `gridladder_vmix_rounds/round7/` (`manifest.sha256`, 34 files).
Ladders stamped clean at `f2dc0dd4dd9d68056a0aea2bb804cb10ac09580a`,
packages tree `a37f6148fdd7b95ff7a83ab0127b168b22b671fb`. The later commit
touches only a test-side pin table.

| file | sha256 prefix |
|---|---|
| `probe_kt1.py` / `probe_lit.py` | cc6e70dc7779ee3f / d4bcf39401f65f37 |
| `probe_etau.py` / `etau_offline.py` / `libm_vs_vector.py` | 4cf0e4a1dbd78c35 / ad6ad1cf40e9931f / 70993b327b3df34d |
| `kt1_base.json` (before) / `kt1_after.json` | d031134f6bbbeb07 / 61f0b28ed0c3e111 |
| `kt1_rhsall.json` / `kt1_etau_exp.json` / `kt1_plant_htau.json` | 23d1cb4adbeb5712 / 096a6ba48e7e89a4 / 2bef47c029b528db |
| `etau_offline.json` / `libm_vs_vector.json` | 066ca8cd2d19f610 / a080ee7ff2eb76f0 |
| `entry_kt2.py` / `entry_kt2.json` | 9fd667ff8e864711 / 5cb435fb3c1f41e1 |
| `kt2_probe.json` / `sh2_probe.json` (round-6 probes, re-run) | b9f4db50a30beeeb / 32da00eab3aa5079 |
| `v1_independent_kt1.json` / `_kt1_5.json` / `_kt6_10.json` | e0e58c9207cc4fdb / af59dbccc64c0fc4 / 6c7bdaeec68ad99d |
| `v1_given_entries.json` | 37a2058910fc6e58 |

## 1. Task 1 — the round-6 flags' wiring is now pinned

Each new test drives the production method and fails when its wiring is cut:

| wiring | new test | plant | result |
|---|---|---|---|
| step-entry N2 bundle passes `source_rounded` | `test_tke_carried_coefficients.py::test_bn2_source_rounded_flag_reaches_the_step_entry_bundle` | argument forced `False` | red (spy saw `[False, False]`) |
| shear call site reads `_nemo_sh2_eta` | `test_nemo_sh2_step_entry_ssh.py::test_production_shear_call_site_routes_through_the_selector` | call site passes the stage eta | red (`0.3 is not 0.1`) |

Both plants were reverted; `git status --porcelain` clean after each.

## 2. Task 2 — kt-1 TKE energy walk (zdftke.f90, compiled order)

Instrument: `probe_kt1.py` runs the card's kt-1 step plain and with the
write-only TKE statement trace (passive: every leaf identical), and scores the
post-`zdf_phy` en against NEMO's kt-1 frame (`oracle_zdf_entry_kt00000001.bin`).
`probe_lit.py` is NEMO's recurrences in numpy, one binary64 rounding per
Fortran operation. At kt 1 the ocean is at rest: shear is 0 everywhere and
the entry en is 1e-6 in every wet cell.

Walk, in `tke_tke` order (`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdftke.f90`,
SUBROUTINE `tke_tke` :199):

| statement | lines | measured |
|---|---|---|
| surface value en(1) | :270-276 | 0 unequal / 16,433 |
| Langmuir source; matrix and RHS | :311-378; :409-427 | exact: the literal chain below closes on the card's own traced matrix and RHS |
| RHS association | :422-426 | private all-boundary arm: no change (31,802 before and after) |
| first and second recurrences, back-substitution | :458-473 | literal replay vs card's post-solve: **3,072 unequal** |
| floor and W mask | :474-476 | part of the solve above |
| nn_etau=1 penetration | :498-502 | card's increment non-bit (below) |

Closure: literal solve plus literal penetration on the card's traced matrix
and RHS equals NEMO's kt-1 en in **all 414,119 wet W cells** (`etau_offline.json`).
Each owner alone: literal solve only leaves 31,180 cells; literal penetration
only leaves 1,005.

**First non-bit statement: the recurrences** `zdftke.f90:466` and `:472`. The
card's code is in source order, but under jit XLA contracts
`zd_lw - (zd_lw/zdiag)*zd_lw` and `zd_lw - zd_up*en` into fused multiply-adds.
NEMO's build has no FMA. Same class as round 6's N2.

**Second: the penetration term** (`:500`) and its depth scale `htau`
(`:864`, SUBROUTINE `zdf_tke_init` :732):
- `htau = MAX(0.5, MIN(30, 45*ABS(SIN(rpi/180*gphit))))`. The card took
  `sin(deg2rad(degrees(lat_T)))`. Its latitude differs from NEMO's `gphit` in
  3,337 cells; `htau` differs in 1,192 columns.
- `EXP`: the V1 and ORCA2 builds link scalar glibc `exp`/`sin`
  (`-fno-tree-vectorize`, no libmvec symbols). XLA's `exp` differs from glibc
  in 59,798 of V1's 414,119 arguments; XLA's `sin` matches glibc on all
  26,640 phases.
- The repo's DINO transcriptions (glibc vector `SIN`/`EXP`) do **not** match
  this build: 15,718 sines and 180,757 exps differ from scalar glibc. So they
  were not reused. The arm without `htau` (literal `EXP` only) made en worse:
  31,802 -> 55,075.

Both are transcription defects. **Landed** (`f2dc0dd4d`), V1-selected only:
- `TKEConfig.nemo_tke_solve_source_rounded` rounds the two products through
  the existing `nemo_source_round` guard;
- new arm `'nemo_scalar_libm'` of the existing `tke_htau_evaluation` and
  `tke_etau_exponential_evaluation` selectors. It uses the native-degree
  phase with the folded `rpi/180` and the precision policy's scalar libm
  `sin`/`exp`, and refuses the native policy;
- the V1 card's grid carries NEMO's `gphit` (`native_lat_T_deg`).

After landing, kt-1 en: **31,802 -> 0 unequal** (P1 CONFIRMED,
`kt1_after.json`). End-to-end plant: reverting only `htau` gives 20,605
unequal (`kt1_plant_htau.json`).

Unit plants (`test_nemo_tke_v1_kt1_statements.py`, 6 tests): reverting the
solver rounding turns the jit-vs-literal test red, as does reverting `EXP` to
XLA's. Reverting the `SIN` routine to XLA's stays green on this host, as
measured above. That statement's defect is the latitude operand, which the
end-to-end plant covers.

Control note: the card's penetration function compiled alone differs from the
in-model compile in 19,428 cells. So it is not an exact control. The
decisive check is the closure above.

## 3. Both labels, kt 1..10

kt 1, both labels: **0 unequal on every row**. Surface ssh, `uu_b`, `vv_b`,
u, v, T, S were 0 / 0 / 0 / 560 / 525 / 169 / 147 in round 6. TKE frame en,
avm, avt were 31,802 / 16,622 / 15,927; now 0 / 0 / 0. kt-2 entry state vs
NEMO's kt-2 step-entry frame, all 30 levels: T, S, u, v, ssh 0 unequal
(`entry_kt2.json`). P2 CONFIRMED.

INDEPENDENT. Each entry is unequal cells / full-depth max abs (partial-cell
max). Every row is DEBT.

| kt | en | avm_k | avt_k | dissl | T |
|---:|---|---|---|---|---|
| 2 | 47,374 / 5.57e-07 (6.9e-18) | 30,354 / 1.17e-05 (2.2e-16) | 28,304 / 9.55e-06 (2.2e-16) | 24,078 / 1.32e-07 (1.7e-18) | 13,802 / 2.95e-04 (3.6e-15) |
| 3 | 67,425 / 1.04e-06 (4.8e-16) | 51,339 / 3.89e-05 (1.4e-14) | 51,177 / 3.89e-05 (1.4e-14) | 45,364 / 4.24e-06 (8.5e-16) | 52,217 / 2.33e-04 (2.1e-09) |
| 4 | 81,962 / 4.68e-06 (1.1e-12) | 70,297 / 5.97e-05 (1.9e-10) | 70,478 / 2.66e-05 (1.5e-06) | 80,376 / 7.94e-06 (8.6e-12) | 71,293 / 1.86e-04 (1.7e-05) |
| 5 | 89,372 / 3.77e-06 (5.0e-07) | 79,034 / 8.38e-05 (4.5e-10) | 79,232 / 8.38e-06 (2.8e-05) | 94,792 / 6.73e-06 (1.3e-07) | 86,161 / 1.85e-04 (3.3e-04) |
| 6 | 96,332 / 2.29e-06 (1.0e-06) | 86,784 / 1.09e-04 (3.6e-06) | 87,145 / 1.09e-04 (8.1e-05) | 107,297 / 1.04e-05 (2.5e-06) | 101,516 / 4.86e-04 (1.2e-03) |
| 7 | 101,533 / 1.85e-06 (1.4e-06) | 93,394 / 5.72e-05 (5.1e-05) | 94,058 / 5.72e-05 (9.0e-05) | 119,433 / 1.24e-05 (9.4e-06) | 118,269 / 8.70e-04 (2.2e-03) |
| 8 | 108,645 / 1.41e-06 (2.6e-06) | 99,833 / 4.23e-05 (1.4e-04) | 100,730 / 4.23e-05 (1.4e-04) | 134,572 / 1.35e-05 (1.7e-05) | 138,779 / 1.43e-03 (3.7e-03) |
| 9 | 113,141 / 2.23e-06 (5.1e-06) | 105,649 / 1.06e-04 (2.6e-04) | 106,709 / 1.06e-04 (2.6e-04) | 153,530 / 1.96e-05 (2.8e-05) | 155,625 / 2.48e-03 (6.3e-03) |
| 10 | 119,428 / 5.11e-06 (9.4e-06) | 110,840 / 2.45e-04 (5.0e-04) | 112,046 / 2.45e-04 (5.0e-04) | 168,960 / 3.18e-05 (4.9e-05) | 171,605 / 4.32e-03 (1.1e-02) |

Round 6, same windows: kt-2 en 73,757 cells, kt-10 en 131,163. Unequal
counts fell 9-36%. Every maximum is unchanged, so the maxima belong to a
different owner. Surface rows at kt 2: ssh, `uu_b`, `vv_b` 0 unequal;
u 5,102 / 2.16e-5; v 4,674 / 7.70e-6; T 2,731 / 2.95e-4; S 2,369 / 5.70e-5.
Surface ssh is first over the bar at kt 3: 14,368 cells / 3.05e-7.

GIVEN-NEMO-ENTRY (two-step projection; full-depth max abs, unequal cells):

| entry -> score | en | avm_k | avt_k | T |
|---|---|---|---|---|
| 1 -> 2 | 5.57e-07 (47,247) | 1.17e-05 (30,279) | 9.55e-06 (28,228) | 2.95e-04 (13,784) |
| 3 -> 4 | 4.92e-06 (71,595) | 6.36e-05 (55,106) | 2.74e-05 (55,153) | 1.88e-04 (54,245) |
| 5 -> 6 | 1.77e-06 (74,888) | 8.68e-05 (58,211) | 8.68e-05 (58,467) | 4.80e-04 (58,002) |
| 7 -> 8 | 1.06e-06 (76,454) | 4.22e-05 (60,074) | 4.22e-05 (60,624) | 8.93e-04 (60,330) |
| 9 -> 10 | 2.04e-06 (79,144) | 6.90e-05 (62,326) | 6.90e-05 (62,943) | 1.79e-03 (62,572) |

The kt-1 given-entry step has no unequal cell. P3 is FALSIFIED: the kt-2 en
maximum stays 5.57e-7. P4 is CONFIRMED: not certified.

**V1 is NOT certified.**

## 4. Retraction (round 6)

Round 6's review concluded that the kt-2 en residual "comes from the inputs —
the kt-1 last-bit u/v/avm". That is false. The kt-2 entry state and the kt-1
viscosity are now bit-identical to NEMO, and the kt-2 maximum is unchanged.

## 5. Next owner (registered, PLAUSIBLE, not landed)

At kt 2 the over-bar en is **entirely on the north-fold pivot row** (j = 147):
76 cells there, max 5.57e-7 at (147,133), w-level 2. Every other row is at most
2.1e-17. At the first interior level the NEMO-backed RHS misses the card's by
up to 7.5e-3 relative, only in mirror pairs on that row: columns 52/128,
53/127, 54/126, 55/125. With exact inputs the interior shear (fold row and
seam excluded) matches the NEMO-literal shear except 10 cells, at most
4.7e-16 relative (`sh2_probe.json`). At kt 1 the shear is zero, which is why the
fold row only shows from kt 2. Candidate: the fold row's shear or energy
(`zdf_sh2`'s pivot-row value, or the north-fold exchange of `en`/`p_sh2`).
Unmeasured.

## 6. Decision 122 (ORCA2_OMIP_L4)

`zdftke.f90` is byte-identical between the V1 build and `ORCA2_OMIP_L4`
(SHA-256 `0f290190a33e4713…`). The `ORCA2_OMIP_L4` executable has the same
flags (`-O3 -fno-tree-vectorize`) and links the same scalar
`exp@GLIBC_2.29` / `sin@GLIBC_2.2.5`.

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| second recurrence `zd_lw - zd_lw/zdiag*zd_lw` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:466` | `tke_tke` (:199) |
| back-substitution `(zd_lw - zd_up*en)/zdiag` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:472` | `tke_tke` (:199) |
| nn_etau=1 `EXP(-gdepw/htau)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:500` | `tke_tke` (:199) |
| `htau = ... SIN(rpi/180*gphit)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:864` | `zdf_tke_init` (:732) |

Pointer only: the full ORCA2 card was not measured with these selections.

## 7. Selection and caller grep (no-silent-default rule)

- The flag and both `'nemo_scalar_libm'` arms are set only in the V1 card's
  vertical block. The V1 selector pins all three, and the selector check fires
  on each when planted back. G1-G5, the full ORCA2 card, DINO, GYRE and every
  library default are unchanged. The V1 config-delta test names the three new
  fields, and G5's grid has no native latitude.
- `_nemo_literal_tke_solve` has one production caller, which passes the flag.
  The model's native-latitude requirement now covers both literal `htau` arms,
  so a card without native latitudes raises.
- **RULE 3:** every fix defaults off on other cards. D130 already answers this,
  so there is no new question.

## Tests

All runs synchronous, CPU, fp64:

| suite | result |
|---|---|
| V1 card + bn2/sh2/new statement units + TKE carried/identity/terms/under-ice/dry-wmask/closure/prognostic/integration + bn2 + Roquet EOS | 311 passed |
| G2 ladder/card + G4 + G5 + G5 top-drag + GYRE TKE walks (rounds 54, 103, 107) + validate-strict, not slow | 189 passed, 11 deselected |
| `tests/test_dispatch_hardening.py` | 1 failed (`slab_land.py step_land`). Pre-existing: it fails identically on a clean worktree of `4ad0634ab` |
| citation gate + private imports, clean tree at `516ce4cb8` | 23 passed |

Review: **codex review: paused (quota)**. GLM unavailable. No independent
review has run on this diff yet; the operator's Claude review is pending.
**UNREVIEWED until then.**

## Choices made

ASKED (this round's brief): wiring tests; the kt-1 en walk; landing
transcription defects V1-selected; re-scoring both labels.

UNASKED, offered for revert:
- `'nemo_scalar_libm'` refuses the native precision policy. That is a new
  hard error, but only on the new arm.
- Extending the two existing selectors with a third value, rather than adding
  new fields.
- Measuring the kt-2 fold-row location (diagnosis only, nothing landed).

UNVERIFIED:
- the kt-2 fold-row owner (section 5);
- that XLA's `sin` keeps matching glibc on other hosts. The landed arm calls
  glibc itself, so V1 does not depend on it;
- the full ORCA2 card's response to these selections.
