# Vertical-mixing lane — V1 round 9 (vmix_r9)

Status: **HELD. V1 is not certified.** Round 9 landed one fix: the TKE matrix diagonal and right-hand side now use NEMO's
`-O0` arithmetic. kt-2 TKE energy misses fall from 47,358 cells to 2. kt-2 surface u misses fall from 5,048 cells to 0.
Every kt-2 field is now AT-BAR under both labels. The first over-bar row is now kt 3: sea level (189 cells, max 2.39e-15 m
absolute, 1.29e-15 normalised), T (13 cells) and diffusivity (12 cells).

Decisions in force: D129 = 1, D130 = 1. Codex review: paused (quota).

Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v1_round9.md`, frozen at `21ff432af` with addendum A at
`4c4b1d6b1`. Both were committed before the measurement they predict.

Commits:
- `21ff432af` preregistration;
- `4c4b1d6b1` addendum A (first candidate falsified, second candidate frozen);
- `4e4f794d7` landing: `TKEConfig.nemo_tke_matrix_rhs_source_rounded` (V1-selected).

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3).
Evidence: `gridladder_vmix_rounds/round9/` (`manifest.sha256`, 40 files). The ladders and traces ran on the package
diff before it was committed. The committed `tke.py` is `cmp`-identical to `tke_landed.py.bak`, which is the file the
ladders ran.

| file | sha256 prefix |
|---|---|
| `probe_avm_seam.py` / `seam_kt2.json` | dfbde7daaed3fc5f / 1f22af59d7373cad |
| `check_r3_forms.py` / `fix_kt1_2.json` (r3 candidate) | a23ca88c4f7ba8e5 / e6032f91d9fbbf29 |
| `probe_tke_trace.py` / `replay_tke_local.py` | 059383c1df53216c / 272c8d5d6861bfc2 |
| `tke_local_kt1.json` (control) / `tke_local_kt2.json` / `tke_local_kt2_fix4.json` | 8980dfed92b7dce3 / 0b5a03bc5b9df20a / b2b4e9a38cd83781 |
| `solve_arms.py` / `solve_arms_kt2.json` | 7aa726a7d0e0e271 / 2dfbd92128b73d28 |
| `v1_independent_kt1_5.json` / `_kt6_10.json` / `v1_given_entries.json` | d7060577e2ff30fe / d6c9a414829d9c89 / 09e9edd5a03a7bd2 |

## 1. kt-2 surface u: the momentum solve is exact; the debt comes from TKE

The ladder prints the absolute miss, 2.22e-15. Normalised by max|u| = 1.31 m/s, the miss is **1.70e-15** at (102,43),
in 5,048 cells. The kt-1 state at HEAD is leaf-identical to round 7's saved state (23 leaves; `kt1_state_check.py`).

Instrument `probe_avm_seam.py`: one kt-2 step from that state. The closure's output is replaced at the point the solves
consume it. Each arm is a fresh model.

| arm | u | v | T | S |
|---|---|---|---|---|
| plain | 44,070 unequal | 34,599 | 13,549 | 11,771 |
| card's own viscosity (identity control) | bit-identical to plain | | | |
| NEMO's kt-2 `avm_k` | **0** | **0** | 13,549 | 11,771 |
| NEMO's kt-2 `avt_k` | 44,070 | 34,599 | **0** | **0** |
| both | 0 | 0 | 0 | 0 |

CONFIRMED: the following run bit-for-bit on V1 at kt 2:
- `dyn_zdf` (`dynzdf.f90:67`), including the RK3 surface stress `:338` and the ice top drag;
- `tra_zdf`.

The first non-bit operand that TKE hands the momentum solve is the viscosity (30,324 unequal cells at kt 2).

## 2. First candidate: the shear's sea-level ratio (falsified, reverted)

NEMO computes the shear's ratio as `r3u = 0.5*(e1e2t*ssh + ..)*r1_hu_0*r1_e1e2u`, with `r1_e1e2u = 1/e1e2u`
(`domqco.f90:188-234`; `domhgr.f90:170`). The card's step-entry shear builder divides by `e1e2u` instead. That is a
real transcription difference. It changes `r3` on 4,002 U and 3,909 V faces, but changes `1 + r3` (what the shear
reads) on 0 U and 3 V faces. kt-2 en, avm and u came out bit-identical (`fix_kt1_2.log`). **P1 falsified.** The
change was reverted (`git status --porcelain` clean) and nothing landed. It is registered as a zero-effect literal
difference, offered below.

Also measured: every wet cell has nonzero shear at kt 2 (414,119 of 414,119). Round 8's observation that "every unequal
en cell has nonzero shear" therefore carried no information.

## 3. The owner: `tke_tke`'s zdiag and RHS (landed)

Instrument: the card's kt-2 `tke_tke` trace, written out passively (the step is bit-identical to plain). Each statement
is replayed in numpy in NEMO's written order with one rounding per operation. The V1 build is gfortran `-O0`
(`arch_nemo.fcm`), so it has no FMA.
- **Control:** at kt 1 the replay reproduces the card's RHS and diagonal in every cell (`tke_local_kt1.json`).
- kt 2: the replay differs from the card's diagonal (`zdftke.f90:420`) in 354,867 cells and from the card's RHS
  (`:423-426`) in 109,781 cells.
- Re-solving with round 7's literal solver, keeping the card's post-solve increment (`solve_arms.py`), kt-2 en
  unequal vs NEMO:
  - card diagonal and RHS: 47,358;
  - NEMO-order diagonal: 24,030;
  - NEMO-order diagonal and RHS: 16,649.

The first in-model attempt (addendum A) put rounding barriers on the diagonal's operations. The traced diagonal did not
move: en stayed at 46,099, the count round 8's RHS-only arm already gave, so **P1b was falsified**. A `+1` plant proved
the branch does run. The cause is a constant, measured in one cell: the card's dissipation weight is
**11339.999999999998**, while NEMO's `zfact2 = 1.5_wp*rn_Dt*rn_ediss` (`zdftke.f90:247`, SUBROUTINE `tke_tke` :199)
is **11340.0**. Under jit, XLA reassociates `(dt*1.5)*0.7` into `dt*(1.5*0.7)`, and `1.5*0.7` rounds to
1.0499999999999998. The result is a one-ulp error on every wet diagonal entry from kt 2 on. At kt 1 the dissipation
term is about 1e-12, which is why the kt-1 control could not see it.

**Landed** (`4e4f794d7`), V1-selected only: `TKEConfig.nemo_tke_matrix_rhs_source_rounded`.
- It computes `zfact2 = r(r(1.5*dt)*rn_ediss)` and builds `zdiag` with each operation rounded.
- The RHS reuses the existing all-boundaries materialization, which follows NEMO's written order.
- After the landing, the replay equals the card's diagonal and RHS in every cell (`tke_local_kt2_fix4.json`).
- kt-2 en vs NEMO: **47,358 -> 2**.

## 4. Both labels, kt 1..10

kt 1, both labels: 0 unequal on every row (P2b CONFIRMED).

INDEPENDENT. Columns en to T give unequal cells / full-depth normalised max. The ssh column gives unequal cells / the
ladder's printed max, which is ABSOLUTE (m), not normalised.

| kt | en | avm_k | avt_k | dissl | T | ssh |
|---:|---|---|---|---|---|---|
| 2 | 2 / 2.2e-19 | 1 / 1.8e-18 | 3 / 7.3e-18 | 2 / 8.7e-19 | 5 / 3.7e-17 | 0 / 0 |
| 3 | 11 / 9.6e-17 | 10 / 5.6e-16 | 12 / 8.36e-14 | 9 / 2.9e-17 | 13 / 6.46e-15 | 189 / 2.39e-15 |
| 4 | 302 / 1.4e-16 | 216 / 8.1e-16 | 442 / 7.9e-16 | 169 / 9.5e-16 | 696 / 2.38e-15 | 1,384 / 6.7e-16 |
| 5 | 2,342 / 3.1e-16 | 1,919 / 7.00e-14 | 2,853 / 2.46e-12 | 1,765 / 3.10e-15 | 3,103 / 6.50e-15 | 3,826 / 7.15e-14 |
| 6 | 7,211 / 1.23e-15 | 6,308 / 1.00e-13 | 8,051 / 1.32e-11 | 5,987 / 4.78e-15 | 7,841 / 4.17e-14 | 6,575 / 4.55e-14 |
| 7 | 15,144 / 6.53e-15 | 13,423 / 9.72e-14 | 15,693 / 3.30e-12 | 13,143 / 1.37e-14 | 14,932 / 4.14e-14 | 10,428 / 3.24e-14 |
| 8 | 26,183 / 3.50e-15 | 23,342 / 1.59e-13 | 26,277 / 6.06e-12 | 23,173 / 1.81e-14 | 25,412 / 8.48e-14 | 13,063 / 6.08e-14 |
| 9 | 39,797 / 6.01e-15 | 36,262 / 1.21e-13 | 39,071 / 2.55e-12 | 36,778 / 2.24e-14 | 37,122 / 1.05e-13 | 13,988 / 4.10e-14 |
| 10 | 52,252 / 1.60e-14 | 48,302 / 3.22e-13 | 50,851 / 2.71e-12 | 50,347 / 2.94e-14 | 48,573 / 7.22e-14 | 14,871 / 7.91e-14 |

Round 8, same windows: kt-2 en 47,231 cells; kt-2 u 5,048 cells; kt-3 ssh 14,148 cells (1.17e-14); kt-10 en 106,823
cells; kt-10 T 7.31e-13.

kt 2: every surface row and every level row is AT-BAR. Surface u, ssh, `uu_b` and `vv_b` are exact (P4b CONFIRMED).
First over the bar: kt-3 ssh, T and `avt_k`.

GIVEN-NEMO-ENTRY (kt 1/3/5/7/9):
- Every single step is AT-BAR on every surface row. The only unequal cells are u/v/T/S at kt 5 (2 cells each at most)
  and T at kt 9 (2 cells).
- The only DEBT is two-step kt 6: T, 2 cells, 3.64e-14, plus level rows T/avm/avt/dissl (6-14 cells).
- Round 8: u at 3.44e-15 and T/S at 8-24e-14 in every single step.

**P5b FALSIFIED: V1 is NOT certified.** P1b (exactly 16,649 cells) was falsified in the other direction: 2 cells
remain, not 16,649, because the zfact2 constant was the larger part.

## 5. Next owner (registered, PLAUSIBLE, not landed)

- kt-2 residue: en 2 cells, avm 1 cell, avt 3 cells, T 5 cells, S 4 cells.
- kt-3 sea level (189 cells) comes from the kt-2 T/S residue. MEASURED by the independent review (scratch, not
  committed): swapping in NEMO's kt-2 T/S makes kt-3 sea level exact; swapping in NEMO's kt-2 TKE fields leaves all 189
  cells; full substitution gives an exact kt 3 (control).
- That the T/S residue itself comes from the 3-cell kt-2 `avt` residue is PLAUSIBLE, not measured.
- kt-3 `avt_k` (12 cells, 8.4e-14) is amplified through the Prandtl-number update (`zdftke.f90:717`). That is a
  consequence of the residue, not a new owner.
- Next round: a cell-level replay of the 2 kt-2 en cells, through the Langmuir source, the penetration increment and
  the mixing length.

## 6. Decision 122 (ORCA2_OMIP_L4)

`zdftke.f90` (`0f290190a33e4713`), `dynzdf.f90` (`f90e8cf3ffc4bf3d`) and `domqco.f90` (`9a3b302cd3f8868f`) are
byte-identical between the V1 build and `ORCA2_OMIP_L4`.

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| `zfact2 = 1.5_wp * rn_Dt * rn_ediss` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:247` | `tke_tke` (:199) |
| `zdiag = 1 - zzd_lw - zzd_up + zfact2*dissl*wmask` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:420` | `tke_tke` (:199) |
| `en = en + rn_Dt*( p_sh2 - p_avt*rn2 + zfact3*dissl*en )*wmask` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:423-426` | `tke_tke` (:199) |
| `p_avm = MAX( zav, avmb ) * wmask` (handed to the momentum solve) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:709` | `tke_avn` (:531) |
| `zzwi/zzws = -zDt_2*(avm+avm)/..`; `puu(1) += rDt*utauU` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:192-201,338` | `dyn_zdf` (:67) |
| `pr3u = 0.5*(..)*r1_hu_0*r1_e1e2u` (not landed) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:188-234` | `dom_qco_r3c` (:188) |

Pointer only: the full ORCA2 card was not measured with this selection.

## 7. Selection, purity, tests

- `nemo_tke_matrix_rhs_source_rounded` is set only in the V1 card's vertical block. It is pinned by the V1 selector
  (the selector check fires when it is planted `False`) and named in the V1 config-delta test. G1-G5, the full ORCA2
  card, DINO, GYRE and every library default are unchanged (default `False`). **RULE 3:** D130 already answers the
  default question.
- Flag-off purity (`probe_jaxpr.py`, round 8's instrument): one-step `make_jaxpr` bodies.
  - Full ORCA2: base `e0351d089a027297`, landed `e0351d089a027297`, **identical**.
  - DINO: base `89418f6a0b2eabae`, landed `89418f6a0b2eabae`, **identical**.
  - These are the same hashes as round 8's base.
- New test `test_nemo_tke_matrix_rhs_source_rounded.py`:
  - x64 guard;
  - flag on under `jax.jit` with traced `dt`: zdiag and RHS equal NEMO's written-order numpy bit for bit, and
    eager == jit;
  - the flag defaults off.
- Plants (`plant_r9.py`, `plants_r9.log`), one statement reverted each, restored with `cp -f` and `cmp`-clean:

| reverted statement | result |
|---|---|
| `zfact2` unrounded `(1.5*dt)*c_eps` | 1 failed (eager != jit) |
| RHS selection back to `elif rhs_materialization:` | 1 failed (eager != jit) |

All runs synchronous, CPU, fp64:

| suite | result |
|---|---|
| new test + fold/sh2/htau + V1 card + TKE terms/carried + GYRE r103/r107 TKE walks | 170 passed |
| citation gate + private imports, clean tree at `4e4f794d7` | 23 passed |
| recipe, freshwater, face-mask rank, prognostic barotropic | **NOT RUN**: killed at the 580 s timeout twice, with the machine at load 25-43 from other lanes; left to the push gate |

Review: **codex review: paused (quota)**. GLM unavailable. Independent Claude review: **HOLD** on three mechanical
items, physics CONFIRMED. All three fixed in the review-fix commit (section 8).

## 8. Independent-review fixes

- **Guard.** The flag on the factored matrix ran and changed nothing. It now raises unless the literal matrix is
  selected; a test asserts the raise and fails with the guard reverted (`DID NOT RAISE`).
- **Citation.** The diagonal statement is `zdftke.f90:420` in both builds (`:419` is `zd_lw`). Fixed here and in the
  code comments; the fix commit `4e4f794d7`'s message still says `:419`. The citation gate's default receipt is the GYRE
  one, so no pinned anchor moved.
- **Label.** kt-3 sea-level max 2.39e-15 m is absolute; normalised it is 1.29e-15 (status line and section 4 header).
- **kt-3 attribution** (reviewer's measurement): recorded in section 5.

## Choices made

ASKED (this round's brief): the walk from surface u into TKE; landing the transcription defect V1-selected; re-scoring
both labels.

ASKED (review fixes): the factored-matrix flag becomes a hard error.

UNASKED, offered for revert:
- one flag covers both statements (diagonal and RHS) instead of one flag per statement;
- the RHS reuses the private all-boundaries materialization as its production form when the flag is set;
- the shear `r3` reciprocal-area difference (section 2) is registered and not landed, because it has zero kt-2 effect.

UNVERIFIED:
- the owner of the 2 remaining kt-2 en cells;
- that XLA's reassociation of `(dt*1.5)*0.7` is version-stable. The test pins the result, not the mechanism;
- the full ORCA2 card's response to this selection;
- the four suites listed as NOT RUN.
