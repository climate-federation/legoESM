# Grid ladder rung G6 — round 7 receipt (gridladder_r43)

Status: **LANDED (two G6-selected statements); G6 NOT CERTIFIED.**  The
walk inside `stp_2D` found the first non-bit operand at kt 3 (from NEMO's
exact restart) to be the first `wzv`'s `ww`, then `dyn_zad`.  Both are
correctly transcribed; under `jax.jit` XLA re-associated them.  With both
landed, kt 3's `stp_2D` exit is bit-exact (was 1,997 / 1,980 cells), the
given-entry kt-3 and kt-5 sea level and barotropic transports are exact,
and the two kt-4 all-level DEBT rows are at the bar.  Remaining DEBT:
independent from kt 4 (first row: kt 4 ssh, 2.25e-15 normalised), and one
given-entry row, kt-6 two-step v (1.023e-15, 2,183 cells).

Preregistration: `PREREG_nemo_testcases_gridladder_g6_round7.md` (commit
`0f9f56548`).  DEVIATION: the kt 1-3 walk ran before it; the landing and
the kt 1..6 re-score ran after.  Record: round 42
`gridladder_rounds/round42/oracle_g6_r42/twin_s` (acquisition PASSED:
`G6R42_ACQUISITION_PASS steps=1..4 terms=kt1..3`; passivity: round-36
record files kt 1-4 byte-identical, restarts 2/4 equal in all 27
variables), for the walk; round 36 `oracle_g6_r36` for the ladder.
Evidence: `gridladder_rounds/round43/` (`final/`, `walk/`).  Runs stamped
`55b7f55e13e8` (clean tree).  SHA-256 prefixes: given kt 3-6
`1d065ea272ae416e`, independent kt 1-3 `b8b1a45e50e03932`, kt 4-6
`ca95efee4698081b`, stp_2D walk kt 1-3 `48dcb25f84ec89e1`.

## 1. The walk (new ladder arm `stp2d_terms`)

Every card value comes from existing write-only observers of the ordinary
step (the step-entry term observer, the ZAD-operand observer, the
barotropic-substep exposure); no operator is re-implemented in the arm.
Bitwise counts on NEMO's wet points; dry points reported apart (0 everywhere).

| boundary, compiled order (`stp_2D`) | kt 1 | kt 3 given, before | kt 3 given, after |
|---|---:|---:|---:|
| after `dyn_hpg` u / v | 0 / 0 | 0 / 0 | 0 / 0 |
| after `dyn_vor` u / v | 0 / 0 | 0 / 0 | 0 / 0 |
| first `wzv` `ww`, levels 2-30 | (see §3) | 90,618 | 22 (2 columns) |
| after `dyn_keg` u / v | 0 / 0 | 0 / 0 | 0 / 0 |
| after `dyn_zad` u / v | 0 / 0 | 4,808 / 4,439 | 0 / 2 |
| depth mean `Ue_rhs` u / v | 0 / 0 | 42 / 32 | 0 / 0 |
| after drag, after wind u / v | 0 / 0 | 40,36 / 31,24 | 0 / 0 |
| `dyn_spg_ts` exit uu_b / vv_b | 0 / 0 | 1,997 / 1,980 | **0 / 0** |

kt 1 is the instrument's calibration (all exact).  kt 2 (independent, the
card's own kt-1 state, 1 v cell off) after landing: exit 109 / 103 cells,
≤6.8e-17 normalised.

How the two statements were separated from transcription errors:

- `ww`: an offline numpy replay of NEMO's `div_hor_RK3_t` + `wzv_RK3_t`
  from restart 2 and the mesh equals NEMO's `ww` on levels 2-30 (0
  unequal).  The card's own function equals it EAGERLY (0) and differs
  under `jit` (90,611 values, 5,744 columns).  Its inputs (per-level
  divergence, both ratios) are equal eager vs jit; the stretch product is
  not.  Materialising `r3_after`/`r3_before` with the shared
  `nemo_source_round` makes the jitted `ww` exact (0).  PLAUSIBLE
  mechanism: XLA strips `optimization_barrier` and factors
  `ssh_a*r1_h0 - ssh_b*r1_h0` into `(ssh_a - ssh_b)*r1_h0`.
- `dyn_zad`: a numpy replay of NEMO's statements from NEMO's own after-KEG
  RHS, `ww` and restart velocity equals NEMO's after-ZAD (0 / 0 unequal);
  with the card's `ww` it still equals NEMO (0 / 2); the card's `dyn_zad`
  differed from it on 3,376 / 2,993 points.  The card wrote
  `-0.25/(e1e2u*e3u)*(G_top+G_bot)` and relied on stripped barriers.

So both are last-bit association differences of correctly transcribed
statements, fixable to exact; both landed.

## 2. Landed (`55b7f55e1`)

| NEMO statement (SUBROUTINE) | G5 r31 build / ORCA2_OMIP_L4 |
|---|---|
| `r3t(Kaa) = ssh(Kaa)*r1_ht_0`, stored (`stp_2D`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:152` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:152` |
| `r3t(Kbb)`: `pr3t = pssh*r1_ht_0`, stored (`dom_qco_r3c_RK3`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/domqco.f90:257` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:257` |
| `pww(jk) = pww(jk+1) - (ze3div + r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb)))*tmask` (`wzv_RK3_t`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/sshwzv.f90:297-298` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297-298` |
| `zWf = e1e2t*ww(jk+1)` (`dyn_zad`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzad.f90:112` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90:112` |
| `zzWdzU = zzWfu*(puu(jk) - puu(jk+1))` (`dyn_zad`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzad.f90:119` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90:119` |
| `puu(Krhs) - 0.25*r1_e1e2u/e3u*(zWdzU + zzWdzU)` (`dyn_zad`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzad.f90:123` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90:123` |
| bottom level, same coefficient (`dyn_zad`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzad.f90:134` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90:134` |
| `e1e2u = e1u*e2u`; `r1_e1e2u = 1/e1e2u` (`dom_hgr`) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/domhgr.f90:159`, `:170` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:159`, `:170` |

`domqco`, `sshwzv`, `dynzad`, `domhgr` are byte-identical between the two
builds.  The ORCA2 `domain_cfg` carries no `e1e2u`, so `:159` runs.

Code: two config bools, `nemo_wzv_stored_r3_ratios` (the ratios pass
through `nemo_source_round` inside the ONE shared `nemo_qco_wzv_operands`,
at its step-entry call and its two RK3 stage calls) and
`nemo_zad_written_association` (the ONE shared
`nemo_advective_vertical_momentum_advection`: `e1e2t*ww`, the interface
term, their sum and `(0.25*(1/e1e2u))/e3u` each rounded).  Defaults False.
Caller grep of both field names: the G6 branch of the card file only; the
non-RK3 `wzv` call site keeps the default.  With False neither new branch
is traced, so every other card's program is unchanged by construction (not
diffed jaxpr by jaxpr).

Tests (each fails when its fix is reverted):
- `test_g6_card_is_g5_plus_g1b_vector_selectors` pins both fields; with
  the zad selection reverted the card refuses to build
  (`disagrees with its deck: ['zad written association']`).
- `test_first_wzv_is_nemos_under_jit_only_with_stored_ratios` (r42 record,
  traced jit): ww levels 2-30 vs NEMO, 0 unequal with the option, >1000
  without (90,611 measured).
- `test_nemo_zad_written_association.py` (traced jit vs a numpy
  transcription): exact with the option; with the branch disabled the
  exactness test FAILS (restored, `grep` shows the branch back).
- Arm tools: record reader and bitwise scorer (with plant), and the shared
  slow-forcing reader's new `(kt, Kbb, Krhs)` argument.

## 3. Registered, not changed

- First `wzv`, level 1 (`ww(1)`): NEMO's `div_hor` adds the runoff
  divergence (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/divhor.f90:142`,
  SUBROUTINE `div_hor_RK3_t`); the card's step-entry call passes none
  (4,671 surface values at kt 3, up to 5.6e-7 m/s).  `dyn_zad` never
  reads `ww(1)` (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzad.f90:119`
  reads `jk+1`), so no field moves.
- kt-1 after-slot: NEMO's first `ww` at kt 1 is nonzero (−8.6e-6 m/s):
  its `ssh(Kaa)` differs from the step entry on 201 cells (max 0.1 m vs
  an entry of −0.093 m).  Round 42's "from-rest fill" reading
  (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/restart.f90:406`) is
  therefore not the whole story for this operand.  Field-invisible: kt-1
  velocity is zero, so `dyn_zad` adds exactly 0, and the slot is rewritten
  at step end (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3.f90:241`).
- Inside `stp_2D` at kt 3, one deep `ww` column (22 values) and 2 v points
  of `dyn_zad` remain inside the full step's fusion; the standalone jitted
  `wzv` is exact there.  They do not reach the exit (0 / 0).

## 4. Both labels, kt 1..6 (level 1 + 2-D; `max_abs / unequal`)

| kt | label | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | indep. = given | 0/0 | 0/0 | 0/0 | 0/0 | 1.39e-17/1 | 0/0 | 0/0 |
| 2 | indep. = given two-step | 3.3e-16/78 | 2.8e-17/109 | 3.3e-17/103 | 1.1e-16/142 | 7.6e-17/141 | 0/0 | 0/0 |
| 3 | independent | 8.9e-16/2869 | 1.5e-16/3422 | 1.4e-16/3541 | 4.4e-16/2909 | 4.4e-16/2859 | 3.6e-15/1 | 7.1e-15/1 |
| 4 | independent | **4.66e-15**/7529 | 2.3e-16/8291 | 2.3e-16/8198 | 2.0e-15/6668 | 2.0e-15/6551 | 4.4e-16/5 | 1.1e-14/5 |
| 5 | independent | **6.97e-15**/10536 | 4.9e-16/11597 | 3.6e-16/11587 | **1.39e-14**/9234 | **1.41e-14**/9476 | 1.8e-15/10 | 7.1e-15/10 |
| 6 | independent | **3.29e-14**/13235 | **2.28e-15**/14072 | 1.0e-15/14202 | **7.41e-15**/11447 | **1.20e-14**/11787 | 7.1e-15/13 | 7.1e-15/12 |
| 3 | given (restart 2) | **0/0** | **0/0** | **0/0** | 1.1e-16/38 | 2.2e-16/46 | 0/0 | 0/0 |
| 4 | given two-step | 8.9e-16/2753 | 1.1e-16/3314 | 1.9e-16/3287 | 2.1e-15/2861 | 2.0e-15/2846 | 2.2e-16/2 | 7.1e-15/2 |
| 5 | given (restart 4) | **0/0** | **0/0** | **0/0** | 8.9e-16/45 | 2.2e-16/46 | 0/0 | 0/0 |
| 6 | given two-step | 6.7e-16/2023 | 1.1e-16/2506 | 8.3e-17/2571 | 3.8e-15/2089 | **3.89e-15**/2183 | 0/0 | 0/0 |

Absolute differences; the verdict uses the normalised error (bar 1e-15).
Bold numbers in the independent and kt-6 v rows = DEBT; bold zeros in the
given kt 3/5 rows = now exact (round 42: 1,666-2,282 cells).  Normalised
DEBT: independent kt 4 ssh 2.25e-15; kt 5 ssh 2.50e-15, u 3.54e-15,
v 2.63e-15; kt 6 ssh 1.48e-14, uu_b 1.06e-15, u 1.79e-15, v 3.14e-15;
given kt-6 two-step v 1.023e-15.  Round 42 for comparison: independent
first over the bar at kt 3 (ssh 3.57e-15), worst 1.91e-14 (kt 4 ssh) /
1.79e-14 (kt 6 v); given kt-6 v 6.84e-16 on 8,365 cells.

All-level rows (normalised, full / partial): given kt-4 two-step u
5.18e-16 / 8.6e-17, v 3.71e-16 / 1.65e-16 — both AT-BAR (round 42: DEBT
1.86e-15 / 1.38e-15 partial); given kt 6 u 9.2e-16 / 1.9e-16, v 6.8e-16 /
8.5e-17 AT-BAR; independent kt 2 and kt 4 AT-BAR (kt 4 u 6.4e-16, v
4.6e-16); independent kt 6 u 4.0e-15 / 4.2e-15, v 3.2e-15 / 3.6e-15 DEBT.
T, S AT-BAR everywhere.  Flushed subnormals (Decision 127): 0 in every run.

D96 registration: the given kt-6 two-step v row moves from AT-BAR
(6.84e-16, 8,365 cells) to 1.023e-15 on 2,183 cells — four times fewer
differing cells, its maximum 2% over the bar.  Every other row of both
labels is equal or better.  Near-zero class; the statements stay selected.

**Certification: NOT certified.**  Independent is over the bar from kt 4;
given-entry at one row (kt-6 v).

Next: walk the given-entry kt-3 stages after `stp_2D` (u 38 / v 46 cells at
step end, with an exact `stp_2D` exit), using the existing `stages` arm on
the r42 record.

## D122 pointer (ORCA2 lane)

Both statements run on ORCA2 at the ORCA2_OMIP_L4 lines in §2 (same
compiled files).  Import waits for G6 certification.

## Tests and review

All CPU fp64, `JAX_ENABLE_X64=1`, PYTHONPATH as in `claude_rounds/land.sh`.
- Default gate (land.sh list), in three chunks: citation gate + private
  imports + TKE terms 91 passed; freshwater closure + stage face-mask rank
  + prognostic barotropic state 26 passed; NEMO recipe 25 passed.
- G2-G7 cards, the G2/G6 ladder and the new `dyn_zad` unit test: 108
  passed (one field-order pin updated for the two fields appended last;
  it failed before the update, as a pin should).
- G1 card, G1b card, G1 tools, G1 ladder: 191 passed.
- Card constructibility, opt-in defaults, after-SSH carry: 57 passed.
- Receipt citation gate on THIS receipt (from its title): 21 citations,
  failures 0, unmapped 0, map audit 0; a planted shift of the
  `dynzad.f90:119` citation FAILS it (`SYMBOL-NOT-AT-LINE`).  Round-43
  entries added to the gate's maps; 78 pins and 21 lines of the GYRE
  phase-3 round-8 receipt re-anchored by the repo's re-anchor tool
  (mechanical, from `6f01650a57`); three spans the landing lengthened
  widened by hand.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

ASKED: the stp_2D walk; land the first non-bit statements G6-selected;
revert-failing tests; traced-jit last-bit asserts; both labels kt 1..6.

UNASKED, offered for revert: the citation re-anchor of the GYRE round-8
receipt (mechanical, tool-applied); the new ladder arm (needed for the walk); the
`levels` argument on the shared slow-forcing reader (needed to read kt 2/3);
the second statement (`dyn_zad`) landed in the same round as the first.

UNVERIFIED: the XLA rewrite named in §1 is inferred from which guard
fixes it, not read from dumped HLO; "other cards' programs unchanged" is
by construction (default False), not a jaxpr diff.
