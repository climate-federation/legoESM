# Vertical-mixing lane — V1 round 8 (vmix_r8)

Status: **HELD. V1 is not certified.** The kt-2 north-fold owner is landed:
kt-2 TKE energy, viscosity, diffusivity and dissipation are now **AT-BAR**
(en max 5.57e-7 -> 2.1e-17). Every field is 8-10 orders of magnitude closer
to NEMO at kt 2..10, but both labels still carry last-bit DEBT from kt 2.

Decisions in force: D129 = 1, D130 = 1. Codex review: paused (quota).

Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v1_round8.md` at
`3bc875b3f`, committed after the offline fold-row walk and before the landing
and the ladders (the file says so).

Commits:
- `ff14675d9` task 1: htau native-latitude selection test;
- `3bc875b3f` preregistration;
- `587ebd30e` landing: pivot-row shear reads NEMO's north-fold images;
- `e091e9234` citation re-anchor (mechanical: nine model pins shifted, two pinned spans grew).
- `8181893215` review fixes: flag-off purity, jit-pinned tests, `sshwzv` citation (section 7);
- `fa072e5e50` citation re-anchor (mechanical: sixteen pins +3, one pinned span 76 -> 79 lines).

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3).
Evidence: `gridladder_vmix_rounds/round8/` (`manifest.sha256`, 20 files).
Ladders, `kt2b.json`, `rhs_arms.json` stamped clean at
`587ebd30e983b4a92d9b2851a68c822a699790c4`, packages tree
`315815df059cfc8a924dd2042e7b9d2fdf8ccf75`. `kt2_probe_fix.json` ran on the
same package diff before it was committed.

| file | sha256 prefix |
|---|---|
| `probe_fold_sh2.py` / `fold_sh2.json` (before) / `fold_sh2_after.json` | 9cdbd37806640c27 / c75d2dbb1ebb9e56 / 7ec0bec5e21ed8ea |
| `probe_kt2.py` (round 7's) / `kt2_probe_fix.json` | 4ea882423ffbc3a9 / 66d9477c5a89771c |
| `probe_kt2b.py` / `kt2b.json` / `walk_kt2.py` / `walk_kt2.json` | 8e46bac1e4dc5967 / 0734e3502b5c75d1 / 493fa2b06c6a24eb / d6d1f8d622125757 |
| `probe_rhs.py` / `rhs_arms.json` | f3cda092093db49b / a72f4e9b73b942c3 |
| `probe_matrix.py` / `matrix_kt2.json` | eca02406a48f62b3 / 65d42307a4a155bd |
| `sh2_interior_after.json` (round 7's `probe_sh2.py`) | 32da00eab3aa5079 |
| `v1_independent_kt1_5.json` / `_kt6_10.json` / `v1_given_entries.json` | 62d2df1bb7cbab41 / 9695e8141ea3d965 / 57f46c0c2138bd7e |

## 1. Task 1 — the htau arm's native-latitude selection is pinned

The card-side half (the V1 grid carries `gphit`) was already tested. The
uncovered revert was the model's call site: putting back
`== "nemo_literal"` for the latitude selection sends the new
`'nemo_scalar_libm'` arm `degrees(lat_T)` again, and every test stayed green.

New test `test_nemo_sh2_step_entry_ssh.py::test_production_htau_arms_read_native_gphit`
drives the production vertical-mixing method and captures the latitude it
hands the closure. Plant (call site reverted): **red**
(`28.64788976 is not gphit 28.0`). Reverted; `git status --porcelain` clean.

## 2. Task 2 — kt-2 pivot-row TKE energy walk

Grid mapping, measured: card row 147 is NEMO's pivot row jj = 148. It is
exactly self-mirrored in NEMO's T (0 unequal), and the mirror column is
`(180 - c) % 180`.

Walk of `tke_tke`'s operands on that row (SUBROUTINE `tke_tke`,
`zdftke.f90:199`):
- surface value, Langmuir source, matrix, solve, penetration: column-local, no
  horizontal operand. kt-2 matrix upper diagonal vs a NEMO-literal rebuild:
  0 unequal (`matrix_kt2.json`). Solve: literal replay equals the card's
  post-sweep in every cell (`walk_kt2.json`);
- **shear `p_sh2` (`zdfsh2.f90:98-113`, SUBROUTINE `zdf_sh2` :56) is the one
  horizontal stencil, and the first non-symmetric operand.**

NEMO's fold treatment of each shear operand on the pivot row's north V face:

| operand | NEMO | card (before) |
|---|---|---|
| `v` on that face | `lbc_lnk 'V' -1` image (`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:785-793`) | the image (already stored) |
| `vmask` / `wvmask` there | wet (raw mesh mask; 1,319 wet faces) | wall (0), so coast factor 2 |
| `avm_k(ji,jj+1)` | `lbc_lnk 'W' +1` image of row 146 (`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:785-793`) | edge-repeat of row 147 |
| `ssh(ji,jj+1)` in `r3v` | `'T' +1` image of row 146 | wrapped to the south row |

Fold routine: `lbc_nfd_dp`, T-pivot, `lbcnfd.f90:584-720` (SUBROUTINE :561).

Instrument (`probe_fold_sh2.py`, numpy `zdf_sh2` on NEMO's kt-2 entry u/v/ssh
and kt-1 viscosity, one rounding per operation):
- control: the card's wall form reproduces the card's pivot-row shear in
  **1,292 / 1,292** wet cells;
- NEMO's fold form is exactly mirror-symmetric and differed from the card in
  1,251 / 1,296 cells. The card was 2 x the south face where NEMO is the south
  face plus its mirror. That is why NEMO's en is the midpoint of the card's
  mirror pair (round-7 review).

**Landed** (`587ebd30e`), V1-selected only: `TKEConfig.nemo_sh2_fold_top_v`.
The step-entry shear builder sets the three operands from the fold images. It
refuses a grid with no fold, the in-solve shear, or a card without NEMO's raw
mask.

After, kt 2 (P1, P3 CONFIRMED):
- card pivot-row shear vs NEMO fold replay: **0 / 1,296 unequal**, symmetric;
- en vs NEMO: max **5.57e-7 -> 2.08e-17** (`kt2_probe_fix.json`);
- en/avm/avt/dissl: all AT-BAR (ladder below).

Unit test `test_nemo_sh2_fold_top_v.py` (4 tests): with symmetric pivot
inputs, the selected shear is mirror-symmetric and only the pivot row moves.
Each operand planted back to the wall form turns it red (mask, viscosity,
`r3v`: 1 failed each). Restored with `cp -f`; `cmp` clean.

## 3. Both labels, kt 1..10

kt 1, both labels: 0 unequal on every row (P2 CONFIRMED).

INDEPENDENT (unequal cells / full-depth max as the ladder prints it):

| kt | en | avm_k | avt_k | dissl | T | ssh |
|---:|---|---|---|---|---|---|
| 2 | 47,231 / 2.08e-17 | 30,249 / 2.92e-16 | 28,198 / 1.17e-16 | 24,009 / 1.39e-17 | 13,531 / 2.66e-15 | 0 / 0 |
| 3 | 67,186 / 1.70e-15 | 51,166 / 4.89e-13 | 50,995 / 3.40e-11 | 45,000 / 1.27e-14 | 32,692 / 9.99e-14 | 14,148 / 1.17e-14 |
| 4 | 80,746 / 9.32e-15 | 66,837 / 4.75e-13 | 66,970 / 2.75e-11 | 62,239 / 1.99e-14 | 48,513 / 1.36e-13 | 15,220 / 5.02e-14 |
| 5 | 87,149 / 5.50e-14 | 75,202 / 9.08e-13 | 75,256 / 4.40e-11 | 74,160 / 2.97e-14 | 60,208 / 7.64e-13 | 15,686 / 7.04e-14 |
| 6 | 92,987 / 3.34e-14 | 81,936 / 2.17e-12 | 82,120 / 2.88e-11 | 83,908 / 7.30e-14 | 69,775 / 7.45e-13 | 15,854 / 5.86e-14 |
| 7 | 96,085 / 3.49e-14 | 86,736 / 1.28e-12 | 87,191 / 5.76e-12 | 91,319 / 7.27e-14 | 77,464 / 7.41e-13 | 15,973 / 7.49e-14 |
| 8 | 100,631 / 3.54e-14 | 91,527 / 1.05e-12 | 92,206 / 1.59e-11 | 98,441 / 3.26e-14 | 85,038 / 7.23e-13 | 16,099 / 1.34e-13 |
| 9 | 103,517 / 2.81e-14 | 95,491 / 5.10e-13 | 96,284 / 4.80e-12 | 104,938 / 5.59e-14 | 90,962 / 7.38e-13 | 16,163 / 5.69e-14 |
| 10 | 106,823 / 5.38e-14 | 98,967 / 3.83e-13 | 99,866 / 3.32e-12 | 110,665 / 7.27e-14 | 96,760 / 7.31e-13 | 16,219 / 9.24e-14 |

Round 7, same windows: kt-2 en 5.57e-7, kt-10 T 4.32e-3, kt-10 avm 2.45e-4.

kt 2: en, avm, avt, dissl, ssh, `uu_b`, `vv_b`, v are AT-BAR. First over the
bar: surface u (5,048 cells, normalised 2.22e-15), T (7.82e-14), S (1.78e-13).

GIVEN-NEMO-ENTRY: every single step from NEMO's entry at kt 3/5/7/9 leaves
ssh, `uu_b`, `vv_b` exact. u/v are at or just over the bar (worst 3.44e-15),
and T/S are DEBT at 8-24e-14. The two-step rows reach 2.1e-12 (T, kt 6).

P4 FALSIFIED: **V1 is NOT certified.**

## 4. Next owner (registered, PLAUSIBLE, not landed)

At kt 2, en is at the bar but unequal in 47,358 cells (max 2.1e-17). Every
one of them has nonzero shear; every zero-shear cell matches.

Ruled out by measurement:
- the solve (literal replay = card post-sweep, all cells);
- the matrix upper diagonal (0 unequal vs NEMO-literal);
- the RHS association (`zdftke.f90:423-426`): the private all-products-rounded
  arm gives 46,099 unequal, not 0 (`rhs_arms.json`);
- the shear itself: interior card vs NEMO-literal shear is 10 unequal cells,
  max 4.7e-16 relative; the pivot row is 0.

Left: a non-shear RHS operand at kt 2 (`rn2`, the Langmuir-updated entry), or
the post-solve penetration increment. Next round: a NEMO-literal replay of
each on the card's traced operands.

## 5. Decision 122 (ORCA2_OMIP_L4)

`zdfsh2.f90`, `lbcnfd.f90`, `stprk3_stg.f90` and `zdfphy.f90` are
byte-identical between the V1 build and `ORCA2_OMIP_L4` (SHA-256 prefixes
`e8bb2398700d0d3d`, `5620dc75e3e75726`, `ac38c2df29ebeaa5`,
`903e0d1220a2f9b3`).

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| `p_sh2 = 0.25*((zsh2u..)*(2-umask*umask) + (zsh2v(jj-1)+zsh2v(jj))*(2-vmask*vmask))` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:112` | `zdf_sh2` (:56) |
| `zsh2v` with `p_avm(ji,jj+1)` and `r3v` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:104-108` | `zdf_sh2` (:56) |
| `lbc_lnk(... avm_k, 'W', 1.)` before `zdf_sh2` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:785,793` | `stp_RK3_stg` (:78) |
| T-pivot fold of T/W and V rows | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/lbcnfd.f90:584-720` | `lbc_nfd_dp` (:561) |
| `lbc_lnk(... pssh(:,:,Kaa), 'T', 1.0)`: the sea level read by `r3v` carries the T fold | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:140` | `ssh_nxt` (:90) |

Pointer only: the full ORCA2 card was not measured with this selection.

## 6. Selection and caller grep (no-silent-default rule)

- `nemo_sh2_fold_top_v` is set only in the V1 card's vertical block and pinned
  by the V1 selector; the selector check fires when it is planted `False`.
  The V1 config-delta test names it. G1-G5, the full ORCA2 card, DINO, GYRE
  and every library default are unchanged.
- `avm_weighted_shear_production` callers: the step-entry builder (passes
  the new argument), `tke.py:3385` (in-solve shear; the flag refuses that
  path) and the GYRE round-54 script (default `None`, unchanged).
- **RULE 3:** the fix defaults off on other cards. D130 already answers
  this, so there is no new question.

## Tests

All runs synchronous, CPU, fp64:

| suite | result |
|---|---|
| new fold + sh2/htau + V1 card + TKE carried/identity/terms + V1 kt-1 statements | 221 passed |
| `test_nemo_sh2_fold_top_v.py` after the in-solve refusal | 4 passed |
| citation gate + private imports, clean tree at `e091e9234` | 23 passed |
| review fixes: fold/sh2/htau + V1 card + TKE carried/terms + V1 kt-1 + recipe + freshwater, at `fa072e5e5` | 197 passed |
| push-gate rest: face-mask rank, prognostic barotropic, ORCA2 parallel receipt, at `fa072e5e5` | 12 passed |
| citation gate + private imports, clean tree at `fa072e5e5` | 23 passed |

Review: **codex review: paused (quota)**. GLM unavailable. The operator's
independent Claude review returned HOLD on two mechanical items (physics
CONFIRMED); both are fixed in section 7. The fixes themselves are
**UNREVIEWED** until the operator re-reviews them.

## Choices made

ASKED (this round's brief): the htau test; the fold-row walk; landing the
transcription defect V1-selected; re-scoring both labels.

UNASKED, offered for revert:
- the flag raises on three conditions (no fold, in-solve shear, no raw mask):
  new hard errors, but only when the new flag is set;
- one TKE flag for the three fold operands rather than one per operand;
- the post-landing kt-2 diagnosis (section 4; nothing landed).

UNVERIFIED:
- the kt-2 last-bit owner (section 4);
- NEMO's own kt-2 `p_sh2` was not dumped. The fold arm is checked by its
  symmetry, by its kt-2 en result, and by the wall-form control;
- the full ORCA2 card's response to this selection.

## 7. Review fixes (independent Claude review, HOLD -> fixed)

Reviewer's kt-2/3 rerun, recorded as given (not re-measured here):
- kt 2 INDEPENDENT, first over the bar: surface u, 5,048 cells, max 2.22e-15
  at (102,43), open interior;
- kt 3 INDEPENDENT: ssh, 14,148 cells, max 1.17e-14 at (123,138);
- kt 3 GIVEN-ENTRY: u, 5,261 cells, max 1.78e-15.

**Flag-off purity (must).** `587ebd30e` reordered the step-entry `r3v`
north-neighbour expression for every card, flag on or off. The flag-off
branch now keeps the base expression verbatim; the fold image is built only
under the flag (`8181893215`). Instrument `probe_jaxpr.py`: `make_jaxpr` of
one `model.step` from each card's initial state (full ORCA2 card with its IWM
maps; DINO through `run_dino.py` with the month gate's card, recipe, dt and
grid), callback object addresses masked, body hashed after the 2-line stamp.

| card | base `6e45bf5a4` | round 8 `601d05b47` (control) | fixed `fa072e5e5` |
|---|---|---|---|
| full ORCA2 | `e0351d089a027297` | `18914db414e607a0` (12 lines differ) | `e0351d089a027297` **identical** |
| DINO | `89418f6a0b2eabae` | `4ebaa1ad2b737305` (12 lines differ) | `89418f6a0b2eabae` **identical** |

Both cards run the step-entry shear with the flag off (logged). The control
column shows the probe sees the defect.

**Tests (must).** `test_nemo_sh2_fold_top_v.py`: x64 guard
(`skipif(not x64_enabled)`, the lane's convention); the shear is now built
from traced `eta, u, v, avm` under `jax.jit`, and every call asserts eager ==
jit bit for bit before the fold asserts read the jit result. The refusals
fire at trace time under jit. The two htau call-site tests carry the same
guard. Plants (`plant_r8b.py`, `plants_r8b.log`), one statement reverted each,
restored by copy and `cmp`-clean:

| reverted statement | result |
|---|---|
| raw `vmask` on the pivot face | 1 failed (symmetry) |
| viscosity fold image | 1 failed (symmetry) |
| `r3v` sea-level fold image | 1 failed (symmetry) |
| tracer-only perturbation of the viscosity image | 2 failed, eager == jit assert fired |
| no-fold refusal | 1 failed |
| in-solve refusal | 1 failed |
| htau call site back to `== "nemo_literal"` | 1 failed (`nemo_scalar_libm`) |

**Citation (should).** The `r3v` sea-level fold is NEMO's `ssh_nxt`
`lbc_lnk(pssh(:,:,Kaa),'T',1.0)`, `sshwzv.f90:140` (SUBROUTINE `ssh_nxt`
:90); `sshwzv.f90` SHA-256 prefix `9d0225953304ecbd`, the same in all four
`ORCA2_OMIP_L4` trees on disk. Added to the model comment and section 5.

Evidence added to `round8/`: `probe_jaxpr.py` (`26ad88a6fac73af1`),
`jaxpr_{orca2,dino}_{base,r8head,fix}.txt`, `plant_r8b.py`, `plants_r8b.log`,
`reanchor_r8b.py`, `focused_r8b.log`, `landgate_r8b.log`; `manifest.sha256` regenerated.

UNASKED, offered for revert: the x64 guard is a skip (lane convention), not a
module-level `jax_enable_x64` update.
