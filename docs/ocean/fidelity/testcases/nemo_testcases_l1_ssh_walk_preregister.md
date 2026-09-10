# OVERFLOW-zps kt>=3 SSH walk: the between-step barotropic memory, the
# kt=2..4 frame walk, and the preregistered seed arm

Written BEFORE the arm was scored (sections 1-5 were composed from the frame
walk and the oracle arithmetic alone; the predictions were not edited after
the arm ran).  Commit order, disclosed: the S-21 landing commit came first,
this document and the gate extension second, the seed fix third; the arm's
frame walks ran between the writing and the committing of this file.
Base: legoESM `59d1fcbb8`
(branch `fidelity/nemo-branch-isomorphism-audit` tip, worktree
`/tmp/wt-ovf-ssh`, branch `fidelity/overflow-ssh-walk`) plus the S-21 stage-1
transport landing (its own commit, below this one).  fp64 throughout
(`PrecisionPolicy.fp64()` + `JAX_ENABLE_X64=1`; every compared array printed
`float64`), CPU.  NEMO 5.0.2 `OVERFLOW-zps`, `key_qco + key_RK3`, resolved
`ln_bt_fw=T`, `nn_bt_flt=1`, `rn_bt_alpha=0`, `nn_e=3` (`icycle=4`),
`ln_dynadv_up3=T`, `ln_drg_OFF=T`, `f=0`, `rn_Dt=10 s`.

## 0. The question

kt=2 entry is bit-identical in T/S/ssh and carries a `4.55e-10` stage-3 `u`
remainder; kt=3 entry SSH is `3.77e-9`, kt=4 `1.07e-6`, kt=10 `9.24e-5`.  The
S-21 round proved nothing inside the RK3 stage ladder can move SSH within a
step, so the owner consumes or carries something the external solve sees
BETWEEN steps.

## 1. Rule 0: every barotropic memory NEMO carries from kt to kt+1

Read from `dynspg_ts.F90`, `stp2d.F90`, `stprk3.F90`, `stprk3_stg.F90`,
`sshwzv.F90`, `domqco.F90`, `oce.F90` (module-level `SAVE` arrays) under the
compiled keys.  Time levels: RK3 calls `dyn_spg_ts(kt, Kbb, Kbb, ...)`
(`stp2d.F90:281`, `Kmm == Kbb`), and `stprk3.F90:213` rotates `Nbb <-> Naa`
at step end.

| NEMO variable | source | level when kt+1 begins | legoESM counterpart | status |
|---|---|---|---|---|
| `ssh(:,:,Nbb)` | `stprk3_stg.F90:224` (`ssha` = boxcar mean of `ssha_e`, `dynspg_ts.F90:835,847`) | N | `state.eta` | VERIFIED (kt entry rows, `1.05e-14` at kt=2) |
| `uu_b/vv_b(:,:,Nbb)` | `dynspg_ts.F90:845-846,883-892` (transport average / `(hu_0+zzsshu)`), imposed on `uu(Kaa)` by `stprk3_stg.F90:439-446` with `e3u_0/hu_0` weights | N | **NONE carried** -- recomputed every step by `barotropic_latlon_cgrid.py:_depth_average_to_faces` from the 3-D `u` | MEASURED: NEMO's carried `uu_b(Kbb)` equals the `e3u_0`-mean of its own `uu(Kbb)` to `4e-19 / 7e-18 / 2e-17` (kt=2/3/4), so no hidden memory; legoESM's recomputation is what differs (section 3) |
| `ssh(:,:,Naa)` = `2*ssh(Nbb) - ssh(Nbb_old)` | `stprk3.F90:217`, consumed ONLY at `stp2d.F90:151` (`r3t(Kaa)` guess) -> `wzv` `stp2d.F90:155` (`sshwzv.F90:334-335`) -> `dyn_adv_up3` vertical flux in the 2-D RHS `stp2d.F90:172` (`dynadv_up3.F90:275-358`, `pUe` branch); overwritten at `stprk3_stg.F90:166` before any other use | N-1 (two-level ssh memory) | **NONE** -- the step-entry `w` is the instantaneous z-star continuity `w` (`ocean_pe_latlon_cgrid.py:1369 diagnose_w_from_flux_div`, surface `w=0`), no `ssh(n-1)` | MEASURED on the oracle: guess = `2 ssh_n - ssh_{n-1}` exactly at kt=2..4 (`oracle_stp2d_kt*.bin`), NEMO surface `ww` guess `1.07e-2 m/s` at kt=2 vs legoESM `0`.  Its only consumer telescopes to the (zero) surface flux on full columns and to the bottom-face flux on truncated u-columns (`umask` cumulation, `dynadv_up3.F90:306,350`).  PLAUSIBLE contributor to the `slow_u` remainder (section 2); NOT the first divergent frame |
| `r3t/r3u/r3v(:,:,Nbb)` | `stprk3_stg.F90:231-233` (`r3ta` from `dom_qco_r3c_RK3(ssha)`) | N | recomputed from `eta` by `vertical.py:nemo_qco_live_face_geometry_from_operands` (same statement as `domqco.F90:219-222`) | VERIFIED by reading; the stage ladder gates use it |
| `r3f` | `stprk3_stg.F90:606` | N | NONE | INERT: consumed by `dyn_vor` only; `f=0` and uniform `e1/e2` make it exactly zero |
| `un_adv/vn_adv` | `dynspg_ts.F90:843-844` | within step | `Hu_avg/Hv_avg` | VERIFIED (S-21 / stage rounds); not carried across steps by either model |
| `sshbb_e, sshb_e, ubb_e, ub_e, vbb_e, vb_e` | `dynspg_ts.F90:469-476` | zeroed EVERY step: `ll_init = ll_bt_av = .TRUE.` (`:199-202`; the `key_RK3` block `:215-227` only touches `ll_init` for `nn_bt_flt==3`) | `nemo_ab3am4_coeff_arrays(ramp=True)` every step, `bt_hist=None` (`barotropic_latlon_cgrid.py:2282-2290`) | VERIFIED by reading + the kt=2..4 `eta_mid/u_mid` frames track their entry frames |
| `wgtbtp1/2, r1_wgt1s/2s, icycle` | `dynspg_ts.F90:245-248`, `ts_wgt` called at `nit000` only under `ln_bt_fw` (`:209-264`) | constant | weights rebuilt identically every step | VERIFIED (kt=1 register at `1e-15`; kt=2..4 entry frames) |
| `ub2_b, vb2_b, un_bf, vn_bf` | `dynspg_ts.F90:914-935` | MLF `#else` branch, NOT COMPILED under `key_RK3` | NONE needed | N/A |
| `wi` (Aimp implicit w) | `stprk3_stg.F90:284,299` | persists, but zeroed at stage 1 before any consumer; `stp2d` consumes `ww` (full `wzv`), not `wi` | per-stage Aimp split | INERT between steps |
| `Ue_rhs, Ve_rhs, sshe_rhs, CdU_u/v` | `stp2d.F90:112,284` | allocated/freed per step | `F_slow_u/v`, `F_slow_eta` | per step |
| `rDt, r1_Dt` | `domain.F90:278`, `stprk3_stg.F90:123,177,221` | `rn_Dt` when `stp_2D` runs | n/a | consistent with the `ssh(Naa)` guess (`r1_Dt` at `sshwzv.F90:335`) |

## 2. Oracle extension and the kt=2..4 frame walk

WRITE-only instrument copy `tests/OVERFLOW_OMIP_L1_BTWALK4` (MY_SRC =
BTWALK's, with the dump gates widened from `kt == nit000` to
`kt <= nit000+3`, `oracle_rhs` named per kt, and a new `stp2d.F90` dump of
`ssh(Kaa)`, `r3t(Kaa)`, `ww`, `Ue_rhs/Ve_rhs`, `uu_b(Kbb)`, `hu(Kbb)`);
run dir `barotropic_walk/oracle_kt1_4` (`nn_itend=4`, copied from
`oracle_kt1_calls`).  Overlap controls: `oracle_step_entry_kt1..4` BYTE-IDENTICAL
to `phase3/overflow_kt1_10`; `oracle_bt_frames_kt1,2`, `oracle_stage_kt1_s1..3`,
`oracle_rhs_kt1` byte-identical to `oracle_kt1_calls`; the kt=1 19-frame
register differs from `oracle_kt1_calls` only at UNASSIGNED halo points
(halo-stripped max |diff| = `0.0`; likewise `oracle_transport_kt1_s*`).

Gate: `nemo_testcase_overflow_barotropic_gate.py --kt N --oracle-root
--entry-root` (new `run_kt_walk`): the production solve of step N is captured
twice, from legoESM's own kt=N entry (`inherited_entry`) and from NEMO's
dumped kt=N prognostic state (`reseeded_from_oracle_entry`, T/S/u/v/ssh
overwritten on the active cells, nothing else exists to carry).  Absolute
max residuals (m or m/s), substep 1 unless noted:

| kt | arm | eta_entry | u_entry | transport_u | eta_continuity | pgf_u | slow_u | u_exit | eta_exit (jn=4) | first DEBT frame |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | inherited | `1.05e-14` | `1.05e-9` | `5.27e-7` | `1.76e-9` | `3.45e-11` | `1.30e-10` | `9.37e-10` | `5.18e-9` | eta_entry (inherited) |
| 2 | re-seeded | `0.0` | `1.05e-9` | `5.27e-7` | `1.76e-9` | `3.45e-11` | `1.30e-10` | `9.37e-10` | `5.18e-9` | **u_entry** |
| 3 | inherited | `3.77e-9` | `3.08e-7` | `1.54e-4` | `5.14e-7` | `1.01e-8` | `9.63e-10` | `2.74e-7` | `1.15e-6` | eta_entry (inherited) |
| 3 | re-seeded | `0.0` | `3.08e-7` | `1.54e-4` | `5.14e-7` | `1.01e-8` | `9.63e-10` | `2.75e-7` | `1.16e-6` | **u_entry** |
| 4 | inherited | `1.07e-6` | `4.16e-6` | `2.08e-3` | `7.84e-6` | `1.52e-7` | `1.90e-9` | `3.65e-6` | `1.56e-5` | eta_entry (inherited) |
| 4 | re-seeded | `0.0` | `4.32e-6` | `2.16e-3` | `7.20e-6` | `1.39e-7` | `1.90e-9` | `3.85e-6` | `1.63e-5` | **u_entry** |

Artifacts: `barotropic_walk/ssh_walk/frame_walk_kt{2,3,4}.json`.
With an EXACT entry state the FIRST frame over the bar is `u_entry` at every
kt: legoESM's loop-entry barotropic velocity is not NEMO's `puu_b(Kmm)`
(`dynspg_ts.F90:487`), and the inherited 3-D `u` residual (`4.55e-10` at kt=2)
is not what produces it (the two arms agree to three digits).  The
`slow_u` row is the second, independent operand (`1.30e-10 -> 9.63e-10 ->
1.90e-9`, `2.5e-8 -> 1.8e-7 -> 3.6e-7` of its own scale); it does not consume
the seed and stays UNMEASURED-owner in this round.

## 3. Owner (CONFIRMED by arithmetic on the oracle's own arrays)

NEMO: `uu_b(Kbb)` = `SUM(e3u_0*uu(Kbb))*r1_hu_0` (`stprk3_stg.F90:440-441`,
imposed at every stage; live `e3u(Kmm) = e3u_0*(1+r3u(Kmm))`,
`domzgr_substitute.h90`), measured to `4e-19..2e-17` above.

legoESM, `barotropic_latlon_cgrid.py:348-411` (`seed_evaluation="nemo_literal"`,
`seed_face_depth="nemo_ssh_avg"`, no carried NEMO mesh on the L1 cards so
`_nemo_literal_seed_from_reference_mesh` returns `None`): the 3-D face
thickness is `min_cell_to_uface(h_k)` -- the per-level MIN of the two
STRETCHED T-cell thicknesses -- rescaled by `H_u_nemo / min(H_west, H_east)`
(`:365-391`), then summed against the literal inverse `1/H_u_nemo`
(`:400-403`).  On a face whose two columns have different reference depths
(the shelf break: west 500 m / 25 levels, east 510 m / 26 levels) the
per-level min takes the LESS-stretched column while the column min takes the
SHALLOWER column, so the weights sum to `hu_0 (1+r3t_e)(1+r3u)/(1+r3t_w)`
instead of `hu_0 (1+r3u)`: a uniform velocity is returned scaled by
`(1+r3t_e)/(1+r3t_w)`.  Predicted relative error `r3t_e - r3t_w` against the
measured seed residual at the flagged face (row 1, NEMO u-column 22,
`hu_0 = 500 m`):

| kt | `r3t_w - r3t_e` from `h_k` | measured `|res|/uu_b` | residual | `uu_b` |
|---:|---:|---:|---:|---:|
| 2 | `8.85e-6` | `8.9e-6` | `-1.055e-9` | `1.19118e-4` |
| 3 | `1.07e-4` | `1.07e-4` | `-3.082e-7` | `2.87210e-3` |
| 4 | `3.04e-4` | `3.0e-4` | `-4.320e-6` | `1.42304e-2` |

(`seed_probe`: `|numpy e3u_0-mean - uu_b| <= 2e-17` on the same arrays.)  Every
other wet face is at `<= 3e-17`.  The seed error alone reproduces the
substep-1 frames: `transport_u = 500 m * 1.055e-9 = 5.27e-7` (measured
`5.27e-7`), `eta_continuity = (H/dx)*5.27e-7/... = 1.76e-9` (measured
`1.76e-9`).  Growth: the relative error is the ssh-slope across the shelf
break, which grows with the overflow, times `uu_b`, which grows with the
adjustment -- `1e-9 -> 3e-7 -> 4e-6` at kt=2..4 entries -- and each step
re-injects it, so the walk is re-injection, not amplification of the kt=2 seed.

LOCK_EXCHANGE-zco has equal column depths everywhere, so both mins pick the
same column and the rescale normalises exactly (algebraically; the arithmetic
differs by ulps).

## 4. The arm and the fix

Fix (inside the WS-RK3 identity, no public selector): the `nemo_literal`
card-mesh seed builds its face thickness and literal inverse from the ONE
shared kernel every other NEMO consumer uses --
`vertical.py:nemo_qco_card_mesh_operands` (`e3u_0` = min-rule of the
REFERENCE ladder, `hu_0 = SUM(e3u_0*umask)`, `domain.F90:145`) +
`nemo_qco_live_face_geometry_from_operands` (`e3u = e3u_0*(1+r3u)`,
`r1_hu = r1_hu_0/(1+r3u)`) -- and then the same source-ordered recurrence
`_nemo_literal_seed_depth_mean`.  The carried-mesh (DINO) path is untouched.
One-variable arm: private `_NEMOWSRK3TestHooks.legacy_seed_min_rule_faces`
(default False) restores the min-rule rescale; threaded like
`primary_transport_average` (`_nemo_legacy_seed_faces_test_override`).  The
legacy side must be bit-identical to the pre-fix tree.

## 5. Predictions (frozen)

| # | quantity | prediction | REFUTED if |
|---|---|---|---|
| P1 | re-seeded kt=2 walk | `u_entry` `1.055e-9 -> <= 1e-15` (AT-BAR); `transport_u`, `eta_continuity`, `eta_pgf` at substep 1 fall to the `1e-15` class; first DEBT frame becomes `slow_u` at `1.30e-10` (unchanged to 3 digits) | `u_entry > 1e-12`, or `slow_u` moves > 10% |
| P2 | re-seeded kt=3, kt=4 walks | `u_entry` `3.08e-7 -> <= 1e-15`, `4.32e-6 -> <= 1e-15` | either stays `> 1e-12` |
| P3 | OVERFLOW kt=2 rows (T, S, u, v, ssh) | BIT-IDENTICAL (the seed multiplies `u = 0` at kt=1) | any move |
| P4 | OVERFLOW kt=3 SSH `3.775e-9` | decreases; remainder is the `slow_u`-driven part, bounded above by `3.5e-9` (`dF = 1.3e-10`, `n^2` accumulation over 4 substeps, boxcar 2-4) | increases |
| P5 | OVERFLOW kt=4 SSH `1.072e-6`, u `1.684e-7`; kt=10 SSH `9.237e-5`, u `2.644e-5` | each drops `>= 10x` | any drops `< 2x` (owner of the WALK refuted; seed stays a confirmed first-divergence operand) |
| P6 | OVERFLOW kt=60 T/u/SSH | drop, no factor claimed (60 steps past the fixed operand) | -- |
| P7 | LOCK kt=2,3,10,60 rows | move by ulps only (`<= 1e-14` relative), kt=2 stays AT-BAR | any row moves `> 1e-12` relative |
| P8 | 19-frame gate kt=1, stage sweep kt=2 (both cards) | bit-identical / unchanged | any move |
| P9 | 6120-step statistics fp64+fp32 (hist `0.062`, census `0.0176`, u_linf `1.99`, plume_descent two-valued, plume_front `117 km`, T_linf `0.3793`) | direction-free; reported before/after; faithful-but-worse disclosed, not reverted (Rule 8) | -- |

Everything above the fix is CONFIRMED on the oracle's own arrays; the growth
mechanism (re-injection) is PLAUSIBLE until P5 lands; the `ssh(Naa)`
guess -> `slow_u` link is PLAUSIBLE and unmeasured.
