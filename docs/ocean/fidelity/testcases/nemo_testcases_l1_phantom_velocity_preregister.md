# OVERFLOW-zps below-seabed ("phantom") velocity: citations, scaling, preregistered arm

Frozen BEFORE the arm was implemented or run, at legoESM `75017dc58`
(branch `fidelity/overflow-phantom-below-seabed`, worktree
`/tmp/wt-ovf-phantom`, cut from `c5275ae48` = the stage-3-remainder tip).
fp64 (`PrecisionPolicy.fp64()` set BEFORE the card is built +
`JAX_ENABLE_X64=1`; every array dtype printed `float64`), CPU.  Instrument:
the committed
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_phantom_velocity_probe.py`
(`census` / `scaling`), receipts under
`/data/abyssal/dbalwada/nemo-testcases-l1/phantom_velocity/before/`.
Lead: `nemo_testcases_l1_stage3_remainder_receipt.md`, "What remains open"
item 1 (CONFIRMED by reading, by both reviewers and the author; UNMEASURED
until this round).

## 0. Rule 0 -- what NEMO does, quoted

**NEMO's rule: a masked cell holds EXACTLY zero, and every flux through a
masked face is zero.**  Both halves are in the source.

`stprk3_stg.F90:365-380` -- the WS-RK3 stage time-stepping, both branches:

```fortran
CASE ( 1 , 2 )    !==  Stage 1 & 2  ==!   time stepping
   IF( ln_dynadv_vec .OR. lk_linssh ) THEN
      uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)   ! :367
   ELSE
#  if defined key_qco
      uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) ) * uu(ji,jj,jk,Kbb )  &        ! :375
         &                 + rDt * ( 1._wp + r3u(ji,jj,Kmm) ) * uu(ji,jj,jk,Krhs)  )   &
         &             /           ( 1._wp + r3u(ji,jj,Kaa) ) * umask(ji,jj,jk)           ! :377
```

`stprk3_stg.F90:439-446` -- the barotropic correction, masked the same way:

```fortran
DO_2D( 0, 0, 0, 0 )             ! barotropic velocity correction
   zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)   ! :440
END_2D
DO_3D( 0, 0, 0, 0, 1, jpkm1 )   ! corrected horizontal velocity
   uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)                        ! :444
```

`dynadv_up3.F90` then READS those zeros -- it does not skip a dry neighbour,
it reads it as `0`:

```fortran
zlu_uu(ji,jj) = (  ( puu(ji+1,jj,jk,Kbb) - puu(ji,jj,jk,Kbb) )  &
   &             + ( puu(ji-1,jj,jk,Kbb) - puu(ji,jj,jk,Kbb) )  ) * umask(ji,jj,jk)   ! :142-143
zFu(ji,jj) = e2u(ji,jj) * e3u(ji,jj,jk,Kmm) * puu(ji,jj,jk,Kmm)                       ! :160
zui = ( puu(ji,jj,jk,Kmm) + puu(ji+1,jj,jk,Kmm) )                                     ! :166
IF( zui > 0 ) THEN   ;   zl_u = zlu_uu(ji  ,jj)
ELSE                 ;   zl_u = zlu_uu(ji+1,jj)                                       ! :169-170
zFu_t(ji+1,jj) = (  zFu(ji,jj) + zFu(ji+1,jj)  ) * ( zui - gamma1 * zl_u )            ! :176
```

## 1. The difference, stated as (i) and (ii)

Both are true; they are separable and were measured separately.

**(i) legoESM lets a masked cell hold a nonzero `u`, and the operator that
writes it is the WS-RK3 stage velocity update.**
`ocean_model_latlon_cgrid.py:5032-5045` (`_replace_stage_mean`, the ONE stage
ladder, called at `:5103,:5120,:5139` for stages 1/2/3) applies
`u_mask_3d = state.u_mask.data[..., jnp.newaxis]` (`:4066`) -- the **2-D**
face mask broadcast over every level -- to
`u_in + (target_u - mean_u)[..., jnp.newaxis]`.  A face that is wet at ANY
level therefore keeps the depth-mean increment at EVERY level below its own
seabed.  The same 2-D rule is at `:5155-5162` (`_transport_stage`, the stage
velocities the tracer transports consume).  MEASURED signature: the
below-seabed column is EXACTLY constant -- 1 unique value over all 75 dry
levels, `ptp = 0.0` -- i.e. a depth-mean broadcast, not an operator residue.

**(ii) legoESM's UP3 curvature omits NEMO's `* umask` at its own point.**
`ocean_pe_latlon_cgrid.py:4132-4151` (`_up3_reconstruct`) evaluates
`(-far_pos + 5*adv_pos + 2*adv_neg)/6`, which is algebraically
`0.5*(adv_pos + adv_neg) - (1/6)*curvature` with NEMO's `zlu_uu` at
`umask = 1`.  NEMO zeroes that curvature when the upwind straddling face is
dry (`dynadv_up3.F90:143`), legoESM does not.  The operands themselves are
fed unmasked -- `u_core = u[:, :-1, :]` at `:4205-4208`, consumed at `:4170`
-- so with (i) present the stencil reads the phantom, and with (i) fixed it
still differs from NEMO by this factor.

Only the T-point (same-direction) stencil has NEMO's single-centre-mask
shape; the F-point ones carry an `fmask` PAIR (`dynadv_up3.F90:146-149`), so
(ii) is scoped to the T-point `u` reconstruction here.

## 2. Scaling -- MEASURED, `phantom_velocity/before/{census,scaling}.json`

Model u-face index `f` == trajectory-gate face index `f-1`.  Shelf staircase,
row 1: T-cells 0..22 carry 25 wet levels, 23..26 carry 26 -- so u-faces 16..23
are dry at `k=25` and faces 24..27 are wet there.

Below-live-seabed `|u|`, free run (the certified card; the initial state is
`0.0` there and NEMO's own kt=2..5 entries are asserted `0.0`):

| kt | max abs below seabed | at | wet max abs | below-seabed column |
|---:|---|---|---|---|
| 2 | `7.920607e-03` | face 21, k=25 | `9.652900e-02` | 1 unique value, ptp `0.0` |
| 3 | `5.294269e-02` | face 21, k=25 | `1.559111e-01` | 1 unique value, ptp `0.0` |
| 4 | `1.223887e-01` | face 21, k=25 | `1.907901e-01` | -- |
| 5 | `1.836966e-01` | face 21, k=25 | `2.336829e-01` | -- |
| 6 | `2.283002e-01` | face 21, k=25 | `2.930978e-01` | -- |

It is REGENERATED every step, not inherited: one step from NEMO's EXACT
(clean) entry gives `4.50e-02 / 6.94e-02 / 6.13e-02 / 4.46e-02` at
kt=2/3/4/5.  Origin arms at kt=1 (from rest, so every below-seabed value was
written during the step): stage-1 velocity `2.737944e-02`, stage-2
`1.855487e-02`, and with the barotropic replacement ABLATED `5.295233e-02`
-- i.e. both halves of `_replace_stage_mean` write there and partly cancel.

Leverage on the horizontal UP3 flux, `dt * d(du/dt)` from legoESM's own
`tendencies()` called twice on the same stage-2 state (one variable), against
the measured one-step injection from NEMO's exact kt-1 entry (the
stage3-remainder receipt's growth table, post-thickness-fix):

| step kt | stage-2 phantom | phantom leverage, model face 24 (gate 23) k=25 | model face 25 (gate 24) k=25 | (ii) curvature-umask, face 24 k=25 | measured injection at kt+1 |
|---:|---|---|---|---|---|
| 1 | `1.8555e-02` | `-3.209e-12` | `-9.22e-14` | `+1.266e-15` | `7.064e-12` (gate face 20, k=24) |
| 2 | `1.9588e-02` | `-4.170e-10` | `-3.084e-10` | `+2.429e-11` | `4.181e-09` (gate face 21, k=24) |
| 3 | `4.2979e-02` | `+8.022e-08` | `-1.585e-08` | `+3.344e-09` | `9.816e-08` (gate face 22, k=24) |

Ratios leverage/injection `0.45 / 0.10 / 0.82`.  (ii) is `4e-4 / 5.8e-2 /
4.2e-2` of (i) -- real, same face and level, and sub-dominant.

READ HONESTLY: the phantom's leverage is at `k=25`, the bottom wet level of
the DEEPER column, on the two faces immediately seaward of the step; the
injection MAXIMUM at kt=2..5 sits one level up (`k=24`) on the faces
immediately shoreward, and only moves to `k=25` from kt=6 (receipt: kt>=6 at
gate face 23 k=25, kt=10 at gate face 24 k=25).  So the amplitude and the
staircase family match; the exact cell does not, and one stage's replay on
the step-entry geometry is a linearisation of a three-stage step.  The
ownership claim is PLAUSIBLE, not CONFIRMED, and that is what the arm below
tests.

Instrument controls, all run before any number above was written: the two
arms differ only in `u` and only on dry faces (asserted); the arm is EXACTLY
`0.0` on a phantom-free state (NEMO's kt=3 entry, 16900 active u points);
both `tendencies()` calls reproducible to `0.0`; NEMO's own entries asserted
`0.0` below the seabed; LOCK's 3-D live u mask asserted EQUAL to the 2-D
broadcast (0 of 3000 differing points on OVERFLOW's staircase, 0 everywhere
on LOCK); every dtype `float64`.

## 3. The arm (inside the WS-RK3 identity; no public selector)

NEMO has ONE mask rule, and legoESM already builds the 3-D live face mask
once for this branch -- `_ws_u_live_mask` / `_ws_v_live_mask` at
`ocean_model_latlon_cgrid.py:4203-4210`, from the shared
`compute_face_masks_3d`, already handed to the stage face-thickness kernel.

Change: `_replace_stage_mean` (both branches) and `_transport_stage` mask
with that 3-D live face mask instead of the 2-D broadcast -- exactly NEMO's
`* umask(ji,jj,jk)` at `stprk3_stg.F90:367,377,444`.  No new mask is built,
no smoothing, no clipping (Rule 9: masking is NEMO's own behaviour).  Private
one-variable control `_NEMOWSRK3TestHooks.legacy_2d_stage_face_mask=True`
restores the 2-D rule (harness/gate arm only; NEMO has no such switch).  No
default keeps the old behaviour on the certified cards (Rule 3).

(ii) is NOT landed in this round: it is a second variable, its NEMO form for
the two F-point stencils is an `fmask` pair rather than a centre mask, and it
is 17-2500x smaller than (i) on every measured row.  It is carried forward as
a measured open item with its numbers, not as an explanation.

## 4. Predictions (frozen)

Baselines: the stage3-remainder receipt's AFTER rows (branch tip `c5275ae48`),
fp64, CPU, byte-identical protocol; each re-measured by this round's before-run.

| # | card | quantity | prediction | REFUTED if |
|---|---|---|---|---|
| P1 | OVERFLOW | `max abs u` below the live seabed, every kt=1..10 of the free run and after each exact-entry step | EXACTLY `0.0` | any nonzero value survives |
| P2 | OVERFLOW | kt=1 stage-1 `u` (`6.522560e-15`) | BIT-IDENTICAL (stage 1's RHS is built from the clean entry `u0`; only its own dry levels change) | it moves |
| P3 | OVERFLOW | kt=1 stage-2 `u` (`1.566344e-12`), stage-3/kt=2 `u` (`7.064252e-12`) | both MOVE by `>= 3e-13`; no direction claimed for stage 2, kt=2 predicted to land in `[1e-13, 1.5e-11]` (the kt=1 leverage is `3.2e-12` per stage over two stages) | either moves by `< 3e-13`, or kt=2 lands outside `[1e-13, 1.5e-11]` |
| P4 | OVERFLOW | kt=3 `u` `4.181282e-09` | drops `>= 2x` | drops `< 1.2x` |
| P5 | OVERFLOW | kt=4 `u` `1.261475e-07`, kt=5 `u` `9.858747e-07` | each drops `>= 1.3x` | either drops `< 1.05x` |
| P6 | OVERFLOW | kt=10 `u` `2.632503e-05`, SSH `1.624474e-06` | `u` drops `>= 2x`; SSH drops `>= 1.5x` | kt=10 `u` drops `< 1.2x` |
| P7 | OVERFLOW | kt=60 `u` `2.506199e-04`, T `1.183929e-05`, SSH `3.394740e-05` | `u` drops `>= 1.3x`; T and SSH reported, no direction | kt=60 `u` WORSENS by more than 5% |
| P8 | OVERFLOW | kt=2 `T` `7.815970e-15`, `S`, `SSH` `1.050549e-14` | `SSH` and `S` BIT-IDENTICAL (the external solve consumes only the Kbb depth mean, and a masked level carries `h = 0`, so no depth mean moves); `T` may move by `<= 3e-14`, no direction | `SSH` or `S` move, or `T` moves `> 3e-14` |
| P9 | LOCK | every row kt=1..60 and the kt=1..2 stage sweep | BIT-IDENTICAL: its 3-D live u mask EQUALS the 2-D broadcast (measured, 0 differing points), so the two rules coincide by construction | any LOCK row moves |
| P10 | both | `legacy_2d_stage_face_mask` arm | reproduces the pre-fix rows BIT-FOR-BIT (stage sweep kt=1..2 and trajectory kt=1..10, both cards) | any difference |
| P11 | OVERFLOW | kt=1 19-frame barotropic gate | 0/152 rows differ: the stage ladder runs AFTER the external solve, and the solve's `F_slow` is a `h`-weighted depth mean in which a masked level contributes `0` | any row differs |
| P12 | DINO | 5-day `nemo_dino_kamm_mlf` twin reach check | NIL by construction -- every DINO card resolves `momentum_time_integrator='euler'` (measured: all 7 recipes), and the change lives inside the `rk3_ws` branch; every archived array differs by `0.000000e+00` | any array differs |
| P13 | OVERFLOW | 6120-step statistics, fp64 + fp32 floor, six rows (`hist 0.0638199`, `census 0.0167508`, `u_linf 1.71839`, `plume_descent 1499.86`, `plume_front 117.139`, `T_linf 0.379253`) | reported before/after, NO direction claimed (the 6120-step state is chaotic; the two legoESM precisions decorrelate to `2.3 m/s` in `u_linf`) | -- |
| P14 | unit | new direct test | FAILS on the reverted code (a synthetic staircase: the faithful and legacy arms differ below the seabed and in the wet bottom-level tendency; a flat-bottom control is identical) | passes on reverted code |

Rule 8: any row that worsens is DISCLOSED, not reverted, and the compensating
term is named.  The kt>=6 `u` and kt>=8 SSH rows were already faithful-but-
worse after the previous landing; if this round does not recover them, that is
reported as such.

## 5. What this instrument cannot see

The scaling replay evaluates ONE stage's horizontal advection on the
step-entry (Kbb) geometry bundle, so it linearises a three-stage step and
cannot resolve the barotropic coupling that compounds the free-run rows at
kt>=3: P4-P7 are factors, not values.  The census resolves the phantom
exactly (it is a mask test, not a norm), but says nothing about whether a
second, independent term also lives at the shelf-break bottom level -- the
kt=2 remainder `7.06e-12` at gate face 20 k=24 and the exact-entry `slow_u`
`5.9e-10` at gate face 22 (stage3-remainder open item 2) sit at `k=24`, where
this arm has no leverage at all, and are expected to survive.
