# OVERFLOW-zps stage-3 baroclinic `u` debt — enumeration, scaling, preregistered predictions

Frozen BEFORE the arm was implemented or run, at legoESM `9070cf276`
(branch `fidelity/overflow-stage3-u-owner`, worktree `/tmp/wt-ovf-u`, clean
tracked tree), fp64 (`PrecisionPolicy.fp64()` + `JAX_ENABLE_X64=1`, every array
dtype printed `float64`), CPU.  Cards: `OVERFLOW-zps` and `LOCK_EXCHANGE-zco`.
Oracle dumps `/data/abyssal/dbalwada/nemo-testcases-l1/phase3/{overflow,lock}_kt1_10/`.
Replay machinery: the hunt probes under
`/data/abyssal/dbalwada/nemo-testcases-l1/hunt_probes/` (`hpg_sco`,
`dynadv_up3`, `dynzdf` transcriptions) re-run against the HEAD stage fields.

## 0. The debt

`OVERFLOW-zps` kt=1 stage 3 (= kt=2 entry) baroclinic `u` L-inf
`2.598797930308122e-07 m/s`; stage 1 `6.50e-15`, stage 2 `9.43e-11`; depth-mean
part `4.75e-15`.  Measured structure (HEAD, `stage_decomp.py`, e3u_0-weighted
depth mean removed), all on the single wet row:

| face | k=0 | k=1 | k=2..24 (uniform) |
|---|---:|---:|---:|
| 19 | `+1.297e-07` | `+6.278e-08` | `-8.50e-09` |
| 20 (front) | `-2.599e-07` | `-1.259e-07` | `+1.678e-08` |
| 21 | `+1.303e-07` | `+6.331e-08` | `-8.14e-09` |

CONFIRMED: horizontal pattern `(+1, -2, +1)` on faces 19/20/21 (a conservative
horizontal flux-form redistribution), vertical pattern confined to the top two
levels with ratio `1.94`, plus the uniform column shift the stage barotropic
correction adds back.  A vertical flux-form operator (viscosity, vertical
advection) CANNOT produce same-sign errors at k=0 and k=1 with a zero bottom
flux, so the vertical operators are ruled out by structure before any number.

## 1. Rule 0 — everything that runs ONLY at stage 3, with dispositions

`stprk3_stg.F90` (NEMO 5.0.2, `key_qco + key_RK3`, flux form, `ln_zad_Aimp`):

| line | call | legoESM executing code (OVERFLOW/LOCK cards, `momentum_time_integrator == tracer_time_integrator == "rk3_ws"`) | disposition |
|---|---|---|---|
| `:284` | `wi = 0` at stage 1 only | `_nemo_ws_stage_transport` returns `wi_stage = 0` for `stage_index != 2` (`omlc:1176`) | MATCH |
| `:299` | `wAimp(kstp, Kmm, zFu, zFv, ww, wi, np_transport)` at stage 3 | `nemo_wicker_aimp_partition_transport` at `omlc:1180` (`stage_index == 2`) | INERT at kt=1: NEMO's stage-3 transport dump gives max `Cu_v = 1.66e-03`, `Cu_h = 4.6e-04` (`Cu_min_v = 0.8`); legoESM's function on the same operands returns `max wi = 0.0` exactly. Any Aimp implementation difference is multiplied by zero |
| `:322-324` | `eos(Kmm)` + `dyn_hpg(Kmm)` on the stage-2 Kaa operands | `_stage_hpg_operands(_T_stage2, _S_stage2, _eta_live_one_half)` -> `tendencies(precomputed_geom_density=...)` (`omlc:5012-5016`) | MATCH: stage-2 tracer operand `max dT 3.55e-15 K`, `dS 0`, `dssh 5.2e-15 m`; `hpg_sco` replay on legoESM vs NEMO operands differs `1.59e-17 m/s` after `dt`; legoESM's own HPG (`tendencies`, `u=0`) vs `hpg_sco` on the stage-2 Kaa operands `1.37e-16 m/s^2` |
| `:327` | `dyn_vor` (ENS, `ntot` = planetary + metric) | structurally eliminated (`f = 0`, Cartesian metrics, one wet row; `validate_nemo_testcase_card`) | INERT |
| `:333` | `dyn_adv(up3)` on `(zFu, zFv, zFw)` | horizontal: `_bc_horizontal_momentum_advection_flux_form` with `momentum_flux_transport_velocity` (`opl:4098`); vertical: `nemo_up3_vertical_momentum_advection` via `_stage_vertical_up3` (`omlc:4942`) | **OWNER of the horizontal part — see section 2**; vertical: transcription on legoESM's `u2` vs NEMO's `u2` differs `1.7e-13 m/s^2`, term magnitude `< 2e-9 m/s` after `dt` |
| `:400` | `dyn_ldf` | `tendencies(skip_lateral_viscosity=False)` at stage 3 only (`omlc:4418`) | INERT: `ln_dynldf_OFF` -> `nldf_dyn = np_no_ldf`, no `CASE` in `dynldf.F90:66-75`; legoESM `A_h = B_h = C_smag = C_leith = 0` -> measured `skip=False minus skip=True` = `0.0` exactly |
| `:402-406` | `dyn_osm`, `bdy_dyn3d_dmp`, `dyn_dmp` | none | OFF in both |
| `:430` | `dyn_zdf(kstp, Kbb, Kmm, Krhs, uu, vv, Kaa)` | qco time step `omlc:5020-5021` (`u3_raw = (q_b u0 + dt q_12 RHS)/q_aa`), implicit solve `_apply_implicit_vertical_mixing` -> `implicit_vertical_diffusion_nemo_momentum` (`implicit_solver.py`), `nemo_literal`, `implicit_vmix_e3t_now_divisor=True` | bounded and structurally excluded: NEMO's whole stage-3 ZDF increment is `1.086e-08 m/s` (k=0 `+1.05e-08`, k=24 `-1.09e-08`, interior `1.75e-11`); a tridiagonal flux-form solve cannot create the observed same-sign top-two-level pattern. Two bottom-localised operand differences found and NOT owned here: legoESM `e3uw(Kmm)` = midpoint of `e3t_now` (12.25 m at the partial-cell interface) vs NEMO `e3uw_0 = e3w_1d = 20 m` (`usrdef_zgr.F90:167`, `domzgr_substitute.h90:127`); legoESM `e3u(Kaa)` = masked cell->face AVERAGE vs NEMO `e3u_0 = min` at the staircase face. Both scale as `<= 1e-8 x O(1)` at the bottom cells only |
| `dynzdf.F90:101` | `zdf_drg_exp` | `bottom_drag_r = 0` | INERT (`ln_drg_OFF`, `rCdU_bot = 0`) |
| `dynzdf.F90:148-171,293-305` | `ln_drgimp` blocks | none | OFF / zero |
| `dynzdf.F90:233-270` | Aimp fusion into the matrix | `implicit_w=nemo_aimp_momentum_w_u` in `implicit_vertical_diffusion_nemo_momentum` | INERT (`wi = 0`) |
| `dynzdf.F90:329` | surface stress `utauU` in the first recurrence | `surface_forcing=None` | INERT (`usrdef_sbc.F90:61`: `utau = 0`) |
| `:439-446` | barotropic correction `zub` at Kaa | `_replace_stage_mean` (`omlc:4818`) | MATCH (depth-mean part `4.75e-15`; S-35 order measured inert) |
| `:468`, `:588` | `bbl` / `tra_bbl` | inactive at kt=1 | exonerated earlier |
| `:598` | `tra_zdf` | tracer solve, `K_v = 0` | tracer only |

`stprk3.F90` after stage 3 (`:229-312`): only diagnostics, restart I/O and the
level swaps touch nothing prognostic.

## 2. The finding (replay, CONFIRMED)

`dynadv_up3.F90:165-177`:

```fortran
zui = ( puu(ji,jj,jk,Kmm) + puu(ji+1,jj  ,jk,Kmm) )        ! :166  VELOCITY sum
IF( zui > 0 ) THEN   ;   zl_u = zlu_uu(ji  ,jj)             ! :169  upwind curvature
ELSE                 ;   zl_u = zlu_uu(ji+1,jj)             ! :170
ENDIF
zFu_t(ji+1,jj  ) = (  zFu(ji,jj) + zFu(ji+1,jj  )  ) * ( zui - gamma1 * zl_u )   ! :176  TRANSPORT (with zub)
```

The upwind branch is selected by the sign of the advected-velocity sum
`uu(Kmm)_i + uu(Kmm)_{i+1}`; the flux magnitude uses the stage transport
`zFu = e2u e3u(Kmm) (uu(Kmm) + zub)` (`stprk3_stg.F90:273`).  legoESM's
`_up3_reconstruct` (`opl:4078-4095`) reproduces both branch formulas
(`(-u_{i-1} + 5u_i + 2u_{i+1})/6` and `(2u_i + 5u_{i+1} - u_{i+2})/6`) but
selects the branch by the sign of the TRANSPORT it is handed
(`Qx_c = 0.5 (Q_i + Q_{i+1})`, `opl:4169-4175`).  On the WS-RK3 cards the
transport carries `zub`, and at kt=1 `zub` is large (the transport time mean
is about half the primary velocity mean after the first ramp):

| stage | T-pairs where `sign(u_i+u_{i+1}) != sign(F_i+F_{i+1})` (NEMO dumps) | curvature there |
|---|---|---|
| 2 | `(16, k=0..19)` and others far from the front | exactly `0.0` (barotropic-only, uniform column) |
| 3 | `(19, k=0,1)`, `(20, k=0,1)` = the front pairs, top two levels; plus `(16, k>=12)`, `(23, k>=13)` | nonzero at the front |

At stage 3, pair 20|21, k=0..1: `zui = +0.0234, +0.0255` while
`Fsum/e2u = -0.076, -0.034 m^2/s` (crosses zero at k=2).

Scaling table (all m/s^2 on the wet U faces, `bc` = e3u_0-weighted depth mean
removed; `D_l` = legoESM's derived stage-3 RHS minus `hpg_sco` minus the
`dynadv_up3` transcription evaluated on legoESM's own `u2` with NEMO's stage-3
transports; NEMO's own closure `max|rhs_n - hpg - up3| = 4.6e-16`):

| suspect | one-variable replay quantity | max | corr with `D_l` | slope | `max|D_l - pattern|` |
|---|---|---:|---:|---:|---:|
| Aimp partition (S-20) | `wi` on NEMO stage-3 transports | `0.0` | — | — | `2.599e-08` (unchanged) |
| implicit ZDF solve | NEMO increment / structure | `1.086e-09` m/s^2-equivalent | excluded by structure | — | — |
| stage-3 HPG operands | `dt * bc(H(lego s2) - H(nemo s2))` | `1.6e-18` | — | — | unchanged |
| lateral viscosity (stage-3-only `skip_ldf=False`) | `tend(skip=False) - tend(skip=True)` | `0.0` | — | — | unchanged |
| vertical UP3 | transcription input sensitivity | `1.7e-13` | — | — | unchanged |
| **horizontal UP3 upwind selector** | `bc(up3[transport-sign]) - bc(up3[velocity-sign])` on legoESM `u2` | `2.5989e-08` | **`+0.99999447`** | **`1.000152`** | **`4.55e-11`** (`4.5e-10 m/s` after `dt`) |

Linearised post-fix stage-3 baroclinic residual: `4.5e-10 m/s` max (face 20
profile `~1e-11`).

`LOCK_EXCHANGE-zco` (from its dumps, no legoESM field): `ssh == 0` so
`un_adv == 0` exactly and `zub = -uu_b(Kaa) = -1.135e-3 m/s` at the front
face; the selectors disagree at 20 cells (pairs 63|64 and 64|65, k=0..9) with
a rough tendency error `1.6e-10 m/s^2 x 1 s` — the same order as LOCK's
entire kt=2 `u` residual `2.1388e-10 m/s`.

## 3. The arm

NEMO's rule is folded into the WS-RK3 scheme identity, no public selector:
when `_bc_horizontal_momentum_advection_flux_form` is handed a separate
`transport_velocity` (only the WS-RK3 stage program does, `omlc:4418`), the
UP3 branch is selected by the sign of the ADVECTED velocity pair
(`adv_pos + adv_neg`); every call without a separate transport keeps the
transport-sign selector bit-for-bit (`Q = h u` there, so the two rules differ
only at a partial-cell face with a near-zero pair sum — a legacy-scheme fact
recorded, not changed).  The private hook
`_NEMOWSRK3TestHooks.legacy_up3_transport_sign_selector=True` restores the old
selector for the stage-sweep gate's one-variable arm; NEMO has no such switch.
A unit test with a curved field and a transport of the opposite sign fails
on the reverted selector.

## 4. Predictions (frozen)

Baseline numbers = the HEAD `9070cf276` gate run in
`scratchpad/ovf_u_before/` (same scripts, same oracle roots, fp64), pasted
into the receipt next to the arm numbers.

| # | card | quantity | prediction | REFUTED if |
|---|---|---|---|---|
| P1 | OVERFLOW | kt=1 stage-3 / kt=2 `u` (baroclinic and instantaneous) | `2.5988e-07 -> < 1.0e-09` (linearised `4.5e-10`) | `> 2.6e-08` or not improving |
| P2 | OVERFLOW | kt=1 stage-1 and stage-2 `u` rows | BIT-IDENTICAL (`6.501744e-15`, `9.433404e-11`): stage-1 advection is zero from rest, the stage-2 disagreement cells carry zero curvature | either row moves `> 1e-16` |
| P3 | OVERFLOW | kt=2 `T`, `S`, `SSH` | BIT-IDENTICAL (`1.1191048e-14`, uninformative, `1.0491608e-14`): stage-3 tracers consume the stage-2 Kmm transport, the external mode is solved before the stage ladder that carries the fix | any of them moves |
| P4 | OVERFLOW | `legacy_up3_transport_sign_selector` arm | reproduces the baseline stage-3 row to `< 1e-15` (the arm IS the old code) | any difference |
| P5 | LOCK | kt=2 `u` | improves by `>= 3x` from `2.138804e-10` (selector share `~1.6e-10`) | `< 1.5x` improvement (then the LOCK residual is NOT selector-owned; the OVERFLOW verdict stands on P1 alone) |
| P6 | LOCK | kt=2 `T`, `SSH` | BIT-IDENTICAL (`1.62773498e-13`, `4.78e-28`) | any move |
| P7 (secondary hypothesis, separate from ownership) | OVERFLOW | kt=10 `u`/`T`/`SSH` and kt=60 `u`/`T`/`SSH` | every row improves; the kt>=3 SSH walk is seeded by this stage-3 error and drops `>= 10x` at kt=10 | walk does not drop `>= 2x` at kt=10 -> the walk has another owner, disclosed as such |
| P8 | both | 6120-step (OVERFLOW) statistics | UNMEASURED before the arm; reported before/after, no direction claimed | — |
| P9 | legacy paths | `tests/ocean/unit/test_flux_form_momentum.py` (no separate transport) | bit-identical, all pass | any failure |

Rule 8: if a row worsens under the faithful selector it is disclosed, not
reverted.
