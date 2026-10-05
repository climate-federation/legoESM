# Preregistration — the S-19 `wzv` arm on the L1 NEMO test cases

Written and committed BEFORE the arm was run. Branch
`fidelity/nemo-wzv-generic-operands`, worktree `/tmp/wt-wzv`, cut from
`646415f02`. Oracle `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE`,
records `/data/abyssal/dbalwada/nemo-testcases-l1/phase3/`. Everything below is
fp64 (`set_policy(PrecisionPolicy.fp64())`, verified by printing the policy and
the array dtypes next to every number).

## 1. What NEMO's `wzv` actually consumes

`wzv_RK3_t`, `sshwzv.F90:273-387`. The quasi-Eulerian (`key_qco`) arm the L1
cards compile is `:331-336`:

```fortran
pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)                       &
   &         + r1_Dt * e3t_0(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) ) ) * tmask(ji,jj,jk)
```

with the bottom boundary `pww(:,:,jpk) = 0` set once at `:305-307`, and
`ze3div` from `div_hor` (`divhor.F90`):

```fortran
CASE ( np_velocity )     ! divhor.F90:108-115
   hdiv = ( ( e2u(i  )*e3u(i  ,Kmm)*pu(i  ) - e2u(i-1)*e3u(i-1,Kmm)*pu(i-1) )
        & + ( e1v(j  )*e3v(j  ,Kmm)*pv(j  ) - e1v(j-1)*e3v(j-1,Kmm)*pv(j-1) ) ) * r1_e1e2t / e3t(Kmm)
CASE ( np_transport )    ! divhor.F90:116-123
   hdiv = ( ( pu(i) - pu(i-1) ) + ( pv(j) - pv(j-1) ) ) * r1_e1e2t / e3t(Kmm)
pe3divUh = hdiv * e3t(:,:,:,Kmm)                          ! divhor.F90:139-141
```

Call sites, all of them:

| oracle site | lane | operands | time levels |
|---|---|---|---|
| `stprk3_stg.F90:297` | RK3, flux form (`ln_dynadv_vec=.false.`) — the L1 cards | `zFu,zFv` = `e2u*e3u(Kmm)*(uu(Kmm)+zub)` (`:273-274`), `np_transport` | T/S-free; `e3t(Kmm)`, `r3t(Kaa)-r3t(Kbb)` |
| `stprk3_stg.F90:290` | RK3, vector-invariant only | `uu(Kmm),vv(Kmm)`, `np_velocity` | same |
| `stp2d.F90:155` | RK3 2-D pre-pass | `uu(Kbb),vv(Kbb)`, `np_velocity`, `Kmm=Kbb` | `r3t(Kaa)` from the `:150-152` guess |
| `stpmlf.F90:227,270` | MLF (`wzv_MLF`, `:150-258`) — DINO's lane | `hdiv` already built by `div_hor_old` | `e3t(Kmm)`, `r3t(Kaa)-r3t(Kbb)` |

`zub` is `un_adv*r1_hu(Kmm) - uu_b(Kmm)` (`stprk3_stg.F90:145-152`), so the
column integral of `zFu` is exactly `e2u*un_adv`.

## 2. Operand table — NEMO vs the two legoESM arms vs the new source

| NEMO operand | generic arm (`vertical.py diagnose_w_from_flux_div`) | NEMO arm (`ocean_pe_latlon_cgrid.py nemo_qco_wzv_operands`) | new generic source (`vertical.py nemo_qco_card_mesh_operands`) |
|---|---|---|---|
| `e2u`, `e1v` | folded inside `divergence_cgrid` | `z_coord.nemo_e2u`, `nemo_e1v` — **hard raise** if absent | `grid.dy_u[:,1:]`, `grid.dx_v[1:,:]` |
| `r1_e1e2t` | `1/grid.area_T` | `1/z_coord.nemo_e1e2t` — hard raise | `1/grid.area_T` |
| `e1e2u`, `e1e2v` | not used | `z_coord.nemo_e1e2u/v` — hard raise | `(dx_u*dy_u)[:,1:]`, `(dx_v*dy_v)[1:,:]`, guarded on `hu_0>0` |
| `hu_0`, `hv_0` | not used | `z_coord.nemo_hu_0/hv_0` — hard raise | `SUM(e3u_0*umask)` — `domain.F90:145` |
| `e3t_0` | `z_coord.h_partial` (via `sigma`) | `z_coord.nemo_e3t_0` — hard raise | `z_coord.h_partial` |
| `e3u_0`, `e3v_0` | not used | `nemo_e3t_0` (full-step mesh) | `min_cell_to_uface/vface(h_partial)` — `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:179-186` |
| `e3t(Kmm)`, `e3u/e3v(Kmm)` | face thickness from the shared `nemo_qco_live_face_geometry_cgrid` | same shared kernel | same |
| `r3t(Kaa)-r3t(Kbb)` | **NOT USED** — replaced by `sigma(k)*deta_dt` with `deta_dt` = the column integral of the same stage transport | the actual barotropic `Kaa`/`Kbb` ssh | same |
| divide-then-multiply by `e3t(Kmm)` | algebraically cancelled | executed literally | executed literally |

The two arms therefore differ in exactly two ways: (a) floating-point
association, and (b) the SOURCE of the free-surface increment.

## 3. Structural finding that reframes the arm (stated before measuring)

`zad_qco_evaluation='nemo_literal'` is rejected by the model constructor
unless `vertical_momentum_scheme='nemo_advective'`
(`ocean_model_latlon_cgrid.py:2709-2713`), and `nemo_advective` is itself
rejected together with `adaptive_implicit_vertadv=True` (`:2598-2606`). The
certified L1 cards run `nemo_up3` + `adaptive_implicit_vertadv=True`
(`nemo_testcase_recipe.py`), which is what NEMO's own RK3 flux-form lane runs
(`dynadv_up3` + `ln_zad_Aimp`; `dynzad` is dead there). So the missing mesh
operands were never the only wall: that pair of selectors gates the **MLF/
dynzad** call sites (`stpmlf.F90:227,270`), and the L1 cards' live `wzv` is the
**WS-RK3 stage** call site (`stprk3_stg.F90:297`), which legoESM reaches
through `_nemo_ws_stage_transport`'s `diagnose_w_from_flux_div`. The arm below
is therefore run at THAT call site, through the private
`_NEMOWSRK3TestHooks.literal_stage_wzv` one-variable control, and it calls the
same `nemo_qco_wzv_operands` the MLF lane calls.

## 4. Scaling estimate, measured on the kt=1 state (Rule 3, before any owner label)

Both arms distribute the free-surface increment with the same reference
thickness fraction (`e3t_0(k)/ht_0` for NEMO; `h_partial(k)/H_col` for the
generic `sigma`, identical on these cards). Hence

```
max |w_NEMO - w_generic|  <=  max | (ssh(Kaa) - ssh(Kbb))/dt  -  (-column_div) |
```

Measured with `scripts/tmp/_probe_wzv_arm_scale.py` on each card's own kt=1
state, fp64 (all inputs printed `float64`):

| card | stage | max abs stage w [m/s] | bound on max abs dw [m/s] | ratio |
|---|---|---|---|---|
| OVERFLOW-zps (dt=10 s) | 0 | 1.208952e-07 | 1.084202e-19 | 8.97e-13 |
| | 1 | 2.218545e-03 | 3.469447e-18 | 1.56e-15 |
| | 2 | 3.327840e-03 | 5.204170e-18 | 1.56e-15 |
| | 3 | 6.655709e-03 | 3.469447e-18 | 5.21e-16 |
| LOCK_EXCHANGE-zco (dt=1 s) | 0 | 1.504633e-36 | 1.504633e-36 | vacuous (at rest) |
| | 1 | 3.785478e-06 | 6.218069e-21 | 1.64e-15 |
| | 2 | 5.678218e-06 | 1.934623e-21 | 3.41e-16 |
| | 3 | 1.135644e-05 | 1.246494e-20 | 1.10e-15 |

So legoESM's split-explicit barotropic `ssh(Kaa)` already equals the column
integral of the stage transport to ~1 ULP. The arm's whole reachable effect is
floating-point association at the 1e-15 relative level.

## 5. Predictions (registered; each names its refuting value)

Baseline, OVERFLOW-zps, generic arm, trajectory gate `--max-step 10
--continue-after-first`: kt=2 `T` 1.1191048e-14, `u` 2.5987979e-07, `ssh`
1.0491608e-14 (all DEBT against the 1e-15 bar); kt=10 `T` 7.71e-08, `u`
2.64e-05, `ssh` 9.24e-05.

| # | prediction | CONFIRMS | REFUTES |
|---|---|---|---|
| P1 | OVERFLOW kt=2 `u` under the NEMO arm | within 1e-12 of 2.5987979e-07 | any move > 1e-10 |
| P2 | OVERFLOW kt=2 `T` and `ssh` | stay in the 1e-14 band | either exceeds 1e-12 |
| P3 | the S-19 `wzv` branch does NOT own the kt=2 `u` debt | P1 holds | kt=2 `u` falls below 1e-8 |
| P4 | LOCK kt=2, all fields | move by <= 1e-13 normalized | any field moves > 1e-11 |
| P5 | kt=10 rows may move (roundoff is amplified ~1.2-13x per step here) but stay within one order of magnitude of baseline | all four kt=10 fields within 10x | any field changes by more than 10x |
| P6 | DINO is untouched | the 5-day `nemo_dino_kamm_mlf` twin is byte-identical base vs HEAD on every numeric array | any nonzero array difference |

If P1-P4 hold, the correct disposition of S-19 for the L1 cards is a pure
structural collapse (one implementation, no measurable numbers change), NOT an
improvement — and it must be reported as such rather than as a fix.

## 6. What this cannot see

* The trajectory gate compares the ENTERING state at each kt; it cannot
  separate the stage-2 and stage-3 `w` contributions from each other. The
  stage sweep gate does that and is run alongside.
* The bound in section 4 is on `w`; it becomes a bound on `u` and `T` only
  through the stage operators, which amplify. P5 exists because of that.
* LOCK's stage-0 row is vacuous (the card is exactly at rest at kt=1), so it
  carries no information; only stages 1-3 do.
