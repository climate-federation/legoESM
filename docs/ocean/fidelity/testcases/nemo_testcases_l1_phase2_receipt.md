# NEMO testcase lane 1 — phase-2 geometry, IC, and kt=1 receipt

Date: 2026-08-31

Session: `01a0591f-a335-7060-9257-6c47bf2149ee`

Original preregistration: `bb718ae7ca60912962187d2c39c5cac358f96e48`

Review-correction preregistration: `b5ee1815853`

Corrective implementation: `680672eaebe`

## Verdict

**VERIFIED for geometry and the informative kt=1 T-alignment scope.** Both
pure-config cards match their certified meshes inside the DINO pointwise
`1e-15` bar. The nonuniform temperature front is bit-identical at the
registered `Nbb/before` entry (`exact=True`, bar zero). S, u, v, and SSH are
also byte-equal, but they are spatially uniform or at-rest zero and therefore
remain loudly **alignment UNMEASURED**. This receipt makes no trajectory,
tendency, EOS, or BBL-transport claim.

| card | mesh registry | geometry | informative kt=1 | alignment controls |
|---|---:|---:|---:|---:|
| LOCK_EXCHANGE-zco | 35/35 disposed (30 VERIFIED, 5 WAIVED) | 30/30 AT-BAR; max `2.212e-16` | T exact (`0`) | S/u/v/SSH exact but UNMEASURED |
| OVERFLOW-zps | 45/45 disposed (39 VERIFIED, 6 WAIVED) | 39/39 AT-BAR; max `1.418e-16` | T exact (`0`) | S/u/v/SSH exact but UNMEASURED |

Every scientific floating legoESM array printed by the gate is `float64`;
masks are boolean and bottom indices integer. The harness entry point sets and
verifies `PrecisionPolicy.fp64()` before card construction
(`nemo_testcase_phase2_gate.py:116-123`). Card builders no longer mutate the
global policy, and a direct unit test proves that property. The comparison ran
on CPU; NEMO was not rerun.

## Pre-implementation search and PGF correction

The original preregistration records the searched recipe catalog, DINO
fidelity harness, Kamm machinery, canonical grid/vertical operators, and
generic lock/overflow experiments. The review search additionally found the
shared `pgf_scheme="nemo_sco"` implementation already used by DINO, with qco
stretch, `gdept_z0` correction, model validation, and a literal recurrence
test. No bespoke solver or fidelity-only physics was added.

The prior cards' `smc03` selection is retracted. In the resolved oracle
namelists, LOCK has `ln_hpg_sco=T`, `ln_hpg_djc=F` at
`lock_exchange_zco/output.namelist.dyn:430-437`; OVERFLOW has the same at
`overflow_zps/output.namelist.dyn:431-438`. NEMO 5.0.2
`src/OCE/DYN/dynhpg.F90:117-123` dispatches SCO to `hpg_sco` and DJC to a
different routine, and `:188-197` requires exactly one. The canonical option
transcribes the selected surface recurrence/correction (`:340-360`) and
interior recurrence/correction (`:367-390`). Both cards now select `nemo_sco`
plus `nemo_trapezoid`; defaults for all other users remain unchanged.

Direct tests cover (a) a vertically stratified, level-isopycnal qco staircase
at exactly zero PGF, (b) a nonzero tilted-SSH/staircase form-stress violation,
and (c) a fully nonuniform-density transliteration matching the NEMO recurrence
to `rtol=1e-12`, `atol=1e-19` on wet faces.

## Exact card resolution and selector reconciliation

| setting | LOCK_EXCHANGE-zco | OVERFLOW-zps |
|---|---|---|
| domain | `130 x 3`, 500 m, closed, `f=0` | `202 x 3`, 1000 m, closed, `f=0` |
| active vertical grid | 20 x 1 m full steps | 100 x 20 m reference steps, zps bottom cells |
| NEMO dummy record | one dry bottom record | one dry bottom record |
| timestep / registered length | 1 s / 61200 | 10 s / 6120 |
| EOS selector | `veros_gsw` | `veros_gsw` |
| tracer / momentum | `fct2`; flux-form `upwind3` | `fct2`; flux-form `upwind3` |
| PGF | `nemo_sco`, NEMO trapezoid | `nemo_sco`, NEMO trapezoid |
| explicit lateral mixing / drag | off / off | off / off |
| constant vertical coefficients | `A_v=1e-4`, `K_v=0` | `A_v=1e-4`, `K_v=0` |
| advective BBL pin | off | option 2, gamma 20 s |

The six review-requested selector dispositions are source-pinned here:

| selector | oracle evidence | disposition |
|---|---|---|
| `pgf_quadrature=nemo_trapezoid` | `dynhpg.F90:343-380` uses a surface half-weight followed by two-level density sums on `e3w` | VERIFIED by the literal direct test |
| `barotropic_solver=explicit_substep` | both resolved namelists set `ln_dynspg_ts=T` (`LOCK:439-447`, `OVERFLOW:440-448`); `dynspg.F90:208-233` maps it to time splitting | selector VERIFIED; evolution UNMEASURED |
| `barotropic_time_filter=cosine` | LOCK resolves Demange `nn_bt_flt=3`, alpha `.07`; OVERFLOW resolves boxcar-width-`nn_e` `nn_bt_flt=1`, alpha zero at the lines above; definitions are `dynspg_ts.F90:1068-1094,1265-1273` | **WAIVED**: the shared cosine selector matches neither oracle filter; selectable parity is trajectory debt |
| `vertical_momentum_scheme=nemo_advective` | active `ln_dynadv_up3=T` is at `LOCK:414-420`, `OVERFLOW:415-421`; `dynadv.F90:78-90` dispatches to `dyn_adv_up3`, including vertical flux at `dynadv_up3.F90:250-358`. `dynzad` is only the inactive vector arm. | **WAIVED**: this legoESM helper transcribes `dynzad`, not active UP3 vertical flux; parity UNMEASURED |
| `tracer_time_integrator=euler` | `stprk3.F90:50-64,194-207` owns three-stage active-tracer/momentum integration; tracer calls occur in `stprk3_stg.F90:519,586,598` | **WAIVED**: the legoESM inner selector is not demonstrated equivalent to NEMO tracer RK3 |
| no independent vorticity selector | both case `usrdef_hgr.F90:100-104` set `f=0`; flux-form `dynvor.F90:879-913` reduces the residual to Coriolis plus Cartesian metric terms | VERIFIED absent on these nonrotating uniform Cartesian grids |

`veros_gsw` remains a selector, not a claim of bit-identical NEMO TEOS-10;
density, `rab`, and BN2 are UNMEASURED.

## Partial-cell and fail-closed corrections

OVERFLOW's `bottom_index_rule="nemo_tpoint"` reproduces the active
`usrdef_zgr.F90:157-180` T-point `k_bot` rule and unsnapped bottom thickness.
It now requires an explicit `t_depth_ref`; arithmetic-midpoint fallback raises.
A stretched-grid unit control proves the two ladders choose different bottom
levels. The cards pin `[0.5,1.5,...,19.5] m` and
`[10,30,...,1990] m`, respectively. The neighboring-min U-face scale is
bit-identical and pays D2.5 (`tools/DOMAINcfg/src/domzgr.F90:1163-1169`;
active specialization `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:182-186`).

Case dispatch checks membership before invoking a builder. The unknown case
still raises `ValueError`, while a planted internal builder `KeyError` now
propagates instead of being mislabeled.

## IC/time level, controls, tests, and artifacts

The time registry labels `oracle_step_entry_kt00000001.bin` **before**. The
instrument writes `ts/uu/vv/ssh(...,Nbb)` before forcing, `stp_2D`, and the
three RK calls (`tests/*_OMIP_L1/MY_SRC/stprk3.F90:88-100,202-225`). All five
rows call `_score(..., exact=True)` (`nemo_testcase_phase2_gate.py:295-307`).
Only T is informative; the gate records the other four as UNMEASURED without
discarding their exact-control result (`:308-318`).

All planted controls exit nonzero:

- file-side unknown: `missing=['PLANTED_UNACCOUNTED_FILE_ARRAY']`;
- LOCK `e1t + 1 m`: `0.002 > 1e-15`;
- wet T `+1 C`: exact error `inf > 0`.

Test-count reconciliation: the reviewer's 29 was the real original count across
the three then-current lane files (6 card + 5 phase-2 gate + 18 oracle-gate
tests). The reported 49 additionally included 20 generic vertical-coordinate
regressions (15 partial-cell phase-0 + 5 full-step), so it was a combined run,
not a lane-file count. At review-round-2 input, the reviewer was exactly right:
the **four lane files contained 41 tests** (9 card + 7 phase-2 gate + 18 oracle
gate + 7 canonical `nemo_sco`), while 20 generic vertical regressions made the
separately combined run 61. Residual r1 adds one direct `nemo_sco` test, so the
current counts are **42 lane tests** and 20 supporting regressions, 62 only when
explicitly described as the combined execution. These are not averaged or
competing measurements.

| artifact | SHA256 |
|---|---|
| LOCK mesh / kt=1 dump / gate JSON | `ec3200f559cb44ee76d00498aac168dd6452cc29bd0abdf955dfc4ab061ed935` / `c9f23d441865c566e3edf0301aca4f6e440259ade1722b725a0aa0fd4c2839e1` / `7a3b8c0cbaff6bdc387188999afe98e672b1d856558e1f8da756062edbc70bc8` |
| OVERFLOW mesh / kt=1 dump / gate JSON | `4692b893eddee3eea5cee2e6f04e1d7fc55e6685100e349369051a6914cbd280` / `cf0183e563aba8b8848bc5dea470c9e50aab2d987ff5da86241e8f82494d5c66` / `8b4e011ab2ee5c02e2f2d9065d75bac9ed7cc0c0556b1a51319709bb1eab8e44` |

The gate JSON keeps kt>1 trajectories, RK-stage tendencies, TEOS-10
density/`rab`/BN2, FCT tendency, adaptive vertical advection, active-UP3
vertical momentum parity, oracle barotropic filters, tracer RK3 parity, and
BBL transport UNMEASURED. No trajectory language is used.
