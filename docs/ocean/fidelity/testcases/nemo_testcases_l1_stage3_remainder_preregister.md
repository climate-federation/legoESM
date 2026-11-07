# OVERFLOW-zps kt=2 stage-3 `u` remainder (`4.55e-10`, faces 19/21): attribution, owner replay, preregistered arm

Frozen BEFORE the arm was implemented or run, at legoESM `fcd0d6057`
(branch `fidelity/overflow-stage3-remainder`, worktree `/tmp/wt-ovf-rem`, cut
from `032bedd3a`), fp64 (`PrecisionPolicy.fp64()` set BEFORE the card is
built + `JAX_ENABLE_X64=1`; every array dtype printed `float64` -- the probe
refused the float32 card the first time), CPU.  Instrument: the committed
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_stage3_remainder_probe.py`
(`growth` / `candidates` / `slow`), receipts under
`/data/abyssal/dbalwada/nemo-testcases-l1/stage3_remainder/before/`.  Oracle
dumps `phase3/overflow_kt1_10/` (entries kt=1..10, kt=1 stages, kt=1 stage-3
transports, kt=1 bt frames, kt=1 rhs) and `barotropic_walk/oracle_kt1_4/`
(external-solve substep traces kt=1..4; its kt=2/kt=4 entries are
bit-identical to the phase3 ones, measured `0.0`).

## 0. (A) Growth attribution -- MEASURED, not a prediction

Arms scored against NEMO's dumped entries with the trajectory gate's own
reducer (normalized L-inf on the entering state):

| kt | free (the gate rows) `u` | exact NEMO kt=2 entry, stepped (`exact2_traj`) | exact kt=2 entry + legoESM's measured kt=2 `u` remainder (`plus_du2`) | one step from the EXACT kt-1 entry (pure injection) |
|---:|---|---|---|---|
| 2 | `4.5517e-10` | -- | `4.5517e-10` | `4.5517e-10` (from t=0) |
| 3 | `8.8437e-09` | `8.3885e-09` (ratio 1.054, corr 0.9994) | `8.8437e-09` (1.000, 1.0000) | `8.3885e-09` |
| 4 | `1.2710e-07` | `1.2594e-07` (1.009, 0.9999) | `1.2594e-07` (1.009, 0.9999) | `9.8735e-08` |
| 5 | `9.9184e-07` | `9.8755e-07` (1.004, 0.9999) | `9.8755e-07` | `4.8900e-07` |
| 6 | `3.4268e-06` | `3.4788e-06` (0.985, 1.0000) | `3.4788e-06` | `8.9385e-07` |
| 10 | `2.6296e-05` | `2.6731e-05` (0.984, 1.0000) | `2.6731e-05` | `4.4079e-07` |

Verdict: the kt=2 remainder is NOT the walk's owner.  An EXACT kt=2 entry
reproduces the free trajectory to 1.6% at every kt>=3 with the same face
pattern (corr >= 0.9994); adding the measured remainder changes nothing
beyond kt=3 (its 0.45e-9 propagates as ~5% of the kt=3 row and is invisible
once the maximum moves to faces 22/23).  The per-step injection from an
exact entry GROWS with the flow -- `4.6e-10, 8.4e-9, 9.9e-8, 4.9e-7, 8.9e-7`
at kt=1..5 -- and its maximum walks with the front (face 21 -> 22 -> 23): a
state-dependent operator error injected every step, not a propagated seed.
Same for `slow_u`: the exact-reseed and inherited frames are `1.296e-10 /
9.63e-10 / 1.90e-9` in BOTH arms at kt=2/3/4.  So there is ONE thing to
find: the per-step operator.

## 1. (B) Candidate replays on NEMO's stage-2 Kaa operands -- MEASURED

Target `R` = e3u_0-baroclinic part of `u3_lego - u3_nemo` at kt=1 stage 3
(= the kt=2 entry): `4.5517e-10` at face 21 k=24; depth-mean part `4.8e-15`;
faces 19 and 21 anticorrelated (`corr(19,21) = -0.9996`), face 20 `1.0e-11`;
per-face linear fit in `gdept` r^2 0.967 (19) / 0.956 (21), i.e. close to
linear with a sign change at mid-depth (a linear column field with its mean
removed).  Transcription controls before any row: `hpg_sco` vs NEMO's rest
RHS `1.1e-15`; `wzv` vs the dumped `zFw` `9.1e-13` (of `|Fw| ~ 1e3`);
NEMO's own stage-3 RHS closure `hpg + up3 = M u3/dt` `4.6e-16`.  The
current-code closure remainder `D_l` is `4.552e-11 m/s^2`, and `dt * D_l`
regresses on `R` with corr `0.999999`, slope `1.0000` -- the remainder IS an
RHS closure defect, as the previous round said.

Each row is a one-variable replay; `E = dt*(1+r3u(Kmm))/(1+r3u(Kaa)) *
bc(RHS_lego - RHS_nemo)`, regressed on `R`:

| candidate (NEMO line vs legoESM line) | max abs E (m/s) | faces 19 / 20 / 21 | corr | slope | max abs R - E |
|---|---:|---|---:|---:|---:|
| **X1 horizontal momentum advection THICKNESS: `tendencies()` builds `h_u = min_cell_to_uface(compute_layer_thickness(eta_stage))` (`ocean_pe_latlon_cgrid.py:1221,1364`, consumed at `:4504`) -- the MIN of the two STRETCHED T thicknesses -- where NEMO's `dyn_adv_up3` consumes `e3u(Kmm) = e3u_0*(1+r3u(Kmm))` (`domzgr_substitute.h90:127`, `domqco.F90:219-220`) in `zFu` (`stprk3_stg.F90:273`) and as the divisor (`dynadv_up3.F90:205-207`).  legoESM's OWN operator, h_u the only variable, NEMO's u2 and un_adv** | `4.612e-10` | `2.04e-10 / 3.1e-12 / 4.61e-10` | **`+0.99979`** | **`0.9891`** | **`7.1e-12`** |
| X1t the same substitution inside the `dynadv_up3` transcription | `4.552e-10` | `2.06e-10 / 1.0e-11 / 4.55e-10` | `+1.00000` | `1.00004` | `1.7e-12` |
| X1 at STAGE 2 (u1, ssh = ssha/3, dt/2) vs the stage-2 residual `9.433e-11` | `9.568e-11` | -- | `+0.99978` | `0.9888` | `1.6e-12` |
| X2 HPG: `hpg_sco` vs `tendencies(u=0)` on NEMO's (T2, S2, ssh2) | `9.0e-16` | -- | -- | -- | unchanged |
| X3 vertical UP3: transcription vs `nemo_up3_vertical_momentum_advection` on NEMO's (u2, zFw, e3u) | `5.0e-21` | -- | -- | -- | unchanged |
| X4 qco stage ratios: `_nemo_ws_qco_stage_faces` vs `dom_qco_r3c_RK3` on NEMO's ssh (rule `0.0`, own operand `1.1e-16`) | `1.4e-21` | -- | -- | -- | unchanged |
| X5 stage transport triplet vs the dumped `zFu`/`zFw` (`1.1e-16` / `4.2e-18`; `Hu_avg` vs `un_adv` `1.1e-12` of 10.4) | `2.0e-20` | -- | -- | -- | unchanged |
| X6 implicit ZDF solve on NEMO's pre-solve field (NEMO's increment: k=0/k=24 `1.05e-8`, interior `1.8e-11`, faces 19/21 `< 4e-12`) | `5.4e-18` | -- | -- | -- | unchanged |
| X7 stage-mean weights `h_u_pre` vs `e3u_0` (eta(Kbb)=0 at kt=1) | `0.0` | -- | -- | -- | -- |
| S-44 reference: the already-fixed UP3 selector pattern | `2.6e-7` | `1.3e-7 / 2.6e-7 / 1.3e-7` | `0.055` | -- | -- |

`R - E(X1)`: `7.1e-12` (faces 19/20/21: `2.0e-12 / 7.1e-12 / 6.0e-12`).
The relative thickness error the candidate corrects is `-4.7e-5` (faces 19,
21), `-9.9e-5` (20), `-2.2e-6` (18, 22): `min(1+r3t_W, 1+r3t_E) - (1+r3u)`
= `-0.5*|ssh_W - ssh_E|/hu_0`, largest at the front, and it cancels at face
20 in the T-point flux divergence (the same `c_20 F_20` enters both fluxes
around face 20) -- which is why the remainder sits at 19/21 and not at the
front face itself.

The same mechanism at the STEP ENTRY (the Kbb `tendencies()` whose
`h_u_pre` depth mean is `F_slow`, NEMO `Ue_rhs` / `zu_frc`) against the
measured `slow_u` residual from NEMO's EXACT kt-entry (the SSH-walk round's
open item 2):

| kt | `slow_u` residual (faces 19/20/21) | X1 prediction (same faces) | corr | slope | max abs residual - E |
|---:|---|---|---:|---:|---:|
| 2 | `1.296e-10` (`+5.9e-11 / -4.1e-12 / -1.30e-10`) | `1.296e-10` (`+5.9e-11 / -4.1e-12 / -1.30e-10`) | `1.00000` | `1.0000` | `8.1e-15` |
| 3 | `9.627e-10` (`+6.0e-10 / -9.1e-11 / -9.63e-10`) | `9.627e-10` (same) | `0.99991` | `0.9997` | `1.5e-11` |
| 4 | `1.903e-09` (`+1.6e-9 / -3.0e-10 / -1.90e-9`) | `1.903e-09` (same) | `0.9735` | `0.9989` | `5.9e-10` (elsewhere than 19-21) |

One operand, three symptoms.  Every other stage-3 operator is at or below
`1e-15 m/s` on the same operands.

## 2. The arm (inside the WS-RK3 identity; no public selector)

NEMO has ONE `e3u(Kmm)`; legoESM already builds it once, in
`_nemo_ws_qco_stage_faces` -> `vertical.nemo_qco_live_face_geometry_cgrid`
(rows S-18/S-21), and the stage program already hands it to the tracer
transport and to the vertical UP3 (`geom[4]`).  The horizontal momentum
advection inside `tendencies()` was the remaining site building its own.

Change: `latlon_cgrid_ocean_baroclinic_tendencies` (and the model's
`tendencies()` wrapper) take an optional `momentum_flux_face_thickness=(h_u,
h_v)` that the flux-form momentum advection uses in place of its own
`min_cell_to_uface(h_k)`; every caller that does not pass it is
bit-identical.  The WS-RK3 stage program passes the stage's NEMO
`(e3u(Kmm), e3v(Kmm))` to all four of its `tendencies()` calls: the
step-entry Kbb call (stage 1 RHS and `F_slow`), the two S-21 stage-1 helper
calls, and the stage-2/3 `_mom_pert_ws` calls.  Private one-variable control
`_NEMOWSRK3TestHooks.legacy_hadv_min_face_thickness=True` restores the old
thickness at every site (harness/gate arm only; NEMO has no such switch).
No default keeps the old behaviour on the certified cards (Rule 3).

## 3. Predictions (frozen)

Baselines: `ssh_walk/seed_after/*` and `stage3_selector/gates_after/*`
rows (= this branch's tip `032bedd3a`, re-measured by the before-run of
this round), fp64, CPU, byte-identical protocol.

| # | card | quantity | prediction | REFUTED if |
|---|---|---|---|---|
| P1 | OVERFLOW | kt=2 `u` (instantaneous) | `4.551736e-10 -> <= 1.0e-11` (linearised `7.1e-12` own operator, `1.7e-12` transcription) | `> 4.6e-11` (less than 10x) or not improving |
| P2 | OVERFLOW | kt=1 stage-2 `u` | `9.433404e-11 -> <= 3.0e-12` (linearised `1.6e-12`); stage-1 `u` BIT-IDENTICAL (`6.501744e-15`; eta(Kbb)=0 so `min(e3t_0) == e3u_0 == e3u(Kbb)`) | stage 2 `> 9.4e-12`; stage 1 moves |
| P3 | OVERFLOW | kt=2 `S`, `SSH` | BIT-IDENTICAL (`SSH`: the external solve consumes the Kbb `F_slow`, unchanged at eta=0; `S` uniform).  `T` may move by `<= 3e-14` normalized (the stage-3 tracer transport carries `u2_corr`, which moves by `9e-11` at faces 19/21 where dT/dx = 0) -- no direction claimed | `SSH` or `S` move; `T` moves `> 3e-14` |
| P4 | both | `legacy_hadv_min_face_thickness` arm | reproduces the pre-fix rows BIT-FOR-BIT (kt=1..10 trajectory both cards, kt=2 stage sweep) | any difference |
| P5 | OVERFLOW | `slow_u` substep 1 from NEMO's exact entry (kt=2/3/4) | `1.296e-10 / 9.63e-10 / 1.90e-9 -> <= 1e-13 / <= 3e-11 / <= 7e-10` (each `>= 10x` except kt=4 `>= 2.5x`, where the replay leaves `5.9e-10` off-front) | any of them drops `< 2x` |
| P6 | OVERFLOW | kt=3 `u`, kt=10 `u` | kt=3 `8.843722e-09` drops `>= 5x`; kt=10 `2.629597e-05` drops `>= 10x` (the walk is this operator's per-step injection, section 0) | kt=10 `< 2x` |
| P7 | OVERFLOW | kt=10 `SSH` `1.473509e-06`, `T` `2.128213e-09`; kt=60 rows | both drop `>= 3x` at kt=10 (`slow_u` is this operand); kt=60 reported, no factor | kt=10 `SSH` or `T` `< 1.5x` |
| P8 | LOCK | every row kt=1..60 | BIT-IDENTICAL: the oracle `ssh` is `0` (`~1e-21` from kt=8: `1 + r3u == 1.0` in fp64), so `min` of the stretched thicknesses equals `e3u_0*(1+r3u)` exactly on the zco mesh | any LOCK row moves |
| P9 | OVERFLOW | 6120-step statistics fp64 + fp32 (six rows) | UNMEASURED before; reported before/after, no direction claimed | -- |
| P10 | unit | new direct test | fails on the reverted code (tilted LOCK entry: faithful vs legacy hook differ; from rest identical) | passes on reverted code |

Rule 8: any row that worsens is disclosed, not reverted; the 6120-step
`u_linf` row is already OUTSIDE on both sides of the previous two landings
and is expected to move either way inside its fp32 floor.

## 4. What this instrument cannot see

The replays act on NEMO's kt=1 operands and the exact kt=2..4 entries; the
free-run rows at kt>=3 compound this operator's injection with the barotropic
coupling, so P6/P7 are factors, not values.  The kt=4 `slow_u` replay leaves
`5.9e-10` off the front faces (PLAUSIBLE: a second, smaller operand at the
partial-cell faces 22-24, where the reseed-vs-free 1.6% also lives); it is
reported, not owned here.
