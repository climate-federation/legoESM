# NEMO testcase lane 1 — phase-2 geometry, IC, and kt=1 receipt

Date: 2026-08-31  
Session: `01a0591f-a335-7060-9257-6c47bf2149ee`  
Preregistration: `bb718ae7ca60912962187d2c39c5cac358f96e48`  
Implementation: `b5fb6b34e6d802b035267da41f2906908e8c3608`

## Verdict

**VERIFIED for the dispatched scope.**  The two legoESM pure-config cards match
their certified NEMO meshes inside the DINO pointwise `1e-15` bar and match all
five native `kt=1` `Nbb/before` fields exactly.  This is not a trajectory,
tendency, EOS, or BBL-transport verdict; those rows remain UNMEASURED.

| card | mesh registry | measured geometry | max normalized error | kt=1 T/S/u/v/SSH |
|---|---:|---:|---:|---:|
| LOCK_EXCHANGE-zco | 35/35 disposed (30 VERIFIED, 5 WAIVED) | 30/30 AT-BAR | `2.212e-16` | 5/5 exact (`0`) |
| OVERFLOW-zps | 45/45 disposed (39 VERIFIED, 6 WAIVED) | 39/39 AT-BAR | `1.418e-16` | 5/5 exact (`0`) |

Every scientific floating legoESM array printed by the gate is `float64`;
masks are boolean and bottom indices are integer.  Both cards call
`set_policy(PrecisionPolicy.fp64())` before allocation
(`nemo_testcase_recipe.py:129,160`).  The comparison ran on CPU; NEMO was not
rerun and its certified CPU artifacts were read-only inputs.

## Pre-implementation search reconciliation

The preregistration records the exact search terms and paths.  The search
found reusable canonical implementations for every mechanism needed here:

* Cartesian beta-plane C-grid geometry and closed masks;
* full-step and partial-cell vertical coordinates plus face-min operators;
* the NEMO pure recipe/config surface (`fct2`, flux-form `upwind3`,
  `rk3_ws`, adaptive vertical advection, NEMO PGF selection);
* NEMO C-grid bridge conventions, fail-closed time-level registry, and the
  DINO coverage/bar machinery;
* the canonical advective BBL implementation in `ocean.physics.bbl_adv`.

The existing generic lock/overflow experiments and Veros fidelity configs were
found but rejected as card sources because their geometry and ICs differ from
the certified NEMO cases.  No bespoke solver was added.  The cards pin the
BBL selectors as data (OVERFLOW `nn_bbl_adv=2`, `nn_bbl_ldf=0`,
`rn_ahtbbl=1000`, `rn_gambbl=20`; LOCK off); applying and measuring that shared
operator belongs to a later step harness and is not claimed here.

## Exact card resolution

| setting | LOCK_EXCHANGE-zco | OVERFLOW-zps |
|---|---|---|
| domain | `130 x 3`, 500 m, closed, `f=0` | `202 x 3`, 1000 m, closed, `f=0` |
| active vertical grid | 20 x 1 m full steps | 100 x 20 m reference steps, zps bottom cells |
| NEMO dummy record | one dry bottom record | one dry bottom record |
| timestep / registered length | 1 s / 61200 | 10 s / 6120 |
| EOS selector | `veros_gsw` | `veros_gsw` |
| tracer / momentum | `fct2`; flux-form `upwind3` | `fct2`; flux-form `upwind3` |
| time / vertical advection | `rk3_ws`; adaptive implicit | `rk3_ws`; adaptive implicit |
| PGF | `smc03`, NEMO trapezoid | `smc03`, NEMO trapezoid |
| explicit lateral mixing / drag | off / off | off / off |
| constant vertical coefficients | `A_v=1e-4`, `K_v=0` | `A_v=1e-4`, `K_v=0` |
| advective BBL pin | off | option 2, gamma 20 s |

`veros_gsw` is the existing recipe's TEOS-10-like selector, not a numerical
claim of bit-identical NEMO TEOS-10 density.  Density, `rab`, and BN2 remain
UNMEASURED exactly for that reason.

## D2.5 partial-cell payoff

The first preregistered OVERFLOW comparison failed, as the gate was intended
to do: the legacy interface-indexed coordinate produced 40 wrong active cells
and a `1.947e-4 m` bottom-thickness mismatch.  Reading the running oracle arm
showed that `usrdef_zgr.F90:157-180` selects `k_bot` from T-point depths and
retains the unsnapped bottom thickness.  The shared canonical constructor now
offers explicit `bottom_index_rule="nemo_tpoint"`; its historical
`"interface"` default is unchanged (`vertical.py:724-832`).

With that selector, every active OVERFLOW `e3t_0` value is bit-identical.  The
U-face field is independently built with canonical neighboring minimum and is
also bit-identical.  This pays the D2.5 rule against the general NEMO source:
`tools/DOMAINcfg/src/domzgr.F90:1163-1169` computes U/V/UW/VW scale factors as
neighboring minima.  The active idealized specialization is
`tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:182-186`.  The three-row tank has no
active V or F transport point, so the registry says that loudly rather than
pretending those rows exercise a transport face.

A second last-bit discrepancy came from vectorized NumPy `tanh` lowering in
one static bathymetry column.  Scalar host-libm `tanh` matches the gfortran
initialization bit-for-bit (`nemo_testcase_recipe.py:167-171`); this is static
card construction, not solver mimicry.

## IC and time-level receipt

The time registry now labels `oracle_step_entry_kt00000001.bin` **before** with
the source citation (`time_levels.py:35-42`).  The instrument writes
`ts/uu/vv/ssh(...,Nbb)` at
`tests/*_OMIP_L1/MY_SRC/stprk3.F90:88-100`, before forcing, `stp_2D`, and the
three RK calls at lines 202-225.  Halo removal yields `202 x 3 x 101` and
`130 x 3 x 21`; the one dummy depth record is disposed explicitly.  Wet/dry
T and S, native east/north-face u/v, and SSH all compare exactly.

## Non-vacuity, artifacts, and withheld rows

All three planted controls exit nonzero:

* file-side unknown array: `missing=['PLANTED_UNACCOUNTED_FILE_ARRAY']`;
* `e1t + 1 m`: normalized `1e-3 > 1e-15`;
* wet IC `T + 1 C`: normalized `5e-2 > 1e-15`.

Gate reports live under the required data root:

| artifact | SHA256 |
|---|---|
| LOCK mesh / kt=1 dump / gate JSON | `ec3200f559cb44ee76d00498aac168dd6452cc29bd0abdf955dfc4ab061ed935` / `c9f23d441865c566e3edf0301aca4f6e440259ade1722b725a0aa0fd4c2839e1` / `780452b1b914b74a238e82fc1f28eed8cbcc6309ea506b424d22ff6b4274ca84` |
| OVERFLOW mesh / kt=1 dump / gate JSON | `4692b893eddee3eea5cee2e6f04e1d7fc55e6685100e349369051a6914cbd280` / `cf0183e563aba8b8848bc5dea470c9e50aab2d987ff5da86241e8f82494d5c66` / `c417995ae4c5325467513185fbf5b280e4c5e7a7187ace2413a4df75565a7d0e` |

The gate JSON lists `kt>1` trajectories, RK-stage tendencies, TEOS-10
density/`rab`/BN2, FCT tendency, adaptive vertical-advection partition, and
BBL transport as UNMEASURED.  No trajectory language is used for this receipt.

The required read of GitHub issue #1455 was attempted, but this environment
could not connect to `api.github.com`; no issue comment was posted and no
external-state claim is made.
