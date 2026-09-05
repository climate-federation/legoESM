# Lane 3b round 22 preregistration — SI3 scalar-libm routing

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `17bf389a6dbe`

## Search first and source map

The pre-implementation search found one existing selectable transcendental
implementation in `packages/core/legoesm/core/transcendentals.py`.  Its
`log`, `log10`, and binary `pow` already dispatch through scalar `libm.so.6`
only under `PrecisionPolicy.fp64(transcendentals="libm")`; the default native
policy remains JAX.  The certified SI3 code already uses the shared
`nemo_source_round` statement guard.  No second policy, thermodynamics model,
bulk module, or albedo implementation will be created.

The live, in-scope statement map is:

| NEMO 5.0.2 source statement | existing legoESM executing site | one change |
|---|---|---|
| `sbc_phy.F90:674-679`, especially `:677` `LOG10(ztmp)` and `:679` `10._wp**zle` | `thermo.py::nemo_si3_saturation_over_ice` | route through shared policy `log10` and `pow` aliases inside the existing source-round nesting |
| `sbc_phy.F90:331-335`, Exner `(rpref/ppa)**rgamma_dry` | `core/bulk_flux.py::nemo_si3_constant_fluxes` | route the existing rounded power through shared policy `pow` |
| `icealb.F90:124-130,159-160`, `LOG(rn_alb_hpiv)`, `LOG(0.05)`, and `LOG(ph_ice)` | `ice/sea_ice.py::_nemo_si3_ice_albedo` | route the three logs through shared policy `log` inside the existing source-round nesting |

Snow/pond exponential statements `icealb.F90:167-175` are already policy
routed and are not a new variable.  The derivative's folded `LOG(10._wp)` is
the pinned `constants.ln10_nemo`, not a runtime transcendental.

## Predictions fixed before measurement

The only changed variable is the transcendental provider under the explicitly
selected libm policy.  For each source statement, the NEMO association and
`nemo_source_round` boundary remain unchanged.  Therefore:

1. the scalar-math C1D SI3 bulk gate will remain exactly **271,560 / 271,560
   bit-identical**, with zero over-bar rows;
2. the C1D exchange gate will remain exactly **448,950 / 448,950
   bit-identical**;
3. the coupled-slab ordered walk will retain the same bits through its current
   first stop, kt2 `PRE_SSM.u = 1.1172865415493005e-7`, which remains the GYRE
   lane's prognostic-`uu_b/vv_b` debt and is not touched here;
4. the registered ice-thermodynamics column results will not move; and
5. LOCK, OVERFLOW, GYRE, and the ORCA2 kt1 entry rows that do not execute these
   SI3 sites will have byte-identical residual/report arrays versus Round 21.

If native and scalar libm differ for any recorded live operand, item 1 is
REFUTED and the first non-bit row is a new measured finding; it will not be
hidden by changing the bar.

## Binding controls

Three private, non-configurable one-variable poison arms will replace only the
newly routed policy result at a named site:

| private arm | poisoned site | required observation |
|---|---|---|
| `libm_si3_saturation` | Goff-ice `LOG10`/power site | at least one scored `q_sat`-downstream row changes and the gate exits nonzero |
| `libm_si3_exner` | constant-flux Exner power | at least one scored `theta_surface`/sensible-flux downstream row changes and the gate exits nonzero |
| `libm_si3_albedo_log` | no-pond ice-thickness log interpolation | at least one scored albedo/shortwave row changes and the gate exits nonzero |

The hooks are private test/probe arguments, not public physics selectors; no
Frankenstein combination is constructible from a card.

## Rule 8/11/12 and scope

Any moved registered row will be reported with the first boundary and owner.
No prior attribution is retracted by prediction.  Native ORCA2 IWM, RGB
shortwave, EOS enthalpy, and non-NEMO bulk-arm sites remain an explicit debt
inventory and are not opportunistically converted.  The target stays the
ORCA1 single-category thermodynamic identity; no multi-category work and no
slab seed or prognostic `uu_b/vv_b` work is authorized.

## ASKED / UNASKED

| choice/action | status | disposition before run |
|---|---|---|
| route the three certified SI3 sites through scalar libm | ASKED | preregistered above |
| add the three row-binding poisoned-policy controls | ASKED | preregistered above |
| add and byte-pin the libc `0**0` edge case | ASKED | expected result `1.0`, exact libc bytes |
| remeasure the named SI3/C1D/cross-card registers | ASKED | predictions fixed above |
| convert unmeasured native NEMO-identity sites | UNASKED | inventory only |
| implement multi-category SI3 or prognostic `uu_b/vv_b` | UNASKED | explicitly excluded |
