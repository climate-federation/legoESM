# Lane 3b round-11 preregistration — scalar libm and coupled slab

Date: 2026-09-04
Tracker: `climate-federation/legoESM#1699`
Starting commit: `b834e734bb82df187c2c9843fa6ad97767c84e23`

## Fixed scope

This round retains the ORCA1-resolved SI3 identity, fp64, CPU-only execution,
the 2018 C1D forcing, and the pointwise normalized bar of `1e-15`.  It neither
alters a production default nor implements rung 3.6 outside the existing
coupler and ocean machinery.  Rung 3.6 follows the independently reviewed
`nemo_testcases_l3thd_rung36_preregister.md` without changing its oracle,
signs, selectors, frame order, coverage seed, private arms, or stopping rule.

## Pre-implementation search

The exact GYRE implementation from source commit
`61180a6776c` was fetched and checked out before local edits:

- `core/precision.py` owns the selectable `transcendentals="native"|"libm"`
  policy;
- `core/transcendentals.py` owns the only scalar-`libm.so.6` `exp`/`tanh`
  callbacks and their custom derivatives;
- its precision and transcendental tests and the oracle-fidelity skill's
  No-Frankenstein paragraph were imported from the same commit.

The imported `precision.py` predates lane 3b's public `nemo_source_round` and
therefore removes that symbol.  Searches found no surviving source-round owner
after the exact checkout.  Compatibility will be restored in one new shared
`core/source_rounding.py` owner, containing the already certified Round-10
implementation; all existing callers will import that owner.  The imported
precision/transcendental modules remain byte-for-byte those of `61180a6776c`.
This composition defect is reported as GYRE-lane debt rather than repaired in
the imported module.

On the certified SI3 bulk path, repository search found two runtime exponential
calls, both in no-pond `ice_alb`; Goff saturation uses native `log10` and power.
No executing constant-coefficient `blk_ice_1`, `blk_ice_2`, or
`ice_flx_other` `tanh` call exists.  The already measured native `sin`, `cos`,
`log`, `log10`, and power calls remain native.  The C1D card, not a default,
will explicitly install `PrecisionPolicy.fp64(transcendentals="libm")` before
tracing the operator.

For rung 3.6, search reconfirmed that
`coupler/ocean_forcing.py` is the single ice/ocean exchange assembly owner;
the lat-lon ocean already owns real freshwater, real salt, RGB penetration,
RK3-WS, split-explicit SSH, implicit vertical solve, surface-stress placement,
and bottom drag.  No top-ice-drag path exists.  The reviewed design's top-drag
and registered 1x1 OFF-operator prerequisites must extend those shared paths;
no second coupler or solver is permitted.

## Measurement A — scalar-libm closure

Prediction: with the C1D card explicitly selecting scalar libm and only the two
certified albedo exponentials routed through policy `exp`, the accepted runtime
will score exactly `271560/271560` bit-identical rows and zero over-bar rows.

- **CONFIRM:** all 18 prior `POST_BLK_ICE_2.{albedo,qsr_ice,qsr_tot}` rows
  become bit-identical, with every ordinary row unchanged.
- **REFUTE:** any non-bit row remains, any formerly exact row moves, or any row
  exceeds the normalized bar.
- **Plant:** a private test hook poisons the policy `exp` return at a registered
  nonzero snow step; the changed value must propagate to scored albedo-derived
  rows and the gate must exit nonzero.  The hook is not a public selector.

## Measurement B — rung 3.6 execution-order walk

The oracle must be a new `C1D_OMIP_L3_COUPLED_SM` copy built from
`arch-conda-scalarmath.fcm`, with zero `_ZGV*`, the reviewed namelist overlay,
and the design's WRITE-only frame registry.  The exact-entry gate walks:
initial state, `PRE/POST_SSM`, `POST_FZP`, `PRE/POST_UPDATE_FLX`,
`POST_UPDATE_TAU`, `POST_FWB`, stress staggering, drag coefficients,
`POST_TRA_SBC_RK3`, `POST_TRA_QSR`, `PRE_DYN_SPG_TS`, every `SSH_SUBSTEP`,
`POST_STP2D`, and `PRE_DYN_ZDF_SOLVE`.  It reports and stops at the first
pointwise row above `1e-15` when resolution needs a new decision.

Input preflight found the official C1D ERA5 member, but did not find ORCA1's
required `merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc` or
`weights_reg05_bilinear.nc` anywhere under `/data/abyssal/dbalwada`.  The ORCA1
deck names them at `EXPREF/namelist_cfg:163-170`; public NEMO documentation
identifies the chlorophyll product but provides no checksum or direct archive
URL.  Rung 3.6 must stop before configuration/build if these real files cannot
be fetched; `CHLA_BATS.nc`, constant chlorophyll, generated weights, or another
substitute are forbidden.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| import shared GYRE libm implementation from `61180a6776c` | ASKED | exact files imported; no local module edits |
| explicitly select libm on the C1D card | ASKED | preregistered above |
| close all 271,560 bulk rows bit-for-bit | ASKED | prediction fixed before run |
| implement reviewed rung 3.6 design | ASKED | conditional on its required real inputs |
| use a synthetic/constant/BATS chlorophyll substitute | UNASKED | forbidden |
| alter the reviewed coupled selectors, signs, cadence, or gate bar | UNASKED | not done |
| implement another coupler, solver, or math policy | UNASKED | forbidden |
| implement rung 3.6 before required input preflight passes | UNASKED | forbidden by reviewed design |
| modify/delete shipped NEMO files or push | UNASKED | not done |
