# Lane 3b rung 3.5b bulk-flux preregistration

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Status: PREREGISTERED before bulk operand capture or legoESM comparison

## Scope and resolved oracle

This rung certifies the ice-side flux path actually executed by the accepted
C1D_OMIP_L3 SAS-ice oracle: `ice_sbc_tau` -> `blk_ice_1`, and
`ice_sbc_flx` -> `ice_alb`, `blk_ice_2`, `ice_flx_other`
(`icesbc.F90:52-108,115-199,310-437`).  It does not certify the ocean-side
NCAR algorithm.

The accepted overlay resolves `ln_NCAR=.true.`, constant ice-air coefficients,
and `Cd=Ce=Ch=1e-3` (`ORCA1-omip/EXPREF/namelist_cfg:129-142`; accepted
`ocean.output:561-582`).  ORCA1 never sets `ln_ECMWF`; its false value comes
from the shared reference and exclusivity with NCAR.  This differs from the
unmodified shipped C1D EXP_SASICE deck, which selects ECMWF and 1.4e-3 at
`cfgs/C1D/EXP_SASICE/namelist_cfg:96-117`.  Therefore the ORCA1-resolved
constant-coefficient ice arm is in scope; the shipped C1D ECMWF/1.4e-3 arm is
explicitly **UNCERTIFIED**.

Other resolved selectors are `nn_snwfra=2`, `rn_snwblow=.66`,
`nn_flxdist=-1`, `nn_qtrice=0`, `ln_cndflx=.false.`, and no ponds (accepted
`ocean.output:701-720,731-743`).  No other ice bulk option is constructible
through the new public selector.

## Mandatory pre-implementation search

The search found:

- `core/bulk_flux.py:68-76,242-258` owns shared scheme validation and
  `simple_bulk_fluxes`/MOST implementations;
- there is no `packages/coupler/legoesm/coupler/bulk_flux.py` in this checkout;
  coupler and surface components import the core implementation rather than
  owning a second one;
- `ice/sea_ice.py:489-537` is the existing ice bulk dispatcher and currently
  accepts constant/MOST/COARE3/Large-Yeager;
- `ice/config.py:359-409` owns `emissivity_ice`, `Cd_ice`, `Ch_ice`, and the
  `bulk_scheme` selector;
- `legoesm/thermo.py:250-357` contains Flatau ice vapour pressure and a
  Clausius-Clapeyron ice saturation helper, but not NEMO's Goff-ice formula or
  its exact derivative;
- no existing legoESM implementation of SI3 `ice_alb`, `blk_ice_2`, or
  `ice_flx_other` was found.

Implementation will therefore extend the existing core validator, thermo
helpers, and `ice/sea_ice.py`; it will not add a second bulk module.  The public
scheme name is `nemo_si3_constant` and cites `sbcblk.F90:1048-1177,1180-1346`.
The selector accepts only the ORCA1-resolved identity.  Private gate-only hooks
may ablate one transcription at a time, but are absent from `SeaIceConfig`.

## Oracle operands and time levels

The end-of-step exchange stream is deterministic SHA-256
`091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4`.
It cannot by itself establish input legitimacy: ZDF updates `qns_ice` and
thermodynamics updates `qml_ice/qcn_ice` before `l3xchg_dump`.  A config-local,
WRITE-only bulk operand stream will therefore register three frames per step:

| frame | time level | source boundary |
|---:|---|---|
| 0 `POST_BLK_ICE_1` | after coefficient selection/stress; before halo link | `icesbc.F90:83-108`; `sbcblk.F90:1085-1168` |
| 1 `POST_BLK_ICE_2` | after albedo and air-ice heat/mass fluxes; before `ice_flx_other` | `icesbc.F90:149-194`; `sbcblk.F90:1218-1346` |
| 2 `POST_ICE_FLX_OTHER` | after lead/basal-ocean exchange; immediately before `ice_thd` | `icesbc.F90:194-199,310-437`; `icestp.F90:182-206` |

Each frame will carry the active C1D scalar of every executing operand and
output.  The writer is observational only and must leave the deterministic
exchange stream, thermodynamics stream, `ocean.output`, and restarts
byte-identical to the exchange-stable run.

## Registered comparison and bar

The gate runs all 8,760 hourly steps in fp64 on CPU and prints JAX/NumPy dtypes.
For each registered output it reports
`abs(lego-nemo)/max(abs(nemo),1)` and requires `<=1e-15`, pointwise.  It reports
the first and largest over-bar row without relaxing the threshold.

The production transcription covers:

- NEMO Goff-ice saturation and analytic derivative
  (`sbc_phy.F90:665-711,727-790`), Exner surface temperature
  (`:321-358`), constant coefficients, wind/stress
  (`sbcblk.F90:1085-1168`);
- SI3 albedo and snow fraction (`icealb.F90:124-185`;
  `icevar.F90:1564-1601`);
- radiative, sensible, latent, sensitivity, precipitation/sublimation, heat
  content, and shortwave-transmission formulas (`sbcblk.F90:1218-1346`);
- the active ice-relative-velocity lead/basal exchange branch
  (`icesbc.F90:310-437`).

The initial hypothesis is **CONFIRMED** only if every registered active output
is at bar.  An over-bar row is **DEBT** and is reported by first owner.  Private
one-variable arms are preregistered for (i) NEMO Goff-ice saturation/derivative,
(ii) NEMO constants and operation order, (iii) SI3 albedo/snow fraction, and
(iv) the active ice-relative-velocity `ice_flx_other` branch.  Each arm changes one
default-true private hook; no mixed public physics is exposed.

## Exchange coverage and controls

Every field in the 36-field exchange schema must be dispositioned.  Direct
bulk outputs are VERIFIED against their registered pre-thermodynamic time
level.  Fields subsequently modified by thermodynamics are VERIFIED in the
bulk frame and WAIVED at end-of-step with the mutation cited.  Ocean-side NCAR,
thermodynamic, diagnostic, and inactive `rCdU_ice` fields are WAIVED with a
specific owner/reason; no field may be absent from the register.

Plants perturb one output of each of `blk_ice_1`, `ice_alb`, `blk_ice_2`, and
`ice_flx_other`, plus an unregistered exchange field and the stream hash.  Each
must exit nonzero.  A selector plant must reject a non-ORCA1 coefficient or
bulk option.  JIT and reverse-mode gradients through the selected existing
ice dispatcher must remain finite.

## Decisions

| Choice | State | Disposition |
|---|---|---|
| Diagnose/stabilize exchange stream first | ASKED | Completed in `e4bf9c3eb1d`. |
| ORCA1-resolved public scope only | ASKED | `nemo_si3_constant` is single-identity. |
| Use the existing ERA5 member | ASKED | No synthetic forcing. |
| Pointwise 1e-15 bar for every year step | ASKED | Gate owns the verdict. |
| Instrument missing operands in a copied source | ASKED | WRITE-only three-frame registry above. |
| Certify unmodified shipped C1D ECMWF/1.4e-3 | UNASKED | Explicitly uncertified. |
| Certify ocean-side NCAR bulk | UNASKED | Outside this ice-side rung. |
| Add a second bulk-flux module | UNASKED | Forbidden; existing modules extended. |
| Expose private ablation mixtures in configuration | UNASKED | Forbidden. |
| Modify shipped NEMO | UNASKED | Forbidden. |

## Flagged for future deletion

Nothing is deleted.  Historical drifting streams and copied run roots remain
retained and are flagged rather than removed.

## Pre-implementation correction

Source/config resolution after the initial preregistration found that the C1D
driver suppresses the dynamics *timestep* with `ln_c1d`
(`icestp.F90:139-140`), while the accepted namelist still resolves
`ln_icedyn=.true.` (`ocean.output:610`).  Consequently `ice_flx_other`
executes its ice-relative-velocity branch (`icesbc.F90:322-341`), not the
fallback atmospheric-stress branch at `:342-346`.  The original
"active no-dynamics branch" statement is retracted before legoESM
implementation.  The operand registry includes `u_ice/v_ice`, `ssu_m/ssv_m`,
`drag_io`, and the resolved branch flags.
