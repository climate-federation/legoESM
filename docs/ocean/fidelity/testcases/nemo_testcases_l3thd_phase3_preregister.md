# Lane 3b SI3 thermodynamics Phase 3 preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED before step-74 branch evaluation, one-variable arms,
or fixes.**

## Fixed protocol and identity

No selector, input, timestep, precision, or comparison bar changes.  The card
remains the ORCA1-resolved one-category HFN identity: BL99 3+3/P07,
`nn_icesal=2`, `rn_sinew=.75`, ponds and lateral melt off, CPU, explicit fp64,
one-hour NEMO-written ZDF inputs, and the `1e-15` boundary bar.  The accepted
Phase-2b thermodynamics and exact-entry-input hashes remain
`7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b`
and `5522eadce595408b00065fa30d8b41fccb3815bee76d6fbf5ba3adbb2656cb27`.

The ordered active chain is `ice_thd_frazil`, category selection/1-D
conversion, `ice_thd_zdf`, `ice_thd_dh`, `ice_thd_temp`, `ice_thd_sal`, a
second `ice_thd_temp`, 2-D conversion, `ice_thd_do`, then `ice_cor`
(`icethd.F90:109-190`).  HFN redistribution, ponds, virtual-ITD melting, and
lateral melt are inactive because `jpl=1`, `ln_pnd=.false.`,
`ln_virtual_itd=.false.`, and `ln_icedA=.false.` (`icethd.F90:158-181`).

## Step-74 discriminator

NEMO computes snow precipitation at `icethd_dh.F90:166-177`, then treats
`evap_ice < 0` as deposition at `:179-202`; if deposition occurs without new
precipitation, it assigns the deposited snow enthalpy from surface temperature
at `:186-188`.  It remaps all resulting segments with `snw_ent` at `:494-507`
and `:535-613`.  The executing legoESM path at
`bitz_lipscomb.py:524-547` removes mass for positive evaporation but has no
negative-evaporation deposition update.

Preregistered primary hypothesis **H74-DEPOSITION**: the first non-roundoff
continuous jump at kt74 is the first active negative-evaporation deposition
event.  It is confirmed only if all of the following hold:

1. the exact kt74 input has `evaporation < 0` and the preceding kt73 input does
   not create the same missing increment;
2. ENTRY and POST_ZDF remain in their prior arithmetic class, and the first
   branch/state disagreement is POST_DH snow thickness/enthalpy;
3. a private `_snow_deposition=False/True` arm changing no other operand moves
   kt74 POST_DH `h_s` and `e_s` by at least 100-fold toward NEMO;
4. the enabled arm follows the NEMO ordering and enthalpy expression above.

It is refuted if evaporation is nonnegative, a mismatch first appears before
POST_DH, or the one-variable arm improves by less than 100-fold.  Independently
evaluate and report these alternatives: first snowfall (`sprecip>0`),
snow-ice flooding (`icethd_dh.F90:441-485`), surface melting
(`:107-120,204-280`), salinity drainage/flushing (`icethd_sal.F90:204-248`),
the snow/no-snow conductivity branch (`icethd_zdf_bl99.F90:159-185`), and
post-chain correction/zapsmall (`icethd.F90:181-190`; `icecor.F90:111-116`).

## Oracle-entry injection census

For every exact-entry step and every registered boundary/field, retain the
normalised error rather than only aggregate counts.  Report:

- the first field row above `1e-12`, with its preceding at-bar boundary and
  evaluated branch conditions;
- the first field row above `1e-3`, likewise;
- all over-bar rows grouped by step, sub-call, and variable in a committed
  machine-readable artifact.

Thresholds apply to the already registered metric
`max(abs(legoesm-NEMO))/max(1,max(abs(NEMO)))`; they are reporting thresholds,
not replacement fidelity bars.  A plant must alter a branch label or omit an
expected row and exit nonzero.

## Melt-season discriminator and scaling

NEMO forms surface melt energy only when `t_su >= rt0`, through
`qml_ice=qns_ice+qsr_ice-qtr_ice_top-qcn_ice_top`, then `zq_top=max(0,qml*dt)`
(`icethd_dh.F90:107-120`).  It consumes snow layers first (`:204-225`) and
then ice layers from the surface (`:231-315`), before basal growth/melt
(`:321-424`), flooding (`:441-485`), and conservative remaps (`:494-519`).
The current legoESM `_dh_step` proceeds from snowfall/sublimation directly to
basal growth/melt (`bitz_lipscomb.py:524-580`) and contains no `qml_ice`
surface-melt sequence.

Preregistered primary hypothesis **HMELT-SURFACE**: missing NEMO surface melt,
including snow-first ordering, owns the summer minimum-thickness deficit.  It
is confirmed only if:

1. exact-entry frames first show a NEMO POST_DH surface thickness loss when
   `t_su >= rt0` and computed `qml_ice > 0`, while the existing legoESM arm
   does not remove the same surface mass;
2. a private `_surface_melt=False/True` arm changes only that ordered path and
   improves the same-step `h_s`, `h_i`, and enthalpy injection by at least
   100-fold at its first active step;
3. the integrated enabled-arm surface ice/snow removal has the dimensional
   scale of the observed `1.2273381762 m` annual minimum-thickness gap;
4. enabling the unbranched NEMO identity moves the annual minimum thickness
   and growth-onset rows toward NEMO without changing the fixed forcing.

Refute ownership if the first melt injection is instead bottom melt
(`zf_tt/qcn/fhld`), ZDF shortwave partition, or snow remapping.  Report each
candidate's exact-entry thickness tendency and its ratio to the observed gap
before attribution.

## Acceptance and controls

Land a fix only for a confirmed owner and only inside the sole SI3 identity;
NEMO supplies no switch for negative-evaporation deposition or the ordered
surface-melt path in active `ice_thd_dh`, so a confirmed implementation is
unbranched.  The post-fix full-year gate must report, without relaxing the
bar: first over-bar frame before/after; the seven requested growth-table
fields at steps 1, 10, 100, 1000, and 8760; and six phenomenology rows
(minimum and maximum thickness and their dates, melt onset, growth onset).
Movement is numeric only; no `matched` or `faithful` label is permitted unless
the row reaches its registered bar/floor.

Controls must make the branch census, deposition arm, surface-melt arm, and
year-frame cursor exit nonzero.  All oracle and legoESM state/operand arrays
must print `float64`; the separate precision-floor trajectory remains
explicitly `float32`.

## Choice register

- ASKED — locate the first step-74 branch disagreement from existing frames
  and active NEMO source order.
- ASKED — enumerate oracle-entry over-bar rows and the first injections above
  `1e-12` and `1e-3`.
- ASKED — scale surface, bottom, shortwave, and snow-first melt candidates
  against the measured minimum-thickness gap.
- ASKED — use one-variable private arms and implement confirmed owners inside
  the fixed SI3 identity where NEMO has no switch.
- ASKED — rerun the complete year gate and report before/after growth and six
  phenomenology rows.
- ASKED — CPU only, copy-only oracle instrumentation, explicit pathspec
  commits/local-git bundle, no shipped-tree edits, and no push.
- UNASKED — alternate SI3 identities, new tunables, changed thresholds,
  changed forcing, or coupled bulk-flux certification.
