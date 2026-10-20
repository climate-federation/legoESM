# GRIDLADDER-VMIX L5 round 21 receipt — ORCA2 mixed-layer eddies

Status: **ACQUISITION_NEEDED**.  This round surveyed mixed-layer eddies,
river-mouth mixing, and (for the record only) interior tracer damping.  It
selected L5 as mixed-layer eddies on certified G9, committed the card and
acquisition tool, and ran only the tool's read-only preflight.  Codex did not
build or run NEMO, and no record was scored.

The frozen preregistration is commit `18498bec4a`.  The L5 implementation and
exact acquisition tool are commit `acda81beb9`.

## Survey

### Mixed-layer eddies — selected L5

ORCA2's experiment deck changes only `ln_mle=T` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:312`.  Its resolved reference group is
`cfgs/SHARED/namelist_ref:950-960`: `rn_ce=0.06`, `nn_mle=1`,
`rn_lf=5000 m`, `rn_time=172800 s`, `rn_lat=20 degrees`, `nn_mld_uv=0`,
`nn_conv=0`, and `rn_rho_c_mle=0.01 kg/m3`.  The L5 deck is therefore exactly
G9 with that one flag changed; every other card remains flag-off.

The running DINO/ORCA2 build contains `tra_mle_trp_RK3_t` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:144`.  It diagnoses mixed-layer
depth and mean buoyancy from the stage's potential density and live layer
thickness at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:234-277`, takes the
`nn_mle=1` streamfunction arm, applies its vertical structure, and adds the
horizontal transport at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:299-347`.  The complete namelist is
read reference-first and experiment-second at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:643-648`; the selected constant-f
coefficient is formed at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:696-697`.

`tra_adv_trp` calls EIV and then MLE only at stage 3, evaluating the EOS at
the Kmm tracer state before MLE, at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:254-263`.  It then reconstructs the
vertical transport from the augmented horizontal pair at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:296`.  MLE therefore enters through
tracer advection and needs certified G9, not G5 or L3.  It consumes live
tracers, density, geometry, metrics, and masks; it has no external input file.

The feature is compiled into the existing binaries: the round-20 L3 smoke log
prints the complete `tra_mle_init` group and its flag-off verdict, and the
compiled source above contains the called RK3 routine.  legoESM already had
the Fox-Kemper diagnosis and C-grid transport construction.  L5 extends that
operator with a `through_fct` route, suppresses its former standalone centred
tendency on that route, adds its existing U/V transports to the existing
stage-3 transport tuple, and reuses L3's WZV reconstruction.  No second MLE
operator was introduced.

### River-mouth mixing — proposed L6

The deck selects `ln_rnf_mouth=T`, `rn_hrnf=15 m`, and
`rn_avt_rnf=0.001 m2/s`, with both runoff and its `socoefr0` mouth mask in
`runoff_core_monthly`, at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:159-170`.  Initialization derives the number
of affected W levels and calls the mouth-mask reader at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:504-526`; that reader obtains the mask
from the configured runoff file and fixes its vertical profile at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:569-588`.

The once-per-step vertical-physics call is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`.  After copying the selected
turbulent closure, `zdf_phy` adds twice the configured coefficient on the
mouth mask at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:353-355`; the resulting
diffusivity is then used by the tracer stages.  This mechanism acts without
tracer advection or eddy velocity, so its base is G5.

`runoff_core_monthly.nc` is already in the ladder's immutable 16-file ORCA2
input set and in the on-disk `ORCA2_ICE_v5.0.0.tar.gz` extraction.  No fetch or
vendoring is required.  legoESM already transcribes runoff mass and tracer
content, but not this mouth-mask augmentation of vertical diffusivity; the
next rung should extend the existing vertical-mixing composition rather than
duplicate runoff forcing.

### Interior T/S damping — record only, deferred

The deck selects full-column tracer damping with `ln_tradmp=T` and
`nn_zdmp=0` at `ORCA2_OMIP_L4/EXP00/namelist_cfg:326-329`; the reference names
`resto.nc` at `cfgs/SHARED/namelist_ref:1012-1018`.  The compiled routine is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:142`.  It reads/interpolates target
T/S and adds `resto * (target - before)` throughout the water column at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:181-194`.  The RK3 program calls it
after lateral diffusion, geothermal heating, and BBL at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`.

`resto.nc` is not in the current 16-file ladder set, but it is already present
both in the on-disk `ORCA2_ICE_v5.0.0.tar.gz` and its extracted directory, so
no download or vendoring decision is needed.  This lane does not start the
damping rung because Decision 141 permits lane A to own it.  legoESM has
generic sponge/restoring machinery, but no ORCA2 card selects this exact
full-column climatology program yet.

The proposed order is therefore: L5 MLE on G9; L6 river-mouth mixing on G5;
interior damping remains unassigned pending lane-A coordination.

## L5 implementation and controls

The L5 card binds every resolved `namtra_mle` value, requires
`through_fct=True`, refuses each single-field wiring plant, and requires the
WS-RK3 FCT program.  The existing standalone MLE configuration remains the
default for every non-L5 caller.  The new horizontal-transport return exits
before the standalone tracer-flux divergence, so the selected path does not
compute and discard a second tendency.

The exact x64 traced transport test pins 18 nonzero U transport words, zero V
transport words in its zonal-front fixture, and an exact zero source-ordered
U sum.  The production one-step control removes only L5 MLE and pins the
resulting unequal-bit counts as 24,588 T cells, 21,880 S cells, and zero u, v,
or SSH cells.  Thus deleting the stage-3 wiring fails while the tracer-only
scope remains non-vacuous.  The focused card/operator run passed; the existing
MLE unit file also passed.  No new division or square root was added: the L5
route reuses the existing guarded MLE diagnosis and adds no new NaN-gradient
source.

## Acquisition preflight

The committed entry point defaults to read-only preflight and reserves
`--run` for the operator.  At clean `bedbcf650e`, while both target
configurations and the evidence parent were absent, it reported:

`L5_PREFLIGHT_OK checklist=7 plants=all_refused`

The seven checks cover: exact `tramle` insertion plus compiler syntax; a real
`makenemo` reference; every rendered clock; strict six-frame layout and
zero-first dry/halo canonicalization; numeric Fortran-log comparison with a
neighbouring-float plant; absent-parent creation plus full completed-run audit
plants; and absence of both target configurations and the evidence root.
Admission rechecks the binary, decks, logs, immutable inputs, smoke completion,
the writer-free boundary arm, and every twin.  The operator workflow is a
two-step smoke, a writer-free run to NEMO's own stop or 100 steps, then three
twins to a common even endpoint no greater than 10 steps.  No score is allowed
until that record is admitted.

## Landing gates

All landing gates used `JAX_ENABLE_X64=1`, the exact `land.sh` `PYTHONPATH`,
and CPU JAX.  At the final substantive tip:

- this receipt's explicit citation gate: **PASS**, 19 citations, zero
  failures, zero unmapped citations, and zero map-audit failures;
- citation-gate unit suite: **17 passed in 5.25 s**;
- L5 card/operator suite, including the exact jitted transport and production
  one-step controls: **16 passed in 188.89 s**;
- exact default ORCA2 landing list: **145 passed in 1000.10 s**;
- private-import ratchet, rerun separately: **6 passed in 5.16 s**; and
- acquisition preflight: `L5_PREFLIGHT_OK checklist=7 plants=all_refused` at
  `bedbcf650e`.

The first default-list attempt reported 140 passed and five failed.  All five
were the same legacy-card compatibility defect: cards with `physics=None`
reached the new MLE selector before doing any physics.  The selector now treats
that legacy shape as MLE-off.  The five exact failures then passed in 634.29 s,
and the complete default list above passed from the clean committed tip.  No
NEMO command was run and no record was scored.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r21-504231288
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r21-504231288/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l5_orca2mle/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — L5 built and preflighted; acquisition needed; record not acquired or scored
