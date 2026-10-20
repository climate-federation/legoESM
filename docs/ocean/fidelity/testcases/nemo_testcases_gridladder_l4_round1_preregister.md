# NEMO testcase grid ladder L4 — round 1 preregistration

Date: 2026-10-20

Starting parent: `7a31b67ef96e3954b64b204671942c84119fa923`

Status: **PREREGISTERED BEFORE L4 IMPLEMENTATION, ORACLE ACQUISITION, OR
SCORING.**  Decision 140 defines L4 as the certified G5 card plus the ORCA2
variable geothermal bottom-temperature boundary condition.  This turn may
prepare a WRITE-only oracle acquisition, but it may not execute NEMO or score
L4 before the new record exists.

## Verified oracle arm and available input

The compiled ORCA2_OMIP_L4 deck selects exactly `ln_trabbc=.true.`,
`nn_geoflx=2`, `cn_dir='./'`, and `sn_qgh=('geothermal_heating.nc', -12.,
'heatflow', .false., .true., 'yearly', '', '', '')`
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:254-264`).  The executed, preprocessed
`trabbc` initializes this arm at `trabbc.f90:177-243`, converts the input from
mW/m2 with `r1_rho0_rcp * heatflow * 1.e-3` at `:243`, and adds that value to
the bottom cell's stage-3 temperature Krhs divided by live Kmm thickness at
`:156-160`.  The call is after tracer lateral diffusion and before BBL in
`stprk3_stg.f90:734-740`; both neighboring processes are off in G5.

Both certified G5 executables contain `__trabbc_MOD_tra_bbc` and
`__trabbc_MOD_tra_bbc_init`, and their preprocessed `trabbc.f90` is bytewise
the same source as the ORCA2_OMIP_L4 build.  The official local
`ORCA2_ICE_v5.0.0.tar.gz` contains
`ORCA2_ICE_v5.0.0/geothermal_heating.nc`; streaming that member and hashing the
already-extracted file each gives SHA-256
`e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb` and
116,499 bytes.  Therefore no fetch or vendoring is needed.

legoesm already has the shared bottom-cell operator in
`physics/geothermal.py:110-172`; it selects one deepest wet cell and evaluates
`Q/(rho*c_sw*h)`.  Its separate coupler wrapper applies a post-step
forward-Euler update (`coupler/geothermal_apply.py:27-87`), not NEMO's
stage-3 Krhs statement.  L4 will extend the shared operator with NEMO's written
association and wire it at the existing stage-3 source boundary.  It adds one
optional traced forcing field, not prognostic model state.

## Frozen L4 card and implementation predictions

The card name is `ORCA2L4-bbc`.  Its complete delta from `ORCA2G5-ice` is one
default-false config selector, `rk3_stage3_geothermal_nemo_written=True`, plus
the required per-column `geothermal_heat_flux_mwm2` forcing.  The forcing keeps
the deck's native mW/m2 units so the production statement includes the
literal `1.e-3` conversion.  The L4 deck differs from the G5 deck only in the
`nambbc` block above.  G9 tracer advection is not selected: the L4 deck keeps
G5's `ln_traadv_OFF=.true.`.

The NEMO association to implement is frozen as:

1. `rho0_rcp = rho0 * rcp`, then `r1_rho0_rcp = 1 / rho0_rcp`
   (`eosbn2.f90:2504-2507`);
2. `qgh_trd0 = (r1_rho0_rcp * heatflow_mwm2) * 1.e-3`
   (`trabbc.f90:242-243`); and
3. `Krhs(bottom) = Krhs(bottom) + qgh_trd0 / e3t_Kmm(bottom)`
   (`trabbc.f90:156-160`).

Each source statement is materialized with the shared `nemo_source_round`.
The bottom mask is reused from the shared geothermal module.  Exact direct
tests must cover eager and JIT execution, traced flux input, x64, and gradients.
Removing the L4 selector or forcing wire must make the selected-card test fail.

## Frozen controls and acquisition boundary

Before/after production JAXPR text must be identical with L4 off for full
ORCA2, DINO, GYRE, SMT, VORTEX, OMT, and rung 0.  The two-step values for every
certified G1--G9 card must be exactly unchanged.  L4 must resolve to every G5
selector plus only its geothermal selector; any other selector delta refutes
the card.

The committed acquisition has three run phases: a two-step instrumented
smoke; a writer-free run to NEMO's own stop or 100 steps using the certified
G5 executable; and three separately staged instrumented twins of at most ten
steps.  The copied geothermal input must have the SHA-256 above.  The new
rank-zero record stores the input `qgh_trd0`, live bottom-cell Kmm thickness,
and the resulting bottom-cell temperature trend.  Admission later requires
header-derived exact EOF, finite values, exact statement reconstruction,
nonzero heating on at least one wet column, canonical land slots, three-way
raw twin identity, and unchanged ordinary restart/history payloads.  Every
numeric log check parses both log and namelist tokens as floating-point values
and compares them exactly.

Default invocation is a read-only seven-item preflight.  `--run` stages the
above runs; `--admit-existing` validates an operator-produced acquisition.
The preflight must pass while the evidence parent does not yet exist, and a
plant for each checklist item must be rejected by the same validator.  This
round stops at `ACQUISITION_NEEDED`; it makes no L4 fidelity verdict.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| add G5 + geothermal as L4 | ASKED, Decision 140 | one card selector and one traced forcing input |
| add G9 tracer advection | conditional, not reached | deck keeps tracer advection off |
| fetch or vendor geothermal input | forbidden stop condition | unnecessary; official local bytes hash-match |
| add prognostic model state | forbidden stop condition | unnecessary |
| execute NEMO in this sandbox | operator-only | never |
| score before a reproducible record exists | forbidden | no score this round |
| change any G1--G9 result or non-L4 JAXPR | forbidden control | exact zero movement |
