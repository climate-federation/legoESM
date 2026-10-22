# Grid ladder rung L8 — round 1 preregistration (gridladder_r56, Codex)

Date: 2026-10-22

Starting commit: `f32c3be7ac9d`

Decision 148 fixes the candidate: L8 is certified G9 tracer advection plus the
certified L1 diffusive BBL, L2 standard isoneutral diffusion, L4b geothermal
heating, and L6 full-column tracer damping.  No L3 eddy-induced velocity or L5
mixed-layer eddy term is selected in this round.  The library defaults and all
existing cards remain unchanged.

NEMO is operator-only.  This round designs and builds L8 and its acquisition;
it does not run NEMO, admit a record, score a trajectory, or certify L8.

## Oracle statement order, frozen before implementation

The executed ORCA2 stage-3 tracer program is:

1. clear every tracer `Krhs` to zero
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:600-605`);
2. add FCT advection
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:633-645`);
3. add the nonlinear surface boundary condition at the call ending on line 645,
   followed by penetrative shortwave heating
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:645-717`);
4. add isoneutral lateral diffusion
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:730-734`);
5. add geothermal bottom heating
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:735-738`);
6. add the bottom-boundary-layer trend
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:739`);
7. add interior damping
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:740`);
8. hand the completed `Krhs` to the implicit vertical tracer solve
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:742-749`).

The resolved ORCA2 deck selects only the diffusive BBL arm:
`ln_trabbl=.true.`, `nn_bbl_ldf=1`, `nn_bbl_adv=0`, and
`rn_ahtbbl=1000 m2/s`
(`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:265-272`).  Consequently `tra_bbl`
calls `tra_bbl_dif` and does not call `tra_bbl_adv`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:182-202`).  The diffusive routine
adds one rounded bottom-cell trend to the incoming `Krhs`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:249-264`).

There is no selected BBL modification to FCT's advective transports.  NEMO
forms `zFu/zFv/zFw`, then computes BBL coefficients
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:551-582`); FCT consumes those
already-formed transports on line 637, while `tra_bbl` writes only `Krhs` on
line 739.  Even the unselected advective BBL arm adds its own trend to `Krhs`
rather than changing FCT's transport arguments
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:271-354`).

## Frozen candidate and controls

The new default-false L8 selector may be true only for this complete
combination: FCT and every certified G9 stage selector; diffusive BBL with
advective BBL off; certified standard isoneutral diffusion; geothermal after
LDF; and damping after BBL.  Removing any member or selecting any additional
BBL arm must fail validation.  Existing G9, L1, L2, L4b, and L6 combinations
must retain their existing programs and values.

The live stage-3 route will expose the seven post-write boundaries in the
order `adv`, `sbc`, `qsr`, `ldf`, `bbc`, `bbl`, `dmp`.  The bit-position test
uses these binary64 inputs, all non-power-of-two values:

```text
adv  0x1.8484bbe636a7cp+25
sbc  0x1.80c4d885778a5p+25
qsr -0x1.e4576fad0145bp+34
ldf -0x1.fbbb660e98d04p+8
bbc  0x1.99a22105eba7ep+34
bbl  0x1.792a95b49c117p+27
dmp -0x1.0452f7986b12bp+33
```

The frozen post-write words are:

```text
adv  0x1.8484bbe636a7cp+25
sbc  0x1.82a4ca35d7190p+26
qsr -0x1.e2d4cae2cb6e9p+34
ldf -0x1.e2d4cb61ba481p+34
bbc -0x1.24caa96f3a80cp+32
bbl -0x1.190154c195a03p+32
dmp -0x1.90d3a1f935e2cp+33
```

Each planted call move must fail at the first displaced boundary.  A final-only
check is insufficient because the first two additions commute.

## Predictions and falsifiers

- **P1 — exact card union.** L8's deck and resolved config differ from the
  certified source cards only by the union Decision 148 names.  Any extra deck
  field, missing certified field, L3/L5 selection, or changed inherited-card
  value refutes P1.
- **P2 — ordered one-step route.** A JIT-compiled x64 test with traced INEXACT
  operands produces every frozen post-write word above.  Moving any one call
  fails at its boundary.  Any unchanged moved-call plant refutes P2.
- **P3 — BBL transport separation.** Toggling the selected diffusive BBL may
  change only its `Krhs` boundary and downstream tracer result; the FCT
  `zFu/zFv/zFw` operands remain bit-identical.  Any changed advective transport
  refutes P3.
- **P4 — selection isolation.** Address-masked two-step JAX programs and values
  for every pre-L8 card are unchanged.  Any inherited-card difference refutes
  P4.
- **P5 — gradients.** The L8 composition introduces no division, square root,
  or forward branch.  Gradients of the ordered sum with respect to every input
  are finite and the corresponding forward bits equal the non-gradient call.
  Any new non-finite gradient or forward-bit change refutes P5.
- **P6 — acquisition preflight.** The read-only preflight passes all seven
  standing checklist items and every planted violation exits nonzero.  It
  executes whitespace-tolerant checks on a real committed `ocean.output`
  excerpt and a stamped rendered L8 deck; runs `check_acting` only on the
  smoke, as its own statement; compiles a gfortran reader/writer layout
  harness; and leaves no scratch.  Missing execution evidence, a surviving
  plant, or any created acquisition target refutes P6.

## Acquisition contract

The operator-only `--run` mode will execute a smoke, a writer-free boundary,
and three byte-identical twins through one endpoint no greater than ten steps.
The smoke alone is checked for acting terms, with damping permitted to be inert
at kt=1 because its target equals `Kbb`.  Observer records contain full
`jpi*jpj*jpk` arrays for each post-write boundary, canonical zero outside the
cells NEMO defines, and are written outside any XIOS-dependent false branch.
The reader must consume the exact Fortran stream order proven by the compiled
gfortran harness.  Collision-safe scratch is removed by an `EXIT` trap.

## Review and choices

- codex review: operator
- independent review: operator (Claude)
- ASKED: Decision 148's complete L8 composition, tests, acquisition, and stop
  at the record boundary.
- UNASKED: none.
