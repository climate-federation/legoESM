# NEMO testcase Lane 4 — ORCA2 card round 10 preregistration

Date: 2026-09-23

Parent: `84ba7f9cd04d4c2b0e5aca0f81f2b73086ca2c1c`

Status: **PREREGISTERED BEFORE ANY ROUND-10 MEASUREMENT.**

Scope is round 9's OPEN items 1 and 3 on the ocean-only `orca2_vector_een_c2`
card, in that order: the three-band chlorophyll shortwave penetration that the
shared physics pipeline refuses, and the second masking legoESM applies to a
coefficient NEMO has already masked.  The six-entry sea-ice registry is frozen
and stays out of scope.  Decision 52's labels are binding: every number is
either **given NEMO's entry** or **independent**, never mixed in one table
without its label.

## What the record fixes, and what nothing in this round may choose

Read from the record's own resolved configuration, not from a deck comment.
The record is the pinned ORCA1-ice reference run
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is printed |
|---|---|---|
| shortwave penetration family | `ln_qsr_rgb = T` | run `ocean.output:1207` |
| chlorophyll source | data file, `nn_chldta = 1` | run `ocean.output:1211` |
| chlorophyll vertical profile | Morel-Berthon analytical, `nn_chlprfl = 1` | run `ocean.output:1212` |
| infrared fraction | `rn_abs = 0.58` | run `ocean.output:1213` |
| infrared attenuation length | `rn_si0 = 0.35` m | run `ocean.output:1214` |
| lateral momentum boundary condition | no-slip, `rn_shlat = 2.0` | run `ocean.output:339` |
| lateral viscosity operator family | div-rot, `nn_dynldf_typ = 0`, laplacian, iso-level | run `ocean.output:1176-1180,1193,1195` |
| viscosity coefficient source | `ahmt_3d`/`ahmf_3d` read whole from `eddy_viscosity_3D.nc` (`nn_ahm_ijk_t = -30`) | run `ocean.output:1184,1197,1200-1201` |

No selector default, tunable, threshold, resolution, timestep, carried state or
data source moves in this round.  The GYRE card selects the two-band identity
(`nemo_qsr_2bd`) and resolves `rn_shlat = 0`; both statements below must leave
it bit-identical.

## Statement one — the three-band chlorophyll penetration

`tra_qsr` dispatches on `nqsr`; with `ln_qsr_rgb` and `nn_chldta = 1` the
resolved arm is `qsr_RGBc`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:213`).  That
routine reads the chlorophyll field, builds a per-level look-up-table index
from the Morel-Berthon analytical profile evaluated on the LIVE interface depth
`gdepw_1d(jk+1)*(1+r3t(Kmm))`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:349`),
partitions the surface flux into an infrared band `rn_abs` and three equal
red/green/blue bands `(1-rn_abs)/3`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:368-374`), and
then walks four depth ranges bounded by the extinction levels `nk0`, `nkR`,
`nkG`, `nkB`, dropping one band at each boundary and multiplying the summed
flux by the w-level mask
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:379-463`).  The
deposit enters the temperature right-hand side as
`r1_rho0_rcp*(zeT - zzeT)/ze3t` with the live thickness
`e3t_3d*(1+r3t(Kmm)*tmask)`, and `nksr = nkV = nkB` is the deepest level the
light reaches
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:1193-1195,1308`).

legoESM already carries this kernel, literal statement by literal statement,
behind `apply_shortwave_penetration(scheme="nemo_qsr_rgb")`
(`packages/ocean/legoesm/ocean/physics/shortwave_penetration.py`), and it is
already gated cell by cell against this record's own `qsr` increment by
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_rgb_gate.py`.
Nothing in this round writes a second RGB kernel: the shared physics pipeline
and the Runge-Kutta stage-three substitution are ROUTED through the existing
one, with the operands NEMO reads.

## Statement two — the second masking of an already-masked coefficient

NEMO stores the F-point viscosity with `fmask` already folded in, and the
compiled operator says so in its own comment: the shearing term is
`ahmf(ji-1,jj-1,jk) * e3f * r1_e1e2f * (...)`, annotated `! ahmf already * by
fmask`, and NO further mask multiplies the vorticity
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynldf_lev.f90:123`; the
divergence twin at `:127` carries the same annotation for `tmask`).  With
`rn_shlat = 2` that stored `fmask` is not a zero/one mask: a coastal F point
whose free-slip value was zero is reset to `rn_shlat` times the neighbouring
velocity mask maximum
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dommsk.f90:269-277`), and
the strait overrides may replace it again
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dommsk.f90:283-303`).  That
non-zero value at the coast IS the no-slip lateral boundary condition.
legoESM's shared div-curl multiplies the vorticity by its own zero/one vertex
mask before applying the coefficient
(`packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py:1484-1489`),
which discards exactly those coastal values.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R10-P1 | The resolved shortwave arm is `qsr_RGBc` with the Morel-Berthon profile, the `rn_abs` / `(1-rn_abs)/3` partition, the four extinction-level ranges, the live `e3t(Kmm)` divisor and `nksr = nkB`, exactly as legoESM's existing `nemo_qsr_rgb` kernel already spells them. | Every one of those five statements reads as stated in the compiled `qsr_RGBc`, and the existing kernel gate stays at its bar. | Any one of them reads otherwise, in which case the kernel — not only its routing — is the round's subject. |
| R10-P2 | Routing the pipeline through the existing kernel removes the `STOP_PRODUCTION_QSR_RGB_PIPELINE_GAP` refusal and the ORCA2 ladder advances past it. | The ladder no longer returns that status. | The same status is returned, or a refusal from the same routine under a different text. |
| R10-P3 | With the pipeline's own operands (not the oracle-supplied ones), the deposit the production step applies at Runge-Kutta stage three equals this record's `qsr` increment on every owned wet cell: 0 unequal doubles of 90x148x30. | Zero unequal doubles. | Any unequal bit. |
| R10-P4 | The shortwave change is inert for GYRE, which selects the two-band identity and carries no chlorophyll field. | Base and tip GYRE ten-step ladders give zero differing rows and array-equal residual arrays, and the thirty-day member snapshots are byte-identical with the day-30 digest `14a7e64b4512860e...`. | Any differing row, unequal residual array, or differing snapshot digest. |
| R10-P5 | NEMO applies NO mask to the F-point vorticity beyond the one already inside `ahmf`; legoESM's extra zero/one vertex mask therefore deletes ORCA2's no-slip boundary condition on the coastal F points round 9 counted (48,287 of 799,200). | The compiled operator's shearing term carries `ahmf` and no `fmask` factor, and the count reproduces. | A mask appears in that term, or the count differs. |
| R10-P6 | Removing the second masking only where NEMO's mask is already inside the coefficient (the file-sourced arm) leaves every other card, GYRE included, bit-identical, because their coefficient is the metric formula and carries no mask. | GYRE's two arms stay byte-identical; the ORCA2 ladder registers a movement on the momentum rows. | GYRE moves, or the ORCA2 momentum rows do not. |
| R10-P7 | The admitted records carry NO isolated lateral-viscosity tendency stream, so statement two cannot be gated against a recorded tendency without a new acquisition. | The instrumented writer list contains no `dyn_ldf` dump. | Such a stream exists, in which case it is used and no acquisition is requested. |

Failed predictions stay in the receipt as **REFUTED** and are never quietly
dropped.

## Controls and stop rules

- A one-representable-value perturbation of the pipeline's shortwave deposit
  must make the new gate refuse and name the offending cell.
- Substituting the two-band kernel for the three-band one must make the new
  gate refuse, so the gate is not passing on a shared prefix.
- Substituting the reference (static) thickness ladder for the live one must
  make the new gate refuse, so the `key_qco` substitution is not vacuous.
- A rigid two-line shift of a round-10 compiled citation must fail the citation
  gate.
- GYRE's trajectory is proven unchanged before anything lands, at the round's
  base tip and at its final tip, with the evaluation protocol byte-identical.
- The DINO card executes the same RGB kernel through its own `rgb_chl`
  selector; its unit gate must stay green and any movement is registered.
- No stabiliser, clip, damp or limiter NEMO lacks may be added, and no
  configuration value is chosen that the record does not resolve.

## Labels

The shortwave operands are the record's own recorded surface inputs, so every
shortwave number in this round is **given NEMO's entry**.  The viscosity
coefficient and its mask come from input files the card already owns, so those
numbers are **independent**.  The ladder's trajectory rows keep their existing
`INDEPENDENT_WITH_DECISION52_SSH` label.
