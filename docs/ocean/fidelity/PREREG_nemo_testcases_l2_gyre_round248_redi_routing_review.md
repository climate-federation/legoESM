# Preregistration — GYRE lane round 248: Redi routing review fixes

Frozen before changing code or running a measurement. Base: `0575754c0`.
Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round248/`.

## Scope and source statement

Decision 104 authorises only the two PR-review dispositions and their
fail-closed coverage. The closed deepest W mask remains the documented
`nemo_iso_lap_tracer_tendency_latlon_cgrid` library default for every lat-lon
C-grid caller. The live tracer-divergence thickness becomes independent of the
horizontal face-thickness selector: the new explicit
`redi_divisor_thickness_evaluation` field defaults to the pre-round-237
reference/Jacobian thickness and selects NEMO's live `Kmm` T thickness only as
`"nemo_qco_live"`.

The compiled SMT-3 program closes the deepest W mask in the coefficient pair at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259` and
divides its regular and deepest-level flux divergences by `e3t(Kmm)` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310` and
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.
No NEMO source, scheme, parameter, timestep, record, carried state, or
acceptance threshold changes.

Pre-implementation search found one existing low-level closed-mask test, no
production-routing test, and six production/fidelity callers of the top-level
lat-lon Redi dispatcher: the two model paths, the shared wrapper, the tendency
probe, the box heat budget, and the MPI benchmark. The existing configuration
already has separate face-thickness and divisor runtime operands, so this round
extends that configuration rather than adding another numerical implementation.

## Frozen predictions and falsifiers

1. **R248-P1 — library default.** `GMRediConfig()` and the low-level
   `nemo_iso_lap` operator retain `closed_bottom_wmask=True`. A caller census
   names every repository call site and proves no caller silently overrides it
   false. Removing the default or omitting a caller makes the test fail.
2. **R248-P2 — independent selector.** The new divisor selector defaults to
   `"reference_jacobian"`. With live face thickness and this default, the
   dispatcher leaves `divisor_thickness=None`; selecting
   `"nemo_qco_live"` builds the live `Kmm` T thickness even when the face
   selector remains `"tpoint_jacobian"`. An unknown value raises. Coupling the
   divisor to either face-selector arm falsifies the prediction.
3. **R248-P3 — production wiring.** A production-step test observes the
   stage-3 half-step SSH at the top-level hook and the dispatcher-built live
   divisor. Reverting either the model hook or the dispatcher construction
   makes that test fail; both reverse plants are run and recorded.
4. **R248-P4 — explicit card census.** GYRE explicitly selects
   `"reference_jacobian"`, preserving its current result. DINO and SMT-3/SMT-4
   explicitly select `"nemo_qco_live"`, preserving the round-237 landing and
   DINO's registered `2.056821682e-03 K` month value. Flat VORTEX and SMT-0/1/2
   do not execute isoneutral Redi and are recorded as non-executing rather than
   assigned hidden physics. Any executing NEMO card inheriting the field is a
   failure.
5. **R248-P5 — certified trajectories.** GYRE's 954-row ladder and certified
   year remain exact to round 247, including day-30/day-240/day-360 T RMS
   `2.3432419318363155e-06`, `6.5816987106668941e-05`, and
   `5.4077212586815052e-05 K`. SMT-1..4 reproduce their 50-row registries and
   day-100 T RMS `4.3321114781972461e-05`, `8.1037591477894766e-06`,
   `1.7729713625071864e-04`, and `2.5527080520554426e-04 K`. DINO remains at
   its registered value and below its fixed bar. Any movement stops the round.
6. **R248-P6 — controls and tests.** Flat VORTEX, tanks, the generic NEMO-GYRE
   recipe, citation gate and push battery introduce no new failure. The shifted
   citation plant and both routing plants exit nonzero. Every pytest invocation
   records its summary line; a new failure stops the round.
7. **R248-P7 — PR disposition.** The PR summary gains a Review section quoting
   both read-only findings and mapping them to the default/census and
   selector/routing controls. A separate read-only Codex pass tries to refute
   the final diff; `DO NOT SHIP` blocks landing. If the sandbox prevents the
   review from starting, the receipt quotes that failure verbatim.

No hidden choice is made: Decision 104 supplies the default and all executing
card selections. **UNASKED list: EMPTY.**
