# Preregistration: production literal EEN coefficient builder, round 28

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen after round 27 stopped at the
production boundary and before implementing or replaying a literal builder.
Bound exact oracle arm: round-25 artifact SHA-256
`a9db28c458bb6e409ec23b74ea371b2d2e4aeaf1d84de322074ab72b46fca16e`.

## Implementation boundary

The existing production path factors NEMO's EEN coefficient through the
generic AL81 operator and fused JAX reductions at
`packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:821-843`.
NEMO instead materializes eight frozen coefficients in source order:

1. three `ff_f/e3f_vor` terms at `dynspg_ts.F90:1517-1528,1544-1555`;
2. surface-to-bottom left accumulation at `:1530-1534,1557-1561`; and
3. ordered `r1_12*r1_e*r1_h*neighbor_metric*accumulator` post factors at
   `:1535-1538,1562-1565`.

Round 25 proves that this complete composition, not face-latitude selection
alone, is exact. The implementation is large enough to stop round 27 because
the literal route must preserve the NEMO bridge's raw reference/QCO operands
through both coefficient consumers: the pre-loop slow-forcing subtraction and
the live substep application. A one-consumer or reconstructed-generic operand
route is forbidden.

## Authorized implementation

Add a `generic | nemo_literal` coefficient-evaluation selector, defaulting to
`generic`. Only `nemo_dino_kamm` and its `nemo_dino_kamm_mlf` child may select
`nemo_literal`; all other recipes and generated grids remain byte-pinned.
The literal builder must:

- consume the bridge-carried raw `e3t_0`, `hu_0`, `hv_0`, T/U/V metric areas,
  and any additional F-point reference/area fields required to reproduce
  `e3f_vor(Kmm)` and `r1_h{u,v}(Kmm)` without a generic z-star reconstruction;
- use `ff_f` at the NEMO F-point latitude and preserve all three registered
  association stages with static Python level loops (JIT/autodiff safe);
- materialize the eight U/V corner coefficients once per baroclinic step and
  reuse them for the Kmm pre-loop subtraction and every live fast substep; and
- map NEMO's native east/north A2D arrays to legoESM's redundant periodic-U
  and wall-V faces only after coefficient construction.

No arithmetic bar changes. Against the retained round-22/25 streams, all
eight coefficients and their applied U/V outputs must have normalized RMS and
maximum/NEMO-RMS `<=1e-15`; row-1.3 substep SSH/U/V must then meet its existing
campaign bars. Identity, four-step-nextafter, association-order, both-consumer,
periodic-seam, north/south-wall, JIT, finite-gradient, and energy/enstrophy
controls must be red-capable. Tests must enumerate exactly the two faithful
cards and prove the generic path byte-identical.

Only a clean current-production replay with row 1.3 AT BAR releases row 1.4.
The existing round-26 bracket is sufficient; no new NEMO writer or SLOT run is
preregistered. Rows 2--6 and all later registry chains remain ordered-blocked.
