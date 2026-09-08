# Preregistration amendment: carried-mesh literal seed correction, round 20

Date: 2026-08-29. Frozen after the round-19 implementation replay stopped and
before its corrected replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Round-19 stop and exact implementation defect

The committed round-19 selector fired, but its literal recurrence consumed the
generic model-native min-face/z-star reconstruction. The authoritative
`forcing_only` arm therefore remained DEBT at row 1.2: U normalized RMS
`6.91633064381619e-6`, V `4.703469755932527e-6`; the exact held-seed control
remained zero. The walk did not advance.

This is an implementation defect, not a retraction of round 18. The round-18
owner used the exact NEMO reference operands: `e3u_0/e3v_0`, `hu_0/hv_0`, and
native T/U/V areas. The NEMO bridge already carries their DINO equivalents on
the vertical coordinate: exact unaveraged `e3t_0` (equal to `e3u_0/e3v_0` on
the executed full-step DINO branch), `hu_0/hv_0`, and `e1e2t/u/v`. Rebuilding
these through live cell thickness and a min-face ratio is mathematically
equivalent but not bit-equivalent.

## Frozen correction and replay

When all six carried reference operands are present, `nemo_literal` must:

1. work in NEMO's native east-/north-face layout;
2. construct `r3u/r3v` with `domqco.F90:166-169` association,
   `0.5*(A_T*ssh + neighbour)/h{u,v}_0/A_{U,V}`;
3. form `e3{u,v}_0*(1+r3*mask)` and the separately associated
   `(1/h{u,v}_0)/(1+r3)`;
4. left-accumulate the masked velocity products in
   `istate.F90:149-155` order; and
5. map the completed native face arrays once into legoESM's redundant
   west/south-face layout.

Generated coordinates without carried NEMO operands retain the registered
source-ordered model-native fallback. A unit red control gives that fallback a
deliberately different thickness ladder and requires a bit-distinct answer.
The corrected unit suite must remain CPU/fp64, JIT/autodiff-safe, and pass the
round-19 byte/dispatch/card-routing controls.

The replay uses the identical round-19 artifact command, inputs, populations,
bars, and ordered release. Row 1.2 advances only at normalized RMS zero,
maximum error/NEMO RMS zero, and zero bit mismatches for both 9,758 U and
9,868 V points. Any nonzero value stops. If exact, row 1.3 resumes at the
already frozen operand order and POINTWISE bars in the round-19 preregistration;
no combined row-1.3 output may assign ownership ahead of those operands.
