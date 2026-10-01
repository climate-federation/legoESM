# Preregistration: dyn_zdf dispatch-operand peel, round 34

Date: 2026-08-30. Frozen before numerical execution. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Question

Round 33 feeds NEMO's reconstructed pre-solve state plus its exact stress
deposit into the full production `dyn_zdf` application, yet leaves U/V
normalized RMS `4.2366e-2`/`1.3397e-2`. The independent ZDF chain-end row-31
receipt proves the same production `nemo_literal` solver reaches the bar with
an oracle operand pack. This round compares the actual arguments passed by the
round-33 fed-plus-stress arm to that already-verified pack. It changes no
physics and uses no new oracle stream.

## Ordered operand ladder

The capture is at
`ocean_model_latlon_cgrid.py:7583-7681`, immediately before
`implicit_vertical_diffusion_ocean_momentum_dispatch`. The reference follows
the committed `zdf_chain_end.py:314-413` construction and NEMO
`dynzdf.F90:137-214,305-371` in this order:

1. dispatch selector is `nemo_literal`, `dt_mom=5400`, and the U/V 3-D wet
   masks equal NEMO `umask/vmask`;
2. solve RHS equals NEMO `Kbb + rDt*Krhs - barotropic(Kaa)`, plus the dumped
   surface-stress increment and the implicit barotropic bottom-drag RHS term;
3. face viscosity equals the two-T-point average of dumped `avm`, masked on
   adjacent wet levels;
4. cell thickness equals live `e3u/e3v(Kaa)` from dumped after-QCO `r3u/r3v`;
5. interface thickness equals live `e3uw/e3vw(Kmm)` from entry NOW SSH and
   the raw mesh face-interface metric;
6. implicit drag diagonal equals `rDt*r_eff/e3{u,v}(Kaa)` at the bottom wet
   cell only.

Each numeric operand is scored U then V. RHS uses the ACCUMULATING `1e-12`
class bar; coefficient, metric, mask, scalar, and diagonal operands use the
POINTWISE `1e-15` bar. The first failing operand owns the row-4 application
interval; later operands are `ORDERED-BLOCKED`. If every operand passes but
the round-33 output remains red, the composition between dispatch return and
row output is `OPEN_UNRESOLVED` and must be instrumented next.

## Admission and controls

The scorer reuses the committed round-33 four-arm execution and captures the
two dispatch calls belonging to the final fed-plus-stress arm. It binds the
official round-33 artifact, the ZDF chain-end artifact SHA-256
`ce6fff6690ff6fbfa1023b4cf3eafbd36d0b86938ef47c5f7524accc410d4975`,
the retained stream manifest, all consumed dumps, entry restart, active NEMO
sources, production modules, scorer, and this preregistration. It requires a
tracked-clean checkout, checkout-first `PYTHONPATH`, CPU/fp64, lane `d180`,
and the exported session ID.

The captured selector/call count and U/V stagger shapes must match exactly.
Identity must pass; a `2x` class-bar perturbation, one-cell roll, wet NaN, and
the wrong `2700 s` scalar must fail their applicable paths. The existing
round-33 null and hook-restoration controls must still pass. A changed/missing
hash, nonfinite capture, failed control, or ambiguous U/V call ordering is
`INVALID`. No SLOT block is allocated because every oracle operand already
exists.

## Post-run retraction

This registration targeted an invalid fed-arm representation boundary.
Round 33 supplied already barotropic-free NEMO `naa_B` to a public method that
strips its mean again, and round 34 captured the resulting double-stripped
arguments. The round-34 scorer is retired and refuses execution. Its nominal
`LOCALIZED_TO_RHS` print is withdrawn; it is not an ownership receipt. Round
35 replaces it with an unmodified production-step capture at the raw dispatch
boundary before mean readdition.
