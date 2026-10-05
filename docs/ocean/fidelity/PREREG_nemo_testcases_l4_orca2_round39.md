# NEMO testcase Lane 4 — ORCA2 card round 39 preregistration

Date: 2026-09-26

Parent: `06d6cf1a68ace46182782069fba5c83f9bfc2ec0`

Status: **PREREGISTERED BEFORE ROUND-39 SCIENTIFIC SCORING.**

Round 38 found that the first two non-bit operands of the ranked slow-forcing
vertical average are a cancelling pair: replacing only `e3u_3d` worsens the
64-cell boundary by seven orders of magnitude, while replacing only
`uu(Krhs)` nearly closes it.  This round walks the compiled expression itself,
left to right, before either operand can be changed.

All numbers are **given NEMO's entry**.  The sea-ice selectors and the card's
`unmeasured_features` tuple remain frozen.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R39-P1 | Round 38 reproduces exactly at the current parent. | Same first `e3u_3d` row and all three frozen arm maxima/counts. | Any movement; stop for instrument drift. |
| R39-P2 | The thickness and RHS differences cancel in their compiled per-level product. | `(e3u_3d*uu(Krhs))*umask` is bit-exact on all 1,920 scored layer values. | Any unequal product value; that product is the first live statement. |
| R39-P3 | If the product is exact, the first non-bit statement is the left-to-right 30-level reduction or the stored reciprocal multiplication. | Product exact; the first unequal partial-sum level or final reciprocal boundary is named with its count and maximum. | Product already differs, or the completed depth mean is exact. |
| R39-P4 | No model statement lands unless one isolated compiled boundary reaches zero unequal cells. | One-variable arm has 0/64 unequal and full landing gates pass. | Any nonzero residual, cancelling pair, or multi-statement requirement; hold. |

The gate must retain the Round-38 admission and self-replay controls.  A
one-ULP mutation of the first newly scored boundary must fire.  Failed
predictions remain REFUTED.

## Choices

ASKED: follow the measured cancelling pair through the compiled product and
reduction before changing either half.

UNASKED: none.
