# Preregistration: mixed W-slope carry and vertical-flux closure, round 92

Date: 2026-08-30. Frozen before implementation and measurement. Admitted
round-91 receipt SHA-256:
`f4d68e6802a48c7c11f52648035bd25be4524bfa7e6214db1cdc20f4630ad63a`.

Implement a DINO-only Redi slope-stage selector. The faithful arm supplies the
already-computed Naa `wslpi/wslpj` pair to the Redi vertical-flux consumer,
while retaining the certified Kmm `uslp/vslp` pair for horizontal Redi fluxes.
The bolus tuple remains wholly Naa. Generic/default cards retain the existing
tuple byte-for-byte. Couple this carry to the already selected literal
`traldf_iso_scheme.h90:109-125` coefficient and gradient association.

The production skew component must pass the unchanged `1e-15` pointwise bar;
the full `zfw_tem` must then pass the same bar. Horizontal temperature rows
78.T.1/T.2 and rows 8.3-8.10 must remain at bar. A planted faithful-to-legacy
W-pair reversion must make the skew row red, and a generic explicit-default
test must be byte-identical.

If skew passes but A33 remains red, stop and peel A33 from existing operands.
If the full temperature vertical flux passes, release and score salinity in
the same NEMO order (zonal, meridional, vertical), then advance to the
registered ZDF and Asselin tail. No GPU or NEMO run is authorized.
