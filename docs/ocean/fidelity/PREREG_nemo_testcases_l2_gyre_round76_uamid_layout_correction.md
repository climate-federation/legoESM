# Preregistration correction: round 76 `ua_e` reduced-domain layout

Date: 2026-09-12. Frozen before implementing or preflighting the acquisition.
This corrects one unsafe extent in the earlier acquisition preregistration;
all other predictions and controls remain unchanged.

The earlier 36 by 26 field prediction is **RETRACTED before implementation**.
The compiled midpoint loops assign `ua_e` over
`ji=ntsi-2:ntei+1, jj=ntsj-1:ntej+1` at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:484-493`, but writing
full 36 by 26 arrays would include cells outside the owned domain and make a
bit replay depend on halo values. The campaign's strict record convention is
an explicit reduced owned domain, not full arrays with unowned slots.

The corrected writer records `ntsi:ntei,ntsj:ntej`, exactly 32 by 22 values
per field. Its one header contains the 16-byte magic and ten int32 values
`(version,kt,ncycle,jpi,jpj,bits,ntsi,ntei,ntsj,ntej)`, making both full and
reduced extents explicit. Each of 50 substeps contains one int32 `jn`, three
float64 coefficients, and the four 32 by 22 float64 U slices. The corrected
exact byte count is 1,127,856:
`16 + 10*4 + 50*(4 + 3*8 + 4*32*22*8)`.

Preflight must prove the compiled writer uses those four explicit slices, the
reader expects the same four-field order and bounds, and both independently
derive 1,127,856 bytes. A plant replacing the `ubb_e` slice with only three
fields must exit nonzero. No full-domain writer may be handed to the operator.
