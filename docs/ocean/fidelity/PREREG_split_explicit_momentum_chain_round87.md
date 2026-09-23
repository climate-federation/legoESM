# Preregistration: held zfw A31/A32 versus A33 split, round 87

Date: 2026-08-30. Frozen before the held run. Round-86 SHA-256 is
`93e39c3ee0fbf973464aa4ea583a404668e4be13dfe044153a28c0e3022c260e`.

Record NEMO's A31+A32 contribution and explicit-A33 contribution separately
for temperature and salinity at `traldf_iso_scheme.h90:119-129`. The two NEMO
components must recompose total zfw at the unchanged `1e-15` pointwise bar.
Score temperature A31/A32 first, then A33; salinity remains ordered-blocked.
The first red component owns the next operand peel. If both components are
AT-BAR but total zfw remains red, the owner is their final addition
association. Identity, finite-stream, exact shared-stream, and one-bit bracket
controls must all pass.
