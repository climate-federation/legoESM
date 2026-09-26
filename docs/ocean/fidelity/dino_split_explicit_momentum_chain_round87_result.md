# Split-explicit / momentum chain: held vertical Redi components, round 87

Official receipt `/tmp/dino_split_explicit_momentum_chain_round87.json`,
SHA-256 `df56a496d6e7ebc14d9e8214cdc6742010d3b8b4286c0d9b2704526593ed6caf`,
classifies `REDI_ZFW_T_DIVERGED_A31_A32`; the first ordered failure is
87.T.3a.  NEMO's independently written A31+A32 and A33 components recompose
the total vertical temperature flux bit-exactly.

The classifier is an ordered-first-red classifier, not an exclusivity claim.
The skew component is red in 9,462/9,920 wet columns with maximum column error
`7.735372278997023e-8`.  A33 is also red in 9,460/9,920 with maximum column
error `1.293072691602089e-7`; it is therefore ordered-later, not exonerated.
The next registered measurement peels skew first, as frozen by round 87, and
must rescore A33 after skew closes before assigning the total residual.

NEMO computes Kbb tracer gradients at `traldf_iso_scheme.h90:21-48`, builds
zA31/zA32 at `:109-120`, and combines the two source-grouped four-point
gradient stencils at `:122-125`.  All component streams are full-halo,
zero-initialized, and bound in the official receipt.
