# Split-explicit / momentum chain: literal ahtu closure, round 85

Official receipt `/tmp/dino_split_explicit_momentum_chain_round85.json`,
SHA-256 `10a5695a804c0e96ec36455ed160169b18897c43b57614c6a979349fcf53ec6c`,
classifies `REDI_ZFU_T_AHTU_AT_BAR`. Rows 8.3--8.10 remain AT-BAR and both
horizontal temperature fluxes are bit-exact: 78.T.1 zfu and 78.T.2 zfv are
0/9,758 with zero maximum error. The next ordered failure is 78.T.3, total
vertical temperature flux zfw, 9,462/9,758 at `1.9659923e-7`. Salinity remains
ordered-blocked.

The fix is DINO-only: the two NEMO cards use the stored U/V face metrics and
the source grouping `(0.5*U_d)*MAX(e1,e2)` from
`ldfc1d_c2d.F90:141-145`; every other path retains the historical grouped
cosine evaluation. The planted missing-velocity control and selector tests are
red-capable.
