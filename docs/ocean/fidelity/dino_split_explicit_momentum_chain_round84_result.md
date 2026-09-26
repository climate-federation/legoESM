# Split-explicit / momentum chain: Redi/bolus stage split, round 84

Official receipt `/tmp/dino_split_explicit_momentum_chain_round84.json`,
SHA-256 `6483fc67d59bea3b5bb81e546a31af47eec85aeea48573f42e11c2e575740451`,
disposes the time-level coupling as
`REDI_ZFU_T_BOLUS_STAGE_SPLIT_FIXED_AHTU_RESIDUAL`. Rows 8.3--8.10 are all
AT-BAR; row 8.8 `pu_bolus` is 0/9,758 at `1.35034e-13`. The Redi Kmm face and
native-slope carries reduce 78.T.1 from 9,758 columns at `3.302117e-3` to 60
columns at `6.283886e-15`. The exact NEMO ahtu arm makes zfu bit-exact.

An existing-dump source ladder then compared the retained
`ldftra_dump_ahtu.bin` with `ldfc1d_c2d.F90:141-145`. The literal expression
`(0.5*rn_Ud)*MAX(e1u,e2u)` matches all 9,758 wet columns bit-for-bit. The
historical grouped `(0.5*rn_Ud*R*rad)*cos(gphiu)` differs by one ULP in 2,079
columns. Thus the final zfu residue is coefficient-construction association,
not a new physical operand.
