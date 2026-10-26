# Preregistration: literal static Redi coefficient, round 85

Date: 2026-08-30. Frozen before replay. Round-84 receipt SHA-256 is
`6483fc67d59bea3b5bb81e546a31af47eec85aeea48573f42e11c2e575740451`.

The two DINO NEMO cards select `nemo_metric_literal`: construct `pUfac =
0.5*U_d`, then independently evaluate `pUfac*MAX(e1u,e2u)` and
`pUfac*MAX(e1v,e2v)` from the stored face metrics, in
`ldfc1d_c2d.F90:141-145` order. All other cards remain on the byte-pinned
`cosine_scaled` path. Rows 8.3--8.10, final uslp, live e3u, and ahtu must be
AT-BAR. If production zfu is AT-BAR, disposition is
`REDI_ZFU_T_AHTU_AT_BAR` and the ordered walk advances to 78.T.2 zfv. Any
regression is `REDI_ZFU_T_AHTU_REGRESSION` and blocks later flux rows. A missing
literal diffusive velocity and an unknown selector are planted red controls.
