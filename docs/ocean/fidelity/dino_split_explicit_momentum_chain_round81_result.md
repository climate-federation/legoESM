# Split-explicit / momentum chain: Kmm face fix, round 81

Official corrected receipt
`/tmp/dino_split_explicit_momentum_chain_round81_rescore.json`, SHA-256
`247acae491d6568febb16d81a7a96e5b2c22a4246cf6b2e613ce640d38eaee71`,
disposes `REDI_ZFU_T_KMM_FACE_FIXED_RESIDUAL_FINAL_USLP`. The production face
thickness is bit-exact over 336,338 wet elements. Temperature zfu improves to
`1.9511075118895188e-7` maximum normalized error and remains red in 8,105 of
9,758 columns; final-uslp substitution plus the already exact ahtu closes it
bit-exactly.

The committed row-30 composite scorer was rerun post hoc from the retained
round-78 raw/post-Shapiro streams against current production. It returns
0/9,758 U and 0/9,868 V failures at both raw and post-Shapiro stages when the
slope kernel receives step-entry eta. Thus triads, limiter, mixed-layer blend,
and Shapiro are exonerated. The full-step residual is the same time-lifetime
mistake already found for zA11: the late call supplied post-dynamics eta even
though NEMO built `ldf_slp` before dynamics on Kmm=Nnn. The next correction
carries the same step-entry eta to both explicit native slopes and implicit
K33, preserving their required shared slope set.

