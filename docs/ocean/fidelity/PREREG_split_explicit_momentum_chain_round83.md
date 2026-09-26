# Preregistration: split Kmm slope consumers, round 83

Date: 2026-08-30. Frozen before replay. Round-82 invalid receipt SHA-256 is
`4177c99481bdbc6996477dad848d1e645ada3a26a08fb9aa1e9dd9e4b20f9e7a`.

The Treguier-kappa/bolus slope consumer retains the prior same-stage eta path;
the Redi explicit tensor and paired implicit K33 alone consume carried Kmm eta.
The replay must restore rows 8.3--8.10 to their unchanged bars and make final
uslp plus live e3u AT-BAR. If zfu is then AT-BAR, disposition is
`REDI_ZFU_T_KMM_OPERATOR_AT_BAR`; if only the exact-ahtu substitution clears a
residual below `1e-12`, disposition is
`REDI_ZFU_T_KMM_OPERATOR_FIXED_AHTU_RESIDUAL`. Any earlier-row regression is
invalid and downstream flux rows remain blocked.
