# Preregistration: live-QCO Redi face-thickness production replay, round 80

Date: 2026-08-30. Frozen before the post-fix replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Input round 79 is `/tmp/dino_split_explicit_momentum_chain_round79.json`,
SHA-256
`dca39985e54bd95f20ee9b1bfb4ab3bc39013953c95aa8fb74965eb6618f6f61`.
It registers live `e3u(Kmm)` as the 99.994091% majority owner of temperature
`zfu`, but retains a `1.9511075e-7` conditional final-slope residual.

The production change replaces only traldf_iso's historical T-point thickness
in `zA11/zA22` with raw-mesh, NOW-SSH QCO `e3u/e3v`, matching NEMO
`traldf_iso_scheme.h90:73-74`. `tpoint_jacobian` remains the GMRediConfig and
DINOConfig default; only `nemo_dino_kamm` and `nemo_dino_kamm_mlf` select
`nemo_qco_live`. The selector requires paired U/V operands, NOW SSH, and raw
NEMO mesh geometry. Unit controls prove the default and an explicit T-point
override are byte-identical, an unpaired override is red, both DINO cards opt
in, and every other DINO recipe remains pinned.

The scorer repeats the unchanged round-79 wet-column bars and factorial from
the already captured streams. Registered dispositions are:

- `REDI_ZFU_T_POSTFIX_AT_BAR` iff production temperature `zfu` is AT-BAR at
  the pointwise `1e-15` threshold; advance to ordered row `78.T.2`;
- `REDI_ZFU_T_POSTFIX_RESIDUAL_FINAL_USLP` iff production live `e3u` itself is
  AT-BAR, production `zfu` remains red below `1e-5`, and the full oracle
  S/H/K current-association arm is AT-BAR; this confirms the live-thickness
  fix and sends the residual to the existing row-30 final-uslp ladder;
- `REDI_ZFU_T_POSTFIX_REGRESSION` for a production error at or above `1e-5`,
  a red production thickness operand, or failed all-oracle closure.

Every inherited planted control must remain green. This is CPU-only replay;
no new NEMO writer or held run is authorized.

