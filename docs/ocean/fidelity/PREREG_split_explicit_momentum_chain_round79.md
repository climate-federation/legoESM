# Preregistration: Redi temperature-U flux operand peel, round 79

Date: 2026-08-30. Frozen before the operand measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

The official corrected round-78 receipt is
`/tmp/dino_split_explicit_momentum_chain_round78_rescore.json`, SHA-256
`30f63d47e11d632e214490c2d6fc8756170a03f12f86635ba6a8b0a8b8d6e7f3`.
It validly stops at `78.T.1`: NEMO temperature `zfu` diverges in 9,758/9,758
wet U columns with maximum normalized error `3.302117026989921e-3`.
Rows `78.T.2` onward remain ordered-blocked.

NEMO assembles the operand at
`src/OCE/TRA/traldf_iso_scheme.h90:70-86`:

1. `zA11 = e2_e1u * e3u(Kmm)` (`:73`), the diagonal horizontal term's live
   QCO U-face thickness;
2. `zmsku = 1 / MAX((w_ip1_k + w_i_kp1) +
   (w_ip1_kp1 + w_i_k), 1)` (`:76-77`);
3. `zA13 = -e2u * uslp * zmsku` (`:81`), consuming the final native U slope
   after its triad construction, limiter/ML ramp, and Shapiro coastal taper;
4. `zfu = ahtu * (zA11*zdit + zA13*((a+b)+(c+d)))` (`:84-86`).

Existing deterministic streams are exhausted before new instrumentation:

- round-78 `ldftra_dump_ahtu.bin`, SHA-256
  `098b95a3548ee8d3c66694ca43c3b720840e2c193895b0df222e8e186b2580da`;
- round-78 `eiv_dump_uslp.bin`, SHA-256
  `ecb3ca3c50c97ef67ad45e96f828a5264c18f94b767a0f9d619f010535d9d0f9`;
- certified row-8 `fct_entry_dump_e3u.bin`, SHA-256
  `d071b046cf5e841d2475b753b3c834ad02e69128e73a59c7dd15576a47d7b4f8`;
- round-78 `redi_dump_zfu_tem.bin`, SHA-256
  `f65fd73ded37ef3fc150f3ba8d7c9869e322d7788819a5a6eafd0d2a99093b75`.

The scorer captures the production replay's own `ahtu`, `e3t`, `uslp`,
`zmsku`, tracer gradients, and metrics without changing the returned tendency.
It first proves that recomposing those captured operands reproduces production
`zfu` byte-for-byte. It then evaluates the full `2^3` oracle-substitution
matrix:

- `S`: substitute NEMO's final `uslp` (triad/limiter/ramp/taper bundle);
- `H`: substitute NEMO's live `e3u(Kmm)` for the current T-thickness in `zA11`;
- `K`: substitute NEMO's `ahtu` coefficient;
- `S/H/K` combinations retain the current JAX association.

Finally, on `S1H1K1`, the `L` arm uses the literal NEMO associations:
`e2u*(1/e1u)`, paired mask sums, paired vertical-gradient sums, and the source
post-factor order at `:73-86`. All arms score the exact round-78 wet U-face
population against the pointwise `1e-15` bar. Main effects and interactions
are reported as signed changes in maximum normalized error; no downstream
operand is promoted across the first red row.

Registered dispositions and bars:

- `REDI_ZFU_T_LOCALIZED_TO_LIVE_E3U` iff `H` alone removes at least 90% of the
  baseline maximum error, `S1H1K1L1` is AT-BAR, and neither `S` nor `K` alone
  removes 50%;
- `REDI_ZFU_T_LOCALIZED_TO_FINAL_USLP` or `_AHTU` by the symmetric rule;
- `REDI_ZFU_T_OPERAND_COMPOSITION` iff no single arm removes 90% but the full
  literal arm is AT-BAR, with the nonzero interaction table naming the coupled
  owners;
- `REDI_ZFU_T_LOCALIZED_TO_ASSOCIATION` iff `S1H1K1` remains red and the `L`
  arm reaches the bar with at least 90% conditional removal;
- `REDI_ZFU_T_OPERANDS_BOUNDED_ASSOCIATION_OPEN` iff `S1H1K1` removes at least
  90% but the literal arm remains red;
- otherwise `REDI_ZFU_T_OPERAND_LADDER_OPEN`.

Controls must all fire: production recompose identity, all eight factorial arms
present, a one-wet-point perturbation red, the live-thickness substitution
non-inert, and source-association recomposition finite. A valid scientific
divergence exits zero; only a failed control is invalid.
