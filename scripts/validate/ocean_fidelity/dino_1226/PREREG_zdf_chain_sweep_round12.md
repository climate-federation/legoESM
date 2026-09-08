# Preregistration: row-18 `htau` operand peel

Date: 2026-08-28. Status: **localization design frozen; not measured**.

Rows 1--17 are verified (row 16 is waived). Row 18 is the ordered stop:
the final `nn_etau=1` field misses the pointwise `1e-15` column bar in
173/9,920 columns (maximum normalized error `2.2779670337896653e-15`), while
all four southern focus columns pass and the pre-injection TKE is exact.

The active NEMO expression is `zdftke.F90:590-591`. Its operands will be
peeled in source order: `rn_efr`, surface `en(1)`, the exponential's live
`gdepw(Kmm)` numerator and `htau` denominator, ice fraction, W mask, T mask,
then the final addition association. The first registered discriminator is
the `htau` construction at `zdftke.F90:1005`: compare the model's current
degrees-to-radians round trip with the raw T-grid radians already carried by
the bridged grid, which are NEMO's literal `rpi/180*gphit` operand.

CONFIRM for that discriminator is a raw-radian substitution that changes row
18 to 0/9,920 failed columns and 4/4 focus passes at the unchanged `1e-15`
bar. REFUTE is any remaining failed column. If confirmed, the registered
production option is `tke_htau_evaluation="nemo_literal_radians"`: thread the
already-carried T-grid radians to the TKE closure and evaluate `SIN` directly.
It becomes the default only on `nemo_dino_kamm` and
`nemo_dino_kamm_mlf`; `degree_roundtrip` remains byte-identical everywhere
else and is the explicit legacy opt-in on those two cards. Required tests are
a hand-computed latitude case, a planted degree-roundtrip discriminator,
eager/JIT and finite reverse-gradient checks, and selector-scope/unchanged-card
identity pins.

If the raw-radian substitution refutes, no fix is implied: continue the
registered operand order and stop at the first substitution that closes the
row. Rows 19--32 remain unmeasured until row 18 is verified.
