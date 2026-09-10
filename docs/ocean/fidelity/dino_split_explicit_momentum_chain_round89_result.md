# Split-explicit / momentum chain: vertical skew production postfix, round 89

Official receipt `/tmp/dino_split_explicit_momentum_chain_round89.json`,
SHA-256 `555520203984430e988c075646e7feb674cc522de33f5cadfc4052ccdc560a76`,
classifies `REDI_ZFW_T_SKEW_POSTFIX_REGRESSION`. All controls pass, including
the explicit receipt that `vertical_skew_evaluation="nemo_literal"` reached
the captured production operator. The skew metric remains the round-87 value:
9,462/9,920 red columns and maximum column error `7.735372278997023e-8`.

This does not refute round 88's exact C1/G1/T0 counterfactual. It shows that
round 88's registered C factor bundled the source arithmetic with substitution
of NEMO's held ahtu, ahtv, wslpi, and wslpj. Source association alone is
necessary but insufficient. The next offline peel separates those four held
operands; no new NEMO run is justified.
