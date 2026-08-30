# Split-explicit / momentum chain: vertical skew coefficient operands, round 90

Official receipt `/tmp/dino_split_explicit_momentum_chain_round90.json`,
SHA-256 `32a79283ab3ed45fbf909ffc03ed8a8afc5ccb9b82e99a5f5de6f22f9ae1dc05`,
classifies `REDI_ZFW_T_SKEW_OWNED_WSLPI_WSLPJ`.

Raw ahtu and ahtv are exact and their substitutions are bit-inert. Raw wslpi
is red in 9,360 columns (maximum column error `3.3041987541636374e-8`);
wslpj is red in 9,461 (`3.1009024512176e-8`). Neither single W-slope arm
passes: wslpi removes 10.01% and wslpj 26.56%. Their joint arm is bit-exact,
with a normalized interaction of `0.6342319998308691`. Every ahtu/ahtv arm
duplicates its no-substitution sibling, confirming the coefficients are not
part of the owner.

This narrows the carried row-30 result: the coefficient fields and
tracer-level slope consumers remain certified, while the W-point slope pair
entering `traldf_iso_scheme.h90:119-125` is the coupled owner.
