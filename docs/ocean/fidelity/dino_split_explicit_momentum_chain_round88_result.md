# Split-explicit / momentum chain: vertical Redi skew operand peel, round 88

Official receipt `/tmp/dino_split_explicit_momentum_chain_round88.json`,
SHA-256 `9107fa743aa3deb6b4f7e2180711f956599f099b908e8eb87cf36bd1bd76bd66`,
classifies `REDI_ZFW_T_SKEW_LOCALIZED_TO_COEFFICIENT_ASSEMBLY`.

The baseline skew flux is red in 9,462/9,920 wet columns with maximum column
error `7.735372278997023e-8`. Rebuilding zA31/zA32 from the existing NEMO
ahtu/ahtv, wslpi/wslpj, metrics, and masks in source order removes
`0.9999999713918843` of that error, leaving 21 red columns at
`2.212944245127673e-15`. NEMO's pair-pair four-gradient association then
closes the faithful C1/G1/T0 arm bit-exactly: 0/9,920 and zero maximum error.
The association-only arm is inert. Kmm/now gradients instead increase the
maximum error to `1.7631274738694178e-2`, independently confirming the Kbb
gradient time level.

The production change is therefore the coupled source-literal zA31/zA32
assembly and final four-gradient grouping at `traldf_iso_scheme.h90:109-125`.
It is selected only by the two DINO NEMO cards; `normalized_sums` remains the
generic byte-pinned default. The default/explicit-default equality and planted
literal-revert tests pass.
