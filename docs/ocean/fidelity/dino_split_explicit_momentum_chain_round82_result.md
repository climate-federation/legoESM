# Split-explicit / momentum chain: over-broad Kmm slope carry, round 82

Receipt `/tmp/dino_split_explicit_momentum_chain_round82.json`, SHA-256
`4177c99481bdbc6996477dad848d1e645ada3a26a08fb9aa1e9dd9e4b20f9e7a`,
is invalid. It made final uslp exact and reduced zfu to a 60-column,
`6.2838858e-15` residue, but regressed certified tracer-entry row 8.8 in
6,024/9,758 columns (`1.6021126e-7`). The carry had been applied both to the
Treguier-kappa/bolus slope consumer and to the later Redi tensor. The certified
row-8 path forbids shipping that coupled change. Round 83 byte-pins the former
and retains Kmm eta only for the Redi explicit/implicit tensor.
