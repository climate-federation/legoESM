# Split-explicit / momentum chain: coupled Redi/bolus slope regression, round 83

Receipt `/tmp/dino_split_explicit_momentum_chain_round83.json`, SHA-256
`f759087db5680d8ee13f8053000812e7b0a89471f8efc28196f2ead0006f8bb3`,
is invalid. Rows 8.3--8.7 remained AT-BAR, but the carried Kmm slope geometry
still moved row 8.8 (`pu_bolus`) in 5,988/9,758 columns, maximum normalized
error `1.6024299e-7`. The low-level `nemo_iso_lap` operator used one native
slope tuple for both the later Redi tensor and the earlier through-FCT bolus
transport. Splitting only the adaptive-kappa slope consumer was therefore
insufficient. The 60-column `6.2838858e-15` zfu residue is not admissible until
the certified bolus row is restored.
