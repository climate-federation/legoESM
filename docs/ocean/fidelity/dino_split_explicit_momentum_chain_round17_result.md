# Split-explicit momentum chain: row-1.2 seed peel, round 17

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

The existing-dump source-ordered reduction using NOW SSH did not clear the
frozen bar. Its artifact is
`/tmp/dino_split_explicit_momentum_chain_round17_seed.json`, SHA-256
`4702ed903c2d4e35a2ecb63c777838279536abab4390beeaf4bfe7e85026ca03`.
The literal arm has U `E=2.2138950837755547e-16`, maximum/NEMO-RMS
`5.5743163337299506e-15`, and V `E=2.6482170558580063e-16`, maximum
`6.396341264411133e-15`; disposition is `INPUT_OR_TIME_LEVEL_OPEN`.

Source inspection resolves the next discriminator: MLF `restart.F90:331-352`
does not read a persistent barotropic restart, and `istate.F90:149-155`
reconstructs the Kbb barotropic seed with the BEFORE thickness, hence `sshb`.
Round 17 used `sshn`. Round 18 freezes that one-variable correction before
measurement. Rows 1.4 and 2--6 remain ordered-blocked.
